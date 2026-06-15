"""
CityMind  -  Simulation Engine  (simulation_engine.py)
=======================================================
Runs the full 20-step integrated simulation demonstrating how
all 5 challenges work together on the SAME shared CityGraph.

Simulation flow per the project statement:
  - Step 0 (setup):
      C1 generates the city layout  (CSP backtracking)
      C2 builds the road network    (Modified Prim's MST)
      C3 places ambulances          (Simulated Annealing)
      C5 predicts crime risk        (KNN) and updates graph weights
  - Steps 1..20:
      Each step picks random civilians needing rescue.
      Every 4th step, random roads flood (removed from graph).
      C4 dynamically routes the ambulance using A*.
      Every 5th step, C3 re-optimises ambulance positions.

The shared graph is the SINGLE SOURCE OF TRUTH:
  - When C5 updates risk indices, C4 sees higher travel costs.
  - When C4 blocks a road (flood), C3's next re-optimisation
    accounts for the missing edge.

Standard-library only: random.
"""

import random
from city_model import CityGraph, LocationType
from csp_layout import CityLayoutCSP
from road_network import RoadNetworkOptimizer
from ambulance_placement import AmbulancePlacementSA
from dynamic_router import DynamicEmergencyRouter, choose_emergency_start_node
from crime_risk import CrimeRiskPredictor


class SimulationEngine:
    """
    Orchestrates the 20-step CityMind simulation.

    Usage:
        engine = SimulationEngine(rows=10, cols=10, seed=42)
        logs   = engine.run(steps=20)
        for line in logs:
            print(line)
    """

    def __init__(self, rows=10, cols=10, seed=42):
        self.seed = seed
        self.rng  = random.Random(seed)
        self.graph = CityGraph(rows=rows, cols=cols)
        self.logs = []

        # Module references (populated during setup)
        self.optimizer      = None
        self.sa_solver      = None
        self.router         = None
        self.risk_predictor = None
        self.last_risk_summary = None
        self.last_police_deployment = {}
        self.protected_edges = set()

    def log(self, message):
        """Append a message to the simulation log."""
        self.logs.append(message)
        print("  " + message)

    # ------------------------------------------------------------------ #
    #  Setup — run Challenges 1, 2, 3, 5
    # ------------------------------------------------------------------ #

    def setup(self):
        """
        Initialise the city by running all foundational challenges.

        Order matters:
          C1 first (layout must exist before roads).
          C2 second (road network defines connectivity for C3, C4).
          C5 third (risk predictions affect C3 and C4 cost weights).
          C3 fourth (ambulance placement uses risk-adjusted graph).
        """
        print("\n" + "=" * 60)
        print("  CityMind — Setting up integrated simulation")
        print("=" * 60)

        # --- Challenge 1: City Layout (CSP) ---
        csp = CityLayoutCSP(self.graph, seed=self.seed)
        if not csp.solve():
            raise RuntimeError(
                "Challenge 1 FAILED: " + str(csp.conflict_reason)
            )
        violations = csp.get_violations()
        self.log("C1 complete: city layout generated ({} violations).".format(
            len(violations)
        ))

        # --- Challenge 2: Road Network (Modified Prim's) ---
        self.optimizer = RoadNetworkOptimizer(self.graph)
        if not self.optimizer.optimize():
            raise RuntimeError("Challenge 2 FAILED: road optimisation error.")
        self.log("C2 complete: {} roads built, cost={:.1f}".format(
            len(self.optimizer.built_roads),
            self.optimizer.calculate_total_cost()
        ))
        # Keep safety corridor edges protected from random floods.
        self.protected_edges = set(self.optimizer.path1_edges) | set(self.optimizer.path2_edges)

        # --- Challenge 5: Crime Risk (KNN) — before C3 so costs are updated ---
        self.risk_predictor = CrimeRiskPredictor(self.graph, seed=self.seed)
        preds = self.risk_predictor.train_and_predict()
        self.last_risk_summary = self.risk_predictor.summary()
        high_risk = 0
        for p in preds.values():
            if p["label"] == "HIGH":
                high_risk += 1
        self.graph.set_risk_cost_weight(0.35)
        self.last_police_deployment = self.risk_predictor.allocate_police_officers(total_officers=10)
        self.log(
            "C5 complete: risk model trained, high-risk={}, accuracy={:.1%}".format(
                high_risk, self.last_risk_summary["test_accuracy"]
            )
        )
        self.log(
            "C5 deployment: {} officers assigned across {} locations".format(
                sum(self.last_police_deployment.values()),
                len(self.last_police_deployment)
            )
        )

        # --- Challenge 3: Ambulance Placement (SA) ---
        self.sa_solver = AmbulancePlacementSA(
            self.graph, n_ambulances=3, seed=self.seed
        )
        self.sa_solver.solve()
        self.log("C3 complete: worst-case response={:.2f}".format(
            self.sa_solver.best_cost
        ))

        # --- Challenge 4: Router initialisation ---
        self.router = DynamicEmergencyRouter(self.graph)

    # ------------------------------------------------------------------ #
    #  Simulation helpers
    # ------------------------------------------------------------------ #

    def _random_civilians(self, count=3):
        """Pick random residential nodes as civilians needing rescue."""
        residential = self.graph.get_nodes_of_type(LocationType.RESIDENTIAL)
        if not residential:
            return []
        sample_size = min(count, len(residential))
        return self.rng.sample(residential, sample_size)

    def _random_floods(self, count=2):
        """Pick random existing edges to flood (block)."""
        edges = set()
        for u, nbrs in self.graph.adjacency.items():
            for v, _cost in nbrs:
                edge = (min(u, v), max(u, v))
                if edge in self.protected_edges:
                    continue
                edges.add(edge)
        edges = list(edges)
        self.rng.shuffle(edges)
        return edges[:min(count, len(edges))]

    # ------------------------------------------------------------------ #
    #  Main simulation loop
    # ------------------------------------------------------------------ #

    def run(self, steps=20):
        """
        Execute the integrated simulation.

        Each step:
          1. Pick random civilians.
          2. Every 4th step, flood random roads.
          3. Route ambulance to civilians (C4 — Dynamic A*).
          4. Every 5th step, re-optimise ambulance positions (C3 — SA).

        Returns list of log messages.
        """
        if self.router is None:
            self.setup()

        print("\n" + "=" * 60)
        print("  CityMind — Running {}-step simulation".format(steps))
        print("=" * 60)

        start = choose_emergency_start_node(self.graph)
        pending_civilians = set()

        # Track active floods so they can be UNBLOCKED after a few steps.
        # Each entry is (expiry_step, list_of_flooded_edges).
        # Floods are temporary (clear after 3 steps) — this prevents
        # irreversible graph degradation over a 20-step simulation.
        active_floods = []

        for step in range(1, steps + 1):

            # --- Unblock expired floods (roads reopen after repair) ---
            for expire_step, flood_edges in list(active_floods):
                if step >= expire_step:
                    for u, v in flood_edges:
                        self.graph.unblock_road(u, v)
                    active_floods.remove((expire_step, flood_edges))
                    self.log("Step {:02d}: Flood cleared, {} roads reopened".format(
                        step, len(flood_edges)
                    ))

            # Pick random civilians
            civilians = self._random_civilians(3)
            pending_civilians.update(civilians)

            # Every 4th step: simulate flooding
            floods = []
            if step % 4 == 0:
                floods = self._random_floods(2)
            flood_events = {}
            if floods:
                flood_events[0] = floods[:1]
                if len(floods) > 1:
                    flood_events[1] = floods[1:]

            # Record new floods with expiry (clear after 3 steps)
            if floods:
                active_floods.append((step + 3, floods))

            # --- Challenge 4: Dynamic routing ---
            route_result = self.router.route_to_nearest_civilians(
                start_id=start,
                civilian_ids=sorted(pending_civilians),
                flood_events=flood_events,
            )

            if route_result["success"]:
                n_visited = len(route_result["visited"])
                n_skipped = len(route_result.get("skipped_civilians", []))
                msg = "Step {:02d}: Routed to {} civilians, cost={:.2f}".format(
                    step, n_visited, route_result["total_cost"]
                )
                if n_skipped > 0:
                    msg += ", {} unreachable (skipped)".format(n_skipped)
                if floods:
                    msg += ", floods={}".format(len(floods))
                if route_result.get("remaining_civilians"):
                    msg += ", pending={}".format(len(route_result["remaining_civilians"]))
                self.log(msg)
                pending_civilians = set(route_result.get("remaining_civilians", []))

                # Update start position for next step
                if route_result.get("visited"):
                    start = route_result["visited"][-1]["target"]
            else:
                self.log(
                    "Step {:02d}: Routing FAILED after {} visits, floods={}".format(
                        step,
                        len(route_result.get("visited", [])),
                        len(floods)
                    )
                )
                pending_civilians = set(route_result.get("remaining_civilians", []))

            # --- Every 5th step: re-optimise ambulance placement AND police ---
            if step % 5 == 0:
                self.sa_solver = AmbulancePlacementSA(
                    self.graph, n_ambulances=3, seed=self.seed + step
                )
                self.sa_solver.solve()
                self.log("Step {:02d}: Re-optimised ambulances, worst={:.2f}".format(
                    step, self.sa_solver.best_cost
                ))
                self.last_police_deployment = self.risk_predictor.allocate_police_officers(total_officers=10)
                self.log(
                    "Step {:02d}: Police redeployed across {} hotspots".format(
                        step, len(self.last_police_deployment)
                    )
                )

        print("\n" + "=" * 60)
        print("  Simulation complete — {} steps executed.".format(steps))
        print("=" * 60)
        return self.logs

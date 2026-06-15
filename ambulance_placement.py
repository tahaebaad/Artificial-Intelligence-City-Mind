"""
CityMind  -  Challenge 3: Ambulance Placement  (ambulance_placement.py)
=======================================================================
Algorithm : Simulated Annealing (SA)

WHY Simulated Annealing?
  The problem is a p-centre optimisation: place N ambulances such that
  the WORST-CASE response time (max distance from any residential node
  to its nearest ambulance) is minimised.

  The search space is  C(|nodes|, N)  which grows combinatorially.
  SA is ideal because:
    1. It needs NO gradient — just a cost function.
    2. It escapes local optima by probabilistically accepting worse
       solutions (controlled by temperature).
    3. It converges to near-optimal with a simple cooling schedule.

Cooling schedule:
    T(t+1)  =  alpha  *  T(t)       (geometric cooling)
    We stop when  T < T_min  or  iterations > max_iter.

Neighbour generation:
    Pick one random ambulance, move it to a random new node.
    This is the simplest possible neighbour — easy to explain in viva.

Acceptance criterion:
    If  delta < 0  (better solution), always accept.
    Otherwise accept with probability  exp(-delta / T).
    This is the standard Metropolis criterion.

Standard-library only: math, random, heapq.
"""

import math
import random
import heapq
from city_model import CityGraph, LocationType


# ---------------------------------------------------------------------------
#  Dijkstra shortest-path  (weighted, single-source)
# ---------------------------------------------------------------------------

def dijkstra(graph, source_id):
    """
    Compute shortest weighted distances from source_id to all reachable nodes.

    Returns dict {node_id: distance}.

    How it works:
      - Priority queue (min-heap) of (distance, node).
      - Pop smallest, relax neighbours.
      - Each node is processed at most once.

    Used here to measure response time from an ambulance to every
    residential node.
    """
    dist = {source_id: 0.0}
    pq = [(0.0, source_id)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue                   # stale entry
        for v, cost in graph.adjacency[u]:
            new_d = d + cost
            if new_d < dist.get(v, math.inf):
                dist[v] = new_d
                heapq.heappush(pq, (new_d, v))
    return dist


# ---------------------------------------------------------------------------
#  Cost function  (the objective SA tries to minimise)
# ---------------------------------------------------------------------------

def worst_case_response(graph, placement, residential_ids):
    """
    Compute the worst-case response time for a given ambulance placement.

    Worst-case = the MAXIMUM shortest distance from any residential
                 node to its NEAREST ambulance.

    Lower is better.

    Parameters
    ----------
    graph           : CityGraph
    placement       : list of ambulance node ids
    residential_ids : list of residential node ids

    Returns
    -------
    float : the worst-case distance (infinity if a node is unreachable)
    """
    if not residential_ids:
        return 0.0

    # Dijkstra from each ambulance position
    all_dists = []
    for amb_id in placement:
        all_dists.append(dijkstra(graph, amb_id))

    worst = 0.0
    for rid in residential_ids:
        # Distance from this residential node to its nearest ambulance
        nearest = math.inf
        for d in all_dists:
            dist_val = d.get(rid, math.inf)
            if dist_val < nearest:
                nearest = dist_val
        if nearest > worst:
            worst = nearest
    return worst


# ---------------------------------------------------------------------------
#  Simulated Annealing Solver
# ---------------------------------------------------------------------------

class AmbulancePlacementSA:
    """
    Solves the p-centre ambulance placement using Simulated Annealing.

    Parameters
    ----------
    graph        : shared CityGraph
    n_ambulances : how many ambulances to place (default 3)
    T_init       : initial temperature (higher = more exploration)
    T_min        : stopping temperature
    alpha        : cooling rate  (T *= alpha each iteration)
    max_iter     : hard cap on iterations
    seed         : random seed for reproducibility
    callback     : function(iteration, temperature, current_cost, best_cost)
                   called every iteration — used by the GUI progress bar
    """

    def __init__(self, graph, n_ambulances=3,
                 T_init=50.0, T_min=0.05, alpha=0.995,
                 max_iter=8000, seed=None, callback=None):
        self.graph        = graph
        self.n_ambulances = n_ambulances
        self.T_init       = T_init
        self.T_min        = T_min
        self.alpha        = alpha
        self.max_iter     = max_iter
        self.callback     = callback

        # Use an instance-level RNG so we don't pollute the global
        # random state. Other modules (simulation engine, CSP) may
        # rely on their own Random instances.
        self.rng = random.Random(seed)

        # Pre-cache node lists (saves time during iterations)
        self.residential_ids = graph.get_nodes_of_type(LocationType.RESIDENTIAL)
        self.all_ids         = graph.all_node_ids()

        # Results (filled after solve())
        self.best_placement  = None
        self.best_cost       = math.inf
        self.cost_history    = []       # [(iteration, current_cost, best_cost)]
        self.temperature_log = []
        self.iterations_run  = 0

    # ── Main solver ─────────────────────────────────────────────────── #

    def solve(self):
        """
        Run Simulated Annealing and return the best placement found.

        Returns list of node ids where ambulances should be placed.
        """
        # --- Initial random placement ---
        current = self.rng.sample(self.all_ids, self.n_ambulances)

        # Filter residential nodes to only those with at least one road.
        # During a flood, some nodes may be temporarily disconnected;
        # including them would inflate worst_cost to infinity. (#17)
        reachable_res = [r for r in self.residential_ids
                         if self.graph.adjacency[r]]
        if not reachable_res:
            reachable_res = list(self.residential_ids)  # fallback

        current_cost = worst_case_response(
            self.graph, current, reachable_res
        )

        self.best_placement = list(current)
        self.best_cost      = current_cost
        self.cost_history   = [(0, current_cost, current_cost)]

        T = self.T_init
        iteration = 0

        while T > self.T_min and iteration < self.max_iter:
            iteration += 1

            # --- Generate a NEIGHBOUR ---
            # Pick one ambulance at random and move it to a new random node.
            idx = self.rng.randrange(self.n_ambulances)
            new_node = self.rng.choice(self.all_ids)
            # Make sure we don't place two ambulances on the same node
            while new_node in current:
                new_node = self.rng.choice(self.all_ids)

            neighbour = list(current)
            neighbour[idx] = new_node

            # Evaluate the neighbour
            neighbour_cost = worst_case_response(
                self.graph, neighbour, reachable_res
            )

            # --- ACCEPTANCE CRITERION (Metropolis) ---
            delta = neighbour_cost - current_cost
            if delta < 0:
                # Better solution → always accept
                accept = True
            else:
                # Worse solution → accept with probability exp(-delta/T)
                probability = math.exp(-delta / T)
                accept = (self.rng.random() < probability)

            if accept:
                current      = neighbour
                current_cost = neighbour_cost

            # Track global best
            if current_cost < self.best_cost:
                self.best_cost      = current_cost
                self.best_placement = list(current)

            # --- COOL DOWN ---
            T = T * self.alpha        # geometric cooling: T(t+1) = alpha * T(t)

            # Logging
            self.cost_history.append((iteration, current_cost, self.best_cost))
            self.temperature_log.append(T)

            if self.callback:
                self.callback(iteration, T, current_cost, self.best_cost)

        self.iterations_run = iteration
        return self.best_placement

    # ── Coverage analysis ─────────────────────────────────────────── #

    def coverage_breakdown(self):
        """
        For the best placement found, compute each node's distance
        to its nearest ambulance.  Returns dict {node_id: distance}.

        Always recomputes fresh Dijkstra distances so results reflect
        the current graph state (e.g. after floods block roads).
        """
        if self.best_placement is None:
            return {}

        all_dists = [dijkstra(self.graph, a) for a in self.best_placement]
        result = {}
        for nid in self.all_ids:
            nearest = math.inf
            for d in all_dists:
                val = d.get(nid, math.inf)
                if val < nearest:
                    nearest = val
            result[nid] = nearest
        return result

    def nearest_ambulance(self, node_id):
        """Return the index of the ambulance closest to node_id."""
        if not self.best_placement:
            return 0
        all_dists = [dijkstra(self.graph, a) for a in self.best_placement]
        dists = [d.get(node_id, math.inf) for d in all_dists]
        return dists.index(min(dists))

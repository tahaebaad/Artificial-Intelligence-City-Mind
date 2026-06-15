"""
CityMind  -  Challenge 1: City Layout Planning  (csp_layout.py)
================================================================
Algorithm : Backtracking + Constraint Satisfaction Problem (CSP)

WHY CSP?
  Placing facilities on a grid with neighbourhood rules is a classic
  constraint-satisfaction problem.  Each empty cell is a VARIABLE,
  the facility types are the DOMAIN, and the urban-planning rules are
  the CONSTRAINTS.  Backtracking systematically explores assignments
  and prunes dead-ends early.

Constraints implemented (from the project statement):
  C1-R1 : Industrial zones CANNOT be adjacent to Hospitals or Schools.
  C1-R2 : Every Residential area must be within 3 road-hops of at
           least one Hospital.
  C1-R3 : Every Power Plant must be within 2 road-hops of at least
           one Industrial zone.
  C1-R4 : If no valid layout exists, report WHICH rule is violated
           and propose a minimum-conflict repair.

Optimisations:
  MRV  (Minimum Remaining Values) selects the facility type with
  the fewest valid candidate cells first — this fails early and
  reduces backtracking.

  Forward checking after each placement ensures remaining types
  still have enough candidates.

Standard-library only: random, time.
"""

import random
import time
from city_model import CityGraph, LocationType


# How many of each facility type to place on the grid.
# Remaining cells become Residential automatically.
PLACEMENT_CONFIG = {
    LocationType.INDUSTRIAL:       6,
    LocationType.POWER_PLANT:      4,
    LocationType.HOSPITAL:         5,
    LocationType.SCHOOL:           5,
    LocationType.AMBULANCE_DEPOT:  3,
}


class CityLayoutCSP:
    """
    Solves the city-layout placement problem using backtracking CSP.

    Usage:
        graph = CityGraph(10, 10)
        csp   = CityLayoutCSP(graph, seed=42)
        ok    = csp.solve()                # True if a valid layout was found
        viols = csp.get_violations()       # list of human-readable strings
    """

    def __init__(self, graph, seed=None):
        """
        Parameters
        ----------
        graph : CityGraph   — the shared city graph (modified in-place)
        seed  : int or None — random seed for reproducibility
        """
        self.graph = graph
        self.config = dict(PLACEMENT_CONFIG)   # copy so we don't mutate global
        self.conflict_reason = None            # set if solve() fails
        # Use instance-level RNG to avoid polluting global random state.
        # Other modules (SimulationEngine, AmbulancePlacementSA) use their
        # own Random instances; a global seed() call here would interfere.
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def solve(self):
        """
        Run the CSP solver.

        Steps:
        1. Clear the grid (set everything to EMPTY).
        2. Use backtracking + MRV to place all facilities.
        3. Fill leftover cells with Residential.
        4. Check coverage constraints (C1-R2); repair if needed.

        Returns True on success, False on failure.
        """
        print("\n[CSP] Starting backtracking CSP layout search...")
        t0 = time.time()

        # Step 1 — blank slate
        self._clear_graph()

        # Step 2 — backtracking placement of facilities
        remaining = dict(self.config)
        solved = self._backtrack_place(remaining)
        if not solved:
            self.conflict_reason = (
                "Backtracking exhausted all options — no feasible placement "
                "exists for the requested facility counts on this grid size."
            )
            return False

        # Step 3 — every remaining EMPTY cell becomes Residential
        self._fill_residential()

        # Step 4 — verify hospital-coverage constraint (C1-R2)
        ok, msg = self._check_coverage()
        if not ok:
            # Attempt automatic repair by converting a well-placed
            # Residential cell into a Hospital.
            ok, msg = self._repair_coverage()

        if ok:
            elapsed = time.time() - t0
            print("[CSP] Solution found in {:.3f}s".format(elapsed))
            return True

        self.conflict_reason = msg or "Coverage repair failed after CSP placement."
        return False

    # ------------------------------------------------------------------ #
    #  Backtracking internals
    # ------------------------------------------------------------------ #

    def _clear_graph(self):
        """Reset every node to EMPTY with population 0."""
        for nid in self.graph.all_node_ids():
            self.graph.set_location(nid, LocationType.EMPTY, pop_density=0)

    def _backtrack_place(self, remaining):
        """
        Recursive backtracking.

        remaining : dict {LocationType: count_still_to_place}

        Base case : all counts are 0 -> success.
        Recursive : pick the type with fewest candidates (MRV),
                    try each candidate cell, recurse.
        """
        # Base case: nothing left to place
        total_left = 0
        for count in remaining.values():
            total_left += count
        if total_left == 0:
            return True

        # MRV heuristic: choose the facility type with fewest valid cells
        choice = self._select_var_mrv(remaining)
        if choice is None:
            return False              # no type has any valid candidate

        loc_type, candidates = choice
        if not candidates:
            return False

        # Try each candidate cell
        for nid in candidates:
            old_pop = self.graph.nodes[nid].population_density

            # Place the facility
            self.graph.set_location(nid, loc_type,
                                    pop_density=self.rng.randint(10, 60))
            remaining[loc_type] -= 1

            # Forward check: do remaining types still have enough room?
            if self._forward_check(remaining):
                if self._backtrack_place(remaining):
                    return True        # propagate success

            # Undo (backtrack)
            remaining[loc_type] += 1
            self.graph.set_location(nid, LocationType.EMPTY,
                                    pop_density=old_pop if old_pop else 0)

        return False                   # all candidates failed

    def _select_var_mrv(self, remaining):
        """
        MRV (Minimum Remaining Values) heuristic.

        Among all facility types that still need placement, pick the one
        with the FEWEST valid candidate cells.  This causes failure early
        and dramatically reduces the search tree.
        Note: this MRV counts currently valid placements only; it does not
        run full arc-consistency propagation over future constraints.

        Special case: Power plants are skipped until at least one
        Industrial zone exists (they need to be within 2 hops of one).
        """
        best_type = None
        best_candidates = None
        best_count = 10 ** 9   # start with "infinity"

        for loc_type, rem_count in remaining.items():
            if rem_count <= 0:
                continue

            # Power plants require an existing industrial zone (C1-R3)
            if loc_type == LocationType.POWER_PLANT:
                if not self.graph.get_nodes_of_type(LocationType.INDUSTRIAL):
                    continue

            candidates = self._candidates_for(loc_type)
            cnt = len(candidates)
            if cnt < best_count:
                best_count = cnt
                best_type = loc_type
                best_candidates = candidates

        if best_type is None:
            return None
        return best_type, best_candidates

    def _candidates_for(self, loc_type):
        """Return a shuffled list of EMPTY cells where loc_type can be placed."""
        candidates = []
        for nid in self.graph.all_node_ids():
            if self.graph.nodes[nid].location_type != LocationType.EMPTY:
                continue
            if self._is_valid_placement(nid, loc_type):
                candidates.append(nid)
        self.rng.shuffle(candidates)     # randomise for variety across seeds
        return candidates

    def _forward_check(self, remaining):
        """
        After placing one facility, verify that every remaining type
        still has at least as many valid candidate cells as it needs.
        If any type is 'starved', prune this branch immediately.

        Performance note: This calls _candidates_for() for each remaining
        type, which runs BFS per candidate.  On a 10×10 grid this is
        fast (~0.05s total), but on larger grids (e.g. 15×15) it could
        become noticeably slow.  The fix would be incremental constraint
        propagation (arc consistency) rather than recomputing all
        candidates from scratch each time.
        """
        for loc_type, rem_count in remaining.items():
            if rem_count <= 0:
                continue
            # Skip power-plant check when no industrial zone exists yet
            if loc_type == LocationType.POWER_PLANT:
                if not self.graph.get_nodes_of_type(LocationType.INDUSTRIAL):
                    continue
            if len(self._candidates_for(loc_type)) < rem_count:
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Constraint checks  (readable and viva-defensible)
    # ------------------------------------------------------------------ #

    def _is_valid_placement(self, nid, loc_type):
        """
        Check whether placing loc_type at nid violates any hard constraint.

        C1-R1 : Industrial CANNOT be adjacent to Hospital or School.
                (and vice-versa: Hospital/School cannot be adjacent to Industrial)
        C1-R3 : Power Plant must be within 2 hops of at least one Industrial.
        """
        neighbours = self.graph.get_adjacent_ids(nid)

        # --- Rule 1: Industrial vs Hospital/School adjacency ---
        if loc_type == LocationType.INDUSTRIAL:
            for nbr_id in neighbours:
                nbr_type = self.graph.nodes[nbr_id].location_type
                if nbr_type == LocationType.HOSPITAL or nbr_type == LocationType.SCHOOL:
                    return False     # VIOLATION: Industrial next to Hospital/School

        if loc_type == LocationType.HOSPITAL or loc_type == LocationType.SCHOOL:
            for nbr_id in neighbours:
                if self.graph.nodes[nbr_id].location_type == LocationType.INDUSTRIAL:
                    return False     # VIOLATION: Hospital/School next to Industrial

        # --- Rule 3: Power Plant within 2 hops of Industrial ---
        if loc_type == LocationType.POWER_PLANT:
            industrial_ids = self.graph.get_nodes_of_type(LocationType.INDUSTRIAL)
            if not industrial_ids:
                return False         # no industrial zone placed yet
            hop_distances = self.graph.bfs_distances(nid)
            found_close_industrial = False
            for ind_id in industrial_ids:
                if hop_distances.get(ind_id, 999) <= 2:
                    found_close_industrial = True
                    break
            if not found_close_industrial:
                return False         # VIOLATION: too far from any industrial

        return True                  # all constraints satisfied

    # ------------------------------------------------------------------ #
    #  Residential fill + coverage
    # ------------------------------------------------------------------ #

    def _fill_residential(self):
        """Every remaining EMPTY cell becomes Residential with random population."""
        for nid in self.graph.all_node_ids():
            if self.graph.nodes[nid].location_type == LocationType.EMPTY:
                pop = self.rng.randint(20, 100)
                self.graph.set_location(nid, LocationType.RESIDENTIAL,
                                        pop_density=pop)

    def _check_coverage(self):
        """
        C1-R2: Every Residential must be within 3 hops of a Hospital.

        Returns (True, "") if satisfied, or (False, reason_string).
        """
        hospital_ids = self.graph.get_nodes_of_type(LocationType.HOSPITAL)
        if not hospital_ids:
            return False, "No hospitals were placed on the grid."

        # Build a coverage map: node_id -> min hops to ANY hospital
        coverage = {}
        for hid in hospital_ids:
            for nid, d in self.graph.bfs_distances(hid).items():
                if nid not in coverage or d < coverage[nid]:
                    coverage[nid] = d

        # Find residential nodes that are too far
        uncovered = []
        for nid, node in self.graph.nodes.items():
            if node.location_type == LocationType.RESIDENTIAL:
                if coverage.get(nid, 999) > 3:
                    uncovered.append(nid)

        if uncovered:
            return False, "{} residential cells are >3 hops from any hospital.".format(len(uncovered))
        return True, ""

    def _repair_coverage(self):
        """
        Minimum-conflict repair: convert the best-placed Residential cell
        into a Hospital so that the maximum number of uncovered cells
        become covered.  Repeat up to 10 rounds.

        This satisfies C1-R4 ("propose minimum conflict solution").
        After each conversion, coverage is rechecked globally.
        """
        for _round in range(10):
            ok, msg = self._check_coverage()
            if ok:
                return True, ""

            # Identify currently uncovered residential nodes
            hospital_ids = self.graph.get_nodes_of_type(LocationType.HOSPITAL)
            coverage = {}
            for hid in hospital_ids:
                for nid, d in self.graph.bfs_distances(hid).items():
                    if nid not in coverage or d < coverage[nid]:
                        coverage[nid] = d

            uncovered = set()
            for nid, node in self.graph.nodes.items():
                if node.location_type == LocationType.RESIDENTIAL:
                    if coverage.get(nid, 999) > 3:
                        uncovered.add(nid)
            if not uncovered:
                break

            # Find the residential cell whose conversion to Hospital
            # would cover the most uncovered cells within 3 hops.
            best_nid = None
            best_score = -1
            for cand in self.graph.get_nodes_of_type(LocationType.RESIDENTIAL):
                # Don't create a new adjacency violation
                neighbours = self.graph.get_adjacent_ids(cand)
                has_industrial_neighbour = False
                for nbr_id in neighbours:
                    if self.graph.nodes[nbr_id].location_type == LocationType.INDUSTRIAL:
                        has_industrial_neighbour = True
                        break
                if has_industrial_neighbour:
                    continue   # can't put Hospital next to Industrial (C1-R1)

                # How many uncovered cells would this new hospital reach?
                hop_map = self.graph.bfs_distances(cand)
                score = 0
                for nid2, d in hop_map.items():
                    if d <= 3 and nid2 in uncovered:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_nid = cand

            if best_nid is None or best_score == 0:
                break   # no useful conversion possible

            self.graph.set_location(
                best_nid, LocationType.HOSPITAL,
                pop_density=self.graph.nodes[best_nid].population_density
            )
            # Log the conversion so the hospital count increase is visible.
            # PLACEMENT_CONFIG requested 5 hospitals, but coverage repair may
            # add more. This is the minimum-conflict repair mechanism (C1-R4).
            node_info = self.graph.nodes[best_nid]
            print("[CSP] Repair: converted Residential at ({},{}) to Hospital (covered {} uncovered cells)".format(
                node_info.row, node_info.col, best_score
            ))

        return self._check_coverage()

    # ------------------------------------------------------------------ #
    #  Violation report  (for GUI display and viva)
    # ------------------------------------------------------------------ #

    def get_violations(self):
        """
        Scan the entire grid and return a list of human-readable strings
        describing every constraint violation found.

        Useful for:
          - Showing in the GUI after layout generation
          - Defending the layout during viva
        """
        violations = []
        nodes = self.graph.nodes

        # --- C1-R1: Industrial cannot be adjacent to Hospital/School ---
        for nid, node in nodes.items():
            if node.location_type != LocationType.INDUSTRIAL:
                continue
            for nbr_id in self.graph.get_adjacent_ids(nid):
                nbr_type = nodes[nbr_id].location_type
                if nbr_type == LocationType.HOSPITAL or nbr_type == LocationType.SCHOOL:
                    violations.append(
                        "C1-R1: Industrial ({},{}) is adjacent to {} ({},{})".format(
                            node.row, node.col,
                            nbr_type.name,
                            nodes[nbr_id].row, nodes[nbr_id].col
                        )
                    )

        # --- C1-R2: Residential within 3 hops of a Hospital ---
        hospitals = self.graph.get_nodes_of_type(LocationType.HOSPITAL)
        coverage = {}
        for hid in hospitals:
            for nid, dist in self.graph.bfs_distances(hid).items():
                if nid not in coverage or dist < coverage[nid]:
                    coverage[nid] = dist
        for nid, node in nodes.items():
            if node.location_type == LocationType.RESIDENTIAL:
                d = coverage.get(nid, 999)
                if d > 3:
                    violations.append(
                        "C1-R2: Residential ({},{}) is {} hops from nearest hospital".format(
                            node.row, node.col, d
                        )
                    )

        # --- C1-R3: Power Plant within 2 hops of an Industrial zone ---
        industrial_ids = self.graph.get_nodes_of_type(LocationType.INDUSTRIAL)
        for pid in self.graph.get_nodes_of_type(LocationType.POWER_PLANT):
            hop_map = self.graph.bfs_distances(pid)
            close_enough = False
            for ind_id in industrial_ids:
                if hop_map.get(ind_id, 999) <= 2:
                    close_enough = True
                    break
            if not close_enough:
                pnode = nodes[pid]
                violations.append(
                    "C1-R3: Power plant ({},{}) is not within 2 hops of any industrial zone".format(
                        pnode.row, pnode.col
                    )
                )

        return violations

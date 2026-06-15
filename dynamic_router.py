"""
CityMind  -  Challenge 4: Dynamic Emergency Routing  (dynamic_router.py)
=========================================================================
Algorithm : Dynamic A* Search

WHY A*?
  A* is an informed search algorithm that finds the SHORTEST path
  while exploring fewer nodes than Dijkstra by using a heuristic.

  It is "dynamic" because we re-run A* after every road change
  (flood / blockage).  The moment a road becomes impassable, it is
  removed from the shared CityGraph, and the NEXT A* call
  automatically avoids it — no stale data.

Heuristic: Manhattan distance  (admissible on a grid with non-negative costs)
  h(n) = |n.row - goal.row| + |n.col - goal.col|
  Since minimum edge cost is 0.8, Manhattan distance (using unit steps)
  never overestimates the true cost → A* is OPTIMAL.

TA Requirement — graceful handling of unreachable civilians:
  "If a road floods and completely cuts off a civilian (making them
   mathematically unreachable), the system must NOT crash or loop.
   It must log the failure gracefully and route to the next civilian."

  Implementation: when A* returns None for a target, we LOG the failure
  and SKIP that civilian, continuing to the next one in the queue.

Standard-library only: heapq.
"""

import heapq
from city_model import LocationType


class DynamicEmergencyRouter:
    """
    Routes a medical team through the city to reach trapped civilians.

    Key features:
      - Uses A* with Manhattan heuristic (optimal, admissible).
      - Re-computes paths dynamically after each flood event.
      - Gracefully skips unreachable civilians instead of crashing.
    """

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : CityGraph  — the shared city graph (read from, never copied)
        """
        self.graph = graph
        self.events = []        # human-readable log of routing decisions

    def reset_events(self):
        """Clear the event log for a fresh routing session."""
        self.events = []

    # ------------------------------------------------------------------ #
    #  A* Search
    # ------------------------------------------------------------------ #

    def _heuristic(self, node_id, goal_id):
        """
        Manhattan-distance heuristic.

        Why Manhattan?
          On a 4-connected grid, the shortest possible path between
          two nodes uses |dr| + |dc| steps.  Since the cheapest edge
          cost is 0.8 (residential), multiplying by 0.8 would make
          the heuristic tighter, but even using 1.0 per step keeps it
          admissible (never overestimates) because actual costs >= 0.8.

          Wait — actually h=|dr|+|dc| CAN overestimate when the true
          optimal path goes through all-residential nodes at cost 0.8
          each.  To be safe and GUARANTEE optimality, we scale by the
          minimum possible edge cost.

        Admissibility proof:
          h(n) = (|dr| + |dc|) * 0.8  <=  true cost  (always)
        """
        n1 = self.graph.nodes[node_id]
        n2 = self.graph.nodes[goal_id]
        manhattan = abs(n1.row - n2.row) + abs(n1.col - n2.col)
        return manhattan * 0.8     # scaled by minimum edge cost for admissibility

    def shortest_path(self, start_id, end_id):
        """
        A* shortest path from start_id to end_id.

        Returns
        -------
        (path, cost) where path is a list of node ids, cost is float.
        If unreachable, returns (None, infinity).

        How A* works:
          - f(n) = g(n) + h(n)
            g(n) = cost from start to n  (known, exact)
            h(n) = heuristic estimate from n to goal  (admissible)
          - Priority queue ordered by f(n).
          - Pop node with smallest f.  If it's the goal, done.
          - Otherwise relax neighbours (update g if cheaper path found).
        """
        g_score = {start_id: 0.0}             # g(n): actual cost from start
        prev    = {start_id: None}            # predecessor map for path
        h_start = self._heuristic(start_id, end_id)
        pq      = [(h_start, 0.0, start_id)]  # (f-score, g-at-push, node_id)

        while pq:
            _f_cost, g_at_push, node = heapq.heappop(pq)

            if node == end_id:
                break                          # found the goal!

            # Stale-entry check:
            # if this heap entry's g is worse than the best known g,
            # a better path was found after it was pushed.
            if g_at_push > g_score.get(node, float("inf")):
                continue

            for nbr, edge_weight in self.graph.adjacency[node]:
                tentative_g = g_score[node] + edge_weight

                if tentative_g < g_score.get(nbr, float("inf")):
                    g_score[nbr] = tentative_g
                    prev[nbr]    = node
                    f_nbr = tentative_g + self._heuristic(nbr, end_id)
                    heapq.heappush(pq, (f_nbr, tentative_g, nbr))

        # --- Check if goal was reached ---
        if end_id not in prev:
            return None, float("inf")          # UNREACHABLE

        # --- Reconstruct path ---
        path = []
        cur = end_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path, g_score[end_id]

    # ------------------------------------------------------------------ #
    #  Dynamic dispatch with graceful unreachable handling
    # ------------------------------------------------------------------ #

    def route_to_nearest_civilians(self, start_id, civilian_ids,
                                    max_visits=None, flood_events=None):
        """
        Dynamically route an ambulance to civilians one at a time.

        At each step:
          1. Apply any pending flood events (block roads on the shared graph).
          2. From current position, find the NEAREST reachable civilian (A*).
          3. If NO civilian is reachable, LOG the failure and STOP gracefully.
          4. If a SPECIFIC civilian is unreachable, SKIP them and try others.
          5. Move to the nearest civilian, mark as visited, repeat.

        Parameters
        ----------
        start_id     : starting node id (ambulance depot or hospital)
        civilian_ids : list of node ids where civilians are trapped
        max_visits   : optional cap on civilians to visit; if None, attempt all
        flood_events : dict { step_index : [(u, v), ...] }
                       roads to block at each routing step

        Returns
        -------
        dict with keys:
          success             : bool
          visited             : list of {target, path, cost}
          total_cost          : float
          events              : list of log strings
          remaining_civilians : list of unvisited civilian ids
          skipped_civilians   : list of civilians that were unreachable
        """
        flood_events = flood_events or {}
        remaining = list(dict.fromkeys(civilian_ids))
        visits = []
        total_cost = 0.0
        current = start_id
        skipped = set()
        self.reset_events()
        max_attempts = len(remaining) if max_visits is None else min(max_visits, len(remaining))

        for visit_idx in range(max_attempts):
            # Reset movement_tick for each civilian visit so that
            # flood_events keyed at tick 0, 1, etc. fire relative to
            # THIS visit, not accumulated globally across all visits.
            movement_tick = 0

            # NOTE: Flood events are ONLY applied inside the inner
            # per-hop while loop below. This avoids the double-fire
            # bug where both the outer loop and inner loop would
            # trigger the same tick-0 flood event.

            if not remaining:
                self.events.append(
                    "Visit {}: All civilians have been reached.".format(visit_idx)
                )
                break

            # --- Find nearest REACHABLE civilian ---
            best_target = None
            best_cost   = float("inf")
            for target in remaining:
                path, cost = self.shortest_path(current, target)
                if path is not None and cost < best_cost:
                    best_target = target
                    best_cost   = cost

            if best_target is None:
                # No civilian is reachable from current position now.
                # Keep them in remaining so the caller can retry later.
                self.events.append(
                    "Visit {}: No reachable civilians from node {} (will retry later)".format(
                        visit_idx, current
                    )
                )
                return {
                    "success": False,
                    "visited": visits,
                    "total_cost": total_cost,
                    "events": list(self.events),
                    "remaining_civilians": remaining,
                    "skipped_civilians": sorted(skipped),
                }

            # --- Route to nearest civilian with mid-journey replanning ---
            # We replan A* on EVERY hop to ensure the path is always optimal
            # in the face of mid-journey floods.  If a flood fires at tick 2,
            # the next hop immediately sees the updated graph.  This is more
            # expensive than replanning only on flood events, but simpler to
            # reason about correctness.  On a 10×10 grid the cost is
            # negligible (~10 A* calls per civilian, each O(N log N)).
            routed_path = [current]
            realised_cost = 0.0
            # Safety guard: max hops = 4 × total nodes.  The theoretical
            # maximum shortest path on a 10×10 grid is 18 hops (corner to
            # corner).  4× provides a generous margin for worst-case paths
            # on larger grids or when detours are needed around floods.
            safety_hops = len(self.graph.nodes) * 4
            hop_guard = 0

            while current != best_target and hop_guard < safety_hops:
                # Apply environment changes at every movement tick.
                for u, v in flood_events.get(movement_tick, []):
                    self.graph.block_road(u, v)
                    self.events.append(
                        "Tick {}: Flood blocked road ({}, {})".format(movement_tick, u, v)
                    )

                replanned_path, _ = self.shortest_path(current, best_target)
                if replanned_path is None:
                    # Target became unreachable mid-journey; keep for retry.
                    node = self.graph.nodes[best_target]
                    self.events.append(
                        "Tick {}: Civilian at ({},{}) became unreachable mid-route; deferred.".format(
                            movement_tick, node.row, node.col
                        )
                    )
                    skipped.add(best_target)
                    break

                if len(replanned_path) < 2:
                    break

                next_node = replanned_path[1]
                edge_cost = None
                for nbr, cost in self.graph.adjacency[current]:
                    if nbr == next_node:
                        edge_cost = cost
                        break

                if edge_cost is None:
                    self.events.append(
                        "Tick {}: Planned road vanished before move; replanning.".format(movement_tick)
                    )
                    movement_tick += 1
                    hop_guard += 1
                    continue

                current = next_node
                routed_path.append(current)
                realised_cost += edge_cost
                movement_tick += 1
                hop_guard += 1

            if current != best_target:
                continue

            visits.append({
                "target": best_target,
                "path": routed_path,
                "cost": realised_cost,
            })
            remaining.remove(best_target)
            total_cost += realised_cost
            target_node = self.graph.nodes[best_target]
            self.events.append(
                "Visit {}: Routed to civilian at ({},{}) cost={:.2f}".format(
                    visit_idx, target_node.row, target_node.col, realised_cost
                )
            )

        return {
            "success": len(remaining) == 0 and len(visits) > 0,
            "visited": visits,
            "total_cost": total_cost,
            "events": list(self.events),
            "remaining_civilians": remaining,
            "skipped_civilians": sorted(skipped),
        }


def choose_emergency_start_node(graph):
    """
    Choose where the ambulance starts from.
    Priority: Ambulance Depot > Hospital > node 0 (fallback).
    """
    depots = graph.get_nodes_of_type(LocationType.AMBULANCE_DEPOT)
    if depots:
        return depots[0]
    hospitals = graph.get_nodes_of_type(LocationType.HOSPITAL)
    if hospitals:
        return hospitals[0]
    return 0

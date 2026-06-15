"""
CityMind  -  Challenge 2: Road Network Optimisation  (road_network.py)
======================================================================
Algorithm : Modified Prim's MST with edge-disjoint safety corridor

WHY Modified Prim's?
  The goal is to connect all locations with MINIMUM total road cost
  (= Minimum Spanning Tree problem).  Prim's algorithm is a greedy
  MST builder that grows the tree one cheapest edge at a time.

  The MODIFICATION is a mandatory pre-step:
    Before running Prim's, we MUST guarantee two completely independent
    (edge-disjoint) routes between the Primary Hospital and the
    Main Ambulance Depot.  This ensures that even if one road fails,
    an alternative path still exists.

TA Requirement:
  "You must write custom logic to explicitly designate one Primary
   Hospital and one Main Depot, and guarantee two completely
   independent (edge-disjoint) routes between them BEFORE running
   Prim's for the rest of the city."

Algorithm steps:
  1. Designate Primary Hospital (first Hospital placed) and
     Main Depot (first Ambulance Depot placed).
  2. Find shortest path between them (Dijkstra) → Route 1.
  3. Remove Route 1 edges from the graph, find ANOTHER shortest
     path → Route 2.  (Sequential Dijkstra for edge-disjointness.)
  4. If Route 2 exists, we have two independent safety routes.
  5. Seed Prim's MST with all nodes already on the safety routes,
     then grow the tree to reach every remaining node.
  6. Apply the optimised road set to the shared graph so all
     downstream modules use the same network.

Standard-library only: heapq.
"""

import heapq
from city_model import CityGraph, LocationType


class RoadNetworkOptimizer:
    """
    Builds an optimised road network on the shared CityGraph.

    After optimize() succeeds:
      - self.built_roads   : set of (min_id, max_id) edges
      - self.path1_edges   : edges of safety route 1
      - self.path2_edges   : edges of safety route 2
      - self.primary_hospital, self.main_depot : designated node ids
    """

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : CityGraph  — the shared city graph (modified in-place)
        """
        self.graph = graph
        self.built_roads = set()     # all edges selected for the road network
        self.path1_edges = set()     # edges of the first safety route
        self.path2_edges = set()     # edges of the second safety route
        self.primary_hospital = None
        self.main_depot = None

        # --- Designate Primary Hospital and Main Depot ---
        # TA requirement: explicitly pick one of each.
        hospitals = self.graph.get_nodes_of_type(LocationType.HOSPITAL)
        depots    = self.graph.get_nodes_of_type(LocationType.AMBULANCE_DEPOT)

        if hospitals:
            self.primary_hospital = hospitals[0]
        if depots:
            self.main_depot = depots[0]

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def optimize(self):
        """
        Build the optimised road network.

        Returns True on success, False if:
          - No hospital or depot exists
          - Hospital and depot are disconnected
          - Cannot find two independent routes
        """
        print("\n[Road Network] Starting optimization...")

        if self.primary_hospital is None or self.main_depot is None:
            print("[Road Network] Error: Missing Hospital or Depot.")
            return False

        # Sanity check: are hospital and depot even connected?
        test_path = self._dijkstra_path(
            self.primary_hospital, self.main_depot, ignore_edges=set()
        )
        if test_path is None:
            print("[Road Network] Error: Hospital and depot are disconnected.")
            return False

        h_node = self.graph.nodes[self.primary_hospital]
        d_node = self.graph.nodes[self.main_depot]
        print("[Road Network] Primary Hospital : ({}, {})".format(h_node.row, h_node.col))
        print("[Road Network] Main Depot       : ({}, {})".format(d_node.row, d_node.col))

        # --- Step 1: Find two independent (edge-disjoint) routes ---
        print("[Road Network] Finding two independent safety routes...")
        success = self._find_two_independent_routes()
        if not success:
            print("[Road Network] FAILED — could not find two independent routes.")
            return False

        print("[Road Network] Safety routes secured (Route 1: {} edges, Route 2: {} edges)".format(
            len(self.path1_edges), len(self.path2_edges)
        ))

        # --- Step 2: Prim's MST to connect the rest ---
        print("[Road Network] Running Prim's Algorithm for remaining locations...")
        self._run_prims_algorithm()

        # --- Step 3: Apply to shared graph ---
        self._apply_built_roads_to_graph()
        if not self._validate_full_connectivity():
            print("[Road Network] FAILED — pruned network is not fully connected.")
            return False

        # NOTE on total road count for viva:
        # A pure MST of N nodes has exactly N-1 edges.  Our total may be
        # slightly higher (e.g. 100 on a 100-node grid) because the two
        # edge-disjoint safety routes are added BEFORE Prim's runs.  Prim's
        # stops once all nodes are in the tree, but the safety routes may
        # have contributed edges that a pure MST would not have selected.
        # Total roads = safety_route_edges + MST_edges (with overlap removed
        # by the built_roads set).
        print("[Road Network] City connected. Total roads: {}".format(len(self.built_roads)))
        return True

    # ------------------------------------------------------------------ #
    #  Edge-disjoint routes
    # ------------------------------------------------------------------ #

    def _find_two_independent_routes(self):
        """
        Find two edge-disjoint paths between primary_hospital and main_depot.

        Method:
          1. Find shortest path P1 (Dijkstra).  Record its edges.
          2. Find shortest path P2 while IGNORING all edges of P1.
          3. If P2 exists, the two paths share no edges → independent.

        NOTE ON EDGE-DISJOINT vs VERTEX-DISJOINT:
          Our approach guarantees EDGE-DISJOINTNESS (no shared roads),
          but NOT vertex-disjointness (the paths may share intermediate
          nodes / intersections).

          The TA requirement says: "if any single road fails, an
          alternative path must remain available."  This is EXACTLY the
          definition of edge-disjointness.  Two paths may pass through
          the same intersection (node), but they will never use the
          same road (edge).  So if any one road is destroyed, the
          other route remains fully intact.

          True Suurballe's algorithm uses edge-reversal in a residual
          graph and optimises combined path cost.
          Our simplified approach uses sequential Dijkstra:
          second path ignores first-path edges to guarantee
          edge-disjointness with simpler implementation.
        """
        # Route 1 — normal shortest path
        path1 = self._dijkstra_path(
            self.primary_hospital, self.main_depot, ignore_edges=set()
        )
        if path1 is None:
            return False

        # Record Route 1 edges
        for i in range(len(path1) - 1):
            u, v = path1[i], path1[i + 1]
            edge = (min(u, v), max(u, v))
            self.path1_edges.add(edge)
            self.built_roads.add(edge)

        # Route 2 — shortest path ignoring Route 1 edges
        path2 = self._dijkstra_path(
            self.primary_hospital, self.main_depot,
            ignore_edges=self.path1_edges
        )
        if path2 is None:
            return False

        # Record Route 2 edges
        for i in range(len(path2) - 1):
            u, v = path2[i], path2[i + 1]
            edge = (min(u, v), max(u, v))
            self.path2_edges.add(edge)
            self.built_roads.add(edge)

        return True

    # ------------------------------------------------------------------ #
    #  Dijkstra (used internally for shortest-path)
    # ------------------------------------------------------------------ #

    def _dijkstra_path(self, start_id, end_id, ignore_edges):
        """
        Standard Dijkstra's shortest-path on the shared graph.

        Parameters
        ----------
        start_id     : source node
        end_id       : destination node
        ignore_edges : set of (min, max) edge tuples to skip

        Returns
        -------
        list of node ids forming the path, or None if unreachable.

        How it works:
          - Maintain a priority queue (min-heap) of (distance, node).
          - Always pop the node with smallest distance.
          - Relax edges: if going through current node is cheaper,
            update neighbour distance.
          - Reconstruct path from predecessor map.
        """
        dist = {start_id: 0}
        prev = {start_id: None}
        pq = [(0, start_id)]         # (distance, node_id)

        while pq:
            d, curr = heapq.heappop(pq)

            if curr == end_id:
                break                 # found shortest path to destination

            if d > dist.get(curr, float("inf")):
                continue              # stale entry, skip

            for nbr, cost in self.graph.adjacency[curr]:
                edge = (min(curr, nbr), max(curr, nbr))
                if edge in ignore_edges:
                    continue          # pretend this road doesn't exist

                new_dist = d + cost
                if new_dist < dist.get(nbr, float("inf")):
                    dist[nbr] = new_dist
                    prev[nbr] = curr
                    heapq.heappush(pq, (new_dist, nbr))

        if end_id not in prev:
            return None               # destination is unreachable

        # Reconstruct path by walking predecessors backward
        path = []
        curr = end_id
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    #  Prim's MST
    # ------------------------------------------------------------------ #

    def _run_prims_algorithm(self):
        """
        Prim's algorithm grows a minimum spanning tree one edge at a time.

        How it works:
          1. Start with the set of nodes already connected (safety routes).
          2. Put all edges from those nodes to non-connected nodes into a
             min-heap ordered by cost.
          3. Pop the cheapest edge.  If the far endpoint is already
             connected, skip.  Otherwise, add the edge and the new node
             to the tree, and push its outgoing edges.
          4. Repeat until all nodes are connected.

        Why Prim's?
          It guarantees the MINIMUM total cost tree, which directly
          matches the project requirement of "minimum total road cost".
        """
        # Nodes already connected by the safety routes
        in_tree = set()
        for u, v in self.built_roads:
            in_tree.add(u)
            in_tree.add(v)

        if not in_tree:
            in_tree.add(self.primary_hospital)

        # Seed the priority queue with edges from connected nodes
        pq = []
        for u in in_tree:
            for v, cost in self.graph.adjacency[u]:
                if v not in in_tree:
                    heapq.heappush(pq, (cost, u, v))

        # Grow the tree
        while pq:
            cost, u, v = heapq.heappop(pq)
            if v in in_tree:
                continue              # already connected

            # Accept this edge
            self.built_roads.add((min(u, v), max(u, v)))
            in_tree.add(v)

            # Push new frontier edges
            for nbr, nbr_cost in self.graph.adjacency[v]:
                if nbr not in in_tree:
                    heapq.heappush(pq, (nbr_cost, v, nbr))

    # ------------------------------------------------------------------ #
    #  Apply optimised network to the shared graph
    # ------------------------------------------------------------------ #

    def _apply_built_roads_to_graph(self):
        """
        Prune the shared graph's adjacency lists to contain ONLY
        the roads we selected.  This ensures downstream modules
        (ambulance placement, dynamic routing) use the optimised network.

        Why this matters:
          The project says "the road network from Challenge 2 defines
          the travel graph".  All later challenges must use these roads.
        """
        for u in self.graph.all_node_ids():
            filtered = []
            for v, _old_cost in self.graph.adjacency[u]:
                edge = (min(u, v), max(u, v))
                if edge in self.built_roads:
                    # Recompute fresh cost (in case risk indices changed)
                    filtered.append((v, self.graph._edge_cost(v)))
            self.graph.adjacency[u] = filtered

    def _validate_full_connectivity(self):
        """
        Verify every node remains reachable after pruning.
        Also prints isolated-node count for quick debugging.
        """
        node_ids = self.graph.all_node_ids()
        if not node_ids:
            return True

        isolated = [nid for nid in node_ids if not self.graph.adjacency[nid]]
        if isolated:
            print("[Road Network] Connectivity check: {} isolated nodes found.".format(len(isolated)))
            return False

        start = node_ids[0]
        visited = set([start])
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, _cost in self.graph.adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        if len(visited) != len(node_ids):
            print("[Road Network] Connectivity check: visited {}/{} nodes.".format(
                len(visited), len(node_ids)
            ))
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def calculate_total_cost(self):
        """
        Sum the construction cost of all built roads.
        Each edge is undirected, so we average the two directional costs.
        """
        total = 0.0
        for u, v in self.built_roads:
            cost_uv = self.graph._edge_cost(v)
            cost_vu = self.graph._edge_cost(u)
            total += (cost_uv + cost_vu) / 2.0
        return total

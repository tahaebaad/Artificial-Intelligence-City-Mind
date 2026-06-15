"""
CityMind  -  Shared City Graph Model  (city_model.py)
=====================================================
This file defines the SINGLE SOURCE OF TRUTH used by ALL 5 challenges.

The city is modelled as a grid-based graph:
  - Each NODE represents a location (Residential, Hospital, etc.).
  - Each EDGE represents a road with a travel cost.

Project requirement satisfied:
  "No module is allowed to maintain its own separate copy of the city."
  Every challenge receives a reference to the SAME CityGraph object.

Standard-library only: enum, collections.
"""

from enum import Enum
from collections import deque


# ---------------------------------------------------------------------------
#  Location types  (the domain values for CSP in Challenge 1)
# ---------------------------------------------------------------------------
class LocationType(Enum):
    """
    Every node on the grid is one of these types.
    The .value string is used for compact display in the terminal and GUI.
    """
    RESIDENTIAL     = "RES"
    HOSPITAL        = "HOS"
    SCHOOL          = "SCH"
    INDUSTRIAL      = "IND"
    POWER_PLANT     = "PWR"
    AMBULANCE_DEPOT = "AMB"
    EMPTY           = "..."


# ---------------------------------------------------------------------------
#  Node  -  one cell on the grid
# ---------------------------------------------------------------------------
class Node:
    """
    Stores all properties the project requires for a single location:
      - location_type        : which facility sits here
      - population_density   : numeric value (people living/working here)
      - risk_index           : updated by Challenge 5, feeds back into edge costs
      - accessible           : False when a flood/accident cuts off the location
    """

    def __init__(self, row, col, node_id):
        self.row = row                          # grid row (0-indexed)
        self.col = col                          # grid column (0-indexed)
        self.node_id = node_id                  # unique int id = row * cols + col
        self.location_type = LocationType.EMPTY
        self.population_density = 0
        self.risk_index = 0.0                   # updated by Challenge 5
        self.accessible = True                  # used by Challenge 4

    def __repr__(self):
        return "Node({},{},{})".format(self.row, self.col, self.location_type.value)


# ---------------------------------------------------------------------------
#  CityGraph  -  the shared graph
# ---------------------------------------------------------------------------
class CityGraph:
    """
    Grid-based city graph of size  rows x cols.

    Internal data structures
    ------------------------
    self.nodes      : dict  node_id -> Node
    self.adjacency  : dict  node_id -> list of (neighbour_id, travel_cost)
    self.blocked_edges : set of (min_id, max_id) pairs currently impassable

    Edge cost rules (from the project statement):
      - Standard road cost = 1.0
      - Road through a Residential zone = 0.8
      - Additional cost multiplier from crime-risk index (Challenge 5)
    """

    def __init__(self, rows=10, cols=10):
        self.rows = rows
        self.cols = cols
        self.nodes = {}           # node_id -> Node
        self.adjacency = {}       # node_id -> [(neighbour_id, cost)]
        self.blocked_edges = set()  # edges currently removed due to flooding
        self.risk_cost_weight = 0.0  # how much crime risk adds to travel cost

        # --- Build all nodes ---
        for r in range(rows):
            for c in range(cols):
                nid = self._nid(r, c)
                self.nodes[nid] = Node(r, c, nid)
                self.adjacency[nid] = []

        # --- Build 4-directional edges (up, down, left, right) ---
        self._build_edges()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _nid(self, r, c):
        """Convert (row, col) to unique integer id."""
        return r * self.cols + c

    def _build_edges(self):
        """
        Create adjacency lists for 4-connected grid.
        Called once during __init__.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
        for r in range(self.rows):
            for c in range(self.cols):
                nid = self._nid(r, c)
                self.adjacency[nid] = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        nbr = self._nid(nr, nc)
                        cost = self._edge_cost(nbr)
                        self.adjacency[nid].append((nbr, cost))

    def _edge_cost(self, dest_id):
        """
        Compute travel cost to enter node dest_id.

        Rules from the project statement:
          - Residential zone destination  -> base cost 0.8
          - Everything else               -> base cost 1.0
          - Crime-risk multiplier applied on top: cost * (1 + weight * risk)
        """
        node = self.nodes[dest_id]
        if node.location_type == LocationType.RESIDENTIAL:
            base_cost = 0.8
        else:
            base_cost = 1.0
        # Crime-risk multiplier (Challenge 5 feeds back into this)
        risk_penalty = self.risk_cost_weight * max(0.0, node.risk_index)
        return base_cost * (1.0 + risk_penalty)

    def _refresh_edges(self, nid):
        """
        After a node's type or risk changes, recompute every edge
        that POINTS TO it so other modules see updated costs immediately.

        Performance note: This is O(N * avg_degree) per call because it
        scans all adjacency lists.  On a 100-node grid with ~4 neighbours
        each, this is ~400 comparisons — fast enough for demo.
        set_risk_cost_weight() calls this for every node = ~40,000
        comparisons total.  For grids larger than ~30x30, incremental
        edge tracking would be needed.
        """
        for src in self.adjacency:
            neighbours = self.adjacency[src]
            for i in range(len(neighbours)):
                dst, _old_cost = neighbours[i]
                if dst == nid:
                    neighbours[i] = (dst, self._edge_cost(nid))

    # ------------------------------------------------------------------ #
    #  Public API  (used by all 5 challenges)
    # ------------------------------------------------------------------ #

    def node_id(self, r, c):
        """Get unique id for grid position (r, c)."""
        return self._nid(r, c)

    def get_node(self, nid):
        """Return the Node object for a given id."""
        return self.nodes[nid]

    def set_location(self, nid, loc_type, pop_density=None):
        """
        Assign a LocationType to a node.  Optionally set population density.
        Automatically refreshes edge costs so downstream modules see the change.
        """
        node = self.nodes[nid]
        node.location_type = loc_type
        if pop_density is not None:
            node.population_density = pop_density
        self._refresh_edges(nid)

    def set_risk_index(self, nid, risk_index):
        """
        Update crime-risk index for a node (Challenge 5).
        Refreshes edge costs so Challenge 4 routing picks up the change.
        """
        self.nodes[nid].risk_index = max(0.0, float(risk_index))
        self._refresh_edges(nid)

    def set_risk_cost_weight(self, weight):
        """
        Set how much crime-risk affects travel costs globally.
        A weight of 0.0 means risk has no effect; 0.35 is a reasonable value.
        """
        self.risk_cost_weight = max(0.0, float(weight))
        # Recompute all edge costs
        for nid in self.nodes:
            self._refresh_edges(nid)

    def get_adjacent_ids(self, nid):
        """Return list of neighbour node ids (ignoring costs)."""
        return [n for n, _cost in self.adjacency[nid]]

    def get_nodes_of_type(self, loc_type):
        """Return list of all node ids that have the given LocationType."""
        result = []
        for nid, node in self.nodes.items():
            if node.location_type == loc_type:
                result.append(nid)
        return result

    def all_node_ids(self):
        """Return list of every node id in the graph."""
        return list(self.nodes.keys())

    def bfs_distances(self, start_id):
        """
        Breadth-first search from start_id.
        Returns dict {node_id: hop_count} for all reachable nodes.

        IMPORTANT: BFS reads self.adjacency[curr], which is the CURRENT
        post-prune, post-flood adjacency list.  After C2 prunes the graph
        to MST edges only, BFS only traverses those roads.  After a flood
        removes an edge, BFS will not cross that road.  This is correct
        because BFS should reflect actual connectivity.

        Used by:
          - Challenge 1 (coverage check: residential within 3 hops of hospital)
          - Challenge 5 (distance-to-nearest-type features)
        """
        dist = {start_id: 0}
        queue = deque([start_id])
        while queue:
            curr = queue.popleft()
            for nbr, _cost in self.adjacency[curr]:
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    queue.append(nbr)
        return dist

    # ---- Road blocking / unblocking (Challenge 4: flooding events) ---- #

    def block_road(self, nid1, nid2):
        """
        Remove the edge between nid1 and nid2 (simulate a flood).
        Every module that later queries the adjacency list will see the
        road is gone — satisfying the 'single source of truth' requirement.
        """
        edge_key = (min(nid1, nid2), max(nid1, nid2))
        self.blocked_edges.add(edge_key)
        # Remove nid2 from nid1's neighbour list and vice versa
        self.adjacency[nid1] = [(n, c) for n, c in self.adjacency[nid1] if n != nid2]
        self.adjacency[nid2] = [(n, c) for n, c in self.adjacency[nid2] if n != nid1]

    def unblock_road(self, nid1, nid2):
        """
        Restore a previously blocked road.
        """
        edge_key = (min(nid1, nid2), max(nid1, nid2))
        if edge_key not in self.blocked_edges:
            return
        self.blocked_edges.discard(edge_key)
        self.adjacency[nid1].append((nid2, self._edge_cost(nid2)))
        self.adjacency[nid2].append((nid1, self._edge_cost(nid1)))

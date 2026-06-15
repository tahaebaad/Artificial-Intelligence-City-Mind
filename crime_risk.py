"""
CityMind  -  Challenge 5: Crime Risk Prediction  (crime_risk.py)
================================================================
Algorithm : K-Means-style Clustering + KNN Classification

Pipeline:
  1. UNSUPERVISED step: cluster neighbourhoods by population density
     and proximity features using K-Means-style centroid refinement.
  2. SUPERVISED step: assign risk labels (LOW/MEDIUM/HIGH) based on
     cluster properties, then train a KNN classifier on the labelled
     data.
  3. INTEGRATION: feed predicted risk indices back into the shared
     city graph so they act as additional travel-cost multipliers
     for Challenge 4 routing and Challenge 3 ambulance placement.

WHY this approach?
  - K-Means-style clustering gives compact risk groups from raw features.
  - KNN then classifies nodes by nearest labelled examples.
  - Both are simple, from-scratch distance-based methods.

Feature vector for each node (6 features):
  [population_scaled, 1/dist_industrial, 1/dist_hospital,
   1/dist_school, 1/dist_depot, neighbourhood_density_scaled]

  Justification:
  - High population + close to industrial = higher crime risk
  - Close to hospital/school/depot = lower crime risk (more surveillance)
  - Dense neighbourhoods correlate with higher incident rates

Standard-library only: math, random.
"""

import math
import random
from city_model import LocationType


class CrimeRiskPredictor:
    """
    Predicts crime risk for every node in the city.

    Usage:
        predictor = CrimeRiskPredictor(graph, seed=42)
        predictions = predictor.train_and_predict()
        summary = predictor.summary()
    """

    RISK_LABELS = ("LOW", "MEDIUM", "HIGH")

    def __init__(self, graph, seed=42):
        self.graph = graph
        self.seed  = seed
        self.rng   = random.Random(seed)

        # Results (populated after train_and_predict)
        self.cluster_centers     = []    # list of centroid feature vectors
        self.cluster_assignments = {}    # node_id -> cluster_index
        self.cluster_risk_levels = {}    # cluster_index -> "LOW"/"MEDIUM"/"HIGH"
        self.training_data       = []    # list of feature vectors
        self.training_labels     = []    # list of risk labels
        self.predictions         = {}    # node_id -> {label, risk_index, features}
        self.metrics             = {}    # test_accuracy etc.
        self.police_deployment   = {}    # node_id -> officer_count

    # ------------------------------------------------------------------ #
    #  Feature extraction
    # ------------------------------------------------------------------ #

    def _feature_vector(self, nid):
        """
        Build a 6-dimensional feature vector for a node.

        Features and their rationale:
          0: population_scaled     — more people = more potential incidents
          1: 1/dist_to_industrial  — industrial proximity correlates with crime
          2: 1/dist_to_hospital    — hospitals provide indirect surveillance
          3: 1/dist_to_school      — schools = lower crime areas
          4: 1/dist_to_depot       — emergency presence deters crime
          5: neighbourhood_density — average population of adjacent cells
        """
        node = self.graph.nodes[nid]
        pop = float(node.population_density)

        # BFS distance to nearest facility of each type
        ind_dist   = self._distance_to_nearest_type(nid, LocationType.INDUSTRIAL)
        hosp_dist  = self._distance_to_nearest_type(nid, LocationType.HOSPITAL)
        school_dist = self._distance_to_nearest_type(nid, LocationType.SCHOOL)
        depot_dist = self._distance_to_nearest_type(nid, LocationType.AMBULANCE_DEPOT)

        # Average population of adjacent cells
        neigh_density = self._neighbourhood_density(nid)

        # Inverse distances (closer = higher value)
        inv_ind   = 1.0 / max(ind_dist, 1.0)
        inv_hosp  = 1.0 / max(hosp_dist, 1.0)
        inv_school = 1.0 / max(school_dist, 1.0)
        inv_depot = 1.0 / max(depot_dist, 1.0)

        # Scale to [0, 1] range approximately
        pop_scaled   = pop / 100.0
        neigh_scaled = neigh_density / 100.0

        return [pop_scaled, inv_ind, inv_hosp, inv_school, inv_depot, neigh_scaled]

    def _distance_to_nearest_type(self, nid, loc_type):
        """
        BFS hop distance from nid to the nearest node of type loc_type.

        Returns 10.0 (a large sentinel value) if no node of that type
        exists.  This compresses the inverse-distance feature toward
        0.1 rather than 0.0 — a known approximation that is acceptable
        on a 10x10 grid where all facility types are always present.
        """
        targets = self.graph.get_nodes_of_type(loc_type)
        if not targets:
            return 10.0              # default large distance
        dists = self.graph.bfs_distances(nid)
        best = 10.0
        for t in targets:
            d = dists.get(t, 10.0)
            if d < best:
                best = d
        return float(best)

    def _neighbourhood_density(self, nid):
        """Average population density of adjacent nodes."""
        nbrs = self.graph.get_adjacent_ids(nid)
        if not nbrs:
            return float(self.graph.nodes[nid].population_density)
        total = 0
        for n in nbrs:
            total += self.graph.nodes[n].population_density
        return total / len(nbrs)

    # ------------------------------------------------------------------ #
    #  Distance metric
    # ------------------------------------------------------------------ #

    def _euclidean(self, a, b):
        """
        Euclidean distance between two feature vectors.
        d = sqrt( sum( (a_i - b_i)^2 ) )

        This is the standard distance metric for KNN.
        """
        total = 0.0
        for i in range(len(a)):
            diff = a[i] - b[i]
            total += diff * diff
        return math.sqrt(total)

    # ------------------------------------------------------------------ #
    #  K-Means-style Clustering  (unsupervised step)
    # ------------------------------------------------------------------ #

    def _kmeans_cluster(self, points, k=3, iterations=20):
        """
        K-Means-style clustering using iterative centroid refinement.

        IMPORTANT NOTE (for viva defense):
          This method is K-Means-style:
            - assign each point to nearest centroid
            - recompute each centroid as mean of assigned points
            - iterate
          The supervised step (_knn_predict) is true KNN classification.

        Steps:
          1. Pick k initial centroids randomly from the data.
          2. Assign each point to the NEAREST centroid (1-NN assignment).
          3. Recompute centroids as the mean of assigned points.
          4. Repeat for 'iterations' rounds.

        Why not use a library?
          - Project requires from-scratch implementation.
          - Easy to explain: "each point joins its nearest centroid."

        Parameters
        ----------
        points     : list of feature vectors
        k          : number of clusters
        iterations : number of refinement rounds

        Returns
        -------
        (centers, labels)  where
          centers = list of k centroid vectors
          labels  = list of cluster indices (one per point)
        """
        if not points:
            return [], []

        # Step 1: random initial centroids
        n_points = len(points)
        sample_size = min(k, n_points)
        centers = []
        chosen = self.rng.sample(range(n_points), sample_size)
        for idx in chosen:
            centers.append(list(points[idx]))
        # If fewer points than k, duplicate the last one
        while len(centers) < k:
            centers.append(list(points[-1]))

        labels = [0] * n_points

        # Steps 2-4: iterate
        for _round in range(iterations):
            # Assign each point to nearest centroid
            for i in range(n_points):
                best_cluster = 0
                best_dist = self._euclidean(points[i], centers[0])
                for c in range(1, k):
                    d = self._euclidean(points[i], centers[c])
                    if d < best_dist:
                        best_dist = d
                        best_cluster = c
                labels[i] = best_cluster

            # Recompute centroids
            dim = len(points[0])
            for c in range(k):
                members = []
                for i in range(n_points):
                    if labels[i] == c:
                        members.append(points[i])
                if members:
                    new_center = []
                    for j in range(dim):
                        total = 0.0
                        for m in members:
                            total += m[j]
                        new_center.append(total / len(members))
                    centers[c] = new_center

        return centers, labels

    # ------------------------------------------------------------------ #
    #  Risk scoring  (for synthetic label generation)
    # ------------------------------------------------------------------ #

    def _risk_score(self, feat):
        """
        Weighted risk score from feature vector.

        Positive weights = factors that INCREASE crime risk:
          - population density (0.30) — more people = more incidents
          - industrial proximity (0.25) — industrial zones lack surveillance
          - neighbourhood density (0.20) — dense areas correlate with crime

        Negative weights = factors that DECREASE crime risk:
          - hospital proximity (-0.12) — hospitals provide indirect surveillance
          - school proximity (-0.08)   — schools deter crime
          - depot proximity (-0.05)    — emergency presence deters crime

        This produces a continuous score used to generate synthetic
        labels for the supervised KNN classifier.
        """
        pop, inv_ind, inv_hosp, inv_school, inv_depot, neigh = feat
        score = (
            0.30 * pop
            + 0.25 * inv_ind
            + 0.20 * neigh
            - 0.12 * inv_hosp
            - 0.08 * inv_school
            - 0.05 * inv_depot
        )
        return score

    # ------------------------------------------------------------------ #
    #  KNN Classifier  (supervised step)
    # ------------------------------------------------------------------ #

    def _knn_predict(self, feat, k=5):
        """
        K-Nearest Neighbours classification.

        How it works:
          1. Compute distance from feat to EVERY training point.
          2. Sort by distance (ascending).
          3. Take the K nearest neighbours.
          4. Majority vote on their labels → predicted label.

        Parameters
        ----------
        feat : feature vector to classify
        k    : number of neighbours to consider

        Returns
        -------
        str : predicted risk label ("LOW", "MEDIUM", or "HIGH")
        """
        if not self.training_data:
            return "LOW"

        # Compute distances to all training points
        distances = []
        for i in range(len(self.training_data)):
            d = self._euclidean(feat, self.training_data[i])
            distances.append((d, self.training_labels[i]))

        # Sort by distance
        distances.sort(key=lambda pair: pair[0])

        # Take K nearest
        k_actual = min(k, len(distances))
        nearest_labels = []
        for i in range(k_actual):
            nearest_labels.append(distances[i][1])

        # Majority vote
        best_label = "LOW"
        best_count = 0
        for label in self.RISK_LABELS:
            count = nearest_labels.count(label)
            if count > best_count:
                best_count = count
                best_label = label

        return best_label

    # ------------------------------------------------------------------ #
    #  Accuracy metric
    # ------------------------------------------------------------------ #

    def _accuracy(self, gold, pred):
        """Simple accuracy = correct / total."""
        if not gold:
            return 0.0
        correct = 0
        for i in range(len(gold)):
            if gold[i] == pred[i]:
                correct += 1
        return correct / len(gold)

    # ------------------------------------------------------------------ #
    #  Main pipeline
    # ------------------------------------------------------------------ #

    def train_and_predict(self):
        """
        Full pipeline:

        Step 1 (Unsupervised): Cluster residential nodes by features.
        Step 2 (Labelling):    Rank clusters by average risk score
                               and assign LOW / MEDIUM / HIGH labels.
        Step 3 (Supervised):   Train KNN classifier on 70% of data,
                               evaluate on 30%.
        Step 4 (Integration):  Predict risk for ALL nodes and update
                               the shared graph's risk_index values.

        Returns dict {node_id: {label, risk_index, features}}.
        """
        residential = self.graph.get_nodes_of_type(LocationType.RESIDENTIAL)

        # Extract features for all residential nodes
        features = []
        for nid in residential:
            features.append(self._feature_vector(nid))

        if not features:
            self.predictions = {}
            self.metrics["test_accuracy"] = 0.0
            return self.predictions

        # --- Step 1: K-Means-style clustering (unsupervised) ---
        self.cluster_centers, cluster_labels = self._kmeans_cluster(
            features, k=3, iterations=25
        )
        for i, nid in enumerate(residential):
            self.cluster_assignments[nid] = cluster_labels[i]

        # --- Step 2: Assign risk levels to clusters ---
        # Average risk score per cluster
        cluster_scores = {0: [], 1: [], 2: []}
        for i in range(len(features)):
            cidx = cluster_labels[i]
            cluster_scores[cidx].append(self._risk_score(features[i]))

        # Rank clusters by average score: lowest = LOW, highest = HIGH
        cluster_avgs = []
        for cidx, scores in cluster_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
            else:
                avg = 0.0
            cluster_avgs.append((cidx, avg))
        cluster_avgs.sort(key=lambda x: x[1])

        if len(cluster_avgs) >= 3:
            self.cluster_risk_levels[cluster_avgs[0][0]] = "LOW"
            self.cluster_risk_levels[cluster_avgs[1][0]] = "MEDIUM"
            self.cluster_risk_levels[cluster_avgs[2][0]] = "HIGH"

        # Build labelled dataset from cluster assignments
        labelled = []
        for i in range(len(features)):
            cidx = cluster_labels[i]
            label = self.cluster_risk_levels.get(cidx, "LOW")
            labelled.append((features[i], label))

        # --- Step 3: Train/test split (70/30) and KNN classifier ---
        # On a standard 10×10 grid this gives ~49 train / 22 test points.
        # On very small grids (few residential nodes), the test set may
        # overlap with training data or be too small for meaningful accuracy.
        # This is acceptable for demonstration but would need stratified
        # splitting for production use.
        self.rng.shuffle(labelled)
        split = max(1, int(0.7 * len(labelled)))
        train_set = labelled[:split]
        test_set = labelled[split:] if split < len(labelled) else labelled[-1:]

        self.training_data   = [x for x, _y in train_set]
        self.training_labels = [y for _x, y in train_set]

        # Evaluate on test set
        test_gold = [y for _x, y in test_set]
        test_pred = [self._knn_predict(x) for x, _y in test_set]
        self.metrics["test_accuracy"] = self._accuracy(test_gold, test_pred)

        # --- Step 4: Predict for ALL nodes and update shared graph ---
        for nid in self.graph.all_node_ids():
            feat = self._feature_vector(nid)
            label = self._knn_predict(feat)

            # Convert label to numeric risk index for the graph
            if label == "HIGH":
                risk_index = 1.0
            elif label == "MEDIUM":
                risk_index = 0.6
            else:
                risk_index = 0.2

            # Update the SHARED graph — this is the integration step!
            # Challenge 4 routing will now see higher costs in high-risk areas.
            self.graph.set_risk_index(nid, risk_index)

            self.predictions[nid] = {
                "label": label,
                "risk_index": risk_index,
                "features": feat,
            }

        return self.predictions

    # ------------------------------------------------------------------ #
    #  Summary for display
    # ------------------------------------------------------------------ #

    def summary(self):
        """Return a summary dict for GUI display and logging."""
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for pred in self.predictions.values():
            counts[pred["label"]] += 1
        return {
            "counts": counts,
            "test_accuracy": self.metrics.get("test_accuracy", 0.0),
            "cluster_centers": self.cluster_centers,
            "cluster_risk_levels": self.cluster_risk_levels,
            "police_deployment": dict(self.police_deployment),
        }

    def allocate_police_officers(self, total_officers=10):
        """
        Allocate officers to the highest-risk nodes.

        Strategy:
          - Build a priority score from risk_index and population density.
          - Reserve assignments for HIGH risk first, then MEDIUM.
          - Distribute officers proportionally with at least 1 officer per selected hotspot.

        Returns dict {node_id: officer_count}.
        """
        if total_officers <= 0 or not self.predictions:
            self.police_deployment = {}
            return {}

        scored_nodes = []
        for nid, pred in self.predictions.items():
            node = self.graph.nodes[nid]
            label = pred["label"]
            if label not in ("HIGH", "MEDIUM"):
                continue
            label_weight = 1.0 if label == "HIGH" else 0.55
            score = (pred["risk_index"] * 0.7 + (node.population_density / 100.0) * 0.3) * label_weight
            scored_nodes.append((score, nid))

        if not scored_nodes:
            self.police_deployment = {}
            return {}

        scored_nodes.sort(reverse=True)
        # Cap hotspots at 4 so officers concentrate visibly (e.g. P:3, P:4)
        # instead of spreading 1 officer each across all HIGH/MEDIUM nodes.
        hotspot_count = min(len(scored_nodes), 4)
        hotspots = scored_nodes[:hotspot_count]
        total_score = sum(max(0.01, score) for score, _nid in hotspots)

        deployment = {}
        for score, nid in hotspots:
            share = max(1, int(round((max(0.01, score) / total_score) * total_officers)))
            deployment[nid] = share

        assigned = sum(deployment.values())
        ranked_ids = [nid for _score, nid in hotspots]
        idx = 0
        while assigned < total_officers:
            nid = ranked_ids[idx % len(ranked_ids)]
            deployment[nid] += 1
            assigned += 1
            idx += 1
        while assigned > total_officers and ranked_ids:
            nid = ranked_ids[idx % len(ranked_ids)]
            if deployment[nid] > 1:
                deployment[nid] -= 1
                assigned -= 1
            idx += 1
            # Safety guard: prevent infinite looping in edge cases where
            # rounding errors make convergence very slow.  In practice
            # the loop converges in < 10 iterations for total_officers=10.
            if idx > 500:
                break

        self.police_deployment = deployment
        return dict(deployment)

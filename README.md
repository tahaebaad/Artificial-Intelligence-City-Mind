# CityMind: Urban Intelligence & Emergency Simulation

CityMind is a dynamic, grid-based urban simulation designed to solve complex city planning and disaster-response problems. The system supports highly customizable grid dimensions (ranging from 6 to 15 rows and columns, allowing for non-square layouts like 6×15 or 15x6), integrating 5 distinct AI challenges into a single, synchronized environment. 

The core of the project is a **Shared Graph Architecture**. Every decision—from where a hospital is built to how an ambulance reroutes during a flood—is handled by a specific algorithm interacting with a single source of truth in memory. If one module modifies a road or updates a risk level, every other algorithm detects that change instantly.

## 🧠 Integrated AI Challenges

### 1. City Layout Planning (CSP)
Uses a **Backtracking Constraint Satisfaction Problem (CSP)** to place facilities (Hospitals, Industrial zones, etc.) according to urban rules.
* **Optimizations:** Implements the **MRV (Minimum Remaining Values)** heuristic and **Forward Checking** to prune the search tree and find valid layouts efficiently.
* **Fail-safe:** Includes a minimum-conflict repair mechanism to fix coverage gaps automatically.

### 2. Road Network Optimization (Prim's MST)
Constructs a minimum-cost road network using **Prim’s Algorithm**.
* **Redundancy:** Before building the full tree, the system uses **Dijkstra** to guarantee two **edge-disjoint safety corridors** between the Hospital and Ambulance Depot. This ensures that if one road floods, a backup route always exists.

### 3. Fleet Deployment (Simulated Annealing)
Solves the "p-centre" problem to find the best home bases for 3 ambulances.
* **Logic:** Uses **Simulated Annealing** with a geometric cooling schedule to minimize the worst-case response time to any residential node.
* **Dynamic:** The fleet re-optimizes every 5 steps to adapt to road closures and changing crime rates.

### 4. Dynamic Emergency Routing (A* Search)
Handles real-time navigation using **A* Search** with an **admissible Manhattan heuristic** (scaled by 0.8).
* **Resilience:** The ambulance performs hop-by-hop re-planning. If a road floods mid-journey, the algorithm detects the missing edge and reroutes instantly without crashing or looping.

### 5. Crime Risk Analytics (K-Means & KNN)
A dual-stage ML pipeline that adjusts city behavior based on data.
* **Clustering:** Uses **K-Means** (unsupervised) to group neighborhoods based on demographics and proximity to industry.
* **Classification:** Uses **KNN** (supervised) to label nodes as Low, Medium, or High risk.
* **Integration:** These risk levels are fed back into the graph as travel-cost multipliers, incentivizing ambulances to avoid high-crime areas.

## 🛠️ Technical Stack
* **Language:** Python 3.x[cite: 10]
* **GUI:** CustomTkinter (Multi-threaded to handle real-time animations and background AI calculations)
* **Graph Logic:** Adjacency lists with dynamic weight refreshing[cite: 18]

## 🚀 How to Run
1. Clone the repository: `git clone https://github.com/yourusername/CityMind.git`
2. Install dependencies: `pip install customtkinter`
3. Run the dashboard: `python citymind_gui.py`

## 👥 Authors

* **Taha Ebaad**
* **Ahmed Cheema**
* **Warisha Ishtiaq**

Built for the Artificial Intelligence Course, Spring 2026.

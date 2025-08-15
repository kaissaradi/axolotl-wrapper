# **Project Plan: Axolotl v3 - The Definitive Refinement Experience**

* **Epic**: `ENG-123: Next-Generation Refinement Engine & Dynamic Visualization`
* **Date**: `August 12, 2025`
* **Status**: `Ready for Development`
* **Priority**: `Highest`

## **1. Vision & Goal** 🎯

### **Why This Project Matters**
Our goal is to make Axolotl the most accurate, intuitive, and efficient tool for neural spike sorting refinement. This requires a two-pronged upgrade: a state-of-the-art **back-end algorithm** that understands the physics of spike propagation, and a fluid, **front-end visualization** that allows scientists to see and understand their data dynamically.

### **Objective & Key Results (OKR)**
* **Objective**: Deliver a trusted, "one-click" refinement solution that drastically reduces manual curation time.
* **Key Result 1 (Accuracy)**: Reduce the ISI violation rate on our benchmark datasets by at least 25% post-refinement compared to the current algorithm.
* **Key Result 2 (Efficiency)**: Decrease user interaction time by 50% through superior automation and clearer visualizations.
* **Key Result 3 (Performance)**: Achieve near-instantaneous UI responsiveness, with all original clusters loading from cache in under 2 seconds.

---

## **2. User Stories**

* **As a neuroscientist**, I want the system to automatically separate contaminated clusters using both waveform shape and spatial location, so I can be confident that refined units are truly single neurons.
* **As a neuroscientist**, I want to visualize spike propagation as a dynamic animation, not a static image, so I can intuitively understand a neuron's spatio-temporal footprint.
* **As a neuroscientist**, I want the application to load my data instantly, so I can begin my analysis without delay.

---

## **3. Scope & Requirements**

#### **In Scope:**
* A complete replacement of the core algorithm within the `refine_cluster_v2` function.
* Implementation of a new **two-stage pipeline**: a "Split Stage" using density-based clustering (HDBSCAN) and a "Merge Stage" using multi-criterion validation.
* Integration of per-spike spatial features (Center of Mass) and cross-cluster temporal features (cross-correlograms).
* A one-time pre-computation step for all original EIs to ensure instant loading.
* Replacement of the static analysis plots with a single, dynamic, and animated visualization component.

#### **Out of Scope (for this Epic):**
* A major GUI redesign beyond the main analysis panel.
* A dedicated settings panel for all algorithm parameters (can be a fast-follow task).
* Support for alternative clustering algorithms beyond HDBSCAN.

---

## **4. Implementation Plan: Two Core Initiatives**

This project is broken into two parallel initiatives that can be developed concurrently.

### **Initiative A: Core Refinement Engine Overhaul (Back-End)** 🧠

The goal is to replace the current algorithm with a spatially-aware, density-based pipeline.
**Location**: `cleaning_utils_cpu.py`

* **Phase 1: Build Foundational Algorithms**
    * **Task 1.1: `[Sub-Task]` Implement `compute_per_spike_features()`**: Create a function that takes raw snippets and channel positions and returns a feature matrix for every individual spike.
        * **Features**: 1) Waveform Shape (first 3 Principal Components), 2) Spatial Location (amplitude-weighted Center of Mass).
    * **Task 1.2: `[Sub-Task]` Implement `test_merge_candidates()`**: Create a function that returns a `True`/`False` merge decision based on a multi-criterion test.
        * **Criteria**: 1) Refractory Dip in cross-correlogram, 2) Spatial Co-location of EIs, 3) Propagation Pattern Similarity.

* **Phase 2: Refactor the Core Pipeline**
    * **Task 2.1: `[Task]` Overhaul `refine_cluster_v2()`**: Replace the entire function body with the new two-stage pipeline.
        1.  **Split**: Use HDBSCAN on the per-spike features to find pure sub-clusters and reject noise.
        2.  **Merge**: Use a graph-based approach (`networkx`) to intelligently merge the purified sub-clusters based on the `test_merge_candidates` function.

### **Initiative B: Dynamic Visualization & UX (Front-End)** 🖥️

The goal is to create a fluid, animated, and intuitive UI.
**Location**: `gui.py`

* **Phase 1: Implement Pre-Computation for Instant Loading**
    * **Task 1.1: `[Task]` Enhance `DataManager`**: Add logic to perform a one-time pre-computation of all original EIs, saving them to a single `all_eis.npy` cache file. The app will prioritize loading this file for an instant startup.

* **Phase 2: Develop the Dynamic Visualization Component**
    * **Task 2.1: `[Task]` Create the "Spike Propagation" View**: Replace the static plot tabs with a single, large interactive widget (using `pyqtgraph`).
    * **Task 2.2: `[Sub-Task]` Implement Interactive Time Slider**: Add a slider widget below the plot that controls the currently displayed time point of the EI.
    * **Task 2.3: `[Sub-Task]` Implement Real-Time Plot Rendering**: Write the update logic that redraws the spatial map whenever the slider's value changes. This function will:
        * **Size** each dot based on voltage amplitude at that time point.
        * **Color** each dot using a divergent blue-to-red colormap.

* **Phase 3: Integration and Polish**
    * **Task 3.1: `[Task]` Integrate the New Visualization**: Embed the new dynamic component into the main window layout, ensuring it correctly receives data from the `DataManager` when a cluster is selected.
    * **Task 3.2: `[Sub-Task]` Add Hover Tooltips**: Implement tooltips on the plot to show channel ID and live voltage when the user mouses over an electrode.

---

## **5. Acceptance Criteria**

* **GIVEN** a contaminated cluster with two spatially distinct units, **WHEN** refinement is run, **THEN** the output is two separate, pure clusters.
* **GIVEN** a single neuron whose waveform drifts and is split into two clusters, **WHEN** refinement is run, **THEN** the output is a single merged cluster with a low ISI violation rate.
* **GIVEN** a dataset is loaded, **WHEN** a user clicks on any original cluster, **THEN** its visualization appears in under 200 milliseconds.
* **GIVEN** the dynamic visualization is open, **WHEN** the user drags the time slider, **THEN** the plot animation is smooth with no discernible lag.
* The end-to-end refinement process on a standard cluster (50k spikes) completes in **under 60 seconds**.

---

## **6. Risks & Mitigation**

* **Risk**: **Algorithm Performance**. Per-spike feature calculation is computationally intensive.
    * **Mitigation**: We will enforce vectorized `numpy` operations and benchmark performance early. If necessary, we will consider downsampling snippets for extremely large clusters (>100k spikes).
* **Risk**: **GUI Performance**. Rendering a real-time animation can be demanding on the CPU.
    * **Mitigation**: We will use `pyqtgraph`, which is highly optimized for this type of data visualization, and profile the drawing loop to ensure efficiency.
* **Risk**: **Parameter Sensitivity**. The new algorithms have thresholds that may need careful tuning.
    * **Mitigation**: We will establish robust, data-driven defaults by testing on diverse datasets. All key parameters will be exposed as optional arguments in the function signatures for easy adjustment and future UI configuration.

---

## **7. Best Practices & Development Guidelines** 🧑‍💻

To ensure success, our development process should adhere to the following principles.

### **Code Quality & Readability**

* **Comprehensive Docstrings**: Every new function must have a clear docstring explaining its purpose, arguments, return values, and any key logic, using the **NumPy docstring format**.
* **Type Hinting**: To improve code clarity and prevent bugs, all new and modified function signatures must include **type hints** (e.g., `def compute_features(snippets: np.ndarray) -> np.ndarray:`).
* **Centralize Configuration**: Avoid hardcoding magic numbers. Group parameters into a configuration dictionary or `dataclass` at the top of `cleaning_utils_cpu.py`.

### **Performance & Optimization**

* **Prioritize Vectorization**: In all numerical computations, aggressively prioritize `numpy`'s vectorized operations. Avoid Python `for` loops over large arrays.
* **Profile Early and Often**: Use a profiler like `cProfile` to benchmark the new algorithms as they are being developed. Do not wait until the end to discover performance issues.
* **Mindful Memory Management**: Be aware that loading snippets for extremely large clusters can consume significant RAM. Ensure that data is passed by reference where possible.

### **Workflow & Version Control**

* **Feature Branches**: All work should be done in separate feature branches corresponding to the initiatives (e.g., `feature/ENG-123-backend-refactor`, `feature/ENG-123-dynamic-viz`).
* **Small, Focused Pull Requests (PRs)**: Create small, logically distinct PRs for review. This makes code review faster and more effective.
* **Test-Driven Mentality**: Write unit tests (`pytest`) for the new algorithms **concurrently** with their development to ensure correctness from the ground up and prevent future regressions.
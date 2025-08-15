# **Developer's Guide: Using Gemini for the Axolotl v3 Project**
USE THE PLAN.MD FILE FOR THE DETAILED PLAN
This guide provides a workflow and best practices for leveraging the Gemini command-line tool to accelerate the development of the Axolotl v3 project. Treat Gemini as an expert pair programmer—it can write boilerplate, suggest optimizations, and help debug, but you are always the pilot.

## **1. Core Principles: The Art of Prompting**

Your success with Gemini depends entirely on the quality of your prompts. Follow these principles.

#### **Principle 1: Context is King**

The model knows nothing about our project by default. You must provide it with all relevant context in every prompt. The best way to do this is by concatenating key files.

**Bad Prompt**: `Write the refine_cluster_v2 function.`

**Excellent Prompt**:

```bash
cat gui.py cleaning_utils_cpu.py project_plan.md | gemini "Based on the provided project plan and existing codebase, write the new implementation for the `refine_cluster_v2` function in `cleaning_utils_cpu.py`. You are an expert Python developer specializing in scientific computing."
```

#### **Principle 2: Assign a Persona**

Always tell the model *who* it should be. This focuses its knowledge and improves the quality of its output.

  * `"You are an expert Python developer specializing in numpy, scipy, and high-performance scientific computing."`
  * `"You are a senior GUI developer with deep expertise in PyQt5 and pyqtgraph."`

#### **Principle 3: Be Specific and Task-Oriented**

Break down large goals into specific, actionable tasks from our project plan.

  * **Instead of**: `"Help with the backend."`
  * **Use**: ` "Implement the function  `compute\_per\_spike\_features`  as specified in the project plan. It should take a numpy array of snippets and the channel positions, and return a feature matrix containing the first 3 waveform PCs and the Center of Mass for each spike." `

#### **Principle 4: Iterate and Refine**

Your first prompt should produce a solid draft. Use follow-up prompts to perfect it.

  * `"That's a good start. Now, can you refactor that function to be fully vectorized and remove the for loop to improve performance?"`
  * `"Please add NumPy-style docstrings and type hints to the function you just wrote."`
  * ` "Now, write a  `pytest`  unit test for this function that covers edge cases like empty input." `

## **2. Practical Workflow for Axolotl v3**

Follow this workflow to tackle the project initiatives.

### **Initiative A: Back-End Refinement Engine**

1.  **Generate Core Functions**: Start by asking Gemini to write the foundational functions in isolation.

    ```bash
    # Prompt for Task 1.1
    cat cleaning_utils_cpu.py | gemini "You are a numpy expert. Implement the function `compute_per_spike_features` as specified in our project plan. Here is the current file for context. Ensure the function is fully vectorized."
    ```

2.  **Orchestrate the Pipeline**: Once the helper functions are written and tested, ask Gemini to assemble them inside the main `refine_cluster_v2` function.

    ```bash
    # Prompt for Task 2.1
    cat cleaning_utils_cpu.py | gemini "Using the functions `compute_per_spike_features` and `test_merge_candidates` that we just created, overhaul the `refine_cluster_v2` function. It should follow the split/merge pipeline detailed in our project plan, using HDBSCAN and a networkx graph."
    ```

### **Initiative B: Front-End Visualization**

1.  **Develop the UI Component**: Ask Gemini for the boilerplate code for the new visualization widget.

    ```bash
    # Prompt for Task 2.1 (Phase 2)
    cat gui.py | gemini "You are a pyqtgraph expert. Create a new QWidget class for our Axolotl GUI. It should contain a `pyqtgraph.PlotWidget` for the spatial map and a `QSlider` for time control. Provide the basic layout and class structure."
    ```

2.  **Implement the Animation Logic**: Request the specific function that will update the plot.

    ```bash
    # Prompt for Task 2.3
    gemini "Write the Python method `update_plot(self, time_index)` for the widget you just created. This method should take a time index, select the corresponding EI data, and update the scatter plot points' size and color using `setData()`. Do not recreate the plot object on each call."
    ```

## **3. The Golden Rule: Review and Verify Everything**

**This is the most important practice.** An LLM is a tool, not an oracle.

  * **NEVER Trust Blindly**: Treat all generated code as if it were written by a talented but unsupervised intern. It may contain subtle bugs, performance issues, or security vulnerabilities.
  * **You Are the Expert**: You must read, understand, and vet every single line of code Gemini produces. If you don't understand what it does, ask Gemini to explain it line-by-line until you do.
  * **Test Rigorously**: All generated logic must be covered by the unit tests you write (or have Gemini help you write). Your validation on benchmark datasets is the final source of truth.
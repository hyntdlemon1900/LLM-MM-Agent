PROBLEM_DESCRIPTION_PROMPT = """\
Problem Background:
{problem_background}

Problem Requirement:
{problem_requirement}

Problem Formulation:
{problem_formulation}
"""

HEURISTIC_ARCHITECT_PROMPT = '''
You are a top-tier **System Architect**. Your task is to design the **architectural skeleton** and **strategic workflow** for a new heuristic method, `solve_with_heuristic()`, extending an **existing** Solver Class.
**Crucial Constraint**: You are strictly an **Architect**, not a Coder. You must define _what_ the modules do and how they interact, but **DO NOT** design the specific algorithmic implementation details (e.g., do not specify sorting keys, specific formulas, or loop structures).

Your workflow must follow "Architecture-Driven Design":
1. **Analyze**: Understand the mathematical problem and the existing class structure.
2. **Strategize**: Decompose the solution process into a sequence of high-level **logical stages** (e.g., "Initialization", "Resource Allocation", "Conflict Resolution").
3. **Map to Architecture**: Define the `function_architecture` (the list of member functions) that represents these stages.

---

## Inputs:
### 1. Mathematical Modeling Problem
{problem_str}
### 2. Existing Solver Class Code (Reference)
Python
```
{heuristic_reference_code}
```

### 3. Target Method Signature
The goal is to implement a method that matches the output schema of the existing solver mechanisms (e.g., the MILP/optimization solver).
Python
```
def solve_with_heuristic(self) -> Any:
    \"\"\"
    Solves the problem using a heuristic strategy.
    Returns:
        A solution object (e.g., dict) strictly matching the schema required by the problem definition.
    \"\"\"
    pass 
```

---

## Your Mission:
Design a **high-level architectural plan** as a single JSON object.

### Design Philosophy:
- **Strategic Abstraction**: Focus purely on the **roles** and **responsibilities** of each function. Do not get bogged down in implementation logic.
- **Modularity**: Each major step in your strategy should map to a distinct, self-contained member function.
- **Data Flow**: Clearly identify what data moves between functions (via member variables), but do not dictate how that data is processed inside the function.   
- **Single-Pass Constructive Flow**: Design a pipeline that builds the solution step-by-step. Avoid proposing complex iterative metaheuristics (like Genetic Algorithms or Simulated Annealing) unless explicitly requested. Focus on a deterministic, rule-based flow.
    
## CRITICAL REQUIREMENT: REUSE EXISTING METHODS
1. **Analyze the Reference Code**: Identify existing helper methods in the provided code (e.g., methods for data preprocessing, path finding, or parameter extraction).  
2. **Integrate, Don't Reinvent**: Your architecture **must** incorporate these existing methods. If a new step requires data already prepared by an existing method, list that method in `dependencies`. 
3. **Real Names Only**: Use the _exact names_ of methods found in the `{heuristic_reference_code}`. Do not hallucinate generic names like `_helper_utility` unless they actually exist. 

---

## Output Format (CRITICAL):
Return **ONLY a valid JSON object** parseable by `json.loads()`. **Do not** use Markdown, explanatory text, or any extra text.
### JSON Structure:
JSON
```
{{
  "problem_analysis": "Brief analysis of the core challenges and critical data structures identified in the problem.",
  "strategy_overview": "A high-level summary of the proposed solution phases.",
  "function_architecture": [
    {{
      "name": "_step_1_function_name",
      "strategic_role": "Strictly define the OBJECTIVE (WHAT this step achieves) and the OUTPUT STATE. Do NOT describe the algorithm or logic (HOW).",
      "inputs": [],
      "outputs": [],
      "member_variables_read": [
        {{
          "name": "self.existing_data",
          "type": "DataType",
          "description": "Data required from the problem instance."
        }}
      ],
      "member_variables_written": [
        {{
          "name": "self._intermediate_state",
          "type": "StateType",
          "description": "State produced/updated by this step."
        }}
      ],
      "dependencies": ["_existing_helper_method_name"]
    }},
    {{
      "name": "_step_2_function_name",
      "strategic_role": "Objective of the second phase.",
      "inputs": [],
      "outputs": [],
      "member_variables_read": [
        {{
          "name": "self._intermediate_state",
          "type": "StateType",
          "description": "State from Step 1."
        }}
      ],
      "member_variables_written": [
        {{
          "name": "self._final_component",
          "type": "ComponentType",
          "description": "Result of Step 2."
        }}
      ],
      "dependencies": ["_step_1_function_name"]
    }},
    {{
      "name": "solve_with_heuristic",
      "strategic_role": "Orchestrator: Coordinates the execution of the pipeline and assembles the final formatted solution.",
      "inputs": [],
      "outputs": [
        {{"name": "solution", "type": "Any", "description": "Final solution object matching problem schema."}}
      ],
      "member_variables_read": [
        {{
          "name": "self._final_component",
          "type": "ComponentType",
          "description": "Data needed to construct final response."
        }}
      ],
      "member_variables_written": [
        {{
          "name": "self.solution",
          "type": "Any",
          "description": "The final result stored in the instance."
        }}
      ],
      "dependencies": ["_step_2_function_name"]
    }}
  ]
}}
```

### Field Requirements:
- **problem_analysis**: String. High-level problem understanding.
- **strategy_overview**: String. The macro-strategy.    
- **function_architecture**: Array of objects. The ordered execution pipeline.    
    - **name**: String. The proposed method name (use `_` prefix for private helpers).        
    - **strategic_role**: String. **CRITICAL**. Describe **WHAT** this function does (e.g., "Determines priority order of tasks", "Assigns resources to consumers"). **DO NOT** describe sorting algorithms, math formulas, or specific code logic.        
    - **inputs/outputs**: Array. Function parameters and return values.
    - **member_variables_read/written**: Array. Data flow via `self`.        
    - **dependencies**: List[String]. Names of **existing methods** (from input code) or **previous steps** that must run before this function.     

## Key Reminders:
1. **NO IMPLEMENTATION DETAILS**: The `strategic_role` must remain abstract. Do not tell the coder _how_ to solve it, only _what_ step to take. 
2. **NO 'DESCRIPTION' FIELD**: The JSON schema above does not have a `description` field. Do not add one.  
3. **Logical Order**: Functions must be listed in the order they should conceptually be designed or executed.   
4. **Integration**: You must explicitly link your new architecture to the **existing class methods** provided in the input code.
'''

# HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT = """
# # Task: Implement a Heuristic Algorithm Member Function
# You are an expert Python programmer implementing a single, modular member method for an existing solver class.

# ---

# ## Mission

# Your task is to write the complete, production-quality Python code for the class method `{function_name}`.
# -   You are implementing **Function {function_id} of {total_functions}**.
# -   This method is part of a larger heuristic algorithm.
# -   Your implementation **must be lean, concise, and strictly follow the specifications** provided.

# ---

# ## Context and Inputs

# ### 1. Mathematical Modeling Problem
# {problem_str}

# ### 2. Existing Solver Class Code
# This is the class that your new method will be added to. You are **NOT** re-writing this class, only implementing a **single new method** for it.

# **Use this code** to understand the class structure and identify existing `self.*` attributes (like `self.problem_data`, etc.) that are available for you to read from.

# ```python
# {heuristic_reference_code}
# ````

# ### 3. Function Specification: `{function_name}` (The ONLY Method to Implement)
# This is the detailed specification for the _only_ method you are allowed to write.
# - **Strategic Role (The "Why"):** {function_strategic_role}
# - **Algorithm Description (The "How"):** {function_description}    
# - **Dependencies (Preceding Functions):** {dependencies}
# You **must** call:
# - Methods explicitly listed in `Dependencies`.
# - These dependency calls are semantically required: they perform earlier steps of the heuristic, initialize/update shared self.* state, or compute intermediate results that later steps rely on. Skipping these calls will leave required variables uninitialized or key logic unexecuted, and will very likely cause runtime errors or incorrect behavior in subsequent methods.
# You may call:
# - Other methods already present in the existing solver class ModelSolver code (these methods may be called as needed, at your discretion).
# You **must NOT**:
# - Call methods that do not appear in `Dependencies` or the existing solver code.

# ### 4. State and Data Flow (CRITICAL)
# This function operates within the class and communicates with other methods _only_ through `self` attributes and the specified parameters/return value.
# **Inputs (Parameters):**
# ```
# {inputs_spec}
# ```
# - This section defines the method's arguments (in addition to `self`). 
# - Your method signature **MUST** use exactly these parameters.    
# - If this section is "None", the signature is just `def {function_name}(self):`.
    
# **Outputs (Return Value):**
# ```
# {outputs_spec}
# ```
# - This section defines the method's **return value**.
# - If this section specifies a return value (i.e., it's not "None"), you **MUST** include a `return` statement returning that value.
# - If this section is "None", the method MUST NOT contain ANY return statement (neither `return` nor `return None`), as the method should perform side effects only without returning any value.
    
# Member Variables Read:
# {member_variables_read}
# - You **MUST** read from these `self.*` attributes to get required state or problem data.  
# - Refer to the "Existing Solver Class Code" to see how these might be structured.
    
# Member Variables Written:
# {member_variables_written}
# - You **MUST** write your results or state changes to these `self.*` attributes.    
# - If the function's purpose is to calculate a value, it should be stored in one of these variables.
    
# ### 5. Previously Completed Functions
# This context shows which parts of the algorithm are already implemented.
# {completed_functions_summary}

# ---

# ## Implementation Requirements
# ### 1. Core Philosophy: Concise & Focused Implementation
# - Implement exactly the algorithm described in **Algorithm Description** and **Strategic Role**.
# - Code must be **lean, concise, and efficient**.    
# - **Forbid unnecessary noise**:    
#     - Do **NOT** include `print()` or logging.    
#     - Do **NOT** create placeholder code like pass or TODO.
#     - Do **NOT** add visualization or test/demo code.
# - Comments:
#     - Keep comments minimal and only for non-obvious logic.
#     - Do **NOT** add boilerplate comments that restate the code.
        

# ### 2. Exception Handling and Debuggability (CRITICAL)
# - **FAIL LOUDLY**: This code is part of a larger system that **requires clear error tracing** for debugging.    
# - **DO NOT HIDE ERRORS**: Do **NOT** wrap standard operations (like dictionary lookups, list indexing, or attribute access) in general `try-except` blocks.    
# - **ALLOW RUNTIME ERRORS**: It is **ESSENTIAL** that potential `KeyError`, `IndexError`, `AttributeError`, `TypeError`, etc., are **allowed to occur**. The system _must_ crash at the exact line of the error to provide a full and accurate traceback.
# - Write **"optimistic" code** that assumes the data and state (managed by `self` and other methods) are correct. Do not write "defensive" code.
    
# ### 3. State Management (via `self`)
# - **Read from `self`**: Access all necessary problem data (e.g., `self.problem_data`) and current algorithm state (e.g., `self.current_solution`) using the `self.*` attributes listed in `Member Variables Read`.    
# - **Write to `self`**: Store all persistent results or state changes (e.g., `self.best_solution_found`) using the `self.*` attributes listed in `Member Variables Written`.
# - **Do NOT pass static data**: Never pass static problem data (attributes initialized in `__init__`) as parameters to this method. Use `self` to access them.
    
# ---

# ## Output Format
# - You must generate **ONLY** the complete Python method code.    
# - Do **NOT** include Markdown code blocks (no ```python).   
# - Do **NOT** include any explanatory text before or after the method.    
# - Do **NOT** include import statements (they are handled globally).    
# - Do **NOT** include test code or example usage.   

# Your generated code **MUST** follow this exact structure:
# Python
# ```
# def {function_name}(self, ...):
#     \"\"\"[One-line description summarizing the Strategic Role.]
    
#     Args:
#         [Define arguments strictly matching 'Inputs (Parameters)'. Omit 'Args' section if Inputs is None.]

#     Returns:
#         [Define return value strictly matching 'Outputs (Return Value)'. Omit 'Returns' section if Outputs is None.]
#     \"\"\"
#     # --- Begin Core Algorithm Logic ---
#     # Implement the {function_name} algorithm as specified.

#     # ... your lean, efficient code goes here ...

#     # --- End Core Algorithm Logic ---

#     # Make sure to:
#     # - Update all required self.* attributes
#     # - Return the correct value (if Outputs specify one)
# ```

# Now generate the complete, lean implementation for the class member method `{function_name}`:
# """

HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT = """
# Task: Design & Implement a Heuristic Algorithm Member Function
You are an expert **Algorithm Designer & Python Developer**. You are working within a predefined architecture to implement a specific member method for a Solver Class.

---

## Mission
Your goal is two-fold:
1.  **Algorithmic Design Phase**: You must bridge the gap between the high-level "Strategic Role" (What to do) and the concrete implementation (How to do it). You need to devise the specific logic (e.g., scoring formulas, filtering criteria, data structures) that best achieves the strategic goal.
2.  **Implementation Phase**: Write the robust, production-quality Python code for `{function_name}`.

**You are implementing Function {function_id} of {total_functions} in the pipeline.**

---

## Context and Inputs

### 1. Mathematical Modeling Problem
{problem_str}

### 2. Existing Solver Class Code (Reference)
Use this code to understand the class structure, available helper methods, and `self.*` attributes.
```python
{heuristic_reference_code}
````

### 3. Function Specification: `{function_name}`
The System Architect has defined the **Strategic Role** (The "What" and "Why") for this function. **Your job is to determine the "How" (The Algorithm).**
- **Strategic Role:** {function_strategic_role}
- **Dependencies (Must Call/Use):** {dependencies}    
    You **must** call:
    - Methods explicitly listed in `Dependencies`.
    - These dependency calls are semantically required: they perform earlier steps of the heuristic, initialize/update shared self.* state, or compute intermediate results that later steps rely on. Skipping these calls will leave required variables uninitialized or key logic unexecuted, and will very likely cause runtime errors or incorrect behavior in subsequent methods.
    You may call:
    - Other methods already present in the existing solver class ModelSolver code (these methods may be called as needed, at your discretion).
    You **must NOT**:
    - Call methods that do not appear in `Dependencies` or the existing solver code.

### 4. Data Flow Contract
**Inputs (Parameters):**
```
{inputs_spec}
```
- This section defines the method's arguments (in addition to `self`). 
- Your method signature **MUST** use exactly these parameters.    
- If this section is "None", the signature is just `def {function_name}(self):`.

**Outputs (Return Value):**
```
{outputs_spec}
```
- This section defines the method's **return value**.
- If this section specifies a return value (i.e., it's not "None"), you **MUST** include a `return` statement returning that value.
- If this section is "None", the method MUST NOT contain ANY return statement (neither `return` nor `return None`), as the method should perform side effects only without returning any value.
    
Member Variables Read (Input State):
{member_variables_read}
- You **MUST** read from these `self.*` attributes to get required state or problem data.  
- Refer to the "Existing Solver Class Code" to see how these might be structured.
   
Member Variables Written (Output State):
{member_variables_written}
- You **MUST** write your results or state changes to these `self.*` attributes.    
- If the function's purpose is to calculate a value, it should be stored in one of these variables.
    
### 5. Previously Completed Functions
This context shows which parts of the algorithm are already implemented.
{completed_functions_summary}

### 6. Historical Insights & Experience Lessons (Guiding Principles)
The following lessons describe key insights derived from previous iterations, including both **successful patterns to maintain** and **pitfalls to avoid**.
**You MUST incorporate these insights into your new design to ensure improvement.**
{experiences}

---

## Implementation Instructions

1. **Design Phase (The "How"):**
    - Analyze the **Strategic Role**.
    - **Orchestration Logic:** If this function depends on other member methods (listed in Dependencies), plan to **INVOKE** them directly. Do not treat this function as a passive consumer of state; treat it as a driver that ensures previous steps happen.
    - Devise a concrete, step-by-step algorithmic procedure to achieve this role efficiently.
    - _Constraint:_ Do not use heavy metaheuristics. Use deterministic, constructive logic (sorting, greedy selection, filtering, math calculation).   
    - _Constraint:_ Ensure you handle the "Member Variables Written" correctly.
        
2. **Coding Phase:**
    - Write production-quality Python code. 
    - **Fail Loudly:** Do not use broad `try-except` blocks. Let errors crash the program for debugging. 
    - **Concise:** No placeholder comments, no print statements.
        
---

## Output Format (CRITICAL):
Return **ONLY a valid JSON object** parseable by `json.loads()`. **Do not** use Markdown, explanatory text, or any extra text.
### JSON Structure:
JSON
```
{{
"func_discription": "A concise text description of the algorithm steps you designed (e.g., '1. Calculate X. 2. Sort Y by Z. 3. Select top K...').",
"func_code": "The complete python code string for the method, starting with 'def {function_name}(self, ...):' including docstrings."
}}
```
"""

HEURISTIC_FUNCTION_CODE_FIX_PROMPT = '''You are a Python bug-fix assistant.

You are given:
1. A short problem description (for context).
2. The full source code of a solver file (including a ModelSolver class).
3. A list of heuristic functions (function architecture).
4. A Python traceback produced when running this solver.
Your job is to:
1) Identify which **heuristic function** is most likely responsible for the error.
2) Rewrite ONLY that function to fix the error.
3) Output ONLY the complete Python definition of the fixed heuristic function.


## Full Solver Code (READ-ONLY CONTEXT)
This is the entire current solver implementation. It may include:
- MILP model construction and Pyomo code.
- Heuristic member functions generated previously.
- Helper methods and utility functions.
You MUST treat all code **outside the buggy heuristic function** as read-only library code.
You are NOT allowed to modify or redesign other methods.
```python
{fullcode_snippet}
```

## Heuristic Functions (candidates for the bug)
This List describes all heuristic member functions that were generated by another agent. Only these functions are allowed to be modified by you.
{functions_list}
You MUST choose the buggy function from this list.
Do NOT pick or modify a function whose name is not in this heuristic functions list.

## Python Traceback (actual error to fix)
This is the real runtime error produced when executing the solver.
- Read it carefully to determine:
- Which file and function frame triggered the error.
- The type of error (AttributeError, KeyError, TypeError, etc.).
- The call stack leading into the failing line.
Traceback:
`{traceback_info}`

## Identification Task: Choose the buggy heuristic function
Using:
- The traceback frames (file, line, function),
- The full solver code,    
- And the heuristic function list,    
you MUST decide **which single heuristic function** is most likely the root cause.

Rules:
- Only consider function that appear in {functions_list}.    
- Prefer the function that:    
    - Appears in the deepest (innermost) relevant frame of the traceback, OR        
    - Contains code that obviously causes the failing operation (e.g., calling a missing method, using a wrong attribute, misusing list indices).        
- If multiple heuristic functions are involved, pick the one that:   
    - Directly contains the line that fails, OR        
    - Immediately calls the line that fails.

Let this chosen function be called `target_function_name`.

## Fixing Task: Rewrite ONLY the chosen function
Once you have selected `target_function_name`, you MUST:
1. Locate its current implementation in the solver code above.
2. Rewrite its body to fix the error indicated in the traceback.
Special rule for AttributeError:
- If the traceback shows an AttributeError like "'ModelSolver' object has no attribute 'X'",
  and the failing line is inside the chosen heuristic function, then your fix MUST:
  - stop calling this missing attribute/method inside that function, OR
  - fully implement the needed logic directly inside this heuristic function (e.g., inline a small feasibility check),
  so that the AttributeError can no longer occur.
You are NOT allowed to leave the call to a missing method in place.

Strict rules:
- Do NOT change:    
    - The function name.        
    - The parameter list/signature (UNLESS the traceback explicitly indicates a `TypeError` regarding argument count mismatch or keyword arguments. In that specific case, you MUST update the signature to match the caller's expectation).
- Do NOT modify any other existing methods in the class.
- You MAY define small local helper functions (inner defs) inside the chosen heuristic function.
- You MUST NOT add new top-level methods on the class (no new "def foo(self, ...)" outside the target function).

Your fix must:
- Address the root cause in the traceback (e.g., missing attribute, wrong variable, type mismatch, bad indexing, etc.).   
- Be as **small and local** as possible.   
- Preserve the existing logical structure and intent of the heuristic.    
- Respect existing data structures and contracts inferred from the full code and heuristic specification.    

Important:
- You MUST actually change the function. Returning code that is identical to the original implementation is invalid.    
- You should use the full solver code to understand variable shapes, list dimensions, and semantic meaning, but the **only** code you are rewriting is the chosen heuristic function.    

## Output Format (IMPORTANT):
You must output only the complete Python definition of the fixed heuristic function:
- Do NOT use Markdown code fences (no orpython).
- Do NOT output any explanation or commentary outside the function.
- Do NOT output more than one function.
- Do NOT add imports or test code.
Your output must follow this general structure:
Python
```
def target_function_name(self, ...):
    \"\"\"[One-line description summarizing the Strategic Role.]
    Args:
        [Parameter]

    Returns:
        [[Return value; omit if None]
    \"\"\"
    # --- Begin Core Algorithm Logic ---
    # Implement the target_function_name algorithm as specified.

    # ... your lean, efficient code goes here ...

    # --- End Core Algorithm Logic ---
```

Now generate the complete, lean implementation for the class member method target_function_name:

'''

HEURISTIC_REFLECT_PROMPT = """
# Task: Heuristic Algorithm Refinement (Logic Optimization under Interface Constraints)
You are a Lead Algorithm Optimization Architect. Your goal is to improve an existing heuristic algorithm (v1.0) based on performance data, while strictly adhering to the existing software architecture.

You will receive the problem definition, the actual code, and a performance report. Your job is to **diagnose logic flaws**, **summarize experience lessons**, and **identify functions to update** to improve solution quality.

---

## Inputs:

### 1. Mathematical Problem & Constraints
{problem_str}

### 2. Full Code Implementation (v1.0)
Analyze the actual Python code to identify implementation weaknesses combined with the performance report.
Python
```python
{fullcode_snippet}
```

### 3. Quality & Bottleneck Report (Performance Feedback)

**Metric Interpretation Guide (CRITICAL):**
- **utilization > 1.0**: **CONGESTION/FAILURE**. The resource is overloaded, directly increasing the Makespan.
- **utilization near 0.0**: **WASTED RESOURCE**. The switch was paid for (Budget K) but the Routing Logic failed to use it.
- **bottleneck_links**: Identify WHERE the jam is. If it's at the `PS_Port`, it means In-Network Aggregation failed to reduce volume effectively.
JSON
```json
{quality_report}
```

### 4. Modifiable Function Set
You are strictly limited to modifying **only** the high-level heuristic functions listed below. Do NOT attempt to modify utility functions, imports, or the solver class structure.
**Valid Targets:** {func_set}

---

## Your Mission:

### Step 1: Diagnose with Evidence Linking (MANDATORY)

**Do not offer vague guesses.** You must explicitly link the **Numerical Symptoms** in the report to the **Logical Causes** in the `v1.0` code.

- **Identify the Gap:** Look for discrepancies between resource capacity and actual usage.
    - **Overload:** Is a specific resource type consistently bottlenecking the system? (Did the code fail to prioritize efficient packing?)
    - **Underutilization:** Are some resources left idle despite pending tasks? (Is the assignment condition too strict? Is the search space definition too narrow?)
- **Trace to Code:** Find the exact line or logic block (e.g., sorting key, if-condition) responsible for this behavior.

### Step 2: Formulate Critical Experience Lessons (Crucial)

Synthesize your diagnosis into actionable "**Experience Lessons**" to guide the next generation of code.
**Objective:** Create a **Causal Narrative** that explains how specific algorithmic choices led to specific outcomes.

Your output will be injected into future prompts as "Historical Insight". It must read naturally, like an engineer explaining a post-mortem.
Structure the lessons to reveal the **Mechanism of Failure**:

1. **State the Strategy:** Explicitly mention the heuristic logic used (e.g., "Sorting by duration," "Assigning to the first valid slot").
2. **State the Outcome:** Connect it to the failure metric (e.g., "caused fragmentation," "overloaded the bottleneck").

**Recommended Pattern:**
"The previous logic of [STRATEGY] failed to handle [SCENARIO], leading to [NEGATIVE OUTCOME]."

Avoid generic statements. Be specific about the *logic* (Sorting, Grouping, Filtering, Thresholds).

These lessons will be appended to the state to guide future code generation.

### Step 3: Identify Functions to Update

Identify which functions in the architecture contain the flawed logic and MUST be updated to apply the lessons derived in Step 2.
- Choose functions **only** from the "Modifiable Function Set".
- Return the list of function names in the `function_update` field.


## Output Format:
Return **ONLY a valid JSON object**.
JSON
```
{{
  "experience_lessons": "The previous logic of sorting tasks by start time caused resource gaps (Utilization < 50%) because it failed to account for task duration.",
  "function_update": ["_step_1_function","_step_4_function", ...
  ]
}}
```
"""

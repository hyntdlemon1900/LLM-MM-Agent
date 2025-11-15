PROBLEM_DESCRIPTION_PROMPT = """\
Problem Background:
{problem_background}

Problem Requirement:
{problem_requirement}

Variable Description:
{variable_description}

Problem Formulation:
{problem_formulation}

Pyomo Reference Code:
The following is a reference Pyomo-based MILP solver implementation for this problem. You can refer to its structure, constraints, and modeling approach when designing your solution:

{pyomo_reference_code}
"""

PROBLEM_CLARIFACATION_PROMPT = """\
{problem_str}

To ensure we have a shared understanding of the task, please confirm the following before starting to write code.

My goal is for you to complete the code implementation based on the information provided above (background, requirements, variables, code template, etc.).

Please respond **only regarding the clarity of the "task description" and "interface definitions"** I provided.
Please confirm:

1.  Are there any ambiguities, unclear points, or contradictions regarding the **task objectives** or **variable descriptions**?
2.  Are the **data formats, structures, or content requirements** for the **function interface**—specifically the Inputs and expected Outputs in the `code_template`—clear?

**Important Note**: You do not need to list the "algorithmic challenges" or "implementation difficulties" of solving this (INA) problem at this stage (e.g., you don't need to ask me how to specifically implement constraint K, or how to extract indicators from `allPathDict`). These are the core problems you will need to solve later when implementing the code.

---
* If you believe my **task description is clear** and the inputs/outputs are well-defined enough for you to begin implementation, please respond with: "**Task description is clear, ready to proceed.**"
* If you find any part of my **task description or interface definitions** to be unclear or conflicting, please list them using a numbered format (1., 2., 3.).
"""

AGENT_FEEDBACK_JUDGE_PROMPT = """\
Agent Feedback:
{agent_feedback}

Analyze the current state of the requested task or question. If there are any remaining ambiguities, unsolved sub-problems, or unfulfilled requirements from the Agent Feedback, respond solely with the string "True". If the task is fully completed, understood, and all requirements are met, respond solely with the string "False". Do not include any additional text or explanation.
"""


CLARIFICATION_ROUND_SUMMARY_PROMPT = """\
You are a professional information extraction assistant. Your task is to extract and summarize only the **new, valid, and actionable** information provided by the user in this round of interaction.

## Current Context

**Current Problem Description:**
{problem_description}

**Agent Feedback/Question:**
{agent_feedback}

**User Reply:**
{user_reply}

---

## Task Requirements

1. **Extract Key Information**: Identify only the new, specific, and actionable details from the user's reply that add clarity or constraints to the problem.

2. **Avoid Redundancy**: Do NOT repeat information already present in the current problem description.

3. **Be Concise**: Output a single paragraph that can be directly appended to the problem description.

4. **No Filler Text**: Do not include pleasantries, acknowledgments, or meta-commentary. Output only the extracted information.

5. **Output Format**: Provide the summary as plain text without any labels, prefixes, or special formatting.

## Output

Please provide the extracted key information directly:
"""


CLARIFICATION_FINAL_SUMMARY_PROMPT = """\
You are a professional information consolidation assistant. Your task is to synthesize all the new and valuable information gathered across multiple rounds of clarification into a single, coherent, and self-contained summary.

## Clarification History

{history_content}

---

## Task Requirements

1. **Consolidate Information**: Merge all new information from the multiple rounds into a unified summary.

2. **Remove Redundancy**: Eliminate duplicate or overlapping information across different rounds.

3. **Ensure Coherence**: The summary should be self-contained and logically structured, suitable for appending to the original problem description.

4. **Maintain Clarity**: Use clear and concise language. Avoid ambiguous or vague statements.

5. **No Meta-Commentary**: Do not include phrases like "The user mentioned..." or "In round X...". Present the information directly.

6. **Output Format**: Provide the consolidated summary as plain text without any labels, prefixes, or special formatting.

## Output

Please provide the consolidated summary directly:
"""

DECOMPOSE_PRINCIPLE_PROMPT = """\
The solution to a mathematical modeling problem is typically broken down into a series of subtasks, each addressing a different aspect of the overall challenge. Based on the examples provided below, summarize what each subtask in tasks 1 through {tasknum} generally involves, with a focus on the principles of task decomposition in mathematical modeling.

<examples>

{examples}

</examples>

Requirements:
1. The summary should focus on the general methods and approaches used in mathematical modeling tasks, not tied to any specific examples or cases provided.
2. The response should not include any details specific to the examples in order to avoid providing any implicit solutions or insights from them.
3. The summary should present a theoretical description of the techniques used at each stage of task decomposition, without any reference to particular problems or contexts.
4. Each subtask should be described as comprehensively and in as much detail as possible within a single paragraph, capturing the essential steps and considerations for that task in a general mathematical modeling framework. The description should be comprehensive, highlighting the key methodologies without resorting to bullet points, numbered lists, or overly formalized structure.
5. Do not provide any form of examples or mention any instances.
"""


TASK_DECOMPOSE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Algorithmic Solution Plan:
{modeling_solution}

---

## Role Definition:
You are an expert task decomposition specialist responsible for breaking down the high-level algorithmic solution plan into a structured sequence of concrete, implementable subtasks.

## Task Objective:
Analyze the provided algorithmic solution plan and decompose it into a logical series of well-defined subtasks that collectively implement the complete algorithm. Each subtask should represent a distinct, cohesive component of the overall solution approach.

## Decomposition Principles:

### 1. Natural Algorithm Structure
Respect the inherent structure of the algorithm as described in the solution plan:
- Identify natural phases or stages in the algorithm (e.g., preprocessing, initialization, iterative optimization, post-processing)
- Recognize logical groupings of related operations
- Maintain the sequential or hierarchical relationships between algorithm components
- Preserve the conceptual integrity of each algorithmic phase

### 2. Clear Separation of Concerns
Each subtask must have a distinct, non-overlapping responsibility:
- Avoid mixing conceptually different operations in a single subtask
- Separate data preparation from algorithmic decision-making
- Distinguish between solution construction and solution validation/refinement
- Keep discrete decision-making separate from continuous optimization
- Isolate constraint handling from objective function optimization

### 3. Completeness and Coverage
Ensure the decomposition covers all aspects of the solution plan:
- Every component, strategy, and mechanism mentioned in the plan must be addressed by at least one subtask
- No algorithmic step or requirement should be left unassigned
- All constraint categories mentioned must be explicitly handled
- Both main algorithmic logic and supporting operations (validation, refinement, output) must be included

### 4. Implementability Focus
Each subtask should be defined at a level suitable for subsequent implementation:
- Describe what computational operations need to be performed
- Specify what data structures or algorithmic techniques are required
- Clarify what inputs the subtask consumes and what outputs it produces
- Indicate how the subtask contributes to the overall algorithm execution
- Make the scope concrete enough that implementation requirements are clear

### 5. Logical Dependencies
Structure subtasks to reflect natural execution dependencies:
- Earlier subtasks should provide necessary data or decisions for later subtasks
- Avoid circular dependencies where tasks mutually depend on each other
- Consider which tasks can potentially execute independently vs. which require sequential ordering
- Ensure prerequisite tasks (data loading, preprocessing) precede tasks that need their results

### 6. Appropriate Granularity
Balance between too coarse and too fine decomposition:
- Each subtask should be substantial enough to represent a meaningful algorithm component
- Avoid trivial subtasks that represent single operations
- Avoid overly complex subtasks that try to do too much
- Aim for subtasks that represent natural units of work in algorithm implementation

### 7. Flexibility in Number
Do NOT assume a fixed number of subtasks. The decomposition should organically follow the structure of the solution plan:
- Simple, direct algorithms may decompose into 2-3 major subtasks
- Complex, multi-phase algorithms may require 5-8 or more subtasks
- Let the problem complexity and solution approach guide the decomposition
- Prioritize logical clarity over hitting an arbitrary subtask count

## Output Format Requirements:

Provide your decomposition as a sequence of subtask descriptions, where:
- Each subtask is described comprehensively in a single, detailed paragraph
- Separate consecutive subtask descriptions with '---' (three dashes on their own line)
- Use plain text format without bullet points, numbered lists, or Markdown formatting
- For each subtask, clearly convey:
  * Its primary purpose and algorithmic role
  * What inputs it requires (data, parameters, or results from previous subtasks)
  * What operations or computations it performs
  * What outputs or results it produces for subsequent subtasks or final output
  * What methods, techniques, or strategies it employs
  * How it relates to the overall algorithm workflow

Ensure that reading all subtasks in sequence provides a complete, step-by-step blueprint for implementing the entire algorithmic solution plan without gaps or ambiguities.
"""


TASK_DESCRIPTION_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Algorithmic Solution Plan:
{modeling_solution}

# All Decomposed Subtasks:
{decomposed_subtasks}

---

## Role Definition:
You are a task specification expert responsible for refining and clarifying the description of Subtask {task_i} to ensure it provides a complete, precise, and self-contained specification that can guide implementation.

## Task Objective:
Enhance the existing description of Subtask {task_i} by adding necessary detail, removing ambiguities, and ensuring all essential information is present for someone to understand exactly what this subtask entails and how to approach its implementation.

## Refinement Requirements:

### 1. Clear Purpose and Scope
Articulate precisely what this subtask accomplishes:
- State the primary objective in clear, concrete terms
- Define the boundaries: what is included and what is explicitly out of scope
- Explain how this subtask contributes to the overall algorithm
- Clarify the algorithmic role this subtask plays (e.g., preprocessing, decision-making, optimization, validation)

### 2. Input Specifications
Explicitly identify all required inputs:
- What data does this subtask need to operate on?
- What parameters or configuration values are required?
- What results from previous subtasks must be available?
- What problem instance information is needed?
- Are there any optional inputs that affect behavior?

### 3. Output Specifications
Clearly define what this subtask produces:
- What are the primary outputs or results?
- What data structures or formats are used for outputs?
- Are outputs intermediate results for other subtasks, or final algorithm outputs?
- What information must be preserved for downstream subtasks?
- Are there any side effects (e.g., files written, state updates)?

### 4. Algorithmic Approach
Describe the methods and techniques to be employed:
- What computational approach should be taken (e.g., greedy heuristic, linear programming, graph algorithm)?
- Are there specific algorithms or data structures that should be used?
- What problem-solving strategies are most appropriate for this subtask?
- Are there established algorithmic patterns that apply?

### 5. Key Considerations
Highlight critical aspects that require attention:
- What constraints must be respected during execution?
- What are potential edge cases or special conditions to handle?
- Where are the main sources of complexity or computational cost?
- What trade-offs exist (e.g., speed vs. accuracy, simplicity vs. optimality)?
- Are there validation or sanity checks that should be performed?

### 6. Integration with Workflow
Explain how this subtask fits into the larger algorithm:
- What prerequisite subtasks must complete first?
- Which subsequent subtasks depend on this subtask's outputs?
- Can this subtask execute independently, or does it require tight integration?
- How does it interact with the overall algorithm control flow?

### 7. Self-Containment
Ensure the description stands alone:
- Provide sufficient context so the subtask can be understood without constantly referring to other subtasks
- Define or clarify any specialized terms or concepts specific to this subtask
- Avoid vague references like "as mentioned earlier" unless you restate the key point
- Make implicit assumptions explicit

## Output Format Requirements:
Provide the refined subtask description as a single, comprehensive paragraph using plain text. Do not use bullet points, numbered lists, or Markdown formatting. The paragraph should be substantial and detailed, integrating all the above elements naturally into a cohesive narrative that fully specifies what Subtask {task_i} entails.

Your refined description should enable someone to clearly understand the subtask's purpose, requirements, approach, and deliverables without needing to consult other documentation.

REFINED SUBTASK {task_i} DESCRIPTION:
"""



PROBLEM_SOLVE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

---

## Role Definition:
You are an expert algorithmic strategist tasked with designing a comprehensive algorithmic solution plan for the given optimization problem. Your role is NOT to implement code or conduct testing, but rather to devise a high-level, systematic strategy that articulates how the problem should be approached and solved algorithmically.

## Task Objective:
Analyze the given problem comprehensively and produce a well-structured algorithmic solution plan that can guide subsequent implementation. Your output should demonstrate deep understanding of the problem structure, constraints, and objectives, while proposing a feasible and effective solving approach.

## Required Analysis and Planning Components:

### 1. Problem Structure Analysis
Begin by thoroughly analyzing the problem to identify:
- Core decision variables and their interdependencies (e.g., discrete placement decisions vs. continuous flow allocations)
- Key constraint categories and their complexity (e.g., capacity constraints, flow conservation, coupling constraints)
- Objective function characteristics (e.g., makespan minimization, throughput maximization)
- Problem scale and computational challenges (e.g., combinatorial explosion, NP-hardness)
- Critical trade-offs inherent in the problem (e.g., solution quality vs. computational tractability)

### 2. Solution Methodology Selection
Propose the fundamental algorithmic approach with clear justification:
- Primary solving paradigm (e.g., decomposition-based methods, heuristic algorithms, relaxation techniques, greedy strategies, iterative refinement)
- Rationale for methodology selection based on problem characteristics (complexity, scale, constraint structure)
- Discussion of why alternative approaches may be less suitable
- Expected computational complexity and scalability properties

### 3. Algorithmic Solution Plan
Provide a detailed, step-by-step plan outlining how the algorithm will solve the problem:
- **Phase 1: Initialization and Preprocessing** - How will the problem data be structured? What preprocessing steps are necessary to facilitate efficient solving?
- **Phase 2: Core Decision Making** - How will discrete decisions be determined? What heuristics or principles guide these choices?
- **Phase 3: Continuous Optimization** - How will continuous variables be allocated given the discrete decisions? What optimization techniques apply?
- **Phase 4: Constraint Satisfaction and Feasibility Enforcement** - How will violations be detected and corrected? What mechanisms ensure all constraints are respected?
- **Phase 5: Solution Refinement and Optimization** - How will the solution quality be iteratively improved? When does the algorithm terminate?

For each phase, clearly specify:
- Input requirements (what information is needed from previous phases)
- Core operations and decision logic
- Output produced (what results are passed to subsequent phases)
- Key algorithmic strategies employed

### 4. Constraint Handling Strategy
Explicitly address how each major constraint category will be satisfied:
- Capacity constraints (switch processing, link bandwidth, port bandwidth)
- Assignment constraints (worker-to-aggregation-point exclusivity)
- Budget constraints (limited INA deployment budget)
- Flow conservation and routing constraints
- Coupling constraints between discrete and continuous decisions

### 5. Optimization Strategy
Articulate how solution quality will be optimized:
- Mechanisms for objective function optimization (makespan minimization)
- Strategies for balancing competing objectives or handling multi-objective aspects
- Techniques for escaping local optima (if applicable)
- Quality guarantees or approximation bounds (if achievable)

### 6. Feasibility Assurance
Explain how the algorithm ensures feasibility throughout:
- Initial feasibility construction methods
- Constraint checking mechanisms during solution generation
- Repair or adjustment strategies when violations occur
- Validation procedures before outputting final solution

### 7. Scalability and Robustness Considerations
Discuss how the approach handles variability:
- Scaling to larger problem instances (more jobs, workers, switches, larger networks)
- Handling edge cases (e.g., insufficient INA budget, highly skewed load distributions, network bottlenecks)
- Robustness to parameter variations or data uncertainty
- Computational efficiency measures to maintain tractability

## Output Format Requirements:
Your response must be comprehensive, detailed, and written in fluent natural language. Use plain text format with LaTeX notation for any necessary mathematical expressions. Structure your response as cohesive, well-connected paragraphs that flow logically from problem analysis to solution plan. Do NOT use bullet points, numbered lists, or Markdown formatting. Each component should be thoroughly explained in paragraph form, demonstrating the logical progression from problem understanding to algorithmic strategy.

Your output should serve as a complete strategic blueprint that enables subsequent agents to understand the solving approach and decompose it into implementable subtasks.
"""


TEMPLATE_GENERATION_PROMPT = """\
You are a code template generator. Your task is to analyze a Pyomo-based MILP solver implementation and generate a clean, simplified template function for an LLM-based solver.

# Pyomo Solver Code:

{pyomo_code}

---

## Your Task:

Analyze the Pyomo solver above and generate a **clean, simplified template function** called `llm_solver` that:

1. **Has the same function signature** as the Pyomo solver's `__init__` method (excluding `self` and solver-specific parameters like `solver_name`, `time_limit`, `mip_gap`, `verbose`)

2. **Has the same return format** as the Pyomo solver's `get_solution` method, but simplified to return only the essential decision variables needed by the evaluation system

3. **Contains comprehensive docstring** that explains:
   - What each input parameter represents (with examples from the Pyomo docstring)
   - What the function should return (format and constraints)
   - Key constraints that must be satisfied
   - Implementation hints or considerations

4. **Includes placeholder implementation** with:
   - Data extraction from inputs
   - Initialization of output data structures
   - TODO comments indicating where the main logic should go
   - A simple example strategy (as comments) that returns a valid (but not optimal) solution

## Requirements:

- **Keep it simple**: Don't include the full Pyomo implementation logic
- **Focus on interface**: Make the input/output specification crystal clear
- **Self-contained docstring**: The docstring should be comprehensive enough that an LLM can implement the function without seeing the Pyomo code
- **Valid default solution**: The placeholder should return a syntactically valid solution (even if performance is poor)
- **No external dependencies**: Don't import Pyomo or other optimization libraries in the template

## Output Format:

Generate **ONLY** the Python function code. Do not include explanations, markdown formatting, or any text outside the function definition.

Example structure:

```python
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    \"\"\"
    [Comprehensive docstring here explaining all inputs, outputs, constraints]
    \"\"\"
    
    # Extract data
    # ...
    
    # Initialize outputs
    # ...
    
    # TODO: Implement your strategy here
    
    # Simple placeholder strategy (REPLACE THIS)
    # ...
    
    return output1, output2
```

Now generate the template function:
"""


TASK_CODE_GENERATION_PROMPT = """\
# Task {task_id} of {total_tasks}: Progressive Solver Implementation

You are implementing **Task {task_id}** in a progressive, incremental development approach. Each task builds upon the previous task's code to gradually complete the full solver implementation.

---

## Overall Algorithm Solution Plan

{algorithm_solution}

---

## Current Task Description (Task {task_id})

{task_description}

---

## Your Base Code ({base_code_type})

Below is the code you should start from:

```python
{base_code}
```

---

## Context: Previous Task

**Previous Task Description**: {previous_task_description}

**Completed Tasks So Far**:
{completed_tasks_summary}

---

{focus_instruction}

---

## Code Generation Requirements

### 1. Code Structure
- Generate a **complete, standalone solver function**
- Function signature must match: `def llm_solver(instance, network, K, jobs_num, Cs, topo_name):`
- Return format must match the expected output: `(ina_placement_expanded, jobs_routing_expanded)`

### 2. Progressive Development Principle
- **Keep ALL code from previous tasks**: Do not remove or modify implementations from completed tasks
- **Add current task's logic**: Implement only the specific algorithm/logic described in Task {task_id}
- **Clear separation**: Use comments to mark which code belongs to which task

### 3. Code Organization
```python
def llm_solver(instance, network, K, jobs_num, Cs, topo_name):
    # Data extraction (common)
    # ...
    
    # ============================================================
    # Task 1: [Task 1 description]
    # ============================================================
    # ... Task 1 implementation (DO NOT MODIFY if task_id > 1) ...
    
    # ============================================================
    # Task 2: [Task 2 description]
    # ============================================================
    # ... Task 2 implementation (DO NOT MODIFY if task_id > 2) ...
    
    # ============================================================
    # Task {task_id}: [Current task description]
    # ============================================================
    # ... YOUR NEW IMPLEMENTATION HERE ...
    
    # Return final solution
    return ina_placement_expanded, jobs_routing_expanded
```

### 4. Implementation Guidelines
- **Preserve working code**: If previous tasks have working implementations, keep them exactly as is
- **Incremental changes only**: Only add/modify code relevant to Task {task_id}
- **Maintain consistency**: Ensure new code integrates smoothly with existing code
- **Comment clearly**: Mark what you added/changed for Task {task_id}

### 5. Output Format
- Return **ONLY** the complete Python code
- No markdown code blocks (no ```python```)
- No explanatory text before or after the code
- The code should be ready to save directly to a .py file

---

## Important Notes

1. **Do not break existing functionality**: Previous tasks have been tested and work correctly
2. **Focus on your task**: Implement only what Task {task_id} requires
3. **Complete function**: Your output must be a complete, runnable solver function
4. **No placeholders in completed sections**: Only use placeholders for future tasks (if any)

Now generate the complete solver code for Task {task_id}:
"""




HEURISTIC_ARCHITECT_PROMPT = """\
# Task: High-Level Algorithm Architecture Design

You are an expert algorithm architect designing a **modular member function architecture** for a HeuristicSolver class.

---

## Inputs:

### 1. Mathematical Modeling Problem
{modeling_problem}

### 2. HeuristicSolver Template Structure
```python
{heuristic_template_code}
```

---

## Your Mission:

Design a **high-level plan** that decomposes the heuristic algorithm into **member functions** (methods of the HeuristicSolver class). Each member function should have:
- A clear, single responsibility
- Well-defined inputs (primarily `self` for accessing class state)
- Well-defined outputs (typically stored as instance variables or returned)

---

## Design Principles:

### 1. Class-Centric Design
- **All functions are member methods** with `self` as the first parameter
- **Leverage instance variables (`self.*`)** for:
  * Problem data (initialized in `__init__` from parameters)
  * Algorithm state (persistent data across method calls)
  * Intermediate results (computed values needed by multiple methods)

### 2. Minimize Parameter Passing
- **Access static problem data via `self`** (don't pass as parameters)
- **Store algorithm state in instance variables** (persistent state in `self.*`)
- **Only pass parameters when necessary** (dynamic values specific to a method invocation)

### 3. Member Variable Management
- **Track evolving class state**: As you design functions, identify which instance variables they will:
  * **Read** (access existing `self.*` attributes)
  * **Write** (create/update `self.*` attributes)
- **Document member variables**: For each function, note which `self.*` variables it uses or creates

### 4. Functional Decomposition
Analyze the algorithmic solution plan and identify the key functional components needed to implement it. Break down the algorithm into member functions based on:
- The natural flow and stages described in the solution plan
- Logical groupings of operations that serve a common purpose
- The data transformations and computations required
- The dependencies between different algorithmic steps

---

## Output Format (CRITICAL):

**Return ONLY a valid JSON array** parseable by `json.loads()`. No markdown, no explanations, no extra text.

### JSON Structure:

```json
[
  {{
    "name": "method_name",
    "description": "What this method does, its algorithmic strategy, and its role in the overall solve workflow",
    "inputs": [
      {{"name": "param", "type": "Type", "description": "Parameter description (if any beyond self)"}}
    ],
    "outputs": [
      {{"name": "result", "type": "Type", "description": "Return value description"}}
    ],
    "member_variables_read": ["self.var1", "self.var2"],
    "member_variables_written": ["self.var3", "self.var4"],
    "dependencies": ["other_method_name"]
  }}
]
```

### Field Descriptions:

- **name**: Method name (lowercase_with_underscores, verb-oriented)
- **description**: Detailed explanation of the method's purpose, algorithm, and integration into the overall workflow
- **inputs**: Parameters **beyond `self`** (only if absolutely necessary; prefer using `self.*`)
  * Most methods should have **empty inputs** or minimal parameters
- **outputs**: Return values (if any)
  * Many methods may return `None` if they update `self.*` directly
  * Return types depend on what the method computes
- **member_variables_read**: List of `self.*` attributes this method **reads**
  * Example format: `["self.param1", "self.param2", "self.computed_data"]`
- **member_variables_written**: List of `self.*` attributes this method **creates or updates**
  * Example format: `["self.result1", "self.result2", "self.state_variable"]`
- **dependencies**: Names of other methods that must execute before this one

### JSON Output Rules:

1. ✓ Start with `[` and end with `]`
2. ✓ Valid JSON syntax (proper quotes, commas, no trailing commas)
3. ✗ NO markdown blocks (```json)
4. ✗ NO explanatory text before/after the JSON
5. ✗ NO comments inside JSON

---

## Example JSON Output Format:

```json
[
  {{
    "name": "function_name_1",
    "description": "Detailed description of what this function does and its algorithmic strategy",
    "inputs": [],
    "outputs": [{{"name": "return_value", "type": "ReturnType", "description": "Description of return value"}}],
    "member_variables_read": ["self.variable_a", "self.variable_b"],
    "member_variables_written": ["self.variable_c"],
    "dependencies": []
  }},
  {{
    "name": "function_name_2",
    "description": "Detailed description of what this function does and its algorithmic strategy",
    "inputs": [{{"name": "param1", "type": "ParamType", "description": "Parameter description"}}],
    "outputs": [{{"name": "result", "type": "ResultType", "description": "Result description"}}],
    "member_variables_read": ["self.variable_c", "self.variable_d"],
    "member_variables_written": ["self.variable_e"],
    "dependencies": ["function_name_1"]
  }}
]
```

---

## Critical Reminders:

1. ✓ Functions are **class methods** (implicit `self` parameter)
2. ✓ **Minimize inputs** - access problem data via `self.*`
3. ✓ **Track member variables** - document what each method reads/writes
4. ✓ **Design for composability** - methods should work together in `solve()`
5. ✓ **Output clean JSON** - parseable by `json.loads()`, no markdown
6. ✓ **No testing functions** - focus on algorithm logic only

Now design the member function architecture and output ONLY the JSON array:
"""


HEURISTIC_TEMPLATE_GENERATION_PROMPT = """\

# Role Definition:
You are an expert code template generator specializing in creating clean, well-structured, and interface-consistent Python frameworks for heuristic solvers.

# Task Objective:
Analyze the user-provided `PyomoSolver` (which inherits from `PyomoTemplateSolver`) code below, and generate a corresponding **`HeuristicSolver` template class**.
This new template **MUST**:
1. **Mirror Public Interface**:    
  - The `__init__` constructor must have the exact same parameter signature as the `PyomoSolver`'s `__init__`.        
  - The return value structure of `solve()` and `get_solution()` (specifically the dictionary structure from `get_solution`) must match the `PyomoSolver`'s `get_solution` _implementation_.
        
2. **Identify & Reuse**:    
  - Automatically identify, copy, and transfer all helper member methods from `PyomoSolver` that **do not depend on Pyomo (`pyo`)** (e.g., data preprocessing, intermediate calculation logic).
        
3. **Be Standalone**:    
  - The generated code **MUST NOT** inherit from `PyomoTemplateSolver` or any Pyomo/Gurobi/CPLEX class. It must be a pure Python class.
        
4. **Provide Framework**:    
  - Provide a `solve()` method with a **placeholder** implementation, ready for a subsequent AI agent to implement the specific heuristic algorithm.        

# Pyomo Solver Reference Code:
Python
```
{pyomo_solver_code}
```

---
## Template Generation Requirements:
### 1. Class Structure
- **Class name**: `HeuristicSolver`    
- **Constructor**: `__init__(self, ...)`    
  - Must extract **all parameters except `self`** from the `PyomoSolver`'s `__init__` above, including type hints and default values.        
- **Core methods**:    
  - `solve(self) -> Optional[Dict[str, Any]]`: Placeholder for the heuristic algorithm implementation.                
- **Reused methods**:    
  - All Pyomo-independent helper functions copied from `PyomoSolver`.        

### 2. Constructor (`__init__`) Requirements
- Strictly replicate the `PyomoSolver.__init__` parameter signature.    
   
- **Store Parameters**:    
  - Store all _problem-specific_ parameters from the `PyomoSolver` signature as instance attributes (e.g., `self.problem_data = problem_data`).
- **Initialization**:  
  - `self.solution = None`
        
### 3. Reusable Helper Method Transfer
- This is a **critical task**. You must analyze all methods in `PyomoSolver`.    
- **Identify**: Find all methods that are _not_ `__init__`, `build_model`, `solve`, or `get_solution`.    
- **Filter**:    
  - **COPY** methods that do _not_ contain `pyo.` calls (e.g., `pyo.Var`, `pyo.Constraint`, `pyo.value`, `pyo.Objective`). These are reusable pure Python logic (e.g., `_preprocess_data`, `_calculate_distances`).        
  - **SKIP** all methods related to Pyomo model building or solver-specific interactions. 

### 4. Solve Method (`solve()`) Requirements
- **Signature**: `def solve(self) -> Optional[Dict[str, Any]]:`  
- **Responsibilities**:    
  1. (To be implemented by another agent) Execute the heuristic algorithm.      
  2. (To be implemented by another agent) Call any reused helper functions for calculations.        
  3. Format the final solution into a dictionary and store it in `self.solution`.        
- **Return Value**: Return the `self.solution` dictionary on success, or `None` on failure.    
- **Placeholder Implementation**:    
  - Print a "not implemented" warning message.      
  - `self.solution = None`      
  - `return self.solution`        

### 5. Get Solution Method (`get_solution()`) Requirements
- **Signature**: `def get_solution(self) -> Optional[Dict[str, Any]]:`    
- **Responsibility**: Maintain interface consistency with `PyomoTemplateSolver`. It only returns the already-computed solution.   
- **Implementation**: `return self.solution`    

### 6. Output Structure Analysis
- You **MUST** carefully analyze the _implementation_ of the **`get_solution`** method in the `PyomoSolver`.    
- **Extract** the **exact dictionary keys and structure** that this method returns.    
- **Annotate** this structure as an example in the **docstring** of the `HeuristicSolver.solve` method to guide the subsequent agent in building the correct output.    

---

## Output Format:
**Strictly** generate ONLY the Python class code. Do not include:
- Markdown code blocks (i.e., do not include `python ...` )   
- Any explanatory text before or after the code.   
- Example usage or test code.
    

**Required Structure:**
from typing import Dict, Any, Optional
# (Add other imports here if required by the reused helper methods)
class HeuristicSolver:

def __init__(self, <parameters extracted from PyomoSolver.__init__>):
    \"\"\"
    Initialize the heuristic solver with the same parameters as the MILP solver.
    
    Args:
      <Preserve parameter docs from Pyomo.__init__>
    
    \"\"\"
    # Store all problem parameters as instance attributes
    # (e.g.: self.problem_data = problem_data)
    
    
    # Initialize the solution
    self.solution = None

def solve(self) -> Optional[Dict[str, Any]]:
    \"\"\"
    Solve the optimization problem using a heuristic algorithm.
    
    (This method is to be implemented by subsequent agents)
    
    Returns:
        Optional[Dict[str, Any]]: A solution dictionary, or None on failure.
        
        On success, the dictionary structure MUST match:
        {{
          '<key1_from_get_solution>': ...,
          '<key2_from_get_solution>': ...,
          # (The keys and structure above must be extracted from the PyomoSolver.get_solution method)
        }}
    \"\"\"
        
    # Placeholder: The heuristic algorithm will compute the solution here

    self.solution = None  # Not yet implemented
    return self.solution

Now generate the HeuristicSolver template based on the Pyomo solver above:
"""


HEURISTIC_FUNCTION_CODE_GENERATION_PROMPT = """\
{problem_str}
---

## HeuristicSolver Template Context

The following HeuristicSolver class template will be used for implementation:

```python
{heuristic_template_code}

Function {function_id} of {total_functions}: {function_name}
You are implementing Function {function_id} as a class member method in a modular heuristic algorithm.

Function Specification
- Function Name: {function_name}
- Description: {function_description}
- Dependencies: {dependencies}

State and Data Flow (CRITICAL)
This function operates within a class and communicates with other methods primarily through `self` attributes.

Inputs (Parameters): {inputs_spec}
- These are ONLY for dynamic data passed as arguments (not static problem data).
- Your method signature MUST use exactly these parameters (in addition to `self`).
- Do NOT add or remove parameters beyond what is specified here.

Outputs (Return Value): {outputs_spec}
- Many functions may return `None` and update `self` instead.
- The return value of your method MUST match this specification (no extra/unmentioned returns).

Member Variables Read: {member_variables_read}
You MUST read from these `self.*` attributes to get the required state or problem data.

Member Variables Written: {member_variables_written}
You MUST write your results or state changes to these `self.*` attributes.

Previously Completed Functions
{completed_functions_summary}

Implementation Requirements
1. Core Philosophy: Concise & Focused Implementation
- Your primary goal is to implement the core algorithm logic described in the function's description.
- The code must be lean, concise, and efficient.
- Avoid all verbosity:
  - Do NOT include `print()` statements.
  - Do NOT include visualization, logging, or placeholder (`pass`) code.

Minimize comments:
- Only add brief comments for non-obvious or complex logic.
- Do NOT add boilerplate comments that restate the code (e.g., `# Initialize variable`).

### 2. Exception Handling and Debuggability (CRITICAL)
- **FAIL LOUDLY**: This code is part of a larger system that **requires clear error tracing** for debugging.
- **DO NOT HIDE ERRORS**: Do **NOT** wrap standard operations (like dictionary lookups, list indexing, or attribute access) in general `try-except` blocks.    
- **ALLOW RUNTIME ERRORS**: It is **ESSENTIAL** that potential `KeyError`, `IndexError`, `AttributeError`, `TypeError`, etc., are **allowed to occur**. The system _must_ crash at the exact line of the error to provide a full and accurate traceback.
- Write **"optimistic" code** that assumes the data and state (managed by `self` and other methods) are correct. Do not write "defensive" code.
- The _only_ time to use `try-except` is if it is a **fundamental part of the heuristic's control flow** (e.g., "try to perform a complex swap, and if it's invalid, revert"). This is rare and should **never** be used for simple data access validation.
    
### 3. State Management (via `self`)
- **Read from `self`**: Access all necessary problem data (e.g., `self.some_problem_parameter`) and current algorithm state (e.g., `self.current_solution_state`) using the `self.*` attributes listed in `Member Variables Read`.    
- **Write to `self`**: Store all persistent results or state changes (e.g., `self.best_solution_found`, `self.updated_state_variable`) using the `self.*` attributes listed in `Member Variables Written`.
- **Do NOT pass static data**: Never pass static problem data (attributes initialized in `__init__`) as parameters to this method. Use `self` to access them.
    

### 4. Method Signature and Docstring
- The method signature **MUST** start with `self`: `def {function_name}(self, ...):`    
- Construct the parameter list directly from the `Inputs` specification above.    
  - If `Inputs` is empty, the signature is exactly: `def {function_name}(self):`        
  - If `Inputs` is non-empty, add those parameters (with type hints) after `self`.
        
- Include type hints for all parameters (except `self`) and for the return value.    
- Write a **CONCISE** docstring:    
  - A single line describing the function's purpose.    
  - Brief `Args:` entries for parameters (only if they exist, beyond `self`).      
  - Brief `Returns:` entry for the return value (only if the function returns something).

Output Format
Generate ONLY the Python method code. Do not include:
- Markdown code blocks (no ```python)
- Explanatory text before or after the method
- Import statements (they are handled globally)
- Test code or example usage

REQUIRED method structure (template):

def {function_name}(self, ...):
    \"\"\"[Concise, one-line description of what this method does.]

    Args:
        [Brief descriptions for parameters beyond self, derived from Inputs spec]

    Returns:
        [Brief description of the return value, derived from Outputs spec]
    \"\"\"
    # Access problem data, e.g.: param = self.some_problem_parameter
    # Access state data, e.g.: state_var = self.some_algorithm_state

    # --- Begin Core Algorithm Logic ---

    # ... implementation ...

    # --- End Core Algorithm Logic ---

    # Write results to self, e.g.: self.some_new_state = ...

    return ...
Now generate the complete, lean implementation for the class member method `{function_name}`:
"""


HEURISTIC_CODE_INTEGRATION_PROMPT = """\
# Role Definition:
You are an expert code integration specialist responsible for combining modular function implementations into a cohesive solver class.

# Task Objective:
Integrate the provided heuristic function implementations into the HeuristicSolver template by filling the empty `solve()` method. The integrated code should orchestrate all functions to create a complete, working heuristic algorithm.

---

## HeuristicSolver Template (with empty solve() method):

```python
{template_code}
```

---

## Heuristic Functions to Integrate ({function_count} functions):

{functions_code}

---

## Function Dependencies (execution order hints):

{function_dependencies}

---

## Integration Requirements:

### 1. Complete the solve() Method
- Fill the empty `solve()` method in the HeuristicSolver class
- Orchestrate all provided functions in a logical sequence
- Respect function dependencies (call dependent functions before those that need their outputs)
- Return `True` on successful solving, `False` on failure

### 2. Function Integration Strategy
- **Data Flow**: Ensure outputs from one function are correctly passed as inputs to dependent functions
- **Error Handling**: Add try-except blocks for robustness, return False on errors
- **Solution Storage**: Store the final solution in `self.solution` dictionary
- **Timing**: Track solve time using `self.solve_time`

### 3. Solution Format
The final `self.solution` dictionary should contain all necessary results that:
- Match the output format expected by `get_solution()` method
- Are compatible with the problem's evaluation framework
- Include all decision variables and performance metrics

### 4. Code Organization
```python
def solve(self) -> bool:
    \"\"\"
    Heuristic solving algorithm.
    
    Orchestrates all helper functions to construct and optimize a solution.
    \"\"\"
    start_time = time.time()
    
    try:
        # Step 1: [First function call]
        # ...
        
        # Step 2: [Second function call]
        # ...
        
        # Step N: [Final processing]
        # Store solution
        self.solution = {{
            # ... solution dictionary ...
        }}
        
        self.solve_time = time.time() - start_time
        
        if self.verbose:
            print(f"✓ Heuristic solving completed in {{self.solve_time:.2f}}s")
        
        return True
    
    except Exception as e:
        if self.verbose:
            print(f"✗ Solving failed: {{e}}")
        self.solve_time = time.time() - start_time
        return False
```

### 5. Code Style
- Clear comments explaining each step
- Meaningful variable names
- Proper error messages
- Verbose output controlled by `self.verbose`

### 6. Important Constraints
- **Include all functions**: Every provided function should be used (unless it's a utility function)
- **Preserve template structure**: Don't modify `__init__`, `get_solution()`, or other template methods
- **Add helper functions below solve()**: Place all function implementations after the solve() method
- **Complete code**: Output must be a fully functional, runnable Python class

---

## Output Format:

Generate ONLY the complete Python class code. Do not include:
- Markdown code blocks (no ```python```)
- Explanatory text before or after the code
- Separate import statements (include them at the top of the class file if needed)
- Test code or example usage

The output should be the complete HeuristicSolver class with:
1. All imports at the top
2. Class definition with filled `solve()` method
3. All helper functions integrated below the class methods

---

Now generate the complete integrated HeuristicSolver code:
"""


PROBLEM_SOLVE_CRITIQUE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Proposed Algorithmic Solution Plan:
{modeling_solution}

---

## Role Definition:
You are a critical algorithmic analyst tasked with rigorously evaluating the proposed algorithmic solution plan. Your objective is to identify weaknesses, gaps, logical inconsistencies, and limitations without providing constructive alternatives. Focus on exposing deficiencies that may compromise correctness, completeness, feasibility, or effectiveness.

## Evaluation Dimensions:

### 1. Problem Understanding and Coverage
Assess whether the solution plan demonstrates complete understanding of the problem:
- Does the plan address all decision variables mentioned in the problem formulation (e.g., INA placement decisions, worker-to-aggregation assignments, flow rate allocations)?
- Are all constraint categories explicitly acknowledged (capacity constraints, assignment constraints, budget constraints, flow conservation, routing constraints)?
- Does the plan recognize all relevant objectives and performance metrics (e.g., makespan minimization, throughput maximization)?
- Are there any problem aspects that have been misunderstood, oversimplified, or entirely overlooked?
- Does the analysis correctly identify the fundamental complexity sources (combinatorial structure, constraint coupling, scale)?

### 2. Methodology Appropriateness and Justification
Critically evaluate the selected algorithmic approach:
- Is the proposed methodology theoretically sound for this problem class? Does it align with established solving paradigms for similar optimization problems?
- Are the stated reasons for methodology selection convincing and technically justified? Or are they superficial and lacking rigor?
- Has the plan considered the computational complexity implications? Does the chosen approach risk exponential growth or intractability?
- Are there obvious methodological alternatives that might be more effective but were not discussed or dismissed without proper justification?
- Does the methodology match the problem's mathematical structure (e.g., using continuous optimization for fundamentally discrete decisions)?

### 3. Algorithmic Logic and Coherence
Examine the internal consistency and logical soundness of the solution plan:
- Is the algorithmic workflow logically structured? Do phases follow a sensible sequence?
- Are there circular dependencies or logical contradictions between different solution steps?
- Does each phase receive the necessary inputs from preceding phases? Are there undefined or missing data flows?
- Are the described operations within each phase clearly specified, or are they vague and ambiguous?
- Is the termination condition well-defined? Could the algorithm get stuck in infinite loops or fail to converge?
- Are there implicit assumptions that have not been stated or justified?

### 4. Constraint Satisfaction Mechanisms
Scrutinize how the plan ensures constraint compliance:
- For each major constraint category, is there an explicit and credible mechanism to ensure satisfaction?
- Are there constraints that are mentioned but lack a clear enforcement strategy?
- Could the proposed decision-making process inadvertently violate constraints (e.g., making greedy choices that lead to capacity violations downstream)?
- Is the constraint checking process comprehensive, or could violations slip through undetected?
- If constraint repair mechanisms are proposed, are they guaranteed to restore feasibility without introducing new violations?
- Does the plan handle constraint interactions and couplings adequately (e.g., discrete placement decisions affecting continuous flow feasibility)?

### 5. Optimization Effectiveness
Evaluate whether the plan is likely to produce high-quality solutions:
- Does the optimization strategy directly target the stated objective function, or does it optimize surrogate metrics that may not correlate well?
- Are there greedy or myopic decision steps that could lead to poor global solutions or getting trapped in low-quality local optima?
- Is there a risk of premature convergence to suboptimal solutions without exploration of the solution space?
- Does the plan lack mechanisms for solution improvement or refinement after initial construction?
- Are there obvious missed opportunities for optimization that a more sophisticated approach could exploit?
- If heuristics are employed, are they well-motivated by problem structure, or are they arbitrary and potentially ineffective?

### 6. Feasibility of Execution
Assess the practical implementability of the proposed plan:
- Are the described algorithmic steps concrete enough to be translated into implementation, or are they too abstract and underspecified?
- Does the plan rely on operations or subroutines that are themselves computationally prohibitive or theoretically uncomputable?
- Are there data structure or computational resource requirements that are unrealistic for the problem scale?
- Could numerical instability, precision issues, or algorithmic pathologies undermine execution?
- Is the plan overly complex, introducing unnecessary complications that increase implementation difficulty without clear benefit?

### 7. Scalability and Robustness Concerns
Identify potential failures under varying conditions:
- How will the algorithm behave as problem instances grow in size (more jobs, workers, switches, larger networks)? Are there bottlenecks that will cause performance degradation or failure?
- Are there edge cases or boundary conditions that could break the algorithm (e.g., zero budget, all-or-nothing scenarios, degenerate network topologies)?
- Does the plan handle asymmetric or highly skewed problem instances (e.g., extreme load imbalances, sparse or dense networks)?
- Is the solution brittle with respect to parameter variations or input noise?
- Are there implicit assumptions about problem structure (e.g., network connectivity, workload balance) that may not hold in practice?

### 8. Clarity, Precision, and Logical Presentation
Critique the communicative quality of the plan:
- Is the explanation clear, or are there ambiguous, vague, or contradictory statements?
- Are technical terms used correctly and consistently throughout?
- Are there logical leaps or unjustified claims (e.g., asserting optimality without proof, claiming feasibility without verification)?
- Is the presentation well-organized, or does it jump between topics without coherent flow?
- Are there missing definitions or unexplained notation that obscures understanding?

## Output Format Requirements:
Present your critique in detailed, cohesive paragraphs using plain text. Use LaTeX notation for any mathematical expressions if necessary. Do NOT use bullet points, numbered lists, or Markdown formatting. Your critique should flow logically through the evaluation dimensions, thoroughly exposing weaknesses and limitations without suggesting solutions or improvements. Be specific in identifying deficiencies, referencing particular aspects of the proposed plan where appropriate.

Your goal is to reveal all substantive flaws so that subsequent refinement can address them systematically.
"""


PROBLEM_SOLVE_IMPROVEMENT_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Current Algorithmic Solution Plan:
{modeling_solution}

# Critical Evaluation:
{modeling_solution_critique}

---

## Role Definition:
You are an expert algorithmic strategist tasked with refining and enhancing the algorithmic solution plan based on the critical evaluation provided. Your objective is to produce an improved, comprehensive solution plan that addresses all identified weaknesses, gaps, and limitations while maintaining or enhancing the strengths of the original approach.

## Improvement Objectives:

### 1. Address Identified Deficiencies
Systematically resolve all substantive issues raised in the critique:
- Correct any misunderstandings of problem structure, constraints, or objectives
- Fill gaps in problem coverage where decision variables, constraints, or objectives were overlooked
- Strengthen weak or unsound methodological choices with more appropriate alternatives
- Resolve logical inconsistencies, circular dependencies, or undefined workflows
- Specify concrete mechanisms for all constraint categories that lacked clear enforcement strategies
- Enhance optimization strategies to avoid myopic decisions, local optima traps, or misaligned objectives
- Improve implementability by providing more concrete, well-defined algorithmic steps
- Address scalability concerns and edge case handling deficiencies
- Clarify ambiguous or vague explanations with precise, technically rigorous descriptions

### 2. Enhance Solution Quality
Beyond addressing critiques, proactively strengthen the overall solution plan:
- Deepen the problem analysis to reveal additional structure or opportunities for exploitation
- Refine the methodology to better align with the problem's mathematical characteristics and computational constraints
- Elaborate algorithmic phases with greater specificity regarding operations, data transformations, and decision logic
- Strengthen constraint satisfaction guarantees through more robust enforcement and verification mechanisms
- Improve optimization effectiveness through better exploration strategies, refinement procedures, or quality bounds
- Enhance feasibility assurance by incorporating validation checks and recovery mechanisms
- Bolster scalability through algorithmic optimizations, preprocessing efficiencies, or adaptive strategies
- Increase robustness by anticipating and handling a broader range of edge cases and parameter variations

### 3. Maintain Clarity and Rigor
Ensure the improved plan is both technically sound and communicatively effective:
- Use precise technical language with consistent terminology throughout
- Provide explicit justifications for all methodological choices and strategic decisions
- Ensure logical flow and coherence across all solution phases
- Eliminate ambiguities, vague statements, or unjustified claims
- Present a complete, self-contained solution plan that can guide subsequent implementation

### 4. Preserve Original Strengths
Retain and build upon the effective elements of the original plan:
- Maintain valid insights and sound methodological choices from the original solution
- Preserve structural elements that were correctly formulated
- Build incrementally on the foundation established, rather than discarding all previous work unnecessarily

## Output Requirements:
Present the improved algorithmic solution plan as a cohesive, comprehensive document. Write in fluent natural language using plain text format with LaTeX notation for mathematical expressions where necessary. Structure your response as well-developed paragraphs that demonstrate logical progression from problem analysis through solution methodology to algorithmic planning. Do NOT use bullet points, numbered lists, or Markdown formatting.

**Critical Instruction**: Do NOT reference or discuss deficiencies from the previous version in your improved solution. Simply present the refined plan as if it were the first and definitive version. The improved solution should stand alone as a complete, authoritative strategic blueprint without meta-commentary about what was changed or why.

Your output should provide a clear, rigorous, and comprehensive algorithmic solution plan that addresses all aspects of the problem and can serve as the definitive guide for subsequent task decomposition and implementation.

IMPROVED ALGORITHMIC SOLUTION PLAN:
"""


PROBLEM_DECOMPOSE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Algorithmic Solution Plan:
{modeling_solution}

---

## Role Definition:
You are an expert task decomposition specialist responsible for breaking down the high-level algorithmic solution plan into exactly {tasknum} concrete, implementable subtasks.

## Task Objective:
Analyze the provided algorithmic solution plan and decompose it into precisely {tasknum} well-defined subtasks that collectively implement the complete algorithm. Each subtask should represent a distinct, cohesive component of the overall solution approach.

## Decomposition Principles:

### 1. Natural Algorithm Structure
Respect the inherent structure of the algorithm as described in the solution plan:
- Identify natural phases or stages in the algorithm (e.g., preprocessing, initialization, iterative optimization, post-processing)
- Recognize logical groupings of related operations
- Maintain the sequential or hierarchical relationships between algorithm components
- Preserve the conceptual integrity of each algorithmic phase

### 2. Clear Separation of Concerns
Each subtask must have a distinct, non-overlapping responsibility:
- Avoid mixing conceptually different operations in a single subtask
- Separate data preparation from algorithmic decision-making
- Distinguish between solution construction and solution validation/refinement
- Keep discrete decision-making separate from continuous optimization
- Isolate constraint handling from objective function optimization

### 3. Completeness and Coverage
Ensure the decomposition covers all aspects of the solution plan:
- Every component, strategy, and mechanism mentioned in the plan must be addressed by at least one subtask
- No algorithmic step or requirement should be left unassigned
- All constraint categories mentioned must be explicitly handled
- Both main algorithmic logic and supporting operations (validation, refinement, output) must be included

### 4. Implementability Focus
Each subtask should be defined at a level suitable for subsequent implementation:
- Describe what computational operations need to be performed
- Specify what data structures or algorithmic techniques are required
- Clarify what inputs the subtask consumes and what outputs it produces
- Indicate how the subtask contributes to the overall algorithm execution
- Make the scope concrete enough that implementation requirements are clear

### 5. Logical Dependencies
Structure subtasks to reflect natural execution dependencies:
- Earlier subtasks should provide necessary data or decisions for later subtasks
- Avoid circular dependencies where tasks mutually depend on each other
- Consider which tasks can potentially execute independently vs. which require sequential ordering
- Ensure prerequisite tasks (data loading, preprocessing) precede tasks that need their results

### 6. Appropriate Granularity
Balance between too coarse and too fine decomposition:
- Each subtask should be substantial enough to represent a meaningful algorithm component
- Avoid trivial subtasks that represent single operations
- Avoid overly complex subtasks that try to do too much
- Distribute the algorithm logic evenly across the {tasknum} subtasks

### 7. Required Subtask Count
You must produce exactly {tasknum} subtasks:
- Analyze the solution plan and identify the {tasknum} most logical decomposition points
- If the natural structure suggests fewer components, combine related operations
- If the natural structure suggests more components, group closely related operations together
- Ensure each of the {tasknum} subtasks has substantial, meaningful content

## Output Format Requirements:

Provide exactly {tasknum} subtask descriptions, where:
- Each subtask is described comprehensively in a single, detailed paragraph
- Separate consecutive subtask descriptions with '---' (three dashes on their own line)
- Use plain text format without bullet points, numbered lists, or Markdown formatting
- For each subtask, clearly convey:
  * Its primary purpose and algorithmic role
  * What inputs it requires (data, parameters, or results from previous subtasks)
  * What operations or computations it performs
  * What outputs or results it produces for subsequent subtasks or final output
  * What methods, techniques, or strategies it employs
  * How it relates to the overall algorithm workflow

Ensure that reading all {tasknum} subtasks in sequence provides a complete, step-by-step blueprint for implementing the entire algorithmic solution plan without gaps or ambiguities.
"""


TASK_DESCRIPTION_REFINEMENT_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Algorithmic Solution Plan:
{modeling_solution}

# All Decomposed Subtasks:
{decomposed_subtasks}

---

## Role Definition:
You are a task specification expert responsible for refining and clarifying the description of **Subtask {task_i}** to ensure it provides a complete, precise, and self-contained specification that can guide implementation.

## Task Objective:
Enhance the existing description of Subtask {task_i} by adding necessary detail, removing ambiguities, and ensuring all essential information is present for someone to understand exactly what this subtask entails and how to approach its implementation.

## Refinement Requirements:

### 1. Clear Purpose and Scope
Articulate precisely what this subtask accomplishes:
- State the primary objective in clear, concrete terms
- Define the boundaries: what is included and what is explicitly out of scope
- Explain how this subtask contributes to the overall algorithm
- Clarify the algorithmic role this subtask plays (e.g., preprocessing, decision-making, optimization, validation)

### 2. Input Specifications
Explicitly identify all required inputs:
- What data does this subtask need to operate on?
- What parameters or configuration values are required?
- What results from previous subtasks must be available?
- What problem instance information is needed?
- Are there any optional inputs that affect behavior?

### 3. Output Specifications
Clearly define what this subtask produces:
- What are the primary outputs or results?
- What data structures or formats are used for outputs?
- Are outputs intermediate results for other subtasks, or final algorithm outputs?
- What information must be preserved for downstream subtasks?
- Are there any side effects (e.g., files written, state updates)?

### 4. Algorithmic Approach
Describe the methods and techniques to be employed:
- What computational approach should be taken (e.g., greedy heuristic, linear programming, graph algorithm)?
- Are there specific algorithms or data structures that should be used?
- What problem-solving strategies are most appropriate for this subtask?
- Are there established algorithmic patterns that apply?

### 5. Key Considerations
Highlight critical aspects that require attention:
- What constraints must be respected during execution?
- What are potential edge cases or special conditions to handle?
- Where are the main sources of complexity or computational cost?
- What trade-offs exist (e.g., speed vs. accuracy, simplicity vs. optimality)?
- Are there validation or sanity checks that should be performed?

### 6. Integration with Workflow
Explain how this subtask fits into the larger algorithm:
- What prerequisite subtasks must complete first?
- Which subsequent subtasks depend on this subtask's outputs?
- Can this subtask execute independently, or does it require tight integration?
- How does it interact with the overall algorithm control flow?

### 7. Self-Containment
Ensure the description stands alone:
- Provide sufficient context so the subtask can be understood without constantly referring to other subtasks
- Define or clarify any specialized terms or concepts specific to this subtask
- Avoid vague references like "as mentioned earlier" unless you restate the key point
- Make implicit assumptions explicit

## Output Format Requirements:
Provide the refined subtask description as a single, comprehensive paragraph using plain text. Do not use bullet points, numbered lists, or Markdown formatting. The paragraph should be substantial and detailed, integrating all the above elements naturally into a cohesive narrative that fully specifies what Subtask {task_i} entails.

Your refined description should enable someone to clearly understand the subtask's purpose, requirements, approach, and deliverables without needing to consult other documentation.

REFINED SUBTASK {task_i} DESCRIPTION:
"""

TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT = """\
Understanding the dependencies among different tasks in a mathematical modeling process is crucial for ensuring a coherent, logically structured, and efficient solution. Given a mathematical modeling problem and its solution decomposition into {tasknum} subtasks, analyze the interdependencies among these subtasks.  

## Input Information:
- **Mathematical Modeling Problem:** {modeling_problem}
- **Modeling Solution:** {modeling_solution}
- **Decomposed Tasks:** {task_descriptions}

## Task Dependency Analysis Instructions:
1. **Identify Task Dependencies:** For each task, determine which preceding tasks provide necessary input, data, or conditions for its execution. Clearly outline how earlier tasks influence or constrain later ones.
2. **Describe Dependency Types:** Specify the nature of the dependencies between tasks. This includes:
   - *Data Dependency:* When one task produces outputs that are required as inputs for another task.
   - *Methodological Dependency:* When a later task builds upon a theoretical framework, assumptions, or models established by an earlier task.
   - *Computational Dependency:* When a task requires prior computations or optimizations to be completed before proceeding.
   - *Structural Dependency:* When a task is logically required to be completed before another due to hierarchical or sequential constraints.
   - *Code Dependency:* When one task relies on code structures, functions, or modules that are defined or executed in a preceding task. This includes shared variables, functions, or libraries that must be defined before their use in later tasks.
3. **Ensure Completeness:** Verify that all tasks in the decomposition are accounted for in the dependency analysis and that no essential dependencies are missing.

## Output Format:  
Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text, without any Markdown formatting or syntax. Written as {tasknum} cohesive paragraphs, each paragraph is a dependency analysis of a task.

The response should be comprehensive and written in a clear, well-structured format without bullet points, ensuring a logical flow of dependency relationships and their implications.
"""

TASK_DEPENDENCY_ANALYSIS_PROMPT = """\
Understanding the dependencies among different tasks in a mathematical modeling process is crucial for ensuring a coherent, logically structured, and efficient solution. Given a mathematical modeling problem and its solution decomposition into {tasknum} subtasks, analyze the interdependencies among these subtasks.  

## Input Information:
- **Mathematical Modeling Problem:** {modeling_problem}
- **Modeling Solution:** {modeling_solution}
- **Decomposed Tasks:** {task_descriptions}

## Task Dependency Analysis Instructions:
1. **Identify Task Dependencies:** For each task, determine which preceding tasks provide necessary input, data, or conditions for its execution. Clearly outline how earlier tasks influence or constrain later ones.
2. **Describe Dependency Types:** Specify the nature of the dependencies between tasks. This includes:
   - *Data Dependency:* When one task produces outputs that are required as inputs for another task.
   - *Methodological Dependency:* When a later task builds upon a theoretical framework, assumptions, or models established by an earlier task.
   - *Computational Dependency:* When a task requires prior computations or optimizations to be completed before proceeding.
   - *Structural Dependency:* When a task is logically required to be completed before another due to hierarchical or sequential constraints.
3. **Ensure Completeness:** Verify that all tasks in the decomposition are accounted for in the dependency analysis and that no essential dependencies are missing.

## Output Format:  
Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text, without any Markdown formatting or syntax. Written as {tasknum} cohesive paragraphs, each paragraph is a dependency analysis of a task.

The response should be comprehensive and written in a clear, well-structured format without bullet points, ensuring a logical flow of dependency relationships and their implications.
"""

DAG_CONSTRUCTION_PROMPT = """\
A well-structured Directed Acyclic Graph (DAG) is essential for visualizing and optimizing the dependencies between different tasks in a mathematical modeling process. Given a problem and its solution decomposition into {tasknum} subtasks, construct a DAG that accurately represents the dependency relationships among these tasks. The DAG should capture all necessary dependencies while ensuring that no cycles exist in the structure.  

## Input Information:
- **Mathematical Modeling Problem:** {modeling_problem}
- **Modeling Solution:** {modeling_solution}
- **Decomposed Tasks:** {task_descriptions}
- **Dependency Analysis:** {task_dependency_analysis}

## Output Format (STRICT REQUIREMENT):
You **MUST** return a valid JSON-formatted adjacency list **without** any additional text, explanations, or comments. **Only** output the JSON object.

### JSON Format (Strictly Follow This Format):
```json
{{
  "task_ID": [dependent_IDs],
  ...
}}

## Example Output: 
```json
{{
"1": []
"2": ['1']
"3": ['1']
"4": ['2', '3']
}}
```
"""

TASK_ANALYSIS_PROMPT = """\
# Task Description:
{task_description}

---
{prompt}

You are collaborating as part of a multi-agent system to solve a complex mathematical modeling problem. Each agent is responsible for a specific task, and some preprocessing or related tasks may have already been completed by other agents. It is crucial that you **do not repeat any steps that have already been addressed** by other agents. Instead, rely on their outputs when necessary and focus solely on the specific aspects of the task assigned to you.

Provide a thorough and nuanced analysis of the task at hand, drawing on the task description as the primary source of context. Begin by elucidating the core objectives and scope of the task, outlining its significance within the larger context of the project or research. Consider the potential impact or outcomes that are expected from the task, whether they relate to solving a specific problem, advancing knowledge, or achieving a particular practical application. Identify any challenges that may arise during the task execution, including technical, logistical, or theoretical constraints, and describe how these might influence the process or outcomes. In addition, carefully highlight any assumptions that are being made about the data, environment, or system involved in the task, and discuss any external factors that could shape the understanding or execution of the task. Ensure that the analysis is framed in a way that will guide future steps or inform the next stages of work.
{user_prompt}
Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text and LaTeX for formulas only, without any Markdown formatting or syntax. Written as one paragraph. Avoid structuring your answer in bullet points or numbered lists.
"""

TASK_FORMULAS_PROMPT = """\
# Reference Modeling Methods:
{modeling_methods}

# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

---
{prompt}

You are collaborating as part of a multi-agent system to solve a complex mathematical modeling problem. Each agent is responsible for a specific task, and some preprocessing or related tasks may have already been completed by other agents. It is crucial that you **do not repeat any steps that have already been addressed** by other agents. Instead, rely on their outputs when necessary and focus solely on the specific aspects of the task assigned to you.

You are tasked with developing a set of precise, insightful, and comprehensive mathematical formulas that effectively model the problem described in the task. Begin by conducting an in-depth analysis of the system, process, or phenomenon outlined, identifying all relevant variables, their interdependencies, and the fundamental principles, laws, or constraints that govern the behavior of the system, as applicable in the relevant field. Clearly define all variables, constants, and parameters, and explicitly state any assumptions, approximations, or simplifications made during the formulation process, including any boundary conditions or initial conditions if necessary.

Ensure the formulation considers the full scope of the problem, and if applicable, incorporate innovative mathematical techniques. Your approach should be well-suited for practical computational implementation, addressing potential numerical challenges, stability concerns, or limitations in simulations. Pay careful attention to the dimensional consistency and units of all terms to guarantee physical or conceptual validity, while remaining true to the theoretical foundations of the problem.

In the process of deriving the mathematical models, provide a clear, step-by-step explanation of the reasoning behind each formula, highlighting the derivation of key expressions and discussing any assumptions or trade-offs that are made. Identify any potential sources of uncertainty, limitations, or approximations inherent in the model, and provide guidance on how to handle these within the modeling framework.

The resulting equations should be both flexible and scalable, allowing for adaptation to different scenarios or the ability to be tested against experimental or real-world data. Strive to ensure that your model is not only rigorous but also interpretable, balancing complexity with practical applicability. List all modeling equations clearly in LaTeX format, ensuring proper mathematical notation and clarity of presentation. Aim for a model that is both theoretically sound and practically relevant, offering a balanced approach to complexity and tractability in its use.
{user_prompt}
Respond as comprehensively and in as much detail as possible, ensuring clarity, depth, and rigor throughout. Using plain text and LaTeX for formulas. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
"""


TASK_FORMULAS_CRITIQUE_PROMPT = """\

# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{modeling_formulas}

---

The goal of this task is to critically evaluate the modeling formulas used to represent a given mathematical modeling problem. Your analysis should address the following dimensions: accuracy and rigor, innovation and insight, and the applicability of the models to real-world scenarios.

1. Accuracy and Rigor:

- Formula Integrity:  
  Evaluate whether the mathematical models and the corresponding formulas are mathematically sound and consistent with the underlying assumptions of the problem. Are the formulas properly derived, free from logical errors, and reflective of the relevant domain knowledge?  
  - Are any simplifications or approximations made, and if so, are they justifiable within the context of the model's scope?
  - Examine the assumptions made in formulating the model. Are these assumptions realistic, and how do they affect the model’s precision and robustness?

2. Innovation and Insight:

- Novelty of Approach:  
  Critique the originality of the modeling approach. Does the model present a new or unconventional way of solving the problem, or does it simply rely on established methodologies without offering new insights?  
  - Consider whether any innovative methods, such as the introduction of novel variables or the use of innovative computational techniques, contribute to improving the model.

- Theoretical Insight:  
  Evaluate the depth of the theoretical insights provided by the model. Does it offer a fresh perspective or new understanding of the problem? How well does it illuminate the key dynamics and relationships within the system under study?  
  - Does the model reveal previously unnoticed phenomena, or does it suggest new directions for further research?

- Integration of Existing Knowledge:  
  Assess the extent to which the model integrates existing mathematical, theoretical, and empirical work. Does it build on prior research, and if so, does it do so in a way that adds substantial value or clarity? Are there gaps where additional cross-disciplinary knowledge could enhance the model?

---

3. Applicable:

- Real-World Relevance:  
  Evaluate the model’s practical applicability. How well does it apply to real-world problems, and to what extent does it provide actionable insights for decision-making or problem-solving in the field?  

Critique the analysis without offering any constructive suggestions—your focus should solely be on highlighting weaknesses, gaps, and limitations within the formulas.
"""


TASK_FORMULAS_IMPROVEMENT_PROMPT = """\


# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{modeling_formulas}

# Task Modeling Formulas Critique:
{modeling_formulas_critique}

---

Based on the provided critique and analysis, refine the existing modeling formulas to address the identified limitations and gaps. 

Respond as comprehensively and in as much detail as possible, ensuring clarity, depth, and rigor throughout. Using plain text and LaTeX for formulas. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
{user_prompt}
Provide a new version of the task modeling formulas that integrates these improvements directly. DO NOT mention any previous formulas content and deficiencies.

IMPROVED TASK MODELING FORMULAS:
"""


TASK_MODELING_PROMPT = """\


# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{modeling_formulas}

---
{prompt}

You are collaborating as part of a multi-agent system to solve a complex mathematical modeling problem. Each agent is responsible for a specific task, and some preprocessing or related tasks may have already been completed by other agents. It is crucial that you **do not repeat any steps that have already been addressed** by other agents. Instead, rely on their outputs when necessary and focus solely on the specific aspects of the task assigned to you.

Please continue the modeling formula section by building upon the previous introduction to the formula. Provide comprehensive and detailed explanations and instructions that elaborate on each component of the formula. Describe the modeling process thoroughly, including the underlying assumptions, step-by-step derivations, and any necessary instructions for application. Expand on the formula by incorporating relevant mathematical expressions where appropriate, ensuring that each addition enhances the reader’s understanding of the model. Make sure to seamlessly integrate the new content with the existing section, maintaining a natural flow and avoiding any repetition or conflicts with previously covered material. Your continuation should offer a clear and in-depth exploration of the modeling formula, providing all necessary details to facilitate a complete and coherent understanding of the modeling process.
{user_prompt}
Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
"""


TASK_MODELING_CRITIQUE_PROMPT = """\


# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{modeling_formulas}

# Task Modeling Process:
{modeling_process}

---

Critically examine the analysis results of the given mathematical modeling solution, focusing on the following aspects:

1. Problem Analysis and Understanding:
- Clarity of the problem definition: Does the solution demonstrate a clear and comprehensive understanding of the problem? Are all relevant variables, constraints, and objectives identified and well-defined? If not, which aspects of the problem may have been misunderstood or overlooked?
- Contextualization and framing: How well does the model account for the context in which the problem is situated? Are there any contextual factors that are essential but were not addressed?
- Scope of the problem: Is the problem's scope appropriately defined? Does the model include all the necessary details, or are there significant components that were neglected or oversimplified?

2. Model Development and Rigor:
- Formulation of the mathematical model: How well is the model constructed mathematically? Does it align with established modeling practices in the relevant domain? Are the mathematical formulations—such as equations, algorithms, or optimization methods—correct and robust?
- Modeling techniques: What modeling approaches or techniques were used (e.g., linear programming, system dynamics, statistical modeling, etc.)? Are they the most appropriate for the problem at hand? What alternative approaches could have been considered, and how might they impact the solution?
- Validation and verification: Was the model tested for consistency and accuracy? Are there validation steps in place to ensure the model behaves as expected under a variety of conditions? What specific methods were used for this validation (e.g., cross-validation, sensitivity analysis, etc.)?

3. Data and Results Analysis:
- Data quality and relevance: Were there any significant issues with data availability or quality that could have influenced the model's results?
- Interpretation of results: How well were the results analyzed and interpreted? Were the outcomes consistent with the problem's real-world implications? Are there any discrepancies between the model’s results and known empirical observations?
- Sensitivity and robustness analysis: Did the model undergo a sensitivity analysis to determine how the results vary with changes in input parameters? Were the results robust across different assumptions, and if not, what are the implications for the solution's reliability?

4. Assumptions and Limitations:
- Explicit and implicit assumptions: What assumptions underlie the model, and are they clearly articulated? Are these assumptions reasonable, and how might they affect the model's predictions? Were any critical assumptions left implicit or unaddressed?
- Limitations of the model: What limitations are inherent in the model, and how do they affect its validity and reliability? Are there elements of the problem that are inherently difficult or impossible to model with the chosen approach? Were simplifications made, and what are the trade-offs involved?
- Model boundaries: Does the model appropriately define its boundaries, and are there any critical factors that lie outside the model’s scope but could significantly influence the results?

5. Practicality and Applicability:
- Real-world applicability: To what extent can the model be applied to real-world scenarios? 
- Practical implementation: How would this model be implemented in practice? What would be the required infrastructure, and what challenges would need to be addressed during implementation? 

Critique the analysis without offering any constructive suggestions—your focus should solely be on highlighting weaknesses, gaps, and limitations within the approach and its execution.
"""


TASK_MODELING_IMPROVEMENT_PROMPT = """\


# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{modeling_formulas}

# Task Modeling Process:
{modeling_process}

# Task Modeling Process Critique:
{modeling_process_critique}

---

Refine and improve the existing modeling process based on the critique provided. The goal is to enhance the formulation, structure, and overall effectiveness of the model while addressing the identified gaps, flaws, or limitations. Propose more appropriate assumptions, more robust mathematical techniques, or alternative modeling approaches if necessary. Focus on improving the model's relevance, accuracy, and computational feasibility while also ensuring its ability to capture the complexity of the problem in real-world contexts.

Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
{user_prompt}
Provide a new version of the modeling process that integrates these improvements directly. DO NOT mention any previous process content and deficiencies.

IMPROVED MODELING PROCESS:
"""

TASK_CODING_PROMPT = """\
# Dataset Path:
{data_file}

# Data Description:


# Variable Description:
{variable_description}

# Other files (Generated by Other Agents):
{dependent_file_prompt}

# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{modeling_formulas}

# Task Modeling Process:
{modeling_process}

# Code Template:
{code_template}

---

## Role & Collaboration:
You are an expert programmer working as part of a multi-agent system. Your role is to implement the code based on the provided dataset (**refer to the Dataset Path, Dataset Description, and Variable Description**) **or preprocessed files generated by other agents** (**refer to "Other Files"**), along with the modeling process and given code template. Other agents will use your results to make decisions, but they will **not** review your code. Therefore, it is crucial that:
1. **Ensure the code is executable** and will successfully run without errors, producing the expected results. **It should be tested to verify it works in the intended environment**.
2. **Reuse files from "Other Files" whenever possible** instead of redoing tasks that have already been completed by other agents.
3. **All data processing steps must save the processed results to local files (CSV, JSON, or pickle) for easy access by other agents.**
4. **The output should be as detailed as possible**, including intermediate results and final outputs.
5. **Ensure transparency** by logging key computation steps and providing clear outputs.

## Implementation Guidelines:
- **Prioritize using files from "Other Files" before processing raw data** to avoid redundant computation.
- Follow the provided **modeling formulas** and **modeling process** precisely.
- The **code must be executable**: ensure that the Python code you generate runs without errors. Do not just focus on producing the correct output format; **focus on producing a working solution** that can be executed successfully in a Python environment.
- **Store intermediate and final data processing results to local** in appropriate formats (e.g., CSV, JSON, or pickle).
- Provide **detailed print/logging outputs** to ensure that other agents can understand the results without needing to read the code.
{user_prompt}

## Expected Response Format:
You **MUST** return the Python implementation in the following format:
```python
# Here is the Python code.
"""


TASK_CODING_DEBUG_PROMPT = """\
# Code Template:
{code_template}

# Modeling Process:
{modeling_process}

# Current Code:
{code}

However, there are some bugs in this version. Here is the execution result:
# Execution Result:
{observation}

---

You are a helpful programming expert. Based on the provided execution result, please revise the script to fix these bugs. Your task is to address the error indicated in the result, and refine or modify the code as needed to ensure it works correctly.
{user_prompt}
Please respond exactly in the following format:
```python
# Provide the corrected python code here.
```
"""


TASK_RESULT_PROMPT = """\
# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{task_formulas}

# Task Modeling:
{task_modeling}

---

Based on the task description, analysis, and modeling framework, present a comprehensive and detailed account of the intermediate results, calculations, and outcomes generated during the task. Clearly articulate the results of any simulations, experiments, or calculations, providing numerical values, data trends, or statistical measures as necessary. If visual representations such as graphs, charts, or tables were used to communicate the results, ensure they are clearly labeled and explained, highlighting their relevance to the overall task. Discuss the intermediate steps or processes that led to the results, including any transformations or assumptions made during calculations. If applicable, compare and contrast these results with expected outcomes or previously known results to gauge the task’s success. Provide a thoughtful interpretation of the findings, considering how they contribute to advancing understanding or solving the problem at hand, and highlight any areas where further investigation or refinement may be needed.
{user_prompt}
Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text and LaTeX for formulas only, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
"""

TASK_RESULT_WITH_CODE_PROMPT = """\
# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{task_formulas}

# Task Modeling:
{task_modeling}

# Code Execution Result:
{execution_result}

---

Based on the task description, analysis, modeling framework, and code execution result, present a comprehensive and detailed account of the intermediate results, calculations, and outcomes generated during the task. Clearly articulate the results of any computations or operations performed, providing numerical values, data trends, or statistical measures as necessary. If visual representations such as graphs, charts, or tables were used to communicate the results, ensure they are clearly labeled and explained, highlighting their relevance to the overall task. Discuss the intermediate steps or processes that led to the results, including any transformations or assumptions made during calculations. If applicable, compare and contrast these results with expected outcomes or previously known results to gauge the task’s success. Provide a thoughtful interpretation of the findings, considering how they contribute to advancing understanding or solving the problem at hand, and highlight any areas where further investigation or refinement may be needed.
{user_prompt}
Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text and LaTeX for formulas only, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
"""


TASK_ANSWER_PROMPT = """\
# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Task Modeling Formulas:
{task_formulas}

# Task Modeling:
{task_modeling}

# Task Result:
{task_result}

---

Craft a comprehensive and insightful answer section that synthesizes the findings presented in the results section to directly address the research questions and objectives outlined at the outset of the study. Begin by clearly stating the primary conclusions drawn from the analysis, ensuring that each conclusion is explicitly linked to specific aspects of the results. Discuss how these conclusions validate or challenge the initial hypotheses or theoretical expectations, providing a coherent narrative that illustrates the progression from data to insight.

Evaluate the effectiveness and reliability of the mathematical models employed, highlighting strengths such as predictive accuracy, robustness, or computational efficiency. Address any limitations encountered during the modeling process, explaining how they may impact the validity of the conclusions and suggesting potential remedies or alternative approaches. Consider the sensitivity of the model to various parameters and the extent to which the results are generalizable to other contexts or applications.

Analyze potential biases that may have influenced the results, including data bias, model bias, and computational bias. Discuss whether the dataset is representative of the problem space and whether any imbalances, selection biases, or sampling limitations might have affected the conclusions. Examine modeling assumptions, parameter choices, and architectural constraints that could introduce systematic deviations in the results. Assess how numerical precision, algorithmic approximations, or implementation details might influence the stability and fairness of the model’s predictions.

Discuss strategies to mitigate identified biases and improve the reliability of the conclusions. Consider adjustments in data preprocessing, such as resampling, normalization, or augmentation, to address distribution imbalances. Explore refinements to the modeling process, including regularization techniques, fairness constraints, and sensitivity analyses, to ensure robustness across different scenarios. Evaluate the impact of alternative modeling approaches and discuss the extent to which the proposed methods can generalize beyond the given dataset or problem context.

Explore the broader implications of the findings for the field of study, identifying how they contribute to existing knowledge, inform future research directions, or influence practical applications. Discuss any unexpected outcomes and their significance, offering interpretations that may reveal new avenues for exploration or theoretical development. Reflect on the societal, economic, or environmental relevance of the results, if applicable, and propose recommendations based on the study’s insights.

Conclude the section by summarizing the key takeaways, emphasizing the contribution of the research to solving the problem at hand, and outlining the next steps for further investigation or implementation. Ensure that the discussion is logically structured, with each paragraph building upon the previous ones to form a cohesive and persuasive argument that underscores the study’s value and impact.

The content of this Task Answer section should be distinct and not merely a repetition of the Task Result section. Ensure that there is no duplication.

{user_prompt}

Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text and LaTeX for formulas only, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
"""


CODE_STRUCTURE_PROMPT = """\
You are a programming expert. Please extract the structure from the following code and output it in the following JSON format, please return an empty list if the corresponding item is not available.:
The code is:
```python
{code}
```
The output format is:
```json
{{
    "script_path": {save_path}
    "class": [
    {{
      "name": class name,
      "description": description of class,
      "class_functions": [
        {{
          "name": function name,
          "description": description of class function,
          "parameters": [
            {{
              "name": param name,
              "type": param type,
              "description": description of param,
            }},
            ...
          ],
          "returns": {{
            "description": "return of the function."
          }},
        }}
      ]
    }}
  ],
  "function": [
    {{
      "name": function name,
      "description": description of class function,
      "parameters": [
        {{
          "name": param name,
          "type": param type,
          "description": description of param,
        }},
        ...
      ],
      "returns": {{
        "description": "return of the function."
      }},
    }}
  ],
  "file_outputs": [
    {{
      "path": "file_path",
      "file_description": "description of the file",
      "column_name": ["column_name_if_csv_else_None"]
    }},
    ...
  ]
}}
```
"""
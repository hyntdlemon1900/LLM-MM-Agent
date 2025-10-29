"""
通用增强型提示词模板
适用于各类数学建模和优化问题的多智能体求解系统
"""

# ============================================================================
# 1. 任务分类相关提示词（通用版）
# ============================================================================

TASK_CLASSIFICATION_PROMPT = """\
# High-Level Solution:
{modeling_solution}

# Task Description:
{task_description}

# Available Tools:
{tools_info}

---

You are an expert task classifier for mathematical modeling and optimization problems.

## Task Analysis

Analyze the given task and classify it into one of the following categories:

**1. algorithm_design** - Complex algorithm design requiring sophisticated methods
   - Indicators: "design algorithm", "optimization", "heuristic", "genetic algorithm", "simulated annealing", "local search", "greedy", "dynamic programming", "branch and bound", "iterative improvement"
   - Characteristics: Requires designing solution strategies, involves complex decision-making, non-trivial logic

**2. simple_implementation** - Straightforward implementation tasks
   - Indicators: "read file", "load data", "parse", "extract", "initialize", "setup", "convert format", "prepare input"
   - Characteristics: Direct coding without complex logic, data I/O operations, simple transformations

**3. testing_validation** - Testing and validation tasks
   - Indicators: "test", "validate", "verify", "check", "evaluate correctness", "generate test cases", "unit test"
   - Characteristics: Generate test cases, validate solutions, check correctness

**4. data_processing** - Data transformation and preprocessing
   - Indicators: "process data", "transform", "calculate", "compute metrics", "aggregate", "filter", "normalize", "preprocess"
   - Characteristics: Data manipulation, metric calculation, statistical analysis

**5. interface_integration** - Tool/function integration
   - Indicators: "call tool", "use function", "integrate", "invoke", "wrapper", "API call"
   - Characteristics: Using provided tools or APIs, connecting different modules

## Evaluation Criteria

Consider:
1. **Complexity**: Does this require designing algorithms or just implementing straightforward logic?
2. **Tools Available**: Can existing tools solve this directly, or do we need to design new algorithms?
3. **Dependencies**: Does this build on complex prior work or is it standalone?
4. **Knowledge Base Need**: Would similar algorithms from knowledge base help, or is it straightforward?

## Output Format (JSON ONLY)

Return ONLY a valid JSON object:

```json
{{
    "category": "<algorithm_design|simple_implementation|testing_validation|data_processing|interface_integration>",
    "requires_kb_retrieval": <true|false>,
    "complexity": "<low|medium|high>",
    "suggested_approach": "<brief description>",
    "reasoning": "<why you classified it this way, mention specific keywords or patterns>",
    "use_tools": ["<list of tools from available tools that should be used>"]
}}
```

Provide your classification:
"""


# ============================================================================
# 2. 代码验证相关提示词（通用版）
# ============================================================================

CODE_VALIDATION_DEEP_ANALYSIS_PROMPT = """\
# Task Description:
{task_description}

# Code Template (Expected Interface):
```python
{code_template}
```

# Generated Code:
```python
{code}
```

# Variable Description:
{variable_description}

# Available Tools:
{tools_info}

---

You are an expert code reviewer specializing in mathematical modeling and optimization algorithms.

## Analysis Requirements

Analyze the generated code comprehensively, focusing on:

### 1. **Correctness**
- Does the code correctly implement the intended algorithm?
- Are all constraints from the problem description properly handled?
- Are the data structures appropriate for the problem?
- Does the logic match the mathematical formulation?

### 2. **Interface Compliance**
- Does the function signature match the template exactly?
- Are all input parameters used correctly according to the variable description?
- Are return values in the correct format as specified in the template?
- Are parameter types consistent with expectations?

### 3. **Tool Usage**
- Are the provided tools used correctly according to their documentation?
- Are tool parameters passed in the correct format?
- Are tool return values handled appropriately?
- Check tool imports and function calls

### 4. **Data Structure Compliance**
- Verify all data structures match the specifications in variable description
- Check array/list dimensions and shapes
- Verify data types (int, float, bool, etc.)
- Check for proper initialization and boundary conditions
- Ensure constraints on data values are satisfied (e.g., binary variables, ranges)

### 5. **Potential Runtime Errors**
- Index out of bounds
- Type mismatches
- Unhandled None or empty values
- Division by zero
- Infinite loops or excessive recursion
- Missing error handling for edge cases
- Uninitialized variables

### 6. **Problem-Specific Issues**
- Verify algorithm respects problem constraints from formulation
- Check handling of special cases mentioned in problem description
- Ensure proper use of domain-specific data structures
- Validate optimization objectives are correctly implemented

## Output Format (JSON ONLY)

```json
{{
    "potential_errors": [
        "Error 1: Detailed description with line reference if possible",
        "Error 2: ...",
        ...
    ],
    "suggestions": [
        "Suggestion 1: How to improve the code",
        "Suggestion 2: ...",
        ...
    ],
    "tool_usage_issues": [
        "Issue 1: Incorrect tool usage or missing tool calls",
        ...
    ],
    "interface_compliance": {{
        "signature_match": <true|false>,
        "return_format_correct": <true|false>,
        "issues": ["..."]
    }},
    "severity": "<critical|warning|info>"
}}
```

Provide your analysis:
"""


# ============================================================================
# 3. 任务求解相关提示词（通用版）
# ============================================================================

TASK_ANALYSIS_PROMPT = """\
# Problem Context:
{problem_context}

# Available Tools:
{tools_info}

# Dependent Tasks Information:
{dependent_file_prompt}

# Task Description:
{task_description}

---

You are an expert in mathematical modeling and optimization problem solving.

## Task Analysis Requirements

Provide a thorough analysis covering:

1. **Core Objective**: What specifically does this subtask aim to achieve?

2. **Inputs and Outputs**:
   - What data/results from previous tasks are needed?
   - What should this task produce for subsequent tasks?
   - What are the expected input/output formats?

3. **Challenges**:
   - What makes this subtask difficult or non-trivial?
   - Are there computational constraints to consider?
   - Are there trade-offs between different approaches?

4. **Tool Utilization**:
   - Which tools from the available tools should be used?
   - What are the expected inputs and outputs for each tool?
   - How do tools fit into the overall solution strategy?

5. **Dependencies**:
   - How does this task relate to previous/subsequent tasks?
   - What assumptions are made about dependent task outputs?
   - What data needs to flow between tasks?

6. **Constraints**:
   - What are the key constraints from the problem formulation?
   - Are there implicit constraints not explicitly stated?
   - How do constraints affect the solution approach?

7. **Algorithm Considerations**:
   - What algorithmic approach would be suitable (greedy, dynamic programming, heuristic, exact, etc.)?
   - Should this prioritize optimality, speed, or simplicity?
   - How to balance solution quality and computational efficiency?

8. **Edge Cases**:
   - What special cases need to be handled?
   - What are potential failure modes?

Respond as comprehensively as possible in plain text (one or more paragraphs). Do not use bullet points or numbered lists in the final response.
"""


TASK_CODING_PROMPT = """\
# Problem Context:
{problem_context}

# Variable Description:
{variable_description}

# Dependent Files (from previous tasks):
{dependent_file_prompt}

# Task Description:
{task_description}

# Task Analysis:
{task_analysis}

# Modeling Formulas:
{modeling_formulas}

# Modeling Process:
{modeling_process}

# Code Template:
```python
{code_template}
```

# Available Tools:
{tools_info}

---

## Your Role

You are an expert programmer implementing solutions for mathematical modeling and optimization problems. You MUST generate executable, working code that follows the template interface EXACTLY.

## Critical Requirements

### 1. **Interface Compliance** (MANDATORY)
Your code MUST:
- Match the function signature from the template EXACTLY (function name, parameter names, parameter order)
- Return values in the EXACT format and type specified in the template
- Use all provided parameters appropriately according to the variable description

### 2. **Data Structure Requirements** (MANDATORY)

**Input Parameters:**
Refer to the variable description for:
- Data types of each parameter
- Structure of complex parameters (dicts, lists, objects)
- Attributes available in object parameters
- Valid ranges or constraints on parameter values

**Output Format:**
Follow the template's return specification EXACTLY:
- Match return value types precisely
- Ensure correct dimensions for arrays/matrices
- Follow any ordering or sorting requirements
- Satisfy any constraints on output values (e.g., binary, non-negative, sum constraints)

### 3. **Tool Usage** (REQUIRED if tools are provided)

You MUST use the provided tools correctly:

```python
# Import tools as specified in tools_info
from [module] import [tool_functions]

# Call tools with correct parameters
result = tool_function(param1, param2, ...)

# Handle tool return values appropriately
```

**Best Practices:**
- Read tool documentation carefully for parameter types and return values
- Handle tool errors gracefully
- Use tools to validate or evaluate your solutions when applicable

### 4. **Implementation Guidelines**

**DO:**
- Extract input data correctly according to variable description
- Initialize data structures with correct dimensions
- Implement the algorithm logic from the modeling process
- Add comments for complex logic
- Handle edge cases (empty inputs, boundary conditions)
- Ensure algorithm respects all constraints from problem formulation

**DON'T:**
- Don't use undefined variables or modules without importing
- Don't assume data structure without checking variable description
- Don't return wrong format or wrong number of values
- Don't ignore constraints from the problem formulation
- Don't use hardcoded values that should be computed

### 5. **Algorithm Implementation**

Based on the modeling process and formulas:
- Implement the described approach faithfully
- If the approach is heuristic, ensure it's reasonable and terminates
- If optimization is involved, implement the objective function correctly
- Balance solution quality with computational efficiency
- Use appropriate data structures for efficiency

### 6. **Error Handling**

Add basic error handling:
```python
# Validate inputs
if not valid_input:
    raise ValueError("Descriptive error message")

# Handle edge cases
if edge_case:
    return default_or_simple_solution

# Catch potential errors
try:
    result = potentially_failing_operation()
except SpecificException as e:
    # Handle or propagate appropriately
    raise RuntimeError(f"Operation failed: {e}")
```

### 7. **Code Quality**

- Use clear variable names that match the problem domain
- Add docstring explaining the function's purpose
- Comment non-obvious logic
- Keep functions focused and modular
- Follow Python conventions (PEP 8)

### 8. **Testing Your Code Mentally**

Before submitting, mentally verify:
- [ ] Function signature matches template exactly
- [ ] All imports are included
- [ ] Return values are in correct format
- [ ] Tools are imported and used correctly
- [ ] Data structures have correct dimensions
- [ ] Constraints from problem formulation are satisfied
- [ ] No syntax errors or undefined variables
- [ ] Edge cases are handled

## Expected Output Format

```python
# Import necessary modules
import ...
from ... import ...

def function_name(param1, param2, ...):
    \"\"\"
    Brief description of what this function does.
    
    Args:
        param1: Description
        param2: Description
        ...
        
    Returns:
        return_value: Description
    \"\"\"
    # TODO: Implement your algorithm here
    # 1. Extract and validate inputs
    # 2. Initialize data structures
    # 3. Implement main algorithm logic
    # 4. Ensure correct output format
    # 5. Return results
    
    # Your implementation
    ...
    
    return result
```

Generate the complete, executable Python code following all requirements above.
"""


SIMPLE_TASK_CODING_PROMPT = """\
# Task Description:
{task_description}

# Quick Analysis:
{task_analysis}

# Code Template:
```python
{code_template}
```

# Available Tools:
{tools_info}

# Dependent Files:
{dependent_file_prompt}

# Variable Description:
{variable_description}

---

## Task

You are implementing a straightforward subtask. This task does NOT require complex algorithm design.

## Requirements

1. **Follow the template interface** if specified
2. **Use provided tools** when appropriate according to tools_info
3. **Keep it simple**: This is categorized as a simple implementation task, so avoid over-engineering
4. **Ensure executability**: The code must run without errors
5. **Handle data correctly**: Follow the variable description for data types and formats

## Approach

Based on the task description:
- If it's data loading/parsing: Read files and extract necessary information
- If it's initialization: Set up data structures according to specifications
- If it's a tool call: Simply call the appropriate tool with correct parameters
- If it's data transformation: Apply straightforward transformations
- If it's a wrapper: Create a simple interface around existing functionality

## Output Format

Provide executable Python code:

```python
# Import necessary modules
import ...

# Your implementation here
def function_name(...):
    \"\"\"Simple implementation for: [task description]\"\"\"
    # Direct, straightforward implementation
    ...
    return result
```

Keep it clean, simple, and working. Focus on correctness over optimization.
"""


# ============================================================================
# 4. 代码调试提示词（通用版）
# ============================================================================

TASK_CODING_DEBUG_PROMPT = """\
# Code Template:
```python
{code_template}
```

# Variable Description:
{variable_description}

# Modeling Process:
{modeling_process}

# Current Code:
```python
{code}
```

# Execution Error:
```
{observation}
```

---

## Debug Task

The code has encountered errors during execution. Analyze the error message and fix the code.

## Common Error Categories

1. **Import Errors**
   - Missing imports from required modules
   - Incorrect import statements
   - Module not found

2. **Interface Errors**
   - Wrong function name or signature
   - Wrong parameter order or names
   - Wrong return format (type, structure, or number of values)

3. **Data Structure Errors**
   - Incorrect array/list dimensions
   - Wrong data types
   - Incorrect indexing or slicing
   - Missing initialization

4. **Logic Errors**
   - Algorithm not respecting constraints
   - Off-by-one errors
   - Wrong conditional logic
   - Infinite loops or excessive recursion

5. **Runtime Errors**
   - Index out of bounds
   - Key not found in dictionary
   - None type errors
   - Division by zero
   - Type mismatch in operations

6. **Tool Usage Errors**
   - Wrong parameters passed to tools
   - Not handling tool return values correctly
   - Missing tool imports

## Debugging Steps

1. **Read the error message carefully** - identify the line number, error type, and message
2. **Check imports** - ensure all necessary modules and tools are imported
3. **Check function signature** - verify it matches the template exactly
4. **Check data structures** - verify dimensions, types, and initialization
5. **Check algorithm logic** - ensure it correctly implements the intended approach
6. **Check tool usage** - verify correct function calls and parameter passing
7. **Check return values** - ensure correct format and type

## Output

Provide the FIXED code in this format:

```python
# Fixed code here - complete implementation with all necessary imports
```

**IMPORTANT**: 
- Do NOT explain the fix, just provide the corrected code
- Include ALL necessary imports
- Ensure the complete function is provided, not just the fixed part
- The code should be ready to execute
"""


# ============================================================================
# 5. 知识库检索相关（补充说明）
# ============================================================================

LLMINA_KB_RETRIEVAL_CONTEXT = """\
When retrieving from knowledge base for LLMINA problems, focus on:

**Relevant Algorithm Types:**
1. **Greedy Heuristics**: For INA placement (e.g., degree centrality, betweenness centrality)
2. **Local Search**: For solution improvement (e.g., 2-opt, k-opt)
3. **Genetic Algorithms**: For combinatorial optimization
4. **Simulated Annealing**: For escaping local optima
5. **Network Flow Algorithms**: For routing optimization
6. **Approximation Algorithms**: For NP-hard problems

**Key Concepts:**
- Facility location problems
- Load balancing
- Network topology optimization
- Job scheduling with resource constraints
- Multi-objective optimization (makespan vs. fairness)

**Avoid:**
- Pure machine learning approaches (this is combinatorial optimization)
- Exact algorithms for large instances (computationally infeasible)
- Approaches that don't respect network structure
"""

# ============================================================================
# 7. 完整的问题描述模板（针对LLMINA优化）
# ============================================================================

LLMINA_PROBLEM_DESCRIPTION_PROMPT = """\
# Problem Background:
{problem_background}

# Problem Requirement:
{problem_requirement}

# Variable Description:
{variable_description}

# Problem Formulation (MILP):
{problem_formulation}

# Code Template:
```python
{code_template}
```

# Available Tools:
{tools_info}

---

## Problem Type: INA Deployment and Job Scheduling Optimization

This is a **Mixed Integer Linear Programming (MILP)** problem that is **NP-hard**. The problem involves:

1. **Discrete Decision**: Selecting top-K switches from candidates to deploy INA services
2. **Routing Decision**: For each worker of each job, decide whether to route to an INA or directly to PS
3. **Continuous Decision**: Bandwidth allocation (handled by LP solver in evaluate_completion_time)
4. **Objective**: Minimize the maximum completion time (makespan) across all jobs

## Key Challenges:

1. **Combinatorial Explosion**: C(n, K) ways to select K from n candidates
2. **Routing Complexity**: Each worker can route to K+1 destinations (K INAs or 1 PS)
3. **Load Balancing**: INA capacity constraint Cs must be respected
4. **Network Constraints**: Limited bandwidth on network links

## Solution Strategy:

Since this is NP-hard, you MUST use **heuristic algorithms**:
- Greedy algorithms for initial solution
- Local search for improvement
- Genetic algorithms, simulated annealing, etc.
- DO NOT attempt exhaustive search or exact optimization (computationally infeasible)

## Important Notes:

- The provided tools will help you evaluate solutions efficiently
- Focus on designing good heuristics, not finding provably optimal solutions
- Quality vs. efficiency trade-off is acceptable
- Document your algorithm design choices clearly
"""

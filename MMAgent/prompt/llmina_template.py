PROBLEM_DESCRIPTION_PROMPT = """\
Problem Background:
{problem_background}

Problem Requirement:
{problem_requirement}

variable description:
{variable_description}

problem formulation:
{problem_formulation}

code Template:
{code_template}
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
# Decompose Principle:
{decomposed_principle}

# Mathematical Modeling Problem:
{modeling_problem}

# Modeling Solution:
{modeling_solution}

---

Please decompose the given modeling solution into a set of distinct and well-defined subtasks that collectively contribute to the overall objective. Do not assume a fixed number of subtasks in advance. These subtasks should be clearly separated in their focus, each addressing a specific aspect of the modeling process. The goal is to break down the solution into key stages or methodologies, ensuring that all components of the solution are covered without redundancy. For each subtask, the approach or technique should be explicitly described, detailing the specific data, algorithms, or models required. The decomposition should reflect a logical and comprehensive path toward completing the task, with each part having a clear purpose and contributing to the final result.
Each subtask should be described as comprehensively and in as much detail as possible within a single paragraph using plain text and separated by '---' for each subtask. All the contents and details of the original solution need to be covered by the subtasks without omission.
"""


TASK_DESCRIPTION_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Problem Analysis:
{problem_analysis}

# Modeling Solution:
{modeling_solution}

# Decomposed Subtasks:
{decomposed_subtasks}

---

You are tasked with refining and improving the description of subtask {task_i} to ensure it is more detailed, clear, and focused. Provide a precise and comprehensive explanation of the task, specifically elaborating on its scope, goals, and methodology without venturing into other subtasks. Make sure the description includes clear and concise language that defines the necessary steps, techniques, or approaches required for this subtask. If applicable, specify the data inputs, tools, or models to be used, but do not introduce analysis, results, or discussions related to other components of the modeling process. The goal is to enhance the clarity, depth, and precision of this subtask description, ensuring it is fully understood on its own without needing further explanation.
The description of subtask {task_i} should be as comprehensive and in as much detail as possible within a single paragraph using plain text.
"""



PROBLEM_SOLVE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

---

You are tasked with designing a comprehensive algorithmic solution to address the given optimization problem. Your response must provide a high-level, coherent strategy that can later be decomposed into specific implementation steps.

## Requirements:

1. **Core Methodology**: State the fundamental approach you will use (e.g., heuristic algorithms, decomposition strategies, greedy methods, iterative optimization, approximation algorithms, etc.) and provide clear rationale for selecting this methodology to handle the inherent complexity and high dimensionality of the problem.

2. **Algorithm Overview**: Provide a structured, step-by-step description of the overall algorithm at a conceptual level. Explain how different components of the problem will be addressed and how they interact with each other.

3. **Key Strategies**: Detail the specific strategies or heuristics that will be employed for each major component of the problem:
   - How will discrete decisions be made?
   - How will continuous values be allocated?
   - How will constraints be satisfied?
   - How will objectives be optimized?

4. **Feasibility and Optimality**: Explain how your algorithm ensures that the final output is feasible (respecting all system constraints) and discuss why the solution is expected to be effective in practice. Include discussion of:
   - Constraint satisfaction mechanisms
   - Optimization techniques
   - Trade-offs between solution quality and computational efficiency

5. **Scalability and Robustness**: Address how the algorithm handles varying problem sizes and edge cases.

The final response should provide a clear, logically structured solution strategy that can serve as the foundation for implementation.

Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.
"""


PROBLEM_SOLVE_CRITIQUE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Proposed Solution:
{modeling_solution}

---

Critically examine the proposed algorithmic solution, focusing on the following aspects:

1. **Correctness and Completeness**: Does the solution correctly address all aspects of the problem? Are there any missing components or overlooked constraints?

2. **Methodology Appropriateness**: Is the chosen approach suitable for this specific problem? Are there better alternatives that should be considered?

3. **Feasibility**: Can the proposed algorithm be practically implemented? Are there computational or resource limitations that might prevent its execution?

4. **Constraint Handling**: Does the solution adequately address all constraints? Are there potential scenarios where constraints might be violated?

5. **Optimization Quality**: Will the algorithm produce high-quality solutions? Are there obvious inefficiencies or areas where the solution could be significantly improved?

6. **Scalability**: How well will the algorithm perform as problem size increases? Are there scalability concerns?

7. **Edge Cases and Robustness**: Does the solution handle edge cases and unusual scenarios appropriately?

8. **Clarity and Logic**: Is the solution clearly explained? Are there logical gaps or ambiguities in the description?

Critique the solution without offering constructive suggestions—your focus should solely be on highlighting weaknesses, gaps, and limitations within the proposed approach.
"""


PROBLEM_SOLVE_IMPROVEMENT_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Current Solution:
{modeling_solution}

# Critique:
{modeling_solution_critique}

---

Refine and improve the existing algorithmic solution based on the critique provided. The goal is to enhance the approach's effectiveness, feasibility, and robustness while addressing identified gaps and limitations.

## Requirements:

1. Address all major issues and concerns raised in the critique
2. Maintain the strengths of the current solution
3. Propose concrete improvements to methodology, constraint handling, and optimization strategies
4. Ensure the improved solution is more complete, clearer, and more implementable
5. Keep the solution at a high conceptual level suitable for subsequent decomposition

Provide the improved solution directly. DO NOT reference the previous solution's deficiencies in the improved version. Simply present a refined, enhanced algorithmic approach.

Respond as comprehensively and in as much detail as possible. Do not format your response in Markdown. Using plain text, without any Markdown formatting or syntax. Written as one or more cohesive paragraphs. Avoid structuring your answer in bullet points or numbered lists.

IMPROVED SOLUTION:
"""


PROBLEM_DECOMPOSE_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Problem Analysis:
{problem_analysis}

# High-Level Solution:
{modeling_solution}

---

Please decompose the given high-level solution into {tasknum} distinct and well-defined subtasks that collectively implement the overall algorithmic approach. Each subtask should represent a specific, concrete step in the solution process.

## Decomposition Guidelines:

1. **Clear Separation**: Each subtask should have a distinct, well-defined purpose that does not overlap with other subtasks.

2. **Logical Flow**: The subtasks should follow a logical sequence that reflects the natural progression of the algorithm.

3. **Completeness**: All aspects of the high-level solution must be covered by the {tasknum} subtasks without omission.

4. **Implementability**: Each subtask should be described at a level of detail that makes it clear what needs to be implemented.

5. **Specificity**: For each subtask, specify:
   - What it accomplishes
   - What inputs/data it requires
   - What outputs/results it produces
   - What methods/techniques it employs
   - How it relates to other subtasks

## Output Format:

Provide exactly {tasknum} subtask descriptions, each as a comprehensive paragraph. Separate each subtask description with '---' (three dashes on a line by themselves).

Each subtask should be described as comprehensively and in as much detail as possible within a single paragraph using plain text. Ensure all contents and details of the original solution are distributed across the {tasknum} subtasks.
"""


TASK_DESCRIPTION_REFINEMENT_PROMPT = """\
# Mathematical Modeling Problem:
{modeling_problem}

# Problem Analysis:
{problem_analysis}

# High-Level Solution:
{modeling_solution}

# All Decomposed Subtasks:
{decomposed_subtasks}

---

You are tasked with refining and clarifying the description of **Subtask {task_i}** to ensure it is detailed, precise, and fully self-contained.

## Requirements:

1. **Focus**: Concentrate solely on Subtask {task_i}. Do not discuss or reference other subtasks unless absolutely necessary for context.

2. **Clarity**: Provide a clear, unambiguous description of what this subtask accomplishes and how it contributes to the overall solution.

3. **Completeness**: Specify:
   - The primary objective and scope of this subtask
   - Required inputs (data, parameters, results from previous steps)
   - Expected outputs (results, decisions, data structures)
   - Methods, techniques, or algorithms to be employed
   - Key considerations, constraints, or edge cases
   - How this subtask integrates with the overall workflow

4. **Detail Level**: The description should be detailed enough that someone could understand exactly what needs to be implemented without referring to other documentation.

5. **Self-Contained**: The description should stand alone and be fully understandable without requiring reference to other subtask descriptions.

Provide the refined description as a comprehensive, detailed paragraph using plain text. Do not use bullet points or numbered lists.

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
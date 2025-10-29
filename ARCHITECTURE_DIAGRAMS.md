# LLMINA 增强多智能体系统架构图

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLMINA Enhanced System                             │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Enhanced Pipeline                             │    │
│  │            (llmina_pipeline_enhanced.py)                         │    │
│  └──────────┬──────────────────────────────────────────────────────┘    │
│             │                                                             │
│             ├──► Stage 1: Problem Understanding                          │
│             │                                                             │
│             ├──► Stage 2: High-Level Solution Design                     │
│             │                                                             │
│             ├──► Stage 3: Problem Decomposition                          │
│             │                                                             │
│             ├──► Stage 4: Task Classification + Dependency Analysis      │
│             │    ┌───────────────────┐    ┌────────────────┐            │
│             │    │ TaskClassifier    │    │  Coordinator   │            │
│             │    │ - Identify type   │    │  - Build DAG   │            │
│             │    │ - Set strategy    │    │  - Order tasks │            │
│             │    └───────────────────┘    └────────────────┘            │
│             │                                                             │
│             ├──► Stage 5: Adaptive Task Solving (Loop)                   │
│             │    ┌─────────────────────────────────────────────┐        │
│             │    │     LLMINATaskSolver                        │        │
│             │    │                                             │        │
│             │    │  ┌────────────────────────────────┐        │        │
│             │    │  │ 1. Task Classification         │        │        │
│             │    │  │    - Get task type             │        │        │
│             │    │  │    - Decide strategy           │        │        │
│             │    │  └──────────┬─────────────────────┘        │        │
│             │    │             │                               │        │
│             │    │  ┌──────────▼─────────────────────┐        │        │
│             │    │  │ 2. Strategy Selection          │        │        │
│             │    │  └──┬───────┬──────────┬──────────┘        │        │
│             │    │     │       │          │                   │        │
│             │    │  ┌──▼──┐ ┌─▼──┐ ┌────▼─────┐             │        │
│             │    │  │Algo │ │Sim │ │Test      │             │        │
│             │    │  │Flow │ │Flow│ │Flow      │             │        │
│             │    │  └──┬──┘ └─┬──┘ └────┬─────┘             │        │
│             │    │     │       │         │                   │        │
│             │    │  ┌──▼───────▼─────────▼─────┐            │        │
│             │    │  │ 3. Code Generation        │            │        │
│             │    │  └──────────┬────────────────┘            │        │
│             │    │             │                              │        │
│             │    │  ┌──────────▼────────────────┐            │        │
│             │    │  │ 4. Code Validation        │            │        │
│             │    │  │    ┌────────────────┐     │            │        │
│             │    │  │    │ CodeValidator  │     │            │        │
│             │    │  │    │ - Syntax       │     │            │        │
│             │    │  │    │ - Imports      │     │            │        │
│             │    │  │    │ - Compatibility│     │            │        │
│             │    │  │    │ - Dependencies │     │            │        │
│             │    │  │    │ - LLM Analysis │     │            │        │
│             │    │  │    └────────┬───────┘     │            │        │
│             │    │  └─────────────┼─────────────┘            │        │
│             │    │                │                           │        │
│             │    │  ┌─────────────▼─────────────┐            │        │
│             │    │  │ 5. Auto Fix (if needed)   │            │        │
│             │    │  └─────────────┬─────────────┘            │        │
│             │    │                │                           │        │
│             │    │  ┌─────────────▼─────────────┐            │        │
│             │    │  │ 6. Execute & Debug        │            │        │
│             │    │  └─────────────┬─────────────┘            │        │
│             │    │                │                           │        │
│             │    │  ┌─────────────▼─────────────┐            │        │
│             │    │  │ 7. Update Coordinator     │            │        │
│             │    │  └───────────────────────────┘            │        │
│             │    └─────────────────────────────────────────────┘      │
│             │                                                           │
│             └──► Stage 6: Result Integration & Reporting               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 任务分类决策树

```
                            Task Description
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │   TaskClassifier       │
                      │   Analyze task         │
                      └────────────┬───────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
        ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
        │ Complex?      │  │ Simple?      │  │ Testing?     │
        │ Keywords:     │  │ Keywords:    │  │ Keywords:    │
        │ - algorithm   │  │ - read       │  │ - test       │
        │ - optimize    │  │ - load       │  │ - validate   │
        │ - design      │  │ - parse      │  │ - verify     │
        └───────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                  │
                ▼                 ▼                  ▼
        ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
        │algorithm_     │  │simple_       │  │testing_      │
        │design         │  │implementation│  │validation    │
        └───────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                  │
                ▼                 ▼                  ▼
        ┌───────────────┐  ┌──────────────┐  ┌──────────────┐
        │Strategy:      │  │Strategy:     │  │Strategy:     │
        │- KB: Yes      │  │- KB: No      │  │- KB: No      │
        │- Modeling: 2  │  │- Modeling: 0 │  │- Modeling: 0 │
        │- Debug: 5     │  │- Debug: 3    │  │- Debug: 2    │
        └───────────────┘  └──────────────┘  └──────────────┘
```

## 代码验证流程

```
                            Generated Code
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   CodeValidator        │
                    └────────────┬───────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐     ┌──────────────────┐
│ Static Check  │      │ Compatibility   │     │ Deep Analysis    │
│               │      │ Check           │     │                  │
│ - Syntax (AST)│      │                 │     │ - LLM Analysis   │
│ - Imports     │      │ - Template      │     │ - Logic Errors   │
│ - Types       │      │ - Dependencies  │     │ - Edge Cases     │
└───────┬───────┘      └────────┬────────┘     └────────┬─────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  Aggregate Results     │
                    │  - Syntax Errors       │
                    │  - Compatibility Issues│
                    │  - Potential Errors    │
                    └────────────┬───────────┘
                                 │
                        ┌────────▼────────┐
                        │  is_valid?      │
                        └────┬───────┬────┘
                             │       │
                         Yes │       │ No
                             │       │
                             ▼       ▼
                      ┌─────────┐  ┌──────────────┐
                      │ Return  │  │ Auto Fix     │
                      │ Code    │  │ - Generate   │
                      └─────────┘  │   fix prompt │
                                   │ - Apply fix  │
                                   │ - Retry (3x) │
                                   └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Return Fixed │
                                   │ Code         │
                                   └──────────────┘
```

## 自适应求解策略

```
┌──────────────────────────────────────────────────────────────┐
│                     Task Type                                 │
└────────┬───────────────────┬───────────────┬──────────────────┘
         │                   │               │
         ▼                   ▼               ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────┐
│ Algorithm       │  │ Simple         │  │ Testing      │
│ Design          │  │ Implementation │  │ Validation   │
└────────┬────────┘  └────────┬───────┘  └──────┬───────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────┐
│ Full Pipeline   │  │ Fast Track     │  │ Test Gen     │
│                 │  │                │  │              │
│ 1. Analysis     │  │ 1. Quick       │  │ 1. Design    │
│    (detailed)   │  │    Analysis    │  │    Test Cases│
│                 │  │                │  │              │
│ 2. KB Retrieval │  │ 2. Direct      │  │ 2. Generate  │
│    (3-5 methods)│  │    Coding      │  │    Test Code │
│                 │  │                │  │              │
│ 3. Modeling     │  │ 3. Quick       │  │ 3. Execute   │
│    (2 rounds)   │  │    Validation  │  │    Tests     │
│                 │  │                │  │              │
│ 4. Iterative    │  │ 4. Execute     │  │              │
│    Coding       │  │                │  │              │
│                 │  │                │  │              │
│ 5. Full         │  │                │  │              │
│    Validation   │  │                │  │              │
│                 │  │                │  │              │
│ 6. Debug (5x)   │  │                │  │              │
└─────────────────┘  └────────────────┘  └──────────────┘
     ~9 min              ~2 min             ~3 min
```

## 依赖关系管理

```
                        Task Decomposition
                               │
                               ▼
                    ┌──────────────────┐
                    │   Coordinator    │
                    │   Build DAG      │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
           ┌────────┐   ┌────────┐   ┌────────┐
           │Task 1  │   │Task 2  │   │Task 3  │
           │(Root)  │   │Dep:1   │   │Dep:1,2 │
           └───┬────┘   └───┬────┘   └───┬────┘
               │            │            │
               │ ┌──────────┘            │
               │ │  ┌────────────────────┘
               ▼ ▼  ▼
        ┌──────────────────────┐
        │  Topological Sort    │
        │  Order: [1, 2, 3]    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Sequential Solving  │
        │                      │
        │  Solve Task 1        │
        │    ↓                 │
        │  Update Memory       │
        │    ↓                 │
        │  Solve Task 2        │
        │    (uses Task 1)     │
        │    ↓                 │
        │  Update Memory       │
        │    ↓                 │
        │  Solve Task 3        │
        │    (uses Task 1, 2)  │
        └──────────────────────┘
```

## 数据流图

```
┌─────────────┐
│ Problem.json│
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Problem String   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐      ┌─────────────────┐
│ High-Level       │──────│ modeling_       │
│ Solution         │      │ solution.txt    │
└──────┬───────────┘      └─────────────────┘
       │
       ▼
┌──────────────────┐      ┌─────────────────┐
│ Task             │──────│ task_           │
│ Descriptions     │      │ descriptions.   │
└──────┬───────────┘      │ json            │
       │                  └─────────────────┘
       ▼
┌──────────────────┐      ┌─────────────────┐
│ Task             │──────│ task_           │
│ Classifications  │      │ classifications.│
└──────┬───────────┘      │ json            │
       │                  └─────────────────┘
       ▼
┌──────────────────┐      ┌─────────────────┐
│ Dependency       │──────│ dependency_     │
│ DAG              │      │ dag.json        │
└──────┬───────────┘      └─────────────────┘
       │
       ▼
┌──────────────────┐
│ For each task    │
│ in order:        │
│                  │      ┌─────────────────┐
│ 1. Classify  ────┼──────│ task_N.py       │
│ 2. Solve     ────┼──────│ task_N_         │
│ 3. Validate  ────┼──────│ result.json     │
│ 4. Execute   ────┼──────│                 │
│ 5. Save          │      └─────────────────┘
└──────┬───────────┘
       │
       ▼
┌──────────────────┐      ┌─────────────────┐
│ Complete         │──────│ complete_       │
│ Solution         │      │ solution.json   │
└──────────────────┘      └─────────────────┘
```

## 组件交互图

```
┌─────────────────────────────────────────────────────────┐
│                         User                            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 llmina_pipeline_enhanced                │
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐     │
│  │Problem     │  │Problem     │  │Problem       │     │
│  │Clarifi-    │  │Solving     │  │Decompose     │     │
│  │cation      │  │            │  │              │     │
│  └────────────┘  └────────────┘  └──────────────┘     │
│                                                         │
│  ┌────────────┐  ┌────────────┐                        │
│  │Task        │  │Coordinator │                        │
│  │Classifier  │  │            │                        │
│  └──────┬─────┘  └──────┬─────┘                        │
│         │               │                              │
└─────────┼───────────────┼──────────────────────────────┘
          │               │
          ▼               ▼
┌─────────────────────────────────────────────────────────┐
│        computational_solving_enhanced                   │
│                                                         │
│  ┌──────────────────────────────────────────────┐      │
│  │         LLMINATaskSolver                     │      │
│  │                                              │      │
│  │  ┌──────────────┐  ┌────────────────┐       │      │
│  │  │Task          │  │Code            │       │      │
│  │  │Classifier    │  │Validator       │       │      │
│  │  └──────────────┘  └────────────────┘       │      │
│  │                                              │      │
│  │  ┌──────────────────────────────────┐       │      │
│  │  │  Strategy-based Solving          │       │      │
│  │  │  - Full Pipeline                 │       │      │
│  │  │  - Fast Track                    │       │      │
│  │  │  - Test Generation               │       │      │
│  │  └──────────────────────────────────┘       │      │
│  └──────────────────────────────────────────────┘      │
│                                                         │
│  ┌──────────────┐  ┌────────────────┐                  │
│  │Method        │  │Execute Script  │                  │
│  │Retriever     │  │& Debug         │                  │
│  │(Optional)    │  │                │                  │
│  └──────────────┘  └────────────────┘                  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Output Files                         │
│  - Classifications, DAG, Solutions, Code, Reports       │
└─────────────────────────────────────────────────────────┘
```

## 关键决策点

```
┌────────────────────────────────────────────────────────────┐
│ Decision Point 1: Should use Knowledge Base?              │
│                                                            │
│ TaskClassifier checks:                                     │
│ - Is task category 'algorithm_design'?                     │
│ - Is complexity 'high' or 'medium'?                        │
│ - Does description contain algorithm keywords?             │
│                                                            │
│ YES → Use KB     NO → Skip KB                             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Decision Point 2: How many modeling rounds?               │
│                                                            │
│ Strategy determines:                                       │
│ - algorithm_design (high complexity) → 2 rounds           │
│ - algorithm_design (medium complexity) → 1 round          │
│ - simple_implementation → 0 rounds (skip)                 │
│ - testing_validation → 0 rounds (skip)                    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Decision Point 3: How to validate code?                   │
│                                                            │
│ CodeValidator severity:                                    │
│ - 'critical' → Must fix before execution                  │
│ - 'warning' → Try to fix, continue if can't              │
│ - 'info' → Just suggestions, proceed                      │
│                                                            │
│ Auto-fix attempts: Up to 3 iterations                     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ Decision Point 4: Continue or fail task?                  │
│                                                            │
│ Criteria:                                                  │
│ - Syntax errors → Fail                                    │
│ - Template mismatch → Fail                                │
│ - Dependency issues (critical) → Fail                     │
│ - Runtime errors after max retries → Fail                │
│ - Validation warnings only → Continue                     │
└────────────────────────────────────────────────────────────┘
```

这些架构图展示了系统的各个层面：
1. **整体架构** - 全局视角
2. **决策树** - 任务分类逻辑
3. **验证流程** - 代码质量保障
4. **自适应策略** - 不同类型任务的处理方式
5. **依赖管理** - 任务间协调
6. **数据流** - 信息如何在系统中流转
7. **组件交互** - 各智能体如何协作
8. **关键决策** - 重要判断点

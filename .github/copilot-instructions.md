GLOBAL EXECUTION RULES
- Before writing any code or modifying any files, output an explicit Chain-of-Thought (CoT) action plan defining the exact sequence of steps, targeted files, and hypotheses to eliminate random actions, minimize token waste, and maximize execution speed. 
- Stay strictly within the requested scope and never access or modify files outside the active project workspace.
- Prefer the smallest safe change that solves the problem; do not refactor, clean up, or modify unrelated code.
- Every tool call, search, command, or file read must directly advance the requested task.
- Never repeat the exact same failed action; if an approach fails without measurable progress, change the hypothesis, tool, command, file, or strategy.
- Before any open-ended search, define what you are looking for, which files or areas to inspect, and a clear stop condition.
- Finish with a structured AI-to-AI handover containing: current project state, actions taken, files changed, commands/tests and their results, blockers or unresolved issues, for another AI to continue the project immediately without repeating previous exploration.
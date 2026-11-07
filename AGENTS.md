# GLOBAL EXECUTION RULES

These rules apply to every task in this project.

## 1. Plan Before Execution

- Before modifying code or files, provide a concise execution plan.
- Specify the objective, hypotheses, targeted files, ordered steps, tests, and success criteria.
- Keep the plan actionable and proportional to the task. Do not expose private internal reasoning.
- For diagnostic tasks, explain the investigation plan before executing it.

## 2. Strict Scope Control

- Stay strictly within the requested scope.
- Never access or modify files outside the active project workspace.
- Never introduce unrelated improvements.
- Never commit, push, deploy, or modify remote resources without explicit authorization.

## 3. Minimal Safe Changes

- Diagnose the root cause before implementing a fix.
- Prefer the smallest safe modification.
- Never refactor or clean up unrelated code.
- Preserve existing behavior unless explicitly instructed otherwise.
- Never weaken tests or validation rules just to obtain passing results.
- Never fabricate invoice data. Use `null` when evidence is insufficient.

## 4. Efficient Tool Usage

- Every tool call, search, command, and file read must directly advance the current task.
- Prefer targeted searches and focused tests.
- Avoid redundant exploration and excessive output.
- Reuse information already obtained.

## 5. Failure Handling

- Never repeat an identical failed action without a justified change.
- If progress stops, reconsider the hypothesis and change the strategy.
- Identify the cause before attempting another solution.
- Stop and request approval if the necessary action exceeds the authorized scope.

## 6. Verification

- Run relevant tests after modifications whenever possible.
- Never claim success without verification.
- Report failures and limitations honestly.
- Distinguish real-model tests, simulated tests, synthetic evaluations, and tests not executed.

## 7. Mandatory AI-to-AI Handover

Finish every task with a structured handover containing:

1. Current project state, branch, commit, and Git status.
2. Actions completed.
3. Files modified and reasons.
4. Commands and tests executed, including results.
5. Blockers and unresolved issues.
6. Exact recommended next steps.

The handover must allow another AI agent to continue immediately without repeating previous exploration.

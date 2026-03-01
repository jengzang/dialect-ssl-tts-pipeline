# Skill: Code Commit Protocol

## Skill Name
`code_commit_protocol`

## Purpose

This skill defines the rules for automatically committing and pushing code changes during development.

Whenever the user explicitly says **"提交"** (commit), the system must perform a `git commit` AND `git push` operation following the rules below.

This skill ensures clean version history, traceability, and structured development workflow with full automation.


---

## Trigger Condition

This skill is triggered only when:

- The user explicitly instructs to "提交"
- Or clearly indicates that the current stage of work should be committed

No automatic commit should occur without explicit instruction.


---

## Commit Rules

### 1. Always Use `git commit` AND `git push`

When triggered:

- Stage relevant changes (excluding data files)
- Execute `git commit` with descriptive message
- **Automatically execute `git push`** to remote repository

Full automation from commit to push.


---

### 2. Commit Message Requirements

Every commit message must:

- Clearly describe what was implemented or modified
- Be specific and informative
- Avoid vague descriptions such as:
  - "update"
  - "fix"
  - "modify"
  - "change"

Instead, use structured and meaningful messages.

Recommended structure:

[Module/Scope] Short summary
- Detailed explanation of what was added or modified
- Any algorithmic changes
- Any structural changes
- Any new files introduced


Clarity and precision are mandatory.


---

### 3. Files to EXCLUDE from Commits

**NEVER commit the following:**

- Database files: *.db, *.sqlite, *.sqlite3
- Data files: *.csv, *.xlsx, *.xls
- Large data files: *.kml, *.gpx, *.json (in data directories)
- User data configuration: KMCounter.ini, AppUsage_*.txt
- Virtual environments: .venv/, venv/
- Generated outputs: pictures/, *.png (charts)
- Temporary files: *.tmp, *.cache

**Only commit:**
- Source code: *.py, *.go, *.js, *.ts, *.tsx
- Configuration: requirements.txt, go.mod, package.json
- Documentation: README.md, *.md
- Project structure files


---

### 4. Commit Frequency Rules

- Every major change should result in a separate commit
- Logical units of work should not be merged into one commit
- Refactoring, feature addition, and structural updates should be separated when possible

Small iterative edits during active development may remain unstaged until a logical milestone is reached.


---

### 5. Explicit Requirements

- **MUST execute `git push`** after commit
- **MUST use Chinese** for commit message descriptions
- **MUST exclude all data files** from commits
- Do NOT amend previous commits unless explicitly instructed
- Do NOT squash commits
- Do NOT modify git history
- Do NOT use vague commit messages


---

## Design Principle

This skill enforces:

- Clean development history
- Traceable analytical evolution
- Professional repository hygiene
- **Fully automated deployment workflow**
- Data privacy (no data files in version control)

Push operations are automatic and integrated into the commit workflow.

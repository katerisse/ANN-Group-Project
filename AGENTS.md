# AGENTS.md — Instructions for AI Coding Agents

This file gives coding agents the rules for working in this repository. Human-readable
project context is in [README.md](README.md). If these instructions conflict with the
associated GitHub Issue, ask for clarification before proceeding.

## Project overview and objectives

This is a university Artificial Neural Networks final project investigating machine
unlearning in animal-image classifiers. The working research question is how withholding
dog-related training information changes what a classifier knows about dogs while
preserving performance on the remaining classes.

The repository currently contains documentation and GitHub templates only. There is no
training code, evaluation code, model, dependency file, or recorded result in this
checkout. Do not assume otherwise, and do not describe planned work as if it exists.

Objectives:

- Make every experiment reproducible from an Issue, a branch, and a reviewed PR.
- Document experimental settings, metrics, results, and conclusions honestly.
- Keep documentation consistent with the actual repository contents.

## Repository structure and responsibilities of important files

```text
.
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── feature-request.md   # planned code changes
│       ├── experiment.md        # experiments with unknown outcomes
│       ├── chore-request.md     # small maintenance tasks
│       ├── bug-fix.md           # bug reports
│       └── pull-request.md      # required PR body template
├── Dataset/                     # local, untracked images; read-only for agents
├── README.md                    # human-facing project documentation
└── AGENTS.md                    # this file: rules for agents
```

Responsibilities:

- `.github/ISSUE_TEMPLATE/` — the only templates in the repo. Do not edit them unless an
  Issue explicitly asks for it. The PR template is stored here too.
- `README.md` — project overview, dataset, workflow, and contribution guidance. Update it
  only when the Issue asks for documentation and the facts are verifiable.
- `AGENTS.md` — agent rules. Changes go through the same PR process as any other change.
- `Dataset/` — downloaded Kaggle images (90 class folders, 5,400 JPG files). Never edit,
  move, delete, or commit it. Scripts must take a dataset path and fail clearly if absent.
- `.DS_Store` — untracked macOS file, not project content.

`TODO: src/, scripts/, tests/, docs/, artifacts/, .gitignore, and requirements.txt do not
exist yet. Create them only when an Issue asks, and update this map in that same Issue.`

## Coding conventions and standards

No code style has been established yet, so do not claim a convention that does not exist.
Until one is agreed:

- Write small, readable functions with clear names and no clever side effects.
- Keep shared logic in one place; do not duplicate training or data code across scripts.
- Prefer explicit arguments and configuration files over hidden global state.
- Do not invent library functions, APIs, flags, or commands. Verify anything you rely on.
- Use deterministic settings for any experiment code; record seeds in the run metadata.
- Never place credentials, tokens, API keys, or personal data in source files.
- Update `README.md` and this file only when in scope for the Issue; keep docs truthful.

## Issue-driven development workflow

1. Every task begins with a GitHub Issue. Use the matching template in
   `.github/ISSUE_TEMPLATE/`.
2. Read the Issue and its approved plan before making any change. The Issue defines the
   goal, definition of done, plan, out-of-scope items, and AI tooling used.
3. Include the rubric-required fields when drafting Issues: Goal, Definition of Done,
   Plan, Out of Scope, Planning Model, and Planning Chat. The existing Feature template
   does not contain all of them; add them to the Issue body rather than editing the
   template unless explicitly asked.
4. Follow the plan and definition of done. Stay inside the stated scope.
5. Ask for clarification when requirements are ambiguous or contradictory; do not guess.
6. Do not begin unrelated work, fix unrelated files, or "helpfully" expand scope.

## Branching and commit conventions

- Create one branch per Issue from `main`, named so the Issue is identifiable.
- Never commit or push directly to `main`. All changes reach `main` through a reviewed PR.
- Do not touch teammates' branches, Issues, or PRs without explicit authorization.
- Commit small, reviewable changes, each tied to the Issue.
- Stage files explicitly (for example by filename). Never use `git add -A` or `git add .`
  while `Dataset/` is untracked; that would stage the entire image collection.
- End AI-assisted commit messages with a `Co-authored-by:` trailer naming the model used.
- Do not commit datasets, images, model binaries, checkpoints, logs containing secrets,
  or any unrelated file.

## Testing and validation requirements

- Run the checks defined in the Issue before submitting a PR. If none are defined, ask
  what validation is expected rather than skipping it.
- Tests must not depend on `Dataset/` being present; use small synthetic fixtures.
- Verify the result yourself: run the code, read the diff, and remove out-of-scope changes.
- Record the commands run and their actual outcomes in the PR. Never claim a passing test
  that was not run.
- For documentation-only changes, validate links, paths, and factual claims against the
  repository, and confirm that only the requested files changed.

## Machine learning experiment and reproducibility requirements

- Record the dataset/path, split protocol, seeds, preprocessing, architecture, optimizer,
  learning rate, batch size, training budget, package versions, and git SHA for every run.
- Keep baseline and comparison models under identical settings except for the single
  variable the experiment is testing.
- Save and report metrics as produced by the evaluation code. Include failed, negative,
  or inconclusive experiments; a null result is a valid result.
- Never fabricate or estimate experimental results, metrics, plots, or logs.
- Do not change data, splits, or evaluation sets to make a result look better.
- Do not commit raw images, downloaded archives, credentials, or model binaries unless an
  Issue explicitly requires an artifact and a human has approved it.
- Document settings, metrics, results, and conclusions, with limitations, in the
  location the Issue specifies. `TODO: docs/ does not exist yet.`

## Pull request and review requirements

- Use `.github/ISSUE_TEMPLATE/pull-request.md` for every PR.
- Link the associated Issue using the closing keyword, `Closes #<issue number>`, so the
  Issue closes when the PR is merged.
- Fill in the testing section with commands and observed results, disclose AI tools/models,
  and list known limitations.
- Require at least one approval from another team member before merging. Never approve
  your own PR or bypass required human review.
- Do not merge PRs or manually close Issues without explicit authorization.
- Be ready to explain every changed line; reviewers must understand and own what they
  approve.

## Restrictions on agent behavior

1. Review the associated GitHub Issue before beginning any work.
2. Follow the approved plan and definition of done.
3. Ask for clarification when requirements are ambiguous.
4. Stay within the defined scope of the issue.
5. Do not modify unrelated files or functionality.
6. Never push directly to main.
7. Work on a separate branch associated with the issue.
8. Test and validate changes before submitting a pull request.
9. Document experimental settings, metrics, results, and conclusions, including unsuccessful experiments.
10. Use the existing pull request template and link the associated issue using the appropriate closing keyword.
11. Never approve your own pull request or bypass required human review.
12. Do not merge pull requests or manually close issues without authorization.
13. Do not commit credentials, API keys, or other secrets.
14. Do not fabricate experimental results.

Additional hard rules:

- Do not edit, move, delete, or commit anything under `Dataset/`.
- Do not change GitHub repository settings (branch protection, permissions, collaborators)
  without owner approval.
- Do not publish Issues, push commits, or open PRs without explicit approval.
- Do not claim that files, commands, models, or results exist when they do not; mark
  missing information as `TODO`.

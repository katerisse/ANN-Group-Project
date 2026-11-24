# ANN Group Project — Machine Unlearning

## Project title and description

**ANN Group Project — Machine Unlearning in Animal-Image Classifiers**

This repository is for a university Artificial Neural Networks final project investigating
machine unlearning. The general setting is an image classifier trained on animal images,
and the question of what changes when information about one class (dogs) is removed from
the training process.

> **Current state:** this checkout contains GitHub issue/PR templates, these documentation
> files, and a local, untracked dataset. No training code, evaluation code, model files,
> or results exist yet. Anything not present in the repository is marked `TODO` below
> rather than assumed.

## Project goals and research question

**Research question (draft):** how does withholding dog training data change what a
classifier knows about dogs, and how can that difference be measured without degrading
performance on the remaining animal classes?

Goals:

- Establish a reproducible baseline and a dog-excluded reference model under otherwise
  identical conditions. `TODO: confirm the final model design with the team.`
- Measure both forgetting of the dog class and retention of the other classes.
  `TODO: define the exact metrics once the evaluation code exists.`
- Keep every experiment reproducible from a GitHub Issue, a dedicated branch, and a
  reviewed pull request.

`TODO: link the canonical experiment issue(s) once published.`

## Repository structure

Actual contents of this checkout:

```text
.
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug-fix.md
│       ├── chore-request.md
│       ├── experiment.md
│       ├── feature-request.md
│       └── pull-request.md
├── Dataset/                 # local, untracked image data (see below)
├── AGENTS.md                # rules for AI coding agents
└── README.md                # this file (human-facing project docs)
```

Notes:

- The pull request template is stored in `.github/ISSUE_TEMPLATE/pull-request.md`.
- `Dataset/` is not committed and must never be committed.
- `.DS_Store` is an untracked macOS file, not project content.
- `AGENTS.md` holds the rules AI coding agents must follow in this repository.
- `TODO: add source/, scripts/, tests/, docs/, and artifacts/ directories when code lands.`
- `TODO: add .gitignore, requirements.txt, and any environment specification.`

## Dataset information

The dataset is present locally but untracked:

- Path: `Dataset/animals/animals/<class_name>/*.jpg`
- 90 class folders, 5,400 JPG files (60 per class), approximately 677 MB on disk
- Class names are listed in `Dataset/name of the animals.txt` (alphabetical order; `dog`
  is one of the classes)
- Source given during project planning:
  <https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals>

The images originate from web sources. Treat the data as research material only and do not
redistribute it in this repository. `TODO: record the exact license/attribution text.`

Rules for agents and contributors:

- Do not edit, move, delete, or commit anything under `Dataset/`. Scripts must accept a
  dataset path and fail clearly when it is missing.
- `TODO: document the final train/validation/test split protocol and any data manifest.`

## Model architecture and methodology

`TODO: no model code exists in this checkout. Document the selected architecture, input
size, output classes, and training configuration once the model is implemented.`

The intended methodology is a controlled comparison: two models trained with the same
pipeline, one with dogs included and one with dog training images excluded, followed by
evaluation on held-out data. `TODO: confirm whether this remains the final design; other
approaches have been proposed in unmerged work and are not part of this checkout.`

`TODO: document preprocessing, augmentation, seeds, optimizer, learning rate, batch size,
training budget, and checkpoint format.`

## Installation instructions and dependencies

No dependency file or environment specification exists in this checkout, so exact
versions cannot be stated yet.

Known local toolchain facts:

- The default `python3` on the development machine is 3.14.6.
- Python 3.12 is also installed on that machine.
- TensorFlow and Keras are not installed in the default Python environment.

`TODO: choose the supported Python version, add requirements.txt (or equivalent), and
write the exact install and verification commands here. Do not guess dependency versions.`

## Training, running, and evaluating the models

There are no training, inference, or evaluation scripts in this checkout, and no model
checkpoints exist.

- `TODO: document the data preparation command once the script exists.`
- `TODO: document the command(s) to train the baseline and dog-excluded models.`
- `TODO: document how to run inference on a single image or batch.`
- `TODO: document the evaluation command and the metrics it produces.`
- `TODO: document where artifacts are written and how to reproduce a stored result.`

Do not run or claim any training or evaluation results until those scripts exist and have
been reviewed.

## Development workflow

All work follows an issue-driven workflow:

1. **Every piece of work begins with a GitHub Issue.** Use the matching template in
   `.github/ISSUE_TEMPLATE/` (feature, experiment, chore, or bug). The issue states the
   goal, definition of done, plan, out-of-scope items, and AI tooling used.
2. **Changes are made on separate branches.** Create one branch per issue from `main`.
   Do not work directly on `main`.
3. **Nobody pushes directly to `main`.** All changes reach `main` only through a reviewed
   pull request. `TODO: enforce this with GitHub branch protection once approved.`
4. **Every pull request requires at least one approval from another team member.** Authors
   cannot approve their own pull requests. Reviewers must run the documented checks and
   confirm the issue's definition of done before approving.
5. **Issues are closed through associated pull requests.** The PR body links the issue
   with `Closes #<issue number>`; merging the PR closes the issue.

Keep changes small and focused on one issue. Do not touch teammates' branches, issues, or
pull requests.

## Contributing

1. Read this README, the relevant issue, and the templates in `.github/ISSUE_TEMPLATE/`.
2. Create or pick an issue, then create a dedicated branch from `main`.
3. Make only the changes described in the issue. Do not commit datasets, model binaries,
   credentials, API keys, or unrelated files.
4. Run any checks the issue defines and record the results in the pull request using
   `.github/ISSUE_TEMPLATE/pull-request.md`.
5. Disclose AI tools and models used in the PR and in commit messages (`Co-authored-by:`
   trailer). `TODO: list the project's approved AI tools/models.`
6. Request review from at least one teammate. Address requested changes before merging.
7. Merge only after approval; the associated issue is then closed by the PR.

Reviewers should verify that the code runs, that the issue's definition of done is met,
that only in-scope files changed, and that AI use is disclosed.

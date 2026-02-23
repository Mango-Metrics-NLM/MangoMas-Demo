# Contributing to MangoMAS Demo

Thank you for your interest in contributing to MangoMAS Demo. This document provides
guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you are expected to uphold this code. Please report unacceptable behavior
to the project maintainers.

## Reporting Bugs

If you encounter a bug, please open a GitHub issue with the following information:

- **Summary**: A clear and concise description of the bug.
- **Steps to Reproduce**: A minimal sequence of steps that reliably triggers the issue.
- **Expected Behavior**: What you expected to happen.
- **Actual Behavior**: What actually happened, including any error messages or stack traces.
- **Environment**: Python version, OS, PyTorch version, and whether you are running on CPU or GPU.
- **Screenshots or Logs**: Attach any relevant output from the Gradio UI or terminal.

Please search existing issues before opening a new one to avoid duplicates.

## Suggesting Features

Feature requests are welcome. To propose a new feature:

1. Open a GitHub issue with the title prefixed by `[Feature Request]`.
2. Describe the problem the feature would solve and why it matters.
3. Outline a proposed solution or approach, if you have one.
4. Indicate whether you are willing to implement the feature yourself.

Feature discussions happen in the issue tracker. Consensus among maintainers is required
before implementation begins.

## Development Setup

### Prerequisites

- Python 3.11 or later
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Mango-Metrics-NLM/MangoMas-Demo.git
cd MangoMas-Demo

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Running the Application

```bash
python app.py
# Open http://localhost:7860 in your browser
```

### Running the Test Suite

```bash
pytest --cov=mangomas_demo --cov-report=term-missing
```

## Code Style

This project enforces consistent code style through automated tooling.

### Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and import sorting:

```bash
ruff check mangomas_demo/ tests/
```

Ruff is configured in `pyproject.toml` with the following rules enabled:
`E`, `F`, `I`, `W`, `UP`, `B`, `SIM`, `RUF`.

### Formatting

- **Line length**: 100 characters maximum.
- **Target version**: Python 3.11.

### Type Checking

All code must pass [mypy](https://mypy-lang.org/) in strict mode:

```bash
mypy mangomas_demo/
```

This includes:
- Fully typed function signatures (no `Any` unless justified).
- No untyped definitions.
- Strict return-type checking.

Third-party stubs for `torch`, `gradio`, and `plotly` are configured to allow missing imports.

## Testing Requirements

- **Framework**: [pytest](https://docs.pytest.org/) with the `pytest-cov` and `pytest-xdist` plugins.
- **Minimum coverage**: 80% line coverage is enforced. The CI pipeline will fail if coverage
  drops below this threshold.
- **Test location**: All tests reside in the `tests/` directory.
- **Requirements for contributions**:
  - Every new feature must include corresponding unit tests.
  - Every bug fix must include a regression test that fails without the fix and passes with it.
  - Tests must be deterministic. Use fixed random seeds where randomness is involved.
  - Do not introduce test dependencies beyond those listed in `pyproject.toml` under
    `[project.optional-dependencies.dev]` without prior discussion.

Run the full test suite before submitting a pull request:

```bash
pytest --cov=mangomas_demo --cov-report=term-missing
```

## Pull Request Process

### Branch Naming

Use descriptive branch names with the following prefixes:

| Prefix       | Purpose                          |
|--------------|----------------------------------|
| `feature/`   | New features                    |
| `fix/`       | Bug fixes                       |
| `refactor/`  | Code restructuring              |
| `docs/`      | Documentation changes           |
| `test/`      | Test additions or modifications |
| `ci/`        | CI/CD pipeline changes          |

Example: `feature/add-planning-cell` or `fix/mcts-ucb1-overflow`.

### Commit Messages

Write clear, imperative-mood commit messages:

- **Good**: `Add deterministic seed to MoE router selection`
- **Good**: `Fix off-by-one error in MCTS rollout depth`
- **Avoid**: `Fixed stuff` or `WIP`

For multi-line commits, include a blank line after the subject, followed by a body that
explains the motivation and context for the change.

### Submitting a Pull Request

1. Create a feature branch from `main`.
2. Make your changes in small, focused commits.
3. Ensure all checks pass locally:
   ```bash
   ruff check mangomas_demo/ tests/
   mypy mangomas_demo/
   pytest --cov=mangomas_demo --cov-report=term-missing
   ```
4. Push your branch and open a pull request against `main`.
5. Fill out the pull request template with a description of your changes, the motivation
   behind them, and any testing you performed.
6. Request a review from at least one maintainer.

### Review Expectations

- All pull requests require at least one approving review before merge.
- Reviewers may request changes. Please address feedback promptly or discuss disagreements
  constructively in the pull request comments.
- CI must pass (lint, type check, tests, coverage) before a pull request can be merged.
- Pull requests should be focused. Avoid combining unrelated changes in a single PR.

---

Thank you for helping improve MangoMAS Demo. Your contributions are valued.

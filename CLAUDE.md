# QuIPS-gen Development Guide

## Environment rules

### Filesystem
- **NEVER** use `find /`, `find /ocean`, or scan outside the project directory.
- **Project root**: the current working directory (use `.` or relative paths)
- **Only search within** `.`
- Use `grep` or `Grep` to search file contents — never `find` with broad paths.

### Python Package Management (uv)

Use uv exclusively — never pip, pip-tools, poetry, or conda.

- **Add/remove**: `uv add <package>` / `uv remove <package>`
- **Sync**: `uv sync`
- **Run scripts/tools**: `uv run <script>.py`, `uv run pytest`, `uv run python`

### PEP 723 Inline Script Dependencies

Manage per-script dependencies via the `dependencies =` header or:
- `uv add --script script.py <package>`
- `uv remove --script script.py <package>`

## What is this?

A simulation pipeline that outputs the measurement record of a levitated nanoparticle undergoing weak, continous measurement, and other interactions that effect the dynamics of its quantum density matrix.
Full architecture and physics details are in **DESIGN.md**. This file covers development instructions and conventions.

## Quick reference

- **Design document**: `DESIGN.md` (read this first)
- **Progress log**: `CHANGELOG.md`

## Repository
- **GitHub**: https://https://github.com/quantum-measurement-group-lbl/quips_generator (private)
- **Always work on a new branch** when starting new work (features, experiments, refactors). Never commit directly to `main`.
  - Branch naming: `feature/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`
  - Create with: `git checkout -b feature/my-thing`
  - Open a PR to merge back into `main` once the work is complete and tests pass.
- Commit at meaningful checkpoints (passing tests, bug fixes, new features)
- Keep commits focused: one logical change per commit

## Principles for autonomous development

### 1. Concise test output (context window hygiene)

LLMs have finite context windows. Every line of noisy test output displaces
useful information and degrades reasoning quality. The C compiler project
learned this the hard way.

**Rules:**
- Tests print at most 5-10 lines on success, ~20 lines on failure.
- Use `pytest -q` by default if testing. Never dump large arrays to stdout.
- Log verbose diagnostics to `test_logs/` files, not stdout.
- Pre-compute aggregate summary statistics. Print them, not raw data.
- When comparing arrays, print: max relative error, the index/value where
  it occurs, and the overall pass rate. Not the full arrays.
- Error messages should be greppable: put ERROR and the reason on one line
  so `grep ERROR logfile` works.

### 2. Keep CHANGELOG.md current (agent orientation)

The C compiler project found that each agent drops into a fresh context with
no memory of what happened before. CHANGELOG.md is the shared memory. Without
it, agents waste time re-discovering what's done and what's broken.

**Rules:**
- Update CHANGELOG.md after every meaningful unit of work.
- Check off completed items with dates.
- Note what worked, what didn't, what's blocked.
- **Record failed approaches** so they aren't re-attempted. E.g.:
  "Tried using Tsit5 for perturbation ODE -- doesn't work, system is too
  stiff. Switched to Kvaerno5."
- Add new tasks discovered during implementation.
- When stuck, maintain a running doc of attempts in CHANGELOG.md.

### 3. Small, testable commits

**Rules:**
- Each commit implements one thing (one function, one module, one bugfix).
- Each commit passes all existing tests.
- Each commit includes or updates tests for the new code.
- Avoid large commits that change multiple modules at once.
- If a refactor touches many files, do it as a separate commit from features.

### 4. Document for the next session, not for users

The C compiler project maintained extensive READMEs and progress files
because each agent starts with zero context. Documentation is not a nicety;
it's a critical coordination mechanism.

## Coding conventions

- **Pure functions only**. No mutable state, no global variables, no side effects.
- **Type hints** on all public functions using `numpy.typing.NDArray` and standard Python types.
- **Naming**:  Use snake_case for everything.
- **Result types** as plain dataclasses with type-annotated fields.

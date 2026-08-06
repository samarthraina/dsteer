# Contributing

Workflow for this repository.

## Branches

- `main`: stable, working code. Only receives merges from `dev` after testing.
- `dev`: integration branch where new work lands.
- `feature/<name>`: optional, for larger features. Branched from `dev`, merged back to `dev`.

## Standard flow

1. Make sure you are on `dev`: `git checkout dev`
2. Pull latest: `git pull origin dev`
3. Make changes, commit with a descriptive message
4. Push: `git push origin dev`
5. When `dev` is stable and a milestone is reached, merge to `main`

## Coding conventions

- Python 3.10+
- Type hints on public API (functions and classes in `src/`)
- Configs live in `configs/`, never hardcode paths in scripts
- All entry-point scripts accept `--config` argument
- Set random seeds at the top of every script that uses randomness

## Testing

Run sanity checks before merging to `main`:

    pytest tests/

## Commit messages

Format: short imperative description, optionally followed by detail.

Good examples:
- `Add IT eval script with IFEval support`
- `Fix tokenizer bug in steering vector extraction`
- `Switch from PC3 to mean steering vector based on comparison results`

Avoid:
- `update`
- `wip`
- `fix`
- `changes`

# vr-foraging-replenishment

## Running

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. `main.py` is a
[marimo](https://marimo.io) notebook.

Install dependencies:

```bash
uv sync
```

Run the notebook as an interactive app:

```bash
uv run marimo edit main.py
```

Or launch it read-only, as a standalone app:

```bash
uv run marimo run main.py
```

## Building a stage

`make_stage.py` rebuilds the `mcm_final_stage` curriculum stage (from
`aind-behavior-vr-foraging-curricula`) with a custom reward amount, wraps it in a `TrainerState`
(`curriculum`, `stage`, `is_on_curriculum`, `active_policies`), writes it to
`mcm_final_stage.json`, and prints a diff against the package's native stage so you can see
exactly which lines (if any) your override changed:

```bash
uv run python make_stage.py
```

Example output when the override actually differs from the package default:

```text
Wrote mcm_final_stage.json

Diff vs native package stage (mcm_final_stage):
--- native (stages.make_s_mcm_final_stage)
+++ custom (make_final_stage)
@@ -52,7 +52,7 @@
                         "family": "Scalar",
                         "distribution_parameters": {
                           "family": "Scalar",
-                          "value": 4.0
+                          "value": 5.0
                         },
                         "truncation_parameters": null,
                         "scaling_parameters": null
```

When run in an interactive terminal the `+`/`-` lines are colored (green/red); when piped to a
file or another program it falls back to plain unified-diff text. If there's no difference, it
prints `No differences vs native package stage.` instead.

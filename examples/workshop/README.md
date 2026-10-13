# Workshop demonstration

This directory contains a small, public-safe demonstration of the model workflow. It is intended to show how the repository is organised and how a synthetic or authorised local input moves through the pipeline.

> **Important:** This is a method demonstration, not an official estimate and not a release of administrative, survey, or unpublished model-output data.

## Demonstration scope

- define a documented observation contract;
- prepare a small synthetic input;
- build the model data structure;
- run a lightweight smoke test;
- inspect diagnostics and record limitations.

The demonstration is deliberately separate from production data and full-scale fitting. It must not be used to infer final country estimates or agreement with official series.

## Related code

- `src/python/` — acquisition, harmonisation, model assembly, and fitting modules;
- `src/stan/` — equation-level reference model;
- `docs/0. Public Project Status.md` — current capabilities and remaining gaps;
- `docs/4. Model Notes.md` — modelling choices and translation notes.

# SDG 4.1.4 Bayesian Model Reproduction

> [!NOTE]
> **This repository is an independent reproduction project.**
> The code and method are substantially developed. The full empirical
> reproduction is not finished because the core administrative input is restricted.

A reconstruction of the Bayesian cohort model used to study **out-of-school
rates for SDG indicator 4.1.4**. It is **not** the official UNESCO / UIS / GEMR
model and does not publish official estimates.

The repository contains:

- a documented translation of the published model equations;
- Python and Stan implementations of the model and data pipeline;
- a transparent record of what is implemented, tested, and still unavailable.

## Start here

| To understand… | Open |
|---|---|
| What is complete and what is missing | [`docs/status.md`](docs/status.md) |
| What each data source is used for | [`docs/data-sources.md`](docs/data-sources.md) |
| How the statistical model was translated | [`docs/model-notes.md`](docs/model-notes.md) |
| How the public-safe demonstration works | [`examples/workshop/README.md`](examples/workshop/README.md) |
| The executable Python pipeline | [`src/python/`](src/python/) |
| The equation-level Stan model | [`src/stan/oosc_cohort_model.stan`](src/stan/oosc_cohort_model.stan) |

## Current status

| Area | Status |
|---|---|
| Project structure | **Complete** |
| Public population-data pipeline | **Implemented and checked** |
| Data harmonisation logic | Implemented in modules; coverage varies by source |
| Probabilistic model translation | **Executable and tested on synthetic / authorised local inputs** |
| Full real-data fit | Pending restricted administrative input |
| Final comparison with the paper | Not claimed yet |

## What this demonstrates

- probabilistic programming with NumPyro / JAX and an equation-level Stan reference;
- country-year-age-sex data contracts and source harmonisation;
- administrative exclusion logic and survey-uncertainty handling;
- spline-basis construction and cohort-process modelling;
- synthetic tests for transformations, constraints, and sampling;
- explicit validation design and documented deviations.

## Repository layout

```text
.
├── docs/            Status, data sources, and model notes
│   ├── status.md
│   ├── data-sources.md
│   └── model-notes.md
├── src/             Implementation
│   ├── python/      Executable data pipeline
│   └── stan/
│       └── oosc_cohort_model.stan   Equation-level model
├── examples/
│   └── workshop/    Public-safe demonstration (see its README)
└── A. Data/         Local data only — excluded from the public repo
```

## Public boundary

Only code, public method documentation, synthetic demonstrations, and high-level
provenance belong in this repository. Raw data, restricted administrative
extracts, credentials, personal information, internal correspondence, and
unpublished estimates are excluded.

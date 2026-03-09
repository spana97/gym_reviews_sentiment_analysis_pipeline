# Gym Reviews Insight Analysis Pipeline

Insight analysis of Google and Trustpilot customer reviews for an undisclosed gym.

---

## Table of Contents

- [Description](#description)
- [Project Status](#project-status)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installing](#installing)
  - [Executing the Program](#executing-the-program)
- [Design Decisions](#design-decisions)
- [Help](#help)
- [Version History](#version-history)
- [Author](#authors)
- [Acknowledgements](#acknowledgements)

---

## Description

This project analyses customer feedback from Google and Trustpilot reviews for a gym, with the aim of understanding common customer pain points, identifying recurring themes in negative feedback, and extracting actionable insights to help improve customer satisfaction and overall service quality.

The pipeline implements an ETL (Extract, Transform, Load) process to combine, clean, and structure both datasets before applying topic modelling techniques. The output is parsed directly into an LLM to automatically generate key insights, recommendations and justifications on grouped topics.

> **Note:** The dataset used in this project is **synthetic**, generated using Python's Faker library.
> All names, reviews and other data points are **fictitious** and do not represent real individuals, actual gym locations, topics or insights into this gym.

---

## Project Status

Actively in development. The core pipeline is implemented. Planned next steps include:

- Adding visualisations for topic modelling outputs
- REST API deployment via FastAPI
- Cloud integration (S3 and broader AWS/GCP storage)
- Docker containerisation for portable deployment
- CD workflows

---

## Getting Started

### Dependencies

The project uses **Python 3.11.3**. Key dependencies are listed below. For the full list see `requirements.txt`.

**Core**

- `pandas`
- `nltk`
- `openai`
- `python-dotenv`
- `PyYAML`
- `certifi`

**Topic Modelling**

- `bertopic`
- `sentence-transformers`
- `transformers` + `tokenizers`
- `torch`
- `umap-learn`
- `hdbscan`
- `numba` _(requires LLVM; can cause install issues on some systems)_

**Dev & Code Quality**

- `pytest`
- `mypy`
- `pre-commit`
- `ruff`
- `detect-secrets`

---

### Installing

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/spana97/gym-reviews-insight-analysis-pipeline.git
cd gym-reviews-insight-analysis-pipeline
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

> **Mac Intel users** — if you encounter issues installing PyTorch, ensure `torch==2.2.2` is being installed as pinned in `requirements.txt`. If you hit SSL errors during NLTK resource downloads, this is handled automatically via `certifi`.

---

### Executing the Program

Run the ETL pipeline first to extract, clean and combine the raw Google and Trustpilot reviews:

```bash
PYTHONPATH=src python -m scripts.run_etl
```

Then run the main pipeline to perform topic modelling and generate insights:

```bash
PYTHONPATH=src python -m scripts.run_pipeline
```

Outputs will be saved to `data/interim` (ETL and BERTopic outputs), `data/processed/` (LLM insight output) and the BERTopic model to `models/`.

#### Configuration

Behaviour can be adjusted via `config.yaml`. Notable parameters include:

- `filters.low_rating_max` — threshold for filtering low-rated reviews (default: `3`)
- `topic_model.nr_topics` — number of topics to extract (default: `auto`)
- `insights_generator.developer_prompt` / `user_prompt` — customise the LLM prompts

---

## Design Decisions

**Dependency management** — This project uses `requirements.txt` rather than `uv` + `pyproject.toml`. This was a deliberate choice to gain experience with the traditional dependency management workflow. Future projects will adopt `uv` + `pyproject.toml` as the default.

**Infrastructure** — The pipeline currently runs locally. Cloud storage (S3), containerisation (Docker), and REST API deployment (FastAPI) are planned for future iterations.

---

## Help

### Known Issues (Mac Intel)

**PyTorch** — Mac Intel users may encounter compatibility issues with recent versions of PyTorch. This project pins `torch==2.2.2` in `requirements.txt` to ensure stability on older Mac hardware.

**NLTK SSL certificates** — Mac users may hit SSL certificate errors when downloading NLTK resources. This is a known macOS issue and is handled explicitly via `certifi` — see `src/text_preprocessing/helpers.py` for the SSL context override.

---

## Version History

> **Note:** Releases prior to v1.0.0 are not tracked on GitHub Releases.
> See version history below for a full changelog.

### v0.6.0 - Refactor full pipeline (09Mar2026)

- Separated ETL pipeline logic into smaller units
- Separated helper functions into smaller modules - column_transforms and filter_rows

### v0.5.1 - Fix BERTopic dependencies (03Mar2026)

- Pinned the following dependencies:
  - `numpy<2.0`
  - `transformers==4.38.2`
  - `sentence-transformers==2.7.0`
  - `bertopic==0.16.4`
  - `umap-learn==0.5.6`
  - `hdbscan==0.8.33`
  - `scikit-learn==1.4.2`

### v0.5.0 — Logging, CI & Pre-commit Updates (26Feb2026)

- Added logging across ETL, text preprocessing, topic modelling, and insights
- Added CI GitHub Actions (ruff, mypy, pytest)
- Replaced flake8, black, isort with ruff in pre-commit
- Added detect-secrets to pre-commit

### v0.4.0 — Full Pipeline Integration (25Feb2026)

- Integrated full end-to-end pipeline
- Added pipeline notebook (03_notebook)
- Downgraded Python to 3.11 for compatibility

### v0.3.0 — Topic Modelling & Insight Generation (24Feb2026)

- Implemented topic modelling with JSON output formatting
- Implemented insight generator
- Added pytests for topic modelling and insights

### v0.2.0 — Text Preprocessing (21Feb2026)

- Implemented text preprocessor
- Added text preprocessor pytests
- Added feature engineering notebook

### v0.1.0 — Initial Release (16Feb2026)

- Initial repository structure and basic CI
- Added ETL pipeline
- Added synthetic gym review datasets
- Added pre-commit config (black, flake8)

---

## Authors

- Sean Panacides
- seanpanacides@gmail.com
- GitHub: https://github.com/spana97

---

## Acknowledgements

The original dataset was provided through a university-industry partnership as part of the [Data Science with Machine Learning & AI Career Accelerator](https://onlinecareeraccelerators.pace.cam.ac.uk/cambridge-data-science-career-accelerator) programme, delivered in collaboration with the University of Cambridge and Forthrev. The client has not been disclosed. This project represents an independent productionisation of that work for learning purposes.

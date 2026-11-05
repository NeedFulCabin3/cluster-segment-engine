# Cluster Segment Engine

Unsupervised customer segmentation and inertia analysis pipeline engineered in Python.

## Overview

Analyzing consumer purchasing behavior often stalls when transaction logs and demographic attributes remain unclassified in raw tabular files. This tool solves that operational friction by ingesting multi-dimensional consumer records, standardizing feature variance, and executing iterative K-Means clustering to partition customer bases into distinct spending tiers.

## How It Works

1. **Data Ingestion & Fallback:** The pipeline attempts to load customer records from a user-specified CSV path. If no path is provided or if loading fails, it generates a synthetic multi-modal dataset using NumPy normal distributions across annual income and spending score axes.
2. **Feature Standardization:** Numeric columns are passed through scikit-learn's `StandardScaler` to ensure features with larger numerical magnitudes do not disproportionately bias Euclidean distance calculations.
3. **Inertia & Elbow Analysis:** The engine iterates cluster counts ($K$) from 1 to 10, fitting the model at each iteration and capturing inertia (within-cluster sum of squares). Matplotlib renders the resulting elbow curve to visualize diminishing returns.
4. **Centroid Inversion & Mapping:** After selecting optimal $K$, final cluster assignments are appended to the dataset, centroids are inverse-transformed back to original scale units, and a spatial scatter plot is generated.

## Key Features

* Automated CSV schema detection with intelligent fallback to synthetic consumer distributions.
* Iterative elbow curve generation for mathematical cluster count validation.
* Variance normalization via standard scaling for equitable distance metric computation.
* Cluster summary statistics generation outputting demographic mean profiles.
* Centroid mapping and visual rendering via Matplotlib.

## Tech Stack & Core Dependencies

* **Python 3.10+**: Core runtime environment.
* **NumPy**: Array manipulation and synthetic data generation using normal distributions.
* **Pandas**: Tabular data structure handling, dataframe filtering, and cluster aggregation.
* **Matplotlib**: Inertia curve plotting and final scatter visualization.
* **Scikit-learn**: Unsupervised clustering estimators (`KMeans`) and feature preprocessing (`StandardScaler`).

## Environment & Web-Based Quick Start

You can run and evaluate this repository entirely within a browser-based GitHub Codespace or a standard virtual environment.

### 1. Setup via Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn
```

### 2. Execution
```bash
Run the script from your terminal:
    python main.py
```

## Repository Structure

```bash
cluster-segment-engine/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
└── main.py
```

## Roadmap

**[ ] Type Hinting Integration:** Add comprehensive type stubs (mypy) across core data loading and transformation routines.

**[ ] Modular Architecture Refactoring:** Separate data intake, scaling logic, and matplotlib plotting into dedicated internal modules.

**[ ] Automated Silhouette Evaluation:** Implement silhouette coefficient scoring alongside the elbow method to provide quantitative cluster separation metrics.
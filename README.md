# HCGNN-DKC: Hierarchical Causal GNN with Domain Knowledge Constraints

**Interpretable Urban Spatial Layout Generation via Neuro-Symbolic Integration**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Research%20Code-orange)

## 📅 Release Date
**January 6, 2026**

## 📖 Introduction

This repository contains the implementation of the **HCGNN-DKC** model.

**HCGNN-DKC** is a neuro-symbolic framework designed to address the "black box" nature of deep learning in urban planning. It integrates three key domain knowledge components into the learning process:
1.  **Ontological Constraints:** Structuring the hierarchical scope of urban indicators.
2.  **Expert Causal DAG:** Initializing the graph topology with heuristic causal flows.
3.  **Normative Logical Rules:** Regularizing the solution space using **DomiKnowS** to ensure physical and regulatory compliance.

## 📂 Directory Structure

```text
.
├── data/                 # USL sample datasets (csv)
├── runs/                 # Checkpoints and results
├── config.py             # Configurations
├── constraints.py        # All logical rules
├── data reader.py        # Data loading utilities
├── experiment.py         # Main experiment
├── main.py               # Entry point for training
├── models.py             # Model architecture 
├── patch_domiknows.py    # Patches
├── sensors.py            # DomiKnowS sensor
└── requirements.txt      # Python dependencies
```

## 🛠️ Prerequisites

### System Requirements
*   **Python:** 3.11
*   **CUDA:** Recommended for GPU acceleration (though CPU is supported).

### Core Dependencies
This project relies on the following key libraries:
*   **[DomiKnowS](https://github.com/HLR/DomiKnowS):** A Python library that facilitates the integration of domain knowledge in deep learning architectures.
*   **[Gurobi Optimizer](https://www.gurobi.com/):** Used for inference-time constraint satisfaction. 
    *   *Note:* You need a valid Gurobi license (Academic License is free for researchers). Please refer to the [Gurobi Documentation](https://www.gurobi.com/documentation/) for setup.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HaochengSun722/HCGNN-DKC.git
    cd HCGNN-DKC
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate on Windows
    venv\Scripts\activate
    # Activate on Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the model:**
    ```bash
    python main.py
    ```
    *Check `config.py` to adjust hyperparameters (learning rate, epochs, etc.) before running.*

## 📊 Data

### Dataset in Repository
The `data/` directory provides the pre-processed dataset used in our experiments, comprising **1,061 blocks** sampled from Central Business Districts (CBDs). This dataset is the result of calculating and discreting 18 USL indicators based on the raw data sources mentioned above.

*   *Note:* Ensure the data is placed in the `data/` directory following the format described in `data reader.py`.

### Data Sources
The USL indicators used in this study were derived and calculated from the following open-source databases:
*   **[OpenStreetMap (OSM)](https://www.openstreetmap.org/):** Used for extracting road networks and building footprints.
*   **[EULUC-China 2.0](https://www.sciencedirect.com/science/article/pii/S2095927325007200):** Used for identifying essential urban land-use categories with block geometry.

## 🧠 Domain Knowledge

The normative domain knowledge embedded in this model includes **364 logical rules** derived from:
*   Planning Common Sense
*   Statistical Correlations
*   Urban Design Guidelines (e.g., *[Hong Kong Planning Standards](https://www.pland.gov.hk/file/tech_doc/hkpsg/full/pdf/ch11.pdf)*)

All constraints are explicitly defined in `constraints.py`.


## 🙏 Acknowledgements

*   We thank the **[HLR Lab](https://hlr.github.io/)** team for their open-source [DomiKnowS](https://github.com/HLR/DomiKnowS) framework.
*   We acknowledge **[Gurobi Optimization](https://www.gurobi.com/)** for providing the academic license.


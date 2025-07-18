# 💹 FCB_Dynamic-Pricing

<p align="left">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/Language-Python-blue" alt="Language">
  <img src="https://img.shields.io/badge/Cloud-GCP-blue" alt="Cloud">
</p>

> A dynamic pricing and decision support system for football match tickets. **Objective:** To evolve a manual price-decision process into a data-driven, semi-automated workflow. The system's core Decision Engine improves the precision of each price variation, with the final goal of optimizing revenue and ticket sales.

### Outline

- [Key Results & Metrics](#key-results--metrics)
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Usage](#usage)

---

## Key Results & Metrics

| Metric                      | Result                               | Description |
| :-------------------------- | :----------------------------------- | :----------------------------------- |
| 📈 Revenue Uplift           | **+9%** Average Revenue per Match    | Achieved by dynamically adjusting prices to match real-time demand forecasts, capturing more value from high-demand matches. Validated via controlled A/B testing.|
| ⚙️ Operational Efficiency   | **7x improvement** in Time-to-Price-Change | Realized by automating the manual data aggregation and analysis pipeline. The system delivers price recommendations directly, shifting the team's focus from data work to strategic approval.|
| 🎯 Demand Forecast Accuracy | **86%** Accuracy (WAPE)              | The result of a model combining internal sales data with external signals. Sales predictions were highly reliable.|
| 🎟️ Optimized Sales          | **+6%** Increase in Ticket Sell-Through Rate | A direct result of modeling price elasticity. Didn't maximize revenue at the cost of empty seats; also improved occupancy, which affects atmosphere and in-stadium sales.|

## Project Overview

The diagram below illustrates the conceptual framework for the Dynamic Pricing project. At its core, the system sits between two main stakeholders, The Club and The Fan, each with opposing goals. The engine's purpose is to find an optimal balance by ingesting various data points, processing them, and providing data-driven answers to both sides. It essentially acts as the *brain* that determines ticket prices based on a range of real-time and historical information.

<p align="left">
  <img src="./assets/dp-hl.png" alt="High-level Project Diagram" width="1500">
</p>

The system operates in a continuous loop: the Dynamic Pricing Engine constantly ingests and analyzes both Internal Factors (like how many seats are left) and External Factors (like social media buzz). Based on this combined data, it **forecasts demand at various price points, which the Decision Engine uses to generate a recommended price**. The Club uses these outputs to set the official ticket prices. The Fan, in turn, sees these prices and uses the system's guidance (e.g., price-drop alerts) to decide when to buy. This entire process creates a responsive, market-driven pricing strategy that is more sophisticated than a static, pre-set price list; this moves from a reactive, manual process to a proactive, automated one with human-in-the-loop (HiTL).

| 🚩 The Problem | 💡 The Solution |
| :--------------------------- | :---------------------------- |
| **Static Pricing**: Prices were set once per season in rigid, inflexible categories (e.g., A++, A, B). | **Dynamic Recommendations**: Generates price proposals for each seating zone based on real-time data analysis. |
| **Manual Adjustments**: The team would slowly analyze various metrics to manually propose price changes. | **Impact Simulation**: Instantly models the projected impact of any price change on revenue and ticket sales. |
| **Data Bottleneck**: Extracting data manually from fragmented systems was slow and operationally complex. | **Centralized Data**: Automatically aggregates all key data points—sales, web analytics, contextual data, etc.—into one place. |
| **Slow Implementation**: The process to act on a decision was manual and disconnected from the sales platform. | **Seamless Integration**: Allows for one-click approval on a dashboard, which triggers a price update to the live ticketing system via REST API. |

## Methodology

This project implemented a complete, production-ready dynamic pricing solution, from initial business discovery to final deployment.

### 1. Scoping

The initial phase involved meeting with business stakeholders (product, legal, and marketing) to define the exact objective. The goal was set to maximize revenue while respecting key business constraints, such as price caps, minumum occupacy and limits on price change frequency.

### 2. Modeling

The modeling strategy follows a two-stage process: first predict, then optimize. The system first forecasts demand with high accuracy and then uses that forecast within a Decision Engine to find the optimal price.

| Modeling Task | Modeling Approach | Key Technology | Rationale for Choice |
| :--- | :--- | :--- | :--- |
| **1. Demand Forecasting** | Forecast future ticket demand for each match **at various potential price points**. Optimized for **predictive accuracy**. | `GradientBoostingRegressor` | Handles complex non-linear relationships essential for accurate "what-if" simulations. |
| **2. Price Optimization** | **Use the demand model to simulate outcomes** and recommend a revenue-maximizing price. Optimized for **business impact** and **constraints**. | `Custom Python Logic` | A simulation and grid-search framework that uses the demand model to find the optimal price, while respecting business rules (e.g., price caps). |

The core of this project is the **Decision Engine**, which translates the demand forecast into actionable business recommendations. It consists of two key components that work together to support a Human-in-the-Loop (HITL) workflow.

#### The Simulation Engine

| Aspect | Description |
| :--- | :--- |
| **Purpose** | To power the 'Impact Simulation' feature for "what-if" analysis by the commercial team. |
| **Question Answered** | "If I set the price to X, what is the likely impact on sales and revenue?" |
| **Core Function** | Takes a hypothetical price and match features as input, and uses the trained Demand Forecast Model to predict the outcome, providing an instant, data-driven preview of any potential pricing decision. |

#### The Optimization Engine

| Aspect | Description |
| :--- | :--- |
| **Purpose** | To proactively generate the official `Price Variation Proposal`. |
| **Question Answered** | "What is the single best price to set for this seat to maximize our total revenue?" |
| **Core Function** | Operates by performing a grid search across a range of valid prices defined by business constraints. For each price point, it simulates the revenue using the demand model and returns the price that yields the highest projected revenue. |

### 3. Feature Engineering

A key part of the strategy was to enrich our models with external data, a common gap in existing research.
* **🏠 Internal factors**: Utilized traditional data such as historical sales, opponent tier, days until the match, and real-time ticket availability.
* **🌍 External factors**: Integrated novel real-time signals including social media sentiment, search engine trends, and competing city events to capture market dynamics.

> For a detailed description of the features in the synthetic dataset, please refer to the [Data Dictionary](reports/data-dictionary.md).

### 4. A/B Testing & Validation

Before a full rollout, the system was rigorously validated through controlled A/B tests. The new dynamic pricing model was applied to a few sections of the stadium, with the rest serving as a control group. This allowed us to scientifically prove the model's positive impact on revenue.

### 5. Deployment

The entire system was deployed within an automated MLOps pipeline. This ensures models are automatically retrained on new data, performance is constantly monitored for degradation, and price recommendations are reliably fed to the ticketing system via an API. All models were designed for batch prediction, running on a daily schedule to balance cost and the need for timely updates.

## Architecture

The architecture is designed for a robust, human-in-the-loop workflow. Data from various internal and external sources is ingested and processed by the core ML models. The resulting proposals and simulations are then presented to the commercial team on a User Control Panel for final review and approval, which triggers the price update via a REST API.

> For a detailed description of the diagram and its' components, please refer to the [Architecture Diagram](reports/architecture-diagram.md).

<p align="left">
  <img src="./assets/dp-ll.png" alt="Low-level Project Diagram" width="950">
</p>

## Project Structure

```
FCB_Dynamic-Pricing/
├── .gitignore                         # Specifies files for Git to ignore.
├── LICENSE                            # Project license (MIT).
├── README.md                          # An overview of the project. <-- YOU ARE HERE
├── requirements.txt                   # The requirements file for reproducing the analysis.
├── config.py                          # Configuration file for paths, parameters, etc.
├── assets/                            # Contains images and diagrams for the README.
├── data/                              # Stores data related to the project.
│   ├── 01_raw/                        # The original, immutable data.
│   └── 02_processed/                  # Processed and cleaned data ready for modeling.
├── models/                            # Stores trained model artifacts.
├── notebooks/                         # Jupyter notebooks for analysis and experimentation.
│   ├── eda.ipynb                      # Exploratory Data Analysis notebook.
├── reports/                           # Contains explanatory documents.
│   ├── data-dictionary.md             # A detailed description of the dataset features.
│   └── architecture-diagram.md        # An explanation of the system architecture.
└── src/                               # Source code for the project.
    ├── __init__.py                    # Makes src a Python package.
    ├── data/                          # Scripts for data ingestion and processing.
    │   └── make_dataset.py            # Script to generate the synthetic dataset.
    └── features/                      # Scripts for feature engineering.
    │    └── build_features.py         # Script to process data into model-ready features. 
    ├── models/                        # Scripts for model training and prediction.
    │   ├── train_demand_model.py      # Script to train the demand prediction model.
    │   └── predict_demand.py          # Script to get a sample demand prediction.
    └── decision_engine/               # Scripts for simulation and optimization.
        ├── simulate.py                # Script to run a what-if simulation.
        └── optimize.py                # Script to find the optimal price.

```

## Usage

### 🚀 Running the Pipeline

To run the project and see the full pipeline in action, follow these steps from your terminal.

1.  **Set up the environment** (only needed once):
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate the dataset:**
    ```bash
    python -m src.data.make_dataset
    ```

3.  **Process features for modeling:**
    ```bash
    python -m src.features.build_features
    ```

4.  **Run the training pipeline:**
    ```bash
    python -m src.models.train_demand_model
    ```

5.  **Run the Decision Engine scripts (optional):**
    ```bash
    # Get a "what-if" analysis for a specific price
    python -m src.decision_engine.simulate

    # Get a revenue-optimal price recommendation
    python -m src.decision_engine.optimize

### Using the Decision Engine's output

The system provides two key outputs for the commercial team via the User Control Panel:

1. **The Price Recommendation**: This is the revenue-maximizing price identified by the Optimization Engine. It serves as a powerful, data-driven starting point.
2. **The Impact Simulation**: This allows the team to test their own hypotheses by entering any price and instantly seeing the predicted impact on ticket sales and revenue.

The workflow is designed to be **Human-in-the-Loop (HiTL)**. The team uses these outputs to make a final, informed decision, blending machine intelligence with their expert knowledge.

### How the Simulation works

The "Impact Simulation" feature is powered by the **Demand Forecast Model**. This model was trained to predict the number of tickets that will be sold based on a given price and other market conditions.

When a user enters a hypothetical price into the control panel, the system feeds this price into the demand model to get a sales forecast. It then calculates the projected revenue (`predicted sales × price`), giving the commercial team an instant preview of the potential outcome of their pricing decisions.

</br>

> ⚠️ **Project Disclaimer:**
>
> * **Illustrative Purpose:** This repository serves as a high-level demonstration of the project's architecture and methodology. Many implementation details and model complexities have been simplified for clarity.
> * **Synthetic Data:** The code runs on synthetic data, as the original data is proprietary and cannot be shared. The purpose is to demonstrate the modeling approach and engineering best practices of the real-world project.

<p align="center">© 🌐 2025 t.r.</p>

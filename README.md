# 💹 FCB_Dynamic-Pricing

<p align="left">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
  <img src="https://img.shields.io/badge/Language-Python-lightgrey" alt="Language">
</p>

> A machine learning powered dynamic pricing and decision support system for ticket pricing in the sports industry. **Objective:** To evolve a manual price-decision process into a data-driven, semi-automated workflow that improves ticketing revenue and sales.

### Outline

- [Key Results](#key-results)
- [Overview](#overview)
- [Architecture](#architecture)
- [Modeling](#modeling)
- [Structure](#structure)

---

## Key Results

| Metric                      | Result                               | Description |
| :-------------------------- | :----------------------------------- | :----------------------------------- |
| 📈 Revenue Uplift           | **+6%** Average Revenue per Match    | Achieved by dynamically adjusting prices to match real-time demand forecasts, capturing more value from high-demand matches. Validated via A/B testing.|
| 🎟️ Optimized Sales          | **+4%** Increase in Ticket Sell-Through Rate | Didn't maximize revenue at the cost of empty seats; also improved occupancy, which positively affects atmosphere and in-stadium sales.|
| ⚙️ Operational Efficiency   | **7x improvement** in Time-to-Price-Change | From weekly to daily changes by automating the manual data aggregation and analysis pipeline. The system delivers price recommendations directly, shifting the team's focus from data work to strategic approval.|
| 🤝 Recommendation Adoption | **91%** of Proposals Approved | Percentage of automated price proposals that were reviewed and approved by the commercial team, indicating trust in the model's business alignment.|
| 🎯 Demand Forecast Accuracy | **14%** Weighted Avg. % Error | The model's predictions have a low average error, performing 60% better than a baseline `DummyRegressor` and indicating that sales forecasts are reliable.|

## Overview

The diagram below illustrates the project's conceptual framework. The system acts as the central *brain* to balance the goals of The Club and The Fan. It operates in a continuous loop by ingesting internal and external factors to forecast demand at various price points. The **Decision Engine** then uses this forecast to recommend an optimal price. This transforms a static, manual pricing strategy into a responsive, automated system with a human-in-the-loop (HiTL), creating a market-driven approach for both setting and responding to ticket prices.

<p align="center">
  <img src="./assets/dp-hl.png" alt="High-level Project Diagram" width="2000">
  <br>
  <em>Fig. 1: A high-level diagram of the Dynamic Pricing Engine.</em>
</p>

The core challenge was to move from a rigid, manual pricing strategy to a data-driven, automated one. The table below summarizes the problem–solution mapping.

| 🚩 The Problem | 💡 The Solution |
| :--------------------------- | :---------------------------- |
| **Static pricing**: Prices were set once per season in rigid, inflexible categories (e.g., A++, A, B), then updated weekly/monthly. | **Dynamic recommendations**: Generates price proposals for each seating zone based on near real-time data analysis, allowing for daily updates. |
| **Manual adjustments**: The team would slowly analyze various metrics to manually propose price changes. | **Impact simulation**: Instantly models the projected impact of any price change on revenue and ticket sales. |
| **Data bottleneck**: Extracting data manually from fragmented systems was slow and operationally complex. | **Centralized data**: Automatically aggregates all key data points—sales, web analytics, contextual data, etc.—into one place. |
| **Slow implementation**: The process to act on a decision was manual and disconnected from the sales platform. | **Seamless integration**: Allows for one-click approval on a dashboard, which triggers a price update to the live ticketing system via REST API. |


## Architecture

The general workflow is as follows:
1. **Data Sources** are collected and fed into the central engine.
2. The **Dynamic Pricing Engine** uses machine learning models and business rules to generate a price recommendation.
3. The pricing team uses the **UI & Integration** layer to review, simulate, and approve the price, which is then updated in the live ticketing system.

<p align="center">
  <img src="./assets/dp-ll.png" alt="Low-level Project Diagram" width="950">
    <br>
  <em>Fig. 2: A low-level diagram of the Dynamic Pricing Engine.</em>
</p>

<details>
<summary><b>Click to see the detailed architecture breakdown</b></summary>

### 1. Data Sources

| Component | Description |
| :--- | :--- |
| **Ticket Sales & Availability** | Historical and real-time data on ticket inventory, sales velocity, and transactions per seating zone. |
| **Competitors Pricing** | Scraped pricing data from secondary markets (e.g., Viagogo, Stubhub, etc.) for competitive analysis. |
| **Web/App Analytics** | Data on user behavior from the official website and app, including page visits, clicks, and conversion funnels. |
| **Matches, Competitions & Channels** | Foundational information about each match, including opponent, date, competition type, and sales channel. |

### 2. Dynamic Pricing Engine

| Component | Description |
| :--- | :--- |
| **Data Ingestion & Centralization** | The entry point that gathers data from all sources and consolidates it into a unified data store for processing. |
| **ML & Analytics Core** | The central "brain" where data is processed, features are engineered, and the machine learning models are trained and executed. |
| **Business Constraints** | A module that receives strategic inputs from the club (e.g., price floors/caps) and applies these rules to the optimization process, ensuring recommendations are compliant with business strategy. |
| **Decision Module** | A container for the core predictive models that feed the optimization engine. |
| ┣ **Demand Forecast Model** | A model that predicts the expected volume of ticket sales at various price points, using historical data and match context to inform its forecast. |
| ┣ **Match Clustering** | An algorithm that groups similar past matches to provide a contextual baseline for the *Demand Forecast Model*. |
| **Decision Engine: Optimization & Simulation** | Takes the predicted demand curve and business rules to find the revenue-maximizing price. It also runs simulations for "what-if" scenarios. |
| **Anomaly Warnings** | An alerting system that flags unusual sales patterns or pricing recommendations that deviate from norms. |
| **Impact Simulation** | A feature that allows a human user to test a hypothetical price and see a projection of its impact on sales and revenue. |
| **Price Variation Proposal** | The final output of the engine: a concrete price recommendation for a given seat or section. |

### 3. UI & Integration

| Component | Description |
| :--- | :--- |
| **User Control Panel** | The dashboard used by the pricing team to view price proposals, run impact simulations, and approve or reject changes, enabling Human-in-the-Loop (HITL) control. |
| **REST API** | The communication layer that allows the User Control Panel to send approved price change commands to the live ticketing system. |
| **Price Drop Logic** | An automated module that can trigger price change events based on predefined rules, such as slow ticket sales. |
| **Ticketing Purchase System** | The club's main backend system that processes transactions and manages ticket inventory. It receives price update commands from the API. |
| **Fan Engagement** | The final layer where fans interact with the system's output. |
| ┗ **Ticketing Purchase UI** | The public-facing website or application screen where fans see the dynamically adjusted prices and make their purchases. |

</details>

## Modeling

The modeling strategy followed a two-stage process: first *predict*, then *optimize*. This phase included the following tasks:
- Select and train a model using the prepared dataset.
- Conduct error analysis to identify improvement areas.
- Iterate on model architecture, hyper-parameters, or data as needed.
  
The system first forecasts demand with high accuracy and then uses that forecast within a *Decision Engine* to find the optimal price.

### Stage 1: 📈 Demand Forecasting

This stage answers the question: *"At a given price, how many tickets are we likely to sell?"*

| Aspect         | Description                                                                                                                                                                             |
| :------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model** | A `GradientBoostingRegressor` forecasts ticket demand (`zone_historical_sales`) by seating zone for each match.                                                                              |
| **Rationale** | Gradient Boosting excels at handling the complex, non-linear relationships discovered during EDA and is robust against outliers, making it a strong choice for this task. |
| **Features** | The model uses a rich set of internal and external factors, including historical sales, opponent tier, social media sentiment, and other engineered features.|
| **Application**| This trained model powers the *Impact Simulation* feature, allowing the commercial team to perform "what-if" analysis by inputting a hypothetical price and instantly seeing the likely impact on revenue and sales. |
| **Design choice**| While `XGBoost` or `LightGBM` are often faster and would probably provide a performance edge, the choice of scikit-learn's `GradientBoostingRegressor`, because of the synthetic dataset size, the difference would be negligible. |

<details>
<summary><b>Click to see the detailed model performance evaluation</b></summary>

To ensure the final pricing decision is effective, the underlying demand forecast must be highly accurate. Therefore, the primary goal of this evaluation was to minimize prediction error. Performance was evaluated against a **baseline model** (`DummyRegressor`) to ensure the model was genuinely learning. The key metric chosen was **WAPE**, as it provides a clear, interpretable measure of percentage error that resonates with business stakeholders.

| Metric                        | Value           | Description & Rationale                                                                                                                                                                                              |
| :---------------------------- | :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **WAPE** (Primary Metric) | **14%** | **Why we chose it:** Weighted Absolute Percentage Error is the most critical metric for this business case. It tells us the average forecast error in percentage terms, making it highly interpretable for revenue planning. A low WAPE is our main goal. |
| **R² Score** | **0.86** | **For model fit:** This shows that the model explains 86% of the variance in ticket sales, confirming it has a strong statistical fit to the data and learns the underlying patterns effectively.                                |
| **Mean Absolute Error (MAE)** | **~254 tickets**| **For business context:** MAE tells us that, on average, our forecast is off by about 254 tickets. This gives stakeholders a concrete sense of the error margin in absolute units.                                      |
| **Root Mean Squared Error (RMSE)**| **~312 tickets**| **For robustness:** RMSE penalizes larger errors more heavily. A higher RMSE relative to MAE suggests the model occasionally makes larger prediction errors, which is useful information for risk assessment.             |

The performance was considered *successful*. A WAPE of 14% and an R² of 0.86 demonstrated a robust and reliable forecasting engine.

</details>

### Stage 2: ⚙️ Price Optimization

This stage answers the business question: *"What is the single best price to maximize total revenue?"*

| Aspect         | Description                                                                                                                                                                                          |
| :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model** | A custom *Optimization Engine* performs an exhaustive grid search over a range of valid prices.                                                                                                        |
| **Rationale** | A grid search is a reliable and straightforward method to find the optimal price within defined business constraints (e.g., price caps and floors). It guarantees finding the maximum projected revenue. |
| **Process** | The engine iterates through potential prices (e.g., from €75 to €350), uses the demand model to predict sales for each, calculates the projected revenue `(Price × Predicted Sales)`, and returns the optimal price. |
| **Output** | The engine's primary output is the official `Price Variation Proposal`, which is sent to the commercial team for review and approval.                                                                   |
| **Design choice**| Bayesian Optimization would likely find a near-optimal price much faster by intelligently exploring the price space. However, it doesn't guarantee finding the absolute maximum. Guaranteeing the optimal recommendation (within the model's predictive power) is often more valuable than the computational speed gained from a heuristic approach. |

<details>
<summary><b>Click to see the detailed model performance evaluation</b></summary>

Since this is an optimization engine, not a predictive model, its performance is measured by its business value and efficiency.

| Metric            | How We Measure It                                                                                              | Success Criteria                                                                                        |
| :---------------- | :------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| **Revenue Lift** | Through A/B testing, comparing the revenue generated by the engine's prices against a control group. | A consistent, statistically significant increase in average revenue per match.                               |
| **Adoption Rate** | Tracking the percentage of `Price Variation Proposals` that are reviewed and approved by the commercial team.    | A high adoption rate (>80%) indicates that the team trusts and values the engine's recommendations.         |
| **Computation Time**| Measuring the wall-clock time it takes for the grid search to complete for a given match.                       | The time must be within acceptable operational limits to allow for rapid, on-demand analysis by the commercial team. |

</details>

### Feature Engineering

A key part of the modeling strategy was to move beyond our internal sales history by enriching our models with external data. Through feature engineering, we combined our own historical performance data with real-world market signals—like opponent rankings and social media hype—to create a more holistic and predictive view of market dynamics. The model's accuracy is dependent on a feature set combining internal and external data:

* **🏠 Internal factors**: `opponent_tier`, `historical_sales`, `zone_seats_availability`, and `days_until_match`.
* **🌍 External factors**: `weather_forecast`, `social_media_sentiment`, `search_engine_trends`, and `competing_city_events`.

<details>
<summary><b>Click to see the detailed list of features</b></summary>

Each row in the synthetic dataset (`synthetic_match_data.csv`) represents the state of a specific seating *zone* for a single *match* at a particular point in time, defined by the `days_until_match`. The primary goal is to predict `zone_historical_sales` based on the other features.

### Identifiers & categorical features

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `match_id` | Integer | A unique identifier for each football match. |
| `zone` | String | The name of the seating zone in the stadium (e.g., 'Gol Nord', 'Lateral', 'VIP'). |
| `opponent_tier` | String | A categorical rating of the opponent's quality and appeal (`A++`, `A`, `B`, `C`). Higher tiers signify more attractive matches, influencing demand. |
| `weather_forecast` | String | The predicted weather for the match day ('Sunny', 'Cloudy', 'Rain'). Can influence last-minute purchase decisions. |
| `competing_city_events` | Boolean | `True` if there are other major events (concerts, festivals) in the city on the same day, which could reduce local demand. `False` otherwise. |

### Time-based & demand signals

These features capture the dynamics of demand over time and external market interest.

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `days_until_match` | Integer | The number of days remaining before the match. A key feature for time-series analysis, as demand typically increases as the match date approaches. |
| `flights_to_barcelona_index`| Integer | A synthetic index (scaled 20-100) representing the volume of inbound flights to the city. This serves as a proxy for tourist demand. |
| `google_trends_index` | Integer | A synthetic index (scaled 20-100) representing public search interest for the match on Google. A proxy for general public interest and hype. |
| `internal_search_trends`| Integer | A synthetic count of searches for match tickets on the club's own website or app. A direct signal of purchase intent from the user base. |
| `web_visits` | Integer | A synthetic count of visits to the ticketing section of the club's official website. A measure of online traffic and interest. |
| `web_conversion_rate` | Float | The synthetic conversion rate on the website (ticket purchases / visits). A measure of how effectively web traffic is converting into sales. |
| `social_media_sentiment`| Float | A synthetic score representing the overall public sentiment (e.g., from -1.0 for strong negative to +1.0 for strong positive) about the match on social media platforms. |

### Sales, availability & pricing

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **`zone_historical_sales`** | **Integer** | **(Target Variable)** The historical number of tickets sold for a similar match in that zone. This is the *primary target variable* for the demand forecast model. |
| `zone_seats_availability` | Integer | The absolute number of seats still available for purchase in that zone. |
| `ticket_availability_pct` | Float | The percentage of total seats in the zone that are still available. |
| `competitor_avg_price` | Float | The average ticket price for a comparable entertainment event (e.g., mobile world congress, a concert) on the same day. Represents the competitive landscape. |
| `ticket_price` | Float | The price of the ticket. This is a *key input* feature for the demand model and the *final output* of the optimization engine. |

</details>

### Validation

Before a full rollout, the system was rigorously validated through a series of controlled **A/B tests** to scientifically measure its impact and mitigate risk. The core principle was to isolate the effect of the dynamic pricing engine from all other market variables. 

The results from the A/B tests confirmed our hypothesis, showing a consistent **+6% lift in average revenue** for the treatment group. Crucially, this was achieved while also increasing the sell-through rate, demonstrating that the model was effective at finding the true market equilibrium. These conclusive, data-backed results gave the business full confidence to proceed with a full-scale rollout of the dynamic pricing system across all stadium zones.

<details>
<summary><b>Click to see the detailed experimental design</b></summary>

### Experimental Design

1.  **Treatment vs. Control Groups**: The stadium was segmented into statistically similar groups of seating zones.
    * **Treatment Group (Dynamic Pricing)**: A select number of zones had their prices set by the new automated engine. These prices could change daily based on the model's recommendations.
    * **Control Group (Static Pricing)**: The remaining zones operated under the existing pricing strategy (e.g., prices set manually at the beginning of the season), serving as our baseline for comparison.

2.  **Hypothesis**: Our primary hypothesis was that the treatment group would generate a statistically significant lift in total revenue per match without negatively impacting the ticket sell-through rate compared to the control group.

3.  **Duration**: The tests were run over several matches of varying importance (e.g., high-demand league matches, lower-demand cup matches) to ensure the results were robust and not skewed by the unique characteristics of a single event.

### Key Metrics Tracked

To evaluate the experiment's outcome, we continuously monitored several KPIs for both groups:

* **Primary Metric**: Avg. Revenue Per Available Seat.
* **Secondary Metrics**:
    * Ticket Sell-Through Rate (Occupancy).
    * Avg. Ticket Price.
    * Sales Velocity (how quickly tickets sold).

</details>


## Structure

While the source code and data for this project are kept private to honor confidentiality agreements, this section outlines the project's structure. This demonstrates a professional, modular, and reproducible approach to building machine learning systems, designed for easy maintenance and scalability.

The project was designed with the following directory structure:

```bash
FCB_Dynamic-Pricing/
├── assets/                         # Diagrams and images for documentation.
├── data/                           # (Private) Stores raw, intermediate, and synthetic data.
├── models/                         # (Private) Stores trained model artifacts.
├── notebooks/                      # (Private) Jupyter notebooks for EDA.
│   └── eda.ipynb                   # (Private) Exploratory Data Analysis notebook.
└── src/                            # (Private) Source code, organized by function.
    ├── data/                       # (Private) Scripts for data ingestion and processing.
    └── features/                   # (Private) Scripts for feature engineering.
    │    └── build_features.py      # (Private) Script to process data into model-ready features. 
    ├── models/                     # (Private) Scripts for model training and prediction.
    │   ├── train_demand_model.py   # (Private) Script to train the demand prediction model.
    │   └── predict_demand.py       # (Private) Script to get a sample demand prediction.
    └── decision_engine/            # (Private) Scripts for simulation and optimization.
        ├── simulate.py             # (Private) Script to run a what-if simulation.
        └── optimize.py             # (Private) Script to find the optimal price.
```

### Descriptions

* **`notebooks/eda.ipynb`**: A Jupyter Notebook was used for all Exploratory Data Analysis. It contained the initial data visualizations and statistical analysis that guided the feature engineering and modeling strategy.

* **`src/features/build_features.py`**: This script handled all preprocessing and feature engineering. It was designed to take raw data and transform it into a clean, model-ready feature set by handling categorical variables, creating interaction terms, and engineering relevant time-based features.

* **`src/models/train_demand_model.py`**: The core machine learning model was trained here. The script loaded the processed features, trained a demand forecasting model (e.g., Gradient Boosting), and saved the final trained model artifact to the `models/` directory for later use.

* **`src/decision_engine/`**: This package contained the logic for the dynamic pricing system.
    * **`simulate.py`**: A script that used the trained model to run "what-if" scenarios, predicting demand and revenue across a range of potential price points.
    * **`optimize.py`**: The final script that orchestrated the simulation to identify the single price point that would maximize projected revenue for a given match.


</br>

> ⚠️ **DISCLAIMER**
>
> * **Illustrative purpose:** This repository serves as a high-level demonstration of the project's architecture and methodology. Many implementation details and model complexities have been simplified for clarity.
> * **Confidentiality:** Source code and data for this project are kept private to honor confidentiality agreements. The purpose is to demonstrate the modeling approach and engineering best practices of the real-world project.

</br>

<p align="center">🌐 © 2025 t.r.</p>

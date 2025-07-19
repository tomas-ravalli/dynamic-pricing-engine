# Modeling –Extended

> This document provides a detailed overview of the modeling strategy for the FCB_Dynamic-Pricing project. The goal is to evolve from manual, intuition-based pricing to a data-driven, semi-automated system that optimizes revenue while respecting business constraints.

## A Two-Stage Framework

The core of the system is a two-stage process that first **predicts** demand and then **optimizes** for the best price. This decoupled approach enhances accuracy and flexibility.

### Stage 1: 📈 Demand Forecasting

This stage answers the question: *"At a given price, how many tickets are we likely to sell?"*

| Aspect | Description |
| :--- | :--- |
| **Model** | A `GradientBoostingRegressor` forecasts ticket demand (`zone_historical_sales`) for each match. |
| **Rationale** | Gradient Boosting excels at handling the complex, non-linear relationships discovered during EDA and is robust against outliers, making it a strong choice for this task. |
| **Features**| The model uses a rich set of internal and external factors, including historical sales, opponent tier, social media sentiment, and the `ticket_price` itself to learn price elasticity. |
| **Application** | This trained model powers the **Simulation Engine**, allowing the commercial team to perform "what-if" analysis by inputting a hypothetical price and instantly seeing the likely impact on sales and revenue. |

### Stage 2: 💹 Price Optimization

This stage answers the business question: *"What is the single best price to maximize total revenue?"*

| Aspect | Description |
| :--- | :--- |
| **Model** | A custom **Optimization Engine** performs an exhaustive grid search over a range of valid prices. |
| **Rationale** | A grid search is a reliable and straightforward method to find the optimal price within defined business constraints (e.g., price caps and floors). It guarantees finding the maximum projected revenue. |
| **Process** | The engine iterates through potential prices (e.g., from €75 to €350), uses the demand model to predict sales for each, calculates the projected revenue (Price × Predicted Sales), and returns the optimal price. |
| **Output** | The engine's primary output is the official `Price Variation Proposal`, which is sent to the commercial team for review and approval. |

### Design Choices & Trade-offs

Key decisions made during the modeling process are summarized below.

| Category | Choice & Rationale | Alternatives & Trade-offs |
| :--- | :--- | :--- |
| **Model Selection** | **`GradientBoostingRegressor`**: Chosen for its high performance and ability to capture non-linearities without the high overhead of more complex models. | **Linear Models**: Too simple for the complex relationships. <br> **Deep Learning**: Higher data/infra requirements; less interpretable. |
| **Feature Engineering** | **`StandardScaler` & `OneHotEncoder`**: Essential for normalizing numerical features with different scales and encoding impactful categorical features like `zone` and `opponent_tier`. | A simpler approach might miss key interactions, while more complex feature engineering could lead to overfitting. |
| **Optimization** | **Grid Search**: Reliable and exhaustive, guaranteeing the optimal price within the defined search space. | **Bayesian Optimization**: Computationally faster but less exhaustive. Reliability was prioritized for this business-critical function. |

## Performance & Business Impact

The model's success is measured by both its predictive accuracy and its tangible business impact.

| Metric | Model | Value | Description |
| :--- | :--- | :--- | :--- |
| **R² Score** | Demand Forecast | **0.86** | The model explains 86% of the variance in the test set, indicating strong predictive performance. |
| **Revenue Uplift** | Optimization Engine | **+9%** | Average revenue increase per match, validated via A/B testing against a static pricing model. |
| **Sell-Through Rate**| Optimization Engine | **+6%** | Improved stadium occupancy by balancing revenue goals with price elasticity, avoiding empty seats. |

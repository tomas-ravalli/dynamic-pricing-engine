# 🌐 Modeling –Extended

> This document provides a detailed overview of the modeling strategy for the FCB_Dynamic-Pricing project.

The core of this project is a two-stage system that first **predicts** ticket demand and then **optimizes** the price to maximize revenue. This decoupled approach enhances both accuracy and flexibility.

## Stage 1: 📈 Demand Forecasting

This stage answers the question: *"At a given price, how many tickets are we likely to sell?"*

| Aspect         | Description                                                                                                                                                                             |
| :------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model** | A `GradientBoostingRegressor` forecasts ticket demand (`zone_historical_sales`) for each match.                                                                                           |
| **Rationale** | Gradient Boosting excels at handling the complex, non-linear relationships discovered during EDA and is robust against outliers, making it a strong choice for this task.                  |
| **Features** | The model uses a rich set of internal and external factors, including historical sales, opponent tier, social media sentiment, and the `ticket_price` itself to learn price elasticity. |
| **Application**| This trained model powers the **Simulation Engine**, allowing the commercial team to perform "what-if" analysis by inputting a hypothetical price and instantly seeing the likely impact on sales and revenue. |
| **Design Choice**| While `XGBoost` or `LightGBM` are often faster and can sometimes provide a performance edge, the choice of scikit-learn's `GradientBoostingRegressor`, because of the synthetic dataset size, the difference would be negligible. |

### Performance Evaluation

The primary goal here is to accurately predict ticket sales. The performance was evaluated against a **baseline model** (a `DummyRegressor` that always predicts the average sales from the training data) to ensure the model was genuinely learning from the features.

| Metric                        | Value           | Description & Rationale                                                                                                                                                             |
| :---------------------------- | :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R² Score** | **0.86** | **Why we chose it:** This is the primary metric. It measures the proportion of the variance in sales that our model can explain. An `R²` of 0.86 signifies a strong fit and a significant improvement over the baseline's `R²` of 0. |
| **Mean Absolute Error (MAE)** | **~254 tickets**| **For context:** MAE tells us, on average, how far off our predictions are in absolute terms. This is more interpretable for business stakeholders than other error metrics.             |
| **Root Mean Squared Error (RMSE)**| **~312 tickets**| **For robustness:** RMSE penalizes larger errors more heavily. A higher RMSE relative to MAE suggests the model makes a few, larger prediction errors, which is useful to know for risk assessment. |

The performance is considered **highly successful**. An `R²` of 0.86 demonstrates a robust and reliable forecasting engine.

Several strategies were employed to achieve this level of performance:

* **Better Data**: Integrated external data like social media sentiment and opponent rankings. This added crucial context that historical sales data alone could not provide.
* **Feature Engineering**: Created interaction terms (e.g., `day_of_week` vs. `opponent_tier`). This helped the model capture nuanced behaviors, such as high demand for a top-tier opponent even on a weekday.
* **Hyperparameter Tuning**: Used `GridSearchCV` to systematically test different model configurations (`n_estimators`, `max_depth`, `learning_rate`), finding the optimal combination that minimized prediction error on the validation set.

## Stage 2: ⚙️ Price Optimization

This stage answers the business question: *"What is the single best price to maximize total revenue?"*

| Aspect         | Description                                                                                                                                                                                          |
| :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Model** | A custom **Optimization Engine** performs an exhaustive grid search over a range of valid prices.                                                                                                        |
| **Rationale** | A grid search is a reliable and straightforward method to find the optimal price within defined business constraints (e.g., price caps and floors). It guarantees finding the maximum projected revenue. |
| **Process** | The engine iterates through potential prices (e.g., from €75 to €350), uses the demand model to predict sales for each, calculates the projected revenue (Price × Predicted Sales), and returns the optimal price. |
| **Output** | The engine's primary output is the official `Price Variation Proposal`, which is sent to the commercial team for review and approval.                                                                   |
| **Design Choice**| Bayesian Optimization would likely find a near-optimal price much faster by intelligently exploring the price space. However, it doesn't guarantee finding the absolute maximum. Guaranteeing the optimal recommendation (within the model's predictive power) is often more valuable than the computational speed gained from a heuristic approach. |

### Performance Evaluation

Since this is an optimization engine, not a predictive model, its performance is measured by its business value and efficiency.

| Metric            | How We Measure It                                                                                              | Success Criteria                                                                                        |
| :---------------- | :------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| **Revenue Lift** | Through A/B testing (simulated or live), comparing the revenue generated by the engine's prices against a control group (e.g., prices set manually or by a simpler heuristic). | A consistent, statistically significant increase in average revenue per match.                               |
| **Adoption Rate** | Tracking the percentage of `Price Variation Proposals` that are reviewed and approved by the commercial team.    | A high adoption rate (>80%) indicates that the team trusts and values the engine's recommendations.         |
| **Computation Time**| Measuring the wall-clock time it takes for the grid search to complete for a given match.                       | The time must be within acceptable operational limits (e.g., under 1 minute) to allow for rapid, on-demand analysis by the commercial team. |


</br>

<p align="center">🌐 © 2025 t.r.</p>

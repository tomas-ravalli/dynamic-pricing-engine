# Modeling –Extended

This document provides a detailed overview of the modeling strategy, design choices, and performance metrics for the FCB Dynamic Pricing project. The goal is to move beyond a manual, intuition-based pricing approach to a data-driven, semi-automated system that optimizes for revenue while respecting business constraints.

## 1. Modeling Approach: A Two-Stage Framework

The core of the system is a two-stage process that first **predicts** demand and then **optimizes** for the best price based on that prediction. This decoupling allows for greater accuracy and flexibility.

### Stage 1: Demand Forecasting 📈

The first stage answers the question: *"At a given price, how many tickets are we likely to sell?"*

* **Model**: A `GradientBoostingRegressor` is used to forecast ticket demand (`zone_historical_sales`) for each match at various potential price points.
* **Rationale**: Gradient Boosting was chosen for its ability to handle complex, non-linear relationships between features and its robustness against outliers. The EDA revealed that ticket price is driven by a mix of factors with varying scales and distributions, making a powerful model like Gradient Boosting a good fit.
* **Features**: The model uses a rich set of features, including:
    * **Internal Factors**: Historical sales, opponent tier, days until the match, and real-time ticket availability.
    * **External Factors**: Social media sentiment, search trends, and competing city events to capture market dynamics.
    * **Price**: The `ticket_price` itself is a key input feature, allowing the model to learn price elasticity.
 
The Simulation Engine

| Aspect | Description |
| :--- | :--- |
| **Purpose** | To power the 'Impact Simulation' feature for "what-if" analysis by the commercial team. |
| **Question Answered** | "If I set the price to X, what is the likely impact on sales and revenue?" |
| **Core Function** | Takes a hypothetical price and match features as input, and uses the trained Demand Forecast Model to predict the outcome, providing an instant, data-driven preview of any potential pricing decision. |

### Stage 2: Price Optimization 💹

The second stage uses the demand model to answer the business question: *"What is the single best price to set to maximize total revenue?"*

* **Model**: A custom **Optimization Engine** performs a grid search over a range of valid prices.
* **Rationale**: A grid search is a straightforward and effective way to find the optimal price within a defined set of business constraints (e.g., price caps and floors). It simulates the outcome for each price point and selects the one that yields the highest projected revenue.
* **Process**:
    1.  The engine iterates through a range of potential ticket prices (e.g., from €75 to €350 in increments of €5).
    2.  For each price, it uses the trained `GradientBoostingRegressor` model to predict the number of sales.
    3.  It calculates the projected revenue (Price × Predicted Sales) for each price point.
    4.  The engine then returns the price that results in the maximum projected revenue.
 
The Optimization Engine

| Aspect | Description |
| :--- | :--- |
| **Purpose** | To proactively generate the official `Price Variation Proposal`. |
| **Question Answered** | "What is the single best price to set for this seat to maximize our total revenue?" |
| **Core Function** | Operates by performing a grid search across a range of valid prices defined by business constraints. For each price point, it simulates the revenue using the demand model and returns the price that yields the highest projected revenue. |

## 2. Key Metrics & Performance

The success of the modeling approach was measured by both its predictive accuracy and its business impact.

| Metric             | Model               | Value  | Description                                                                                                                            |
| :----------------- | :------------------ | :----- | :------------------------------------------------------------------------------------------------------------------------------------- |
| **R² Score** | Demand Forecast     | **0.86** | The model explains 86% of the variance in the test set, indicating a strong predictive performance.                                    |
| **Revenue Uplift** | Optimization Engine | **+9%** | An average revenue increase per match, validated through controlled A/B testing against a static pricing model.                      |
| **Sell-Through Rate**| Optimization Engine | **+6%** | The model improved stadium occupancy by balancing revenue maximization with price elasticity, avoiding overly aggressive pricing that could lead to empty seats. |

## 3. Design Choices & Trade-offs

Several key decisions were made during the modeling process:

* **Model Selection**:
    * **Choice**: `GradientBoostingRegressor`.
    * **Alternatives Considered**: Linear models were considered but were not able to capture the complex, non-linear relationships revealed in the EDA. Deep learning models were also an option but would have required more data and a more complex MLOps pipeline.
    * **Trade-off**: Gradient Boosting offered a good balance of performance and interpretability without the high overhead of more complex models.

* **Feature Engineering**:
    * **Choice**: A combination of `StandardScaler` for numerical features and `OneHotEncoder` for categorical features was used.
    * **Rationale**: The EDA showed that numerical features had vastly different scales, justifying the need for normalization. The strong relationship between categorical features like `zone` and `opponent_tier` and the ticket price made one-hot encoding a natural choice.

* **Optimization Approach**:
    * **Choice**: Grid search within the Optimization Engine.
    * **Alternatives Considered**: More advanced optimization algorithms like Bayesian optimization could have been used.
    * **Trade-off**: While a grid search can be computationally intensive, it is exhaustive and guarantees finding the best price within the specified range. Given the business context, this reliability was prioritized over computational speed.

## 4. Potential Improvements & Future Work

While the current model is highly effective, there are several avenues for future improvement:

* **Real-time Feature Integration**: Currently, the models run on a daily batch schedule. Integrating a real-time feature store would allow for more responsive pricing based on intra-day changes in demand signals.
* **Personalized Pricing**: The model could be extended to offer personalized prices based on user segments (e.g., season ticket holders, first-time buyers). This would require more granular data but could lead to further revenue uplift.
* **Incorporate Causal Inference**: Use causal inference techniques to better distinguish the impact of price changes from other confounding factors (e.g., a star player's injury).
* **Advanced Optimization**: Explore more sophisticated optimization algorithms that can handle more complex business constraints, such as ensuring a certain level of occupancy in specific zones to improve the stadium atmosphere.

# 📖 Data Dictionary
> This document provides a detailed description of each feature in the synthetic dataset (`synthetic_match_data.csv`), which is used to model and predict football match ticket prices.

## Overview

Each row in the dataset represents the state of a specific seating **zone** for a single **match** at a particular point in time, defined by the `days_until_match`. The primary goal is to predict the `ticket_price` based on the other features.

## Identifiers & Categorical Features

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `match_id` | Integer | A unique identifier for each football match. |
| `zone` | String | The name of the seating zone in the stadium (e.g., 'Gol Nord', 'Lateral', 'VIP'). |
| `opponent_tier` | String | A categorical rating of the opponent's quality and appeal (`A++`, `A`, `B`, `C`). Higher tiers signify more attractive matches, influencing demand. |
| `weather_forecast` | String | The predicted weather for the match day ('Sunny', 'Cloudy', 'Rain'). Can influence last-minute purchase decisions. |
| `competing_city_events` | Boolean | `True` if there are other major events (concerts, festivals) in the city on the same day, which could reduce local demand. `False` otherwise. |


## Time-Based & Demand Signals

These features capture the dynamics of demand over time and external market interest.

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `days_until_match` | Integer | The number of days remaining before the match. A key feature for time-series analysis, as demand typically increases as the match date approaches. |
| `flights_to_barcelona_index`| Integer | A synthetic index (scaled 20-100) representing the volume of inbound flights to the city. This serves as a proxy for tourist demand. |
| `google_trends_index` | Integer | A synthetic index (scaled 20-100) representing public search interest for the match on Google. A proxy for general public interest and hype. |
| `internal_search_trends`| Integer | A synthetic count of searches for the match on the club's own website or app. A direct signal of purchase intent from the user base. |
| `web_visits` | Integer | A synthetic count of visits to the ticketing section of the club's official website. A measure of online traffic and interest. |
| `web_conversion_rate` | Float | The synthetic conversion rate on the website (ticket purchases / visits). A measure of how effectively web traffic is converting into sales. |
| `social_media_sentiment`| Float | A synthetic score representing the overall public sentiment (e.g., from -1.0 for very negative to +1.0 for very positive) about the match on social media platforms. |


## Sales, Availability & Pricing

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **`zone_historical_sales`** | **Integer** | **(Target Variable)** The historical number of tickets sold for a similar match in that zone. This is the *primary target variable* for the demand forecast model. |
| `zone_seats_availability` | Integer | The absolute number of seats still available for purchase in that zone. |
| `ticket_availability_pct` | Float | The percentage of total seats in the zone that are still available. |
| `competitor_avg_price` | Float | The average ticket price for a comparable entertainment event (e.g., another major football match, a concert) on the same day. Represents the competitive landscape. |
| `ticket_price` | Float | The price of the ticket. This is a *key input* feature for the demand model and the *final output* of the optimization engine. |

</br>

<p align="center">🌐 © 2025 t.r.</p>

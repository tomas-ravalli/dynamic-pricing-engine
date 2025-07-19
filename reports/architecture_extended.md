# **Architecture –Extended**

> This document outlines the architecture of the Dynamic Pricing Engine system. The system is designed as a three-layer structure that ingests data, processes it through a core analytics engine, and integrates with user-facing systems to manage and apply pricing decisions.

The general workflow is as follows:
1.  **Data Sources** are collected and fed into the central engine.
2.  The **Dynamic Pricing Engine** uses machine learning models and business rules to generate a price recommendation.
3.  The pricing team uses the **UI & Integration** layer to review, simulate, and approve the price, which is then updated in the live ticketing system.

<p align="left">
  <img src="../assets/dp-ll.png" alt="Low-level Project Diagram" width="950">
</p>

## **Component Descriptions**

### 1. Data Sources

| Component | Description |
| :--- | :--- |
| **Ticket Sales & Availability** | Historical and real-time data on ticket inventory, sales velocity, and transactions. |
| **Competitors Pricing** | Scraped pricing data from secondary markets (e.g., Viagogo, Stubhub) for competitive analysis. |
| **Web/App Analytics** | Data on user behavior from the official website and app, including page visits, clicks, and conversion funnels. |
| **Matches, Competitions & Channels** | Foundational information about each match, including opponent, date, competition type, and sales channel. |

### 2. Dynamic Pricing Engine

| Component | Description |
| :--- | :--- |
| **Data Ingestion & Centralization** | The entry point that gathers data from all sources and consolidates it into a unified data store for processing. |
| **ML & Analytics Core** | The central "brain" where data is processed, features are engineered, and the machine learning models are trained and executed. |
| **Business Constraints** | A module containing hardcoded business logic (e.g., price floors/caps) that provides rules directly to the Decision Engine to ensure recommendations are compliant with club strategy. |
| **Decision Module** | A container for the core predictive models that feed the optimization engine. |
| ┣ **Price Elasticity Model** | A model that calculates how sensitive ticket demand is to changes in price. |
| ┣ **Demand Forecast Model** | A model that predicts the expected volume of ticket sales at various price points. |
| ┣ **Match Clustering** | An algorithm that groups similar matches together (e.g., "Weekday league match vs. mid-tier team") to improve model accuracy. |
| **Decision Engine: Optimization & Simulation** | Takes the outputs from the Decision Module and Business Constraints to find the revenue-maximizing price. It also allows for "what-if" simulations. |
| **Anomaly Warnings** | An alerting system that flags unusual sales patterns or pricing recommendations that deviate from norms. |
| **Impact Simulation** | A feature that allows a human user to test a hypothetical price and see the model's prediction for its impact on sales and revenue. |
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

<p align="center">© 🌐 2025 t.r.</p>

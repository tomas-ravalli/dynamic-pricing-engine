# **Architecture Overview**

This document outlines the architecture of the Dynamic Pricing Engine system. The system is designed as a three-layer structure that ingests data, processes it through a core analytics engine, and integrates with user-facing systems to manage and apply pricing decisions.

The general workflow is as follows:
1.  **Data Sources** are collected and fed into the central engine.
2.  The **Dynamic Pricing Engine** uses machine learning models and business rules to generate a price recommendation.
3.  The pricing team uses the **UI & Integration** layer to review, simulate, and approve the price, which is then updated in the live ticketing system.

## **Diagram**

<p align="left">
  <img src="./assets/fcb-dp-architecture.svg" alt="Dynamic Engine Architecture">
</p>

## **Component Descriptions**

| Layer | Component | Description |
| :--- | :--- | :--- |
| *Data Sources* | **Ticket Sales & Availability** | Historical and real-time data on ticket inventory, sales velocity, and transactions. |
| *Data Sources* | **Competitors Pricing** | Scraped pricing data from secondary markets (e.g., Viagogo, Stubhub) for competitive analysis. |
| *Data Sources* | **Web/App Analytics** | Data on user behavior from the official website and app, including page visits, clicks, and conversion funnels. |
| *Data Sources* | **Matches, Competitions & Channels** | Foundational information about each match, including opponent, date, competition type, and sales channel. |
| *Dynamic Pricing Engine* | **Data Ingestion & Centralization** | The entry point that gathers data from all sources and consolidates it into a unified data store for processing. |
| *Dynamic Pricing Engine* | **ML & Analytics Core** | The central "brain" where data is processed, features are engineered, and the machine learning models are trained and executed. |
| *Dynamic Pricing Engine* | **Business Constraints** | A module containing hardcoded business logic, such as price floors/caps, that ensures all price recommendations are compliant with club strategy. |
| *Dynamic Pricing Engine* | **Decision Module** | A container for the core predictive models that feed the optimization engine. |
| *Dynamic Pricing Engine* | ┣ Price Elasticity Model | A model that calculates how sensitive ticket demand is to changes in price. |
| *Dynamic Pricing Engine* | ┣ Demand Forecast Model | A model that predicts the expected volume of ticket sales at various price points. |
| *Dynamic Pricing Engine* | ┣ Match Clustering | An algorithm that groups similar matches together (e.g., "Weekday league match vs. mid-tier team") to improve model accuracy. |
| *Dynamic Pricing Engine* | **Optimization & Simulation Engine** | Takes the outputs from the Decision Module and Business Constraints to find the revenue-maximizing price. It also allows for "what-if" simulations. |
| *Dynamic Pricing Engine* | **Anomaly Warnings** | An alerting system that flags unusual sales patterns or pricing recommendations that deviate from norms. |
| *Dynamic Pricing Engine* | **Impact Simulation** | A feature that allows a human user to test a hypothetical price and see the model's prediction for its impact on sales and revenue. |
| *Dynamic Pricing Engine* | **Price Variation Proposal** | The final output of the engine: a concrete price recommendation for a given seat or section. |
| *UI & Integration* | **User Control Panel** | The dashboard used by the pricing team to view price proposals, run impact simulations, and approve or reject changes, enabling Human-in-the-Loop (HITL) control. |
| *UI & Integration* | **REST API** | The communication layer that allows the User Control Panel to send approved price change commands to the live ticketing system. |
| *UI & Integration* | **Price Drop Logic** | An automated module that can trigger price change events based on predefined rules, such as slow ticket sales. |
| *UI & Integration* | **Ticketing Purchase System** | The club's main backend system that processes transactions and manages ticket inventory. It receives price update commands from the API. |
| *UI & Integration* | **Fan Engagement** | The final layer where fans interact with the system's output. |
| *UI & Integration* | ┗ Ticketing Purchase UI | The public-facing website or application screen where fans see the dynamically adjusted prices and make their purchases. |

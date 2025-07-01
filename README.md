**FlightAssistant** is an intelligent conversational assistant that predicts and compares flight ticket prices based on user inputs such as airline, source and destination cities, travel dates, class type, number of stops, and more. It integrates machine learning with a conversational AI interface, making it intuitive, interactive, and highly user-friendly.

With just a few details from the user, the assistant instantly provides the **three lowest-priced flight options**, helping travelers make informed decisions.

---

## How It Works

FlightAssistant is powered by advanced machine learning techniques to accurately predict flight prices. Multiple models were tested, and after evaluating their performance, a **Stacking Regressor** was selected for deployment due to its superior performance by combining the strengths of multiple base models.

### Model Comparisons

| Model                  | MAPE       | MSE             | R²          | MAE          |
| ---------------------- | ---------- | --------------- | ----------- | ------------ |
| Linear Regression      | 43.67%     | 49,118,095      | 0.905       | ₹4,627       |
| Ridge Regression       | 43.67%     | 49,118,099      | 0.905       | ₹4,627       |
| Decision Tree          | 7.32%      | 12,043,906      | 0.977       | ₹1,158       |
| Random Forest          | 6.91%      | 7,490,065       | 0.985       | ₹1,072       |
| **Stacking Regressor** | **6.5%**   | **7,000,000**   | **0.987**   | **₹1,000**   |

> The **Stacking Regressor** outperformed individual models by combining them into a single, more powerful ensemble. It leverages multiple base learners ( Random Forest, Gradient Boosting, and Ridge) and a meta-learner to generate highly accurate predictions.

---

## Conversational AI

To enhance the user experience, FlightAssistant includes an intelligent conversation flow powered by:

* **LangGraph** for stateful conversation management
* **Llama3 LLM** via **ChatOllama** for natural language understanding and generation
* **LangChain** for tool and prompt orchestration

Users interact with the assistant through a conversational interface, asking for price comparisons or specific flight quotes.

---

##  Built With

* **Scikit-learn** – for model training and ensemble learning
* **LangGraph** – for building structured conversational flows
* **LangChain** – to integrate tools and prompt templates
* **ChatOllama** – for running Llama3 locally
* **Streamlit** – for building an interactive frontend interface

---

## Key Features

* Predicts flight prices with high accuracy
* Intelligent conversation using LLMs
* Supports inputs like airline, stops, source/destination cities, and travel dates
* Returns the **top 3 cheapest flight options**
* Backend powered by an ensemble ML model
* User-friendly frontend with Streamlit

---

##  Demo

https://github.com/user-attachments/assets/bdfe06c2-1541-4ca4-9923-5efca972e3df


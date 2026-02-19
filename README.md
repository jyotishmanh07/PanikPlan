# PanikPlan

**PanikPlan** is a self-evolving restaurant recommendation system designed to plan dinners with friends. The end goal would be to demonstrate a professional MLOps lifecycle—from automated data collection and experiment tracking to real-time agentic orchestration and human-in-the-loop feedback.


---

## Getting Started

This project uses **Poetry** for dependency management to ensure a consistent, isolated virtual environment.

### 1. Environment Setup
First, install the project dependencies and activate the virtual environment:

```bash
# Install dependencies from pyproject.toml
poetry install

# Activate the virtual environment shell
poetry shell
```
### 2. Configure API Keys
The system requires a **Tavily API Key** for real-time restaurant searching and automated data mining.

* **Streamlit**: Add your key to `.streamlit/secrets.toml`.
* **Collector/Orchestrator**: Ensure the key is correctly set in `src/stages/data_collector.py` and `agents/orchestrator.py`.

---

### Execution Sequence
To properly initialize and test, run the scripts in the following order. Running them out of sequence may result in a `ModelNotFoundError` if the required artifacts do not yet exist.



#### **Step 1: Data Collection**
Get fresh training data from the web using the Tavily API. This script searches for top-rated restaurants in major cities to create a "ground truth" dataset. Change cities to your liking

```bash
python -m src.stages.data_collector
```
#### **Step 2: Model Training & Evolution**
Merge the original historical records from `TA_restaurants_curated.csv` with your fresh API data to train a new version of the **DineSync** model using MLflow.

```bash
python -m src.stages.train
```

#### **Step 3: Launch the Planner**
Start the interactive Streamlit app to search for restaurants and provide feedback to the model. The app loads the model artifact directly from your MLflow directory.

```bash
streamlit run streamlit_app.py
```

#### **Experiment Tracking**
MLflow to track every training run, logging parameters, RMSE metrics, and model artifacts.
To view your model's evolution and performance history:
Bash
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your browser

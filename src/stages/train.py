import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.utils.helpers import load_and_clean_data
from src.stages.data_collector import collect_training_data_from_api
import os

def run_evolving_train():
    mlflow.set_experiment("PanikPlan_Evolving_Brain")

    api_csv = collect_training_data_from_api()

    X_orig, y_orig = load_and_clean_data("data/raw/TA_restaurants_curated.csv")

    df_api = pd.read_csv(api_csv)

    X_api = df_api[['price_score', 'number_of_reviews', 'is_unknown']]
    y_api = df_api['rating']
    
    X_final = pd.concat([X_orig, X_api]).fillna(0)
    y_final = pd.concat([y_orig, y_api])

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2)

    with mlflow.start_run(run_name="Evolving_RandomForest"):
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, "randomforest_model")
        print(f"Brain updated! Now trained on {len(X_final)} rows.")

if __name__ == "__main__":
    run_evolving_train()
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from src.utils.helpers import load_and_clean_data

def run_train():
    mlflow.set_experiment("Restaurant_Satisfaction_Comparison")
    
    X, y = load_and_clean_data("data/raw/TA_restaurants_curated.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of models to try
    models = [
        {"name": "XGBoost", "model": XGBRegressor(n_estimators=100, max_depth=5)},
        {"name": "RandomForest", "model": RandomForestRegressor(n_estimators=100, max_depth=8)}
    ]

    for m_info in models:
        with mlflow.start_run(run_name=m_info["name"]):
            model = m_info["model"]
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, preds)
            
            # Log params, metrics, and signature
            mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
            mlflow.log_metric("rmse", rmse)
            
            signature = infer_signature(X_test, preds)
            mlflow.sklearn.log_model(model, m_info["name"].lower() + "_model", signature=signature)
            
            print(f"{m_info['name']} trained with RMSE: {rmse:.4f}")

if __name__ == "__main__":
    run_train()
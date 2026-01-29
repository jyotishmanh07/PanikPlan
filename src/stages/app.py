import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN_ID = "0996947b65954485a4f9b0fa6df2a92d"
EXP_ID = "584193143075774228"

model_uri = os.path.join(base_path, "mlruns", EXP_ID, RUN_ID, "artifacts", "randomforest_model")

# Load model
model = mlflow.sklearn.load_model(model_uri)
app = FastAPI(title="PanikPlan DineSync API")



class SurveyData(BaseModel):
    price_score: int
    number_of_reviews: int
    is_european: int = 0
    is_italian: int = 0
    is_french: int = 0
    is_mediterranean: int = 0
    is_vegetarian_friendly: int = 0
    is_unknown: int = 0

@app.post("/predict")
def predict(data: SurveyData):

    input_df = pd.DataFrame([data.model_dump()])

    expected_features = [
        'price_score', 
        'number_of_reviews', 
        'is_european', 
        'is_italian', 
        'is_french', 
        'is_mediterranean', 
        'is_vegetarian_friendly', 
        'is_unknown'
    ]
    
    # Reindex reorders columns and fills missing ones with 0
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    
    try:
        prediction = model.predict(input_df)
        return {
            "predicted_rating": round(float(prediction[0]), 2),
            "status": "success",
            "model_id": RUN_ID
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
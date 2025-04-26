from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from typing import Any, Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your Shopify domain(s) instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/kwikathon_model_v1.pkl')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), '../model/label_encoder.joblib')
TARGET_ENCODER_PATH = os.path.join(os.path.dirname(__file__), '../model/target_encoder.joblib')

class PredictRequest(BaseModel):
    api_data_1: Dict[str, Any]
    api_data_2: Dict[str, Any]

class PredictResponse(BaseModel):
    prediction: Any

model = None
label_encoders = None
target_le = None

def load_artifacts():
    global model, label_encoders, target_le
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    if label_encoders is None:
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_PATH}")
        label_encoders = joblib.load(LABEL_ENCODER_PATH)
    if target_le is None:
        if not os.path.exists(TARGET_ENCODER_PATH):
            raise FileNotFoundError(f"Target encoder file not found at {TARGET_ENCODER_PATH}")
        target_le = joblib.load(TARGET_ENCODER_PATH)
    return model, label_encoders, target_le

def get_kwikintent_signal(user_data, mdl, label_encs, target_encoder):
    user_data = user_data.copy()
    user_data.pop('time_spent_on_pdp', None)
    user_data = {k.replace(' ', '_').lower(): v for k, v in user_data.items()}
    df = pd.DataFrame([user_data])
    feature_list = mdl.get_booster().feature_names

    # Fill missing features with default values (e.g., 0 or "")
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0  # or "" for categorical, or another sensible default

    df = df[feature_list]
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = label_encs[col]
        # Handle unseen labels by mapping them to a default value (e.g., first class)
        df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
    pred_class = mdl.predict(df)[0]
    pred_label = target_encoder.inverse_transform([pred_class])[0]
    return pred_label

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        mdl, label_encs, target_encoder = load_artifacts()
        # Merge the two dicts
        merged_data = request.api_data_1.copy()
        merged_data.update(request.api_data_2)
        pred_label = get_kwikintent_signal(merged_data, mdl, label_encs, target_encoder)
        return PredictResponse(prediction=pred_label)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

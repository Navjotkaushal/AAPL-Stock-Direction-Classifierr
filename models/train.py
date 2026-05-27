import joblib 
import os
from pathlib import Path

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import RANDOM_STATE 

def build_models() -> dict:
    
    rf = Pipeline(steps=[
        ("scaling", StandardScaler()),
        ("clf", RandomForestClassifier(
            
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])
    
    xgb = Pipeline(steps=[
        ("scaling", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators = 300,
            max_depth = 4,
            learning_rate = 0.05,
            subsample = 0.8,
            colsample_bytree = 0.8,
            eval_metric = "logloss",
            random_state = RANDOM_STATE
        )),
    ])
    
    return {"Random Forest": rf, "XGBoost": xgb}


def train_all(models: dict, X_train, y_train) -> dict:
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
    return models 

def save_models(models, path = "saved_models/"):
    os.makedirs(path, exist_ok = True)
    for name, model in models.items():
        joblib.dump(model, f"{path}{name}.pkl")
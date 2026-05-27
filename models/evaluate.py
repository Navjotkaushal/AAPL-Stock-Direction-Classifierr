import sys
import os 
from pathlib import Path 

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

from config import FEATURE_COLS 

def evaluate_all(models: dict, X_test, y_test) -> dict:
    """
    Run evaluation for every fitted model
    and returns the results in dict format
    """
    
    results = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        cm = confusion_matrix(y_test, preds)
        
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {accuracy_score(y_test, preds):.4f}")
        print(f"  ROC-AUC  : {roc_auc_score(y_test, proba[:, 1]):.4f}")
        print(f"\n{classification_report(y_test, preds, target_names=['Down', 'Up'])}")
        
        results[name] = (preds, proba, cm)
        
    return results 

def plot_results(results: dict, models: dict):
    
    """
    3-panel chart per model:
    -confusiion matrix
    -predicted probability distribution 
    -top 15 feature importance 
    """
    
    n = len(results)
    fig = plt.figure(figsize=(16, 5* n))
    fig.suptitle(
        "AAPL - Price Direction Classification",
        fontsize = 16,
        fontweight = "bold"
    )
    
    for i , (name, (preds, proba, cm)) in enumerate(results.items()):
        gs = gridspec.GridSpec(n, 3, figure = fig)
        
        # Confusion matrix 
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(cm, cmap="Blues")
        ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
        ax1.set_xticklabels(["Down","Up"]); ax1.set_yticklabels(["Down","Up"])
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
        ax1.set_title(f"{name}\nConfusion Matrix")
        for r in range(2):
            for c in range(2):
                ax1.text(c, r, cm[r, c], ha = "center", va = "center",
                         color = "white" if cm[r, c] > cm.max() / 2 else "black",
                         fontsize = 14)
                
        
        # Probability distribution 
        
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.hist(proba[:, 1], bins = 30, edgecolor = "white", color = "#4C72B0")
        ax2.axvline(0.5, color = "red", linestyle = "--", label = "threshold = 0.5")
        ax2.set_xlabel("Predicted Probability (Up)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{name} \n Prediction Probability")
        ax2.legend()
        
        
        # Feature Importance 
        
        ax3 = fig.add_subplot(gs[i, 2])
        clf = models[name].named_steps["clf"]
        imps = clf.feature_importances_
        idx = np.argsort(imps)[-15:]
        ax3.barh(np.array(FEATURE_COLS)[idx], imps[idx], color = "#55A868")
        ax3.set_xlabel("Importance")
        ax3.set_title(f"{name}\n Top 15 Features")
        
        
    plt.tight_layout()
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)  # creates folder if it doesn't exist
    save_path = out_dir / "model_results.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved -> {save_path}")
    
    
def predict_tomorrow(models: dict, df):
    """Run the latest available row through every fitted model"""
    latest = df[FEATURE_COLS].dropna().iloc[[-1]]
    
    print(f"\n{'=' * 50}")
    print(f"Tomorrow's prediction (based on {df.index[-1].date()})")
    
    for name, model in models.items():
        prob = model.predict_proba(latest)[0,1]
        direction = "⬆  UP" if prob >= 0.5 else "⬇  DOWN"
        print(f"  {name:20s}  {direction}   (confidence: {prob:.2%})")
        
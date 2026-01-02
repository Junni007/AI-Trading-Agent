import numpy as np
import pandas as pd
import joblib
import logging
from src.data_loader import MVPDataLoader
from src.ticker_utils import get_extended_tickers

# Sklearn Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainClassic")

def flatten_data(X):
    # X shape: (N, 50, 7) -> Take last time step: (N, 7)
    return X[:, -1, :]

def train_classic():
    logger.info("Loading Data (Tickers)...")
    tickers = get_extended_tickers(limit=500000) # Scale up to Max
    loader = MVPDataLoader(tickers=tickers, window_size=50) 
    splits = loader.get_data_splits()
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    
    X_train_flat = flatten_data(X_train)
    X_val_flat = flatten_data(X_val)
    
    logger.info(f"Data Shape: {X_train_flat.shape}")
    
    # Define Model Suite
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=50), # High neighbors for noisy data
        "Gaussian NB": GaussianNB(),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.01, dual=False, random_state=42)), # Calibrated for probs
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*50)
    print("🚀 STARTING MASSIVE MODEL SHOOTOUT 🚀")
    print("="*50)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train_flat, y_train)
            preds = model.predict(X_val_flat)
            acc = accuracy_score(y_val, preds)
            results[name] = acc
            logger.info(f"--> {name}: {acc:.4f}")
            
            # Save Model
            filename = f"{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, filename)
            
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results[name] = 0.0

    # Sort Results
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*50)
    print("🏆 FINAL LEADERBOARD 🏆")
    print("="*50)
    for rank, (name, acc) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:<25} : {acc:.4%}")
    print("="*50 + "\n")
    
    # Save Best Predictions for Stacking
    best_name, best_acc = sorted_results[0]
    best_model = models[best_name]
    logger.info(f"Saving predictions from champion: {best_name}")
    
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_val_flat)[:, 1]
        np.save("val_probs_best_classic.npy", probs)
    else:
        # Fallback for SVM if not calibrated (but we used CalibratedClassifierCV)
        preds = best_model.predict(X_val_flat)
        np.save("val_probs_best_classic.npy", preds)

    np.save("y_val.npy", y_val)
    logger.info("Shootout Complete.")

if __name__ == "__main__":
    train_classic()

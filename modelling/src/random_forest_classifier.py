import os
import pickle
import json
import numpy as np
import cudf 
import cuml
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# --- Config ---
ARTIFACTS_DIR = '../../pre_processing/artifacts/'
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'X_train.parquet')
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'X_test.parquet')
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, 'y_train.parquet')
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, 'y_test.parquet')

SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'artifacts/scaler.pkl')

SAVE_PATH = os.path.join("../artifacts")

# --------------------

# Functions to handle user inputs
def get_int_input(prompt_text, default = None):
    val_str = input(prompt_text).strip()
    if not val_str: # If user press Enter (Null)
        return default
    
    try:
        return int(val_str)
    except ValueError:
        print(f"Invalid input (not an integer). Using default: {default}")
        return default

def get_model_params():
    print("--- Define parameters for RandomForestClassifier: ---")
    
    # 1. Model name
    model_name = ""
    while not model_name:
        model_name = input("Enter a unique name for this model run (e.g., 'rf_v1_depth5'): ").strip()
        if not model_name:
            print("Error: Model name is required to save artifacts.")

    # 2. cuML RandomForestClassifier parameters
    n_estimators = get_int_input("Parameter 'n_estimators' (int, num trees) [Default: 100]: ", default = 100)
    max_depth = get_int_input("Parameter 'max_depth' (int) [Default: 16]: ", default = 16)
    min_samples_leaf = get_int_input("Parameter 'min_samples_leaf' (int) [Default: 1]: ", default = 1)
    min_samples_split = get_int_input("Parameter 'min_samples_split' (int) [Default: 2]: ", default = 2)
    random_state = get_int_input("Parameter 'random_state' (int) [Default: 42]: ", default = 42)

    # 3. Model params dict
    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "random_state": random_state
    }
    
    print("\n--- Configuration Selected ---")
    print(f"Model Name: {model_name}")
    print(f"Parameters: {model_params}\n")
    
    return model_name, model_params

def train_model():
    
    # --- 1. Load Data ---
    print("Loading data directly into GPU memory (VRAM)...")
    X_train = cudf.read_parquet(X_TRAIN_PATH)
    X_test = cudf.read_parquet(X_TEST_PATH)
    y_train = cudf.read_parquet(Y_TRAIN_PATH)
    y_test = cudf.read_parquet(Y_TEST_PATH)

    print("------------------------------------------\n")
    print(f"X Train shape (on GPU): {X_train.shape}")
    print(f"y Train shape (on GPU): {y_train.shape}")
    print("------------------------------------------\n")

    # --- 2. Model Configuration ---
    model_name, params = get_model_params()
    MODEL_PATH = os.path.join(SAVE_PATH, f"{model_name}.pkl")
    METRICS_PATH = os.path.join(SAVE_PATH, f"{model_name}_metrics.json")

    # --- 3. Model Training ---
    print("Initializing cuML RandomForestRegressor model (GPU)...")
    model = RandomForestClassifier(**params)

    print("Fitting model on GPU...")
    model.fit(X_train, y_train)

    print("------------------------------------------\n")
    print("Model fitting complete.")
    print(f"Params: {model.get_params()}")
    print("------------------------------------------\n")

    # --- 4. Prediction & Evaluation ---
    print("Predicting on test set (GPU)...")
    y_pred = model.predict(X_test)

    print("Calculating final metrics on test set...")
    y_test_cpu = y_test['Churn'].to_numpy()
    y_pred_cpu = y_pred.to_numpy()

    class_report = classification_report(y_test_cpu, y_pred_cpu)
    class_report_dict = classification_report(y_test_cpu, y_pred_cpu, output_dict=True)

    accuracy = accuracy_score(y_test_cpu, y_pred_cpu)

    cm_numpy = confusion_matrix(y_test_cpu, y_pred_cpu)
    cm_list = cm_numpy.tolist()

    metrics = {
        "test_metrics": {
            "accuracy": round(float(accuracy), 4),
            "confusion_matrix": cm_list,
            "classification_report": class_report_dict
        }
    }

    # --- 5. Print Metrics ---
    print(f"\n--- Model Metrics: {model_name} (GPU Accelerated) ---")
    print(f"Accuracy: {metrics['test_metrics']['accuracy']}")
    print(f"Confusion Matrix:\n{np.array(cm_list)}")
    print("\n--- Classification Report ---")
    print(class_report)
    print("------------------------------------------\n")

    # --- 6. Cross-Validation --- 
    print("\nCross-Validation")
    X_train_cpu = X_train.to_numpy()
    y_train_cpu = y_train['Churn'].to_numpy()
    cv_scores = cross_val_score(model, X_train_cpu, y_train_cpu, cv = 5)
    print(f"Cross_validation scores: {cv_scores}")

    # These results provide a more robust view of the model's performance, as cross-validation 
    # assesses the model's ability to generalize to new data. The variation in accuracy scores 
    # across different folds indicates that the model may behave inconsistently across different 
    # subsets of the data. This may be due to data characteristics, such as class imbalance, or the 
    # need for finer tuning of the model's hyperparameters.

    print(f"Saving model artifact to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Saving metrics to {METRICS_PATH}...")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Training script complete.")

if __name__ == "__main__":
    train_model()
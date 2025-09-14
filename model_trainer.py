from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


def train_models(X_train, y_train, X_test, y_test, best_params, log_callback):
    """Trains a suite of models and returns their results."""
    log_callback("--- Starting Model Training ---\n")
    if best_params:
        log_callback("Using best parameters found during tuning.\n\n")
    else:
        log_callback("Using default model parameters.\n\n")

    # Ensure SVM always has probability=True for confidence scores
    svm_params = {'probability': True, 'random_state': 42}
    if "SVM" in best_params:
        svm_params.update(best_params["SVM"])

    models_to_train = {
        "Random Forest": RandomForestClassifier(**best_params.get("Random Forest", {'random_state': 42})),
        "SVM": SVC(**svm_params),
        "KNN": KNeighborsClassifier(**best_params.get("KNN", {}))
    }

    trained_models = {}
    model_metrics = {}
    model_reports = {}
    model_predictions = {}

    for name, model in models_to_train.items():
        log_callback(f"--- Training {name} ---\n")
        log_callback(f"Parameters: {model.get_params()}\n")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        report_dict = classification_report(y_test, y_pred, target_names=['bad', 'good'], output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        f1_good = report_dict['good']['f1-score']

        trained_models[name] = model
        model_predictions[name] = y_pred
        model_reports[name] = report_dict
        model_metrics[name] = {'Accuracy': accuracy, 'AUC': auc, 'F1-Score (Good)': f1_good}

        report_str = f"Accuracy: {accuracy:.4f}\nAUC Score: {auc:.4f}\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['bad', 'good'])}\n"
        log_callback(report_str)

    return trained_models, model_metrics, model_reports, model_predictions


def tune_models(X_train_tuned, y_train_tuned, log_callback):
    """Performs hyperparameter tuning for all models."""
    log_callback("--- Starting Hyperparameter Tuning (This may take several minutes) ---\n")

    param_grids = {
        "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
        "SVM": {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        "KNN": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    }

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    best_params = {}
    for name in models:
        log_callback(f"--- Tuning {name} ---\n")
        grid_search = GridSearchCV(models[name], param_grids[name], cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_tuned, y_train_tuned)
        best_params[name] = grid_search.best_params_
        log_callback(f"Best parameters for {name}: {grid_search.best_params_}\n")
        log_callback(f"Best F1-score: {grid_search.best_score_:.4f}\n\n")

    return best_params

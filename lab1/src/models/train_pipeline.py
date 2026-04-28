import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.sklearn
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
from src.models.pipeline import create_nb_pipeline

def load_data():
    df = pd.read_csv('data/raw/dataset.csv')
    return df['text'].fillna(''), df['target']

def train_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    params = {
        "alpha": 0.5,
        "max_features": 15000,
        "ngram_range": "(1,2)",
        "test_size": 0.2,
        "random_state": 42
    }

    mlflow.set_experiment("pipeline-experiment")
    with mlflow.start_run(run_name="NB_Pipeline_best"):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "MultinomialNB_Pipeline")
        mlflow.set_tag("dataset", "20newsgroups-full")

        pipeline = create_nb_pipeline(alpha=0.5, max_features=15000)

        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_std",  round(cv_scores.std(), 4))
        print(f"CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mlflow.log_metrics({
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision":     round(precision_score(y_test, y_pred,
                                   average='weighted'), 4),
            "recall":        round(recall_score(y_test, y_pred,
                                   average='weighted'), 4),
            "f1_score":      round(f1_score(y_test, y_pred,
                                   average='weighted'), 4),
        })

        mlflow.sklearn.log_model(pipeline, "pipeline")

        os.makedirs('models', exist_ok=True)
        with open('models/pipeline.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        print("Збережено: models/pipeline.pkl")

if __name__ == "__main__":
    train_pipeline()
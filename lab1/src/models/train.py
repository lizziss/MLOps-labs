import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report)

def load_data():
    df = pd.read_csv('data/raw/dataset.csv')
    return df['text'].fillna(''), df['target']

def run_experiment(model, model_name, params,
                   X_train, X_test, y_train, y_test):
    mlflow.set_experiment("newsgroups-classification")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "20newsgroups-full")

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_std",  round(cv_scores.std(), 4))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_metrics({
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision":     round(precision_score(y_test, y_pred,
                                   average='weighted'), 4),
            "recall":        round(recall_score(y_test, y_pred,
                                   average='weighted'), 4),
            "f1_score":      round(f1_score(y_test, y_pred,
                                   average='weighted'), 4),
        })
        print(f"\n=== {model_name} ===")
        print(classification_report(y_test, y_pred))
        return model

def main():
    X, y = load_data()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)

    # Multinomial Naive Bayes — 3 експерименти
    for alpha in [0.1, 0.5, 1.0]:
        run_experiment(
            MultinomialNB(alpha=alpha),
            f"MultinomialNB_alpha{alpha}",
            {"alpha": alpha},
            X_train, X_test, y_train, y_test)

    # Logistic Regression — 2 експерименти
    for C in [1.0, 10.0]:
        run_experiment(
            LogisticRegression(C=C, max_iter=1000, random_state=42),
            f"LogisticRegression_C{C}",
            {"C": C, "max_iter": 1000},
            X_train, X_test, y_train, y_test)

    # Linear SVC — 2 експерименти
    for C in [0.5, 1.0]:
        run_experiment(
            LinearSVC(C=C, max_iter=2000, random_state=42),
            f"LinearSVC_C{C}",
            {"C": C, "max_iter": 2000},
            X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
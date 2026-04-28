import pickle
import mlflow.sklearn

def load_model_from_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_model_from_mlflow(run_id, artifact_path="pipeline"):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)

def predict(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    return model.predict(texts)

if __name__ == "__main__":
    model = load_model_from_file('models/pipeline.pkl')
    sample = ["NASA launches new satellite for weather monitoring"]
    print("Prediction:", predict(model, sample))
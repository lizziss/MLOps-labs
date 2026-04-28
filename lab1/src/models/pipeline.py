from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def create_nb_pipeline(alpha=1.0, max_features=10000):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )),
        ('classifier', MultinomialNB(alpha=alpha))
    ])

def create_lr_pipeline(C=1.0, max_features=10000):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            C=C, max_iter=1000, random_state=42))
    ])

def create_svc_pipeline(C=1.0, max_features=10000):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )),
        ('classifier', LinearSVC(
            C=C, max_iter=2000, random_state=42))
    ])

if __name__ == "__main__":
    pipe = create_nb_pipeline()
    print("Pipeline steps:", list(pipe.named_steps.keys()))
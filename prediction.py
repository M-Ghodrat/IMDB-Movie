import pickle
import os
import openai
from dotenv import load_dotenv
import time

load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

PATH = os.getcwd()
PATH = os.path.join(PATH, 'models')

with open(os.path.join(PATH,'BernoulliNB_classifier.pkl'), 'rb') as f:
    BernoulliNB_classifier = pickle.load(f)
with open(os.path.join(PATH,'LogisticRegression_classifier.pkl'), 'rb') as f:    
    LogisticRegression_classifier = pickle.load(f)
with open(os.path.join(PATH,'LGBMClassifier_classifier.pkl'), 'rb') as f:
    LGBMClassifier_classifier = pickle.load(f)

with open(os.path.join(PATH,'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)


def predict(x_test):
    x_test_vectorized = vectorizer.transform([x_test]) 
    predictions_BernoulliNB = BernoulliNB_classifier.predict(x_test_vectorized)
    predictions_LogisticRegression = LogisticRegression_classifier.predict(x_test_vectorized)
    predictions_LGBMClassifier = LGBMClassifier_classifier.predict(x_test_vectorized)
    predictions_ChatGPT = oai(x_test)
    return predictions_BernoulliNB, predictions_LogisticRegression, predictions_LGBMClassifier, predictions_ChatGPT

with open(os.path.join(PATH,'genre.pkl'), 'rb') as f:
    genre_columns = pickle.load(f)

def oai(prompt):
    prompt = "Suggest a list of genre, separated by comma, for this movie plot synopsis: " + prompt
    prompt = prompt[:4096]
    time_start = time.time()
    while time.time() < time_start + 5:
        suggested_tags = []
        try:
            response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0.6,
                    # max_tokens=64
                )
            suggested_tags = response["choices"][0]["text"].strip().split(',')
            break
        except:
            pass
    return suggested_tags

        

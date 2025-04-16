import joblib
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Define the Lemmatizer class (needed for unpickling the vectorizer)

# Function to clean text
def clear_text(data):
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in tqdm(data.posts):
        sentence = sentence.lower()
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)  # Remove links
        sentence = re.sub('[^0-9a-z]', ' ', sentence)  # Remove symbols
        cleaned_text.append(sentence)
    return cleaned_text

# Function to predict MBTI type
def predict_mbti(text):
    class Lemmatizer(object):
        def __init__(self):
            self.lemmatizer = WordNetLemmatizer()
        def __call__(self, sentence):
            return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

    # Load the saved model, vectorizer, and label encoder
    model_xgb = joblib.load('xgb_mbti_model.pkl')
    target_encoder = joblib.load('label_encoder.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    text_data = pd.DataFrame({'posts': [text]})
    text_cleaned = clear_text(text_data)
    text_vectorized = vectorizer.transform(text_cleaned).toarray()
    predicted_class = model_xgb.predict(text_vectorized)
    predicted_label = target_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a text sample: ")
    prediction = predict_mbti(user_input)
    print("Predicted Personality Type:", prediction)

def predict_personality(user_input):
    return predict_mbti(user_input)

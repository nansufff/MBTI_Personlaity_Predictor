import time
import joblib
import re
import tweepy
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request
from predict_module import predict_personality
# Initialize Flask App
app = Flask(__name__)

# Define the Lemmatizer class (needed for unpickling the vectorizer)
class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]

# # Load ML Model & Preprocessing Tools
# model_xgb = joblib.load('xgb_mbti_model.pkl')
# target_encoder = joblib.load('label_encoder.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Twitter API Authentication
BEARER_TOKEN="enter here"
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# MBTI personality descriptions
mbti_descriptions = {
    "INTJ": "The Mastermind – Strategic, logical, and loves planning for the future.",
    "ENTP": "The Debater – Charismatic, energetic, and enjoys intellectual challenges.",
    "INFJ": "The Advocate – Deep thinker, compassionate, and insightful.",
    "ENFP": "The Campaigner – Enthusiastic, creative, and free-spirited.",
    "ISTP": "The Virtuoso – Practical, independent, and highly skilled at troubleshooting.",
    "ESFP": "The Entertainer – Fun-loving, spontaneous, and enjoys being the center of attention.",
    "INTP": "The Logician – Analytical, curious, and loves exploring abstract ideas.",  # Added description
    "ENTJ": "The Commander – Bold, strong-willed, and an efficient leader.",
    "ISFJ": "The Defender – Loyal, practical, and highly caring for others.",
    "ESTJ": "The Executive – Organized, driven, and values traditions.",
    "ISTJ": "The Inspector – Responsible, practical, and highly reliable.",
    "ISFP": "The Adventurer – Artistic, spontaneous, and enjoys living in the moment.",
    "ESTP": "The Entrepreneur – Energetic, action-oriented, and loves excitement.",
    "INFJ": "The Advocate – Thoughtful, insightful, and deeply in tune with emotions.",
    "ENFJ": "The Protagonist – Charismatic, inspiring, and a natural leader.",
    "INFP": "The Mediator – Idealistic, creative, and deeply values personal meaning."
}


import time

def get_recent_tweets(username, max_tweets=30):
    """Fetch recent tweets while handling rate limits."""
    try:
        user = client.get_user(username=username, user_fields=["id"])
        if not user or not user.data:
            print(f"⚠ User '{username}' not found.")
            return [], None
        
        user_id = user.data.id
        
        while True:
            try:
                tweets = client.get_users_tweets(id=user_id, max_results=max_tweets)

                if not tweets or not tweets.data:
                    print(f"⚠ No tweets found for '{username}'.")
                    return [], None

                print(f"\n🔹 Retrieved Tweets for {username}:")
                for tweet in tweets.data:
                    print(f"➡ {tweet.text}")  # Print each tweet line by line
                
                return [tweet.text for tweet in tweets.data], None
            
            except tweepy.errors.TooManyRequests as e:
                reset_time = int(e.response.headers.get("x-rate-limit-reset", time.time() + 60))
                wait_time = reset_time - int(time.time())

                print(f"🚫 Rate limit exceeded! Retrying in {wait_time} seconds...")
                return [], wait_time  # Return wait time instead of sleeping
    
    except tweepy.errors.TweepyException as e:
        print(f"⚠ Error fetching tweets for '{username}': {e}")
        return [], None




# Function to Clean and Lemmatize Text
def clean_text(data):
    """Preprocess tweets: remove links, symbols, and apply lemmatization."""
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in data:
        sentence = sentence.lower()
        sentence = re.sub(r'https?://\S+|www\.\S+', ' ', sentence)  # Remove links
        sentence = re.sub(r'[^0-9a-z]', ' ', sentence)  # Remove symbols
        sentence = " ".join([lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2])
        cleaned_text.append(sentence)
    return cleaned_text

# Function to Predict MBTI Type
def predict_mbti(username):
    tweets, wait_time = get_recent_tweets(username)

    if wait_time:
        return "Rate Limit Exceeded", f"Please wait {wait_time} seconds before retrying.", wait_time

    if not tweets:
        return "Could not retrieve tweets for prediction.", "No tweets found.", None
    
    combined_tweets = " || ".join(tweets)
    print(combined_tweets)
    personality = predict_personality(combined_tweets)
    
    description = mbti_descriptions.get(personality, "A reserved and analytical individual.")
    
    return personality, description, None  # No wait time





# Flask Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("Mbti_Twitter_Flask.html")

@app.route("/predict", methods=["POST"])
def predict_module():
    username = request.form.get("username")
    personality, description, wait_time = predict_mbti(username)
    
    return render_template("Mbti_Twitter_Flask.html", 
                           personality=personality, 
                           description=description, 
                           username=username,
                           wait_time=wait_time)  # Send wait time to frontend



if __name__ == "__main__":
    #app.run(debug=True)
    app.run( port=8080)


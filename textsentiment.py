import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the necessary NLTK data (only the first time)
nltk.download('vader_lexicon')

# Function to analyze the sentiment of a given text
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    # Determine if the sentiment is positive, negative, or neutral
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Main function
if __name__ == "__main__":
    user_input = input("Enter some text to analyze: ")
    result = analyze_sentiment(user_input)
    print(f"The sentiment of the input is: {result}")

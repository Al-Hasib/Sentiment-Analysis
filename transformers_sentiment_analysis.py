from transformers import pipeline

# Initialize the sentiment analysis pipeline
# if model doesn't initialized, the given model will taken by default
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")# Sample data

# Take the user input
print("\nPlease enter the string: ")
data = input()

# Perform sentiment analysis
results = sentiment_pipeline(data)
label = results[0]['label']
score = results[0]['score']*100

print(f"\nThe sentiment of the String is : {label}")
print(f"The Confidence Score is : {score:.2f}%\n")
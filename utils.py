from textblob import TextBlob
# Valence Aware Dictionary for Sentiment Reasoning (VADER) is rule based sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def textblob_score(text):
    # create textBlob string
    text = TextBlob(text)
    # ranges -1 to 1 whether close to -1 is negative and close to 1 is posivite sentiment and neither close to -1 and 1 is neutral
    polarity = text.sentiment.polarity
    if polarity <= -0.05:
        return f"negative sentiment ({polarity*100:.2f}%)"
    elif polarity >-0.05 and polarity <0.05:
        return f"neutral sentiment ({polarity*100:.2f}%)"
    else:
        return f"positive sentiment ({polarity*100:.2f}%)"
    


# create sentiment Intensity Analyzer object
vader_sia = SentimentIntensityAnalyzer()
# calculate the polarity scores of sentiment text which provides the percentage of positivity, negativity, neutral & compound
# compound range -1 to 1 and other ranges 0 to 1
def vader_score(text):
    # provides dictionary with pos, neg, neu & compound key and their values
    scores = vader_sia.polarity_scores(text)
    compound = scores['compound'] * 100

    # compound more than 5% is positive, from -5% to 5% neutral and less than 5% 
    if compound > 10:
        return f"positive sentiment ({compound:.2f}%)"
    elif compound <=10 and compound >=-10:
        return f"neutral sentiment ({compound:.2f}%)"
    else:
        return f"negative sentiment ({compound:.2f}%)"
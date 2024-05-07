# Valence Aware Dictionary for Sentiment Reasoning (VADER) is rule based sentiment analyzer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# calculate the polarity scores of sentiment text which provides the percentage of positivity, negativity, neutral & compound
# compound range -1 to 1 and other ranges 0 to 1
def sentiment_scores(text):
    # create sentiment Intensity Analyzer object
    vader_sia = SentimentIntensityAnalyzer()

    # provides dictionary with pos, neg, neu & compound key and their values
    scores = vader_sia.polarity_scores(text)
    positive = scores['pos'] * 100
    negative = scores['neg'] * 100
    neutral = scores['neu'] * 100
    compound = scores['compound'] * 100
    return scores, positive, neutral, negative, compound

    

if __name__=="__main__":
    # take the text from users
    text = input("\nEnter your sentiment: ")

    # calculate the sentiment scores of the text
    scores, positive, neutral, negative, compound = sentiment_scores(text)

    print(f"\nThe Output : {scores}")
    #print(f"The text is {positive:.2f}% Positive, {negative:.2f}% Negative & {neutral:.2f}% neutral")

    # compound more than 5% is positive, from -5% to 5% neutral and less than 5% 
    if compound > 5:
        print("The text is Positive\n")
    elif compound <=5 and compound >=-5:
        print("The Text is Neutral\n")
    else:
        print("The text is Negative\n")


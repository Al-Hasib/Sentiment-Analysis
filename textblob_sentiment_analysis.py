# It lexical based approach to perform sentiment analysis
# It return polarity and subjectivity where polarity for sentiment measures and subjectivity  determines the text factual information or personal information

from textblob import TextBlob

# take the sentiment from users
text = input("\nEnter your sentiment: ")

# create textBlob string
text = TextBlob(text)

def getSubjectivity(text):
    # it ranges from 0 to 1 whether close to 0 indicates the factual information and close to 1 indicates the personal opinion
    subjectivity =  text.sentiment.subjectivity
    if subjectivity < 0.05:
        return f"factual information ({subjectivity})"
    else:
        return f"personal information ({subjectivity})"

def getPolarity(text):
    # ranges -1 to 1 whether close to -1 is negative and close to 1 is posivite sentiment and neither close to -1 and 1 is neutral
    polarity = text.sentiment.polarity
    if polarity <= -0.05:
        return f"negative sentiment ({polarity})"
    elif polarity >-0.05 and polarity <0.05:
        return f"neutral sentiment ({polarity})"
    else:
        return f"positive sentiment ({polarity})"

print("\nSubjectivity of text: ",getSubjectivity(text))
print("Polarity of text: ", getPolarity(text))
print("\nDone!!!\n")

# issue
# this is the best
# this is not the best
# let's try VADER
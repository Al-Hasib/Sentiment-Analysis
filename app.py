import streamlit as st
from PIL import Image

def main():
    st.title("Sentiment Analysis App")
    # You can load an image from a file path or URL
    image_path = "Sentiment-Analysis.png"
    st.image(image_path, use_column_width=True)
    # Create a text input box for the user to input the string
    user_input = st.text_input("Enter your string:")
    # create textBlob string
    text = TextBlob(user_input)
    subjectivity = ("Subjectivity of text: ",getSubjectivity(text))
    polarity = ("Polarity of text: ", getPolarity(text)) 

    # Display the string entered by the user
    st.write("**Your String:**")
    st.write(subjectivity)
    st.write(polarity)







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


   

if __name__ == "__main__":
    main()

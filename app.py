import streamlit as st
from PIL import Image
from utils import *


def textblobSentiment(user_input):
    textblob_sentiment = textblob_score(user_input)
    textblob_str  = (f"**TextBlob :**      {textblob_sentiment}")
    st.write(textblob_str)

def VaderSentiment(user_input):
    vader_sentiment = vader_score(user_input)
    vader_str = (f"**VADER :**      {vader_sentiment}")
    st.write(vader_str)

def transformersSentiment(user_input):
    label, score = transformers_score(user_input)
    transformers_str = f"**Transformers :** {label} ({score:.2f}%)"
    st.write(transformers_str)

def main():
    st.title("Sentiment Analysis App")
    # You can load an image from a file path or URL
    image_path = "Sentiment-Analysis.png"
    st.image(image_path, use_column_width=True)
    # Create a text input box for the user to input the string
    user_input = st.text_area(label="**Enter your Opinion:**", height= 200)

    # Sample data
    data = ["TextBlob", "Vader", "Transformers", "All"]

    # Filter options
    selected_option = st.sidebar.selectbox("Select an option", data)

    if selected_option=="TextBlob":
        textblobSentiment(user_input)

    elif selected_option=="Vader":
        VaderSentiment(user_input)

    elif selected_option=="Transformers":
        transformersSentiment(user_input)
    
    else:
        textblobSentiment(user_input)
        VaderSentiment(user_input)
        transformersSentiment(user_input)

if __name__ == "__main__":
    main()

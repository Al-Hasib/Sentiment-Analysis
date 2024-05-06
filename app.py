import streamlit as st
from PIL import Image
from utils import *


def textblobSentiment(user_input):
    textblob_sentiment = textblob_score(user_input)
    textblob_str  = (f"**TextBlob Score :** {textblob_sentiment}")
    st.write(textblob_str)

def VaderSentiment(user_input):
    vader_sentiment = vader_score(user_input)
    vader_str = (f"**VADER Score :** {vader_sentiment}")
    st.write(vader_str)

def main():
    st.title("Sentiment Analysis App")
    st.subheader("You typed:")
    # You can load an image from a file path or URL
    image_path = "Sentiment-Analysis.png"
    st.image(image_path, use_column_width=True)
    # Create a text input box for the user to input the string
    user_input = st.text_area(label="Enter your string:", height= 200)

    # Sample data
    data = ["TextBlob", "Vader", "Transformers", "All"]

    # Filter options
    selected_option = st.sidebar.selectbox("Select an option", data)

    if selected_option=="TextBlob":
        textblobSentiment(user_input)

    elif selected_option=="Vader":
        VaderSentiment(user_input)

    elif selected_option=="Transformers":
        st.write("Not added yet")
    
    else:
        textblobSentiment(user_input)
        VaderSentiment(user_input)
        Transformers_str = (f"**Transformer Score :** Not added yet")
        st.write(Transformers_str)

if __name__ == "__main__":
    main()

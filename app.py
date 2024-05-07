import streamlit as st
from PIL import Image
from utils import *


def textblobSentiment(user_input, performance_visualization):
    textblob_sentiment = textblob_score(user_input)
    textblob_str  = (f"**TextBlob :**      {textblob_sentiment}")
    st.write(textblob_str)
    if performance_visualization =='Yes':
        # st.write("**TextBlob**")
        st.subheader("See the Performance of the methodology for a dataset")
        img_path = "data/textblob.png"
        st.markdown(f"**TextBlob Accuracy : 52.36%**")
        st.image(img_path, use_column_width=True)

def VaderSentiment(user_input,performance_visualization):
    vader_sentiment = vader_score(user_input)
    vader_str = (f"**VADER :**      {vader_sentiment}")
    st.write(vader_str)
    if performance_visualization =='Yes':
        # st.write("**VADER**")
        st.subheader("See the Performance of the methodology for a dataset")
        img_path = "data/vader.png"
        st.markdown(f"**VADER Accuracy : 62.69%%**")
        st.image(img_path, use_column_width=True)

def transformersSentiment(user_input, performance_visualization):
    label, score = transformers_score(user_input)
    transformers_str = f"**Transformers :** {label} ({score:.2f}%)"
    st.write(transformers_str)
    if performance_visualization =='Yes':
        # st.write("**Transformers**")
        st.subheader("See the Performance of the methodology for a dataset")
        img_path = "data/transformers.png"
        st.markdown(f"**Transformers Accuracy : 65.89%**")
        st.image(img_path, use_column_width=True)


def main():
    st.title("Sentiment Analysis App")
    # You can load an image from a file path or URL
    image_path = "data/Sentiment-Analysis.png"
    st.image(image_path, use_column_width=True)
    # Create a text input box for the user to input the string
    user_input = st.text_area(label="**Enter your Opinion:**", height= 150)

    # Sample data
    data = ["TextBlob", "Vader", "Transformers", "All"]

    # Filter options
    selected_option = st.sidebar.selectbox("Select an option", data)
    # Create radio buttons for plot selection
    plot_selection = st.sidebar.radio("Performance Visualization", ('Yes','No'))

    if selected_option=="TextBlob":
        textblobSentiment(user_input,plot_selection)

    elif selected_option=="Vader":
        VaderSentiment(user_input,plot_selection)

    elif selected_option=="Transformers":
        transformersSentiment(user_input,plot_selection)
    
    else:
        textblobSentiment(user_input,'No')
        VaderSentiment(user_input,'No')
        transformersSentiment(user_input,'No')
        
        if plot_selection=='Yes':
            st.subheader("See the Performance of the methodology for a dataset")
            st.markdown(f"**TextBlob     Accuracy : 52.36%**")
            st.markdown(f"**VADER        Accuracy : 62.69%**")
            st.markdown(f"**Transformers Accuracy : 65.89%**")

            # Load the three images
            image1 = Image.open("data/textblob.png")
            image2 = Image.open("data/vader.png")
            image3 = Image.open("data/transformers.png")

            # Combine the images into one horizontally
            combined_image = Image.new("RGB", (image1.width + image2.width + image3.width, max(image1.height, image2.height, image3.height)))
            combined_image.paste(image1, (0, 0))
            combined_image.paste(image2, (image1.width, 0))
            combined_image.paste(image3, (image1.width + image2.width, 0))

            # Display the combined image
            st.image(combined_image, caption='Combined Image', use_column_width=True)
    st.subheader("Conclusion")
    st.write("TextBlob and VADER is a rule based Approach but transformers is a pretrained model that is deep learning based approach. \
             We just performed inference from those methodology and measure the performance based on a dataset. The transformers model provides \
             highest performance over rule based approach. Also we can improve the performance of the transformer by fine tuning the model \
             with our own dataset. Then we may get accuracy close to 100%. Thanks!!  ")

if __name__ == "__main__":
    main()

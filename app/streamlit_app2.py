import streamlit as st
import requests
from inference2 import *

# Streamlit app logic
def main():
    # Add an image to the page
    col1, col2, col3 = st.columns([1,2,1])  # Create three columns
    with col2:  # Center the image in the middle column
        st.image("app/image.png", use_column_width=True)

    # Add input for start word
    start_word = st.text_input("Enter a start word for story generation:")

    # Center-align the button
    st.markdown(
        """
        <style>
        .stButton>button {
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("Generate Story"):
        story = generate_text(start_word)
        st.header("Story:")
        st.write(f"{story}")

if __name__ == "__main__":
    main()

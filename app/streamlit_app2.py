import streamlit as st
import requests
from inference2 import *

# Streamlit app logic
def main():
    # Add an image to the page
    st.image("app/image.png", width=600, use_column_width=False)

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

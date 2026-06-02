import streamlit as st
import requests

api_url = "http://127.0.0.1:8000/predict"

st.title('AG NEWS TEXT CLASSIFIER')
st.write("Enter news text (in any language), and the model will determine its category.")

user_input = st.text_area("✍Enter news text here:")

if st.button("Classify"):
    if user_input.strip():
        response = requests.post(api_url, json={"word": user_input})

        if response.status_code == 200:
            data = response.json()

            # Displaying the response from FastAPI like in the screenshot
            st.success(f" **Class: {data['label']}**")
            st.markdown(f"**Translated text:** {data['translated_text']}")
        else:
            st.error("Error connecting to the API.")
    else:
        st.warning("Please enter some text!")
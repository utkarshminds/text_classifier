import streamlit as st
import pickle
import numpy as np
# streamlit_app.py
import hmac



def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()


# Load the pre-trained models and transformer
with open('tfidf_transformer.pkl', 'rb') as tfidf_file:
    tfidf_transformer = pickle.load(tfidf_file)

with open('selector.pkl', 'rb') as selector_file:
    feature_selector = pickle.load(selector_file)

with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Function to classify text
def classify_text(text):
    # Transform the text using the TF-IDF transformer
    tfidf_features = tfidf_transformer.transform([text])
    
    # Select features using the feature selector
    selected_features = feature_selector.transform(tfidf_features)
    
    # Predict using the machine learning model
    prediction = best_model.predict(selected_features)
    
    # Return the label
    return "Religious" if prediction[0] == 0 else "Technology"

# Streamlit app interface
st.title("Text Classification: Religious vs Technology")
st.write("Enter some text below to classify it as *Religious* or *Technology*.")

# Text input from the user
user_text = st.text_area("Enter your text here:", "")

# Classification button
if st.button("Classify"):
    if user_text.strip():
        # Perform classification
        result = classify_text(user_text)
        st.success(f"The entered text is classified as: **{result}**")
    else:
        st.error("Please enter some text to classify.")

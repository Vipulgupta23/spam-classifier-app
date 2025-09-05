import streamlit as st
import pickle
import re
import pandas as pd
import io
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Add these lines to import nltk and os
import nltk
import os

# Add the NLTK download code block here
# Download NLTK data if not already downloaded in the environment
# This is necessary for Streamlit Cloud deployment
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except nltk.downloader.DownloadError:
    nltk.download('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Point NLTK data to a writable directory if needed (sometimes necessary in deployment environments)
# Although Streamlit Cloud usually handles this, it's a potential troubleshooting step
# nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))


# Initialize lemmatizer (assuming NLTK data is already downloaded)
lemmatizer = WordNetLemmatizer()

# Define the cleaning function (should be the same as used during training)
def clean_text_for_prediction(text):
    if not isinstance(text, str):
        return "" # Corrected: Should return an empty string or handle appropriately
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(cleaned_words)

# Load the saved model and vectorizer
# Add error handling for file loading
try:
    with open('tuned_svm_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Please ensure 'tuned_svm_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as the app.py file.")
    model_loaded = False
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    model_loaded = False

# ... (rest of your app.py code) ...


# Define the prediction function
def predict_spam(message):
    if not model_loaded:
        return "Model not loaded", None

    if not isinstance(message, str):
        return "Invalid input", None # Return error for non-string input

    cleaned_message = clean_text_for_prediction(message)
    message_features = loaded_vectorizer.transform([cleaned_message])

    prediction = loaded_model.predict(message_features)
    predicted_label = 'Spam' if prediction[0] == 1 else 'Ham'

    # Confidence score (requires predict_proba, which might not be available for all SVMs)
    confidence = None
    if hasattr(loaded_model, 'predict_proba'):
        # Get the probability of the predicted class
        confidence = loaded_model.predict_proba(message_features)[0][prediction[0]]
    else:
        # For SVM without predict_proba, confidence is not directly available
        # We could potentially use decision_function, but predict_proba is more intuitive for confidence
        pass # Confidence remains None

    return predicted_label, confidence

# --- Streamlit App Layout ---
st.set_page_config(page_title="Spam Classifier App", layout="wide")

st.title("Spam Message Classifier")

st.markdown("""
This application classifies messages as 'Spam' or 'Ham' using a pre-trained SVM model.
You can enter a single message or upload a file for bulk testing.
""")

# --- Single Message Prediction ---
st.header("Test a Single Message")
message_input = st.text_area("Enter your message here:", height=150)

if st.button("Classify Message"):
    if message_input:
        predicted_label, confidence = predict_spam(message_input)
        if predicted_label != "Model not loaded" and predicted_label != "Invalid input":
            st.write(f"**Prediction:** {predicted_label}")
            if confidence is not None:
                st.write(f"**Confidence:** {confidence:.4f}")
            else:
                 st.info("Confidence score not available for this model.")
        elif predicted_label == "Invalid input":
             st.error("Invalid input. Please enter a valid text message.")
    else:
        st.warning("Please enter a message to classify.")

# --- Bulk Testing with File Upload ---
st.header("Bulk Testing (Upload File)")
uploaded_file = st.file_uploader("Upload a CSV or text file (one message per line)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # Read the file
        # Assuming text file with one message per line for simplicity
        # For CSV, you might need to adjust based on column structure
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        messages = stringio.read().splitlines()

        if not messages:
            st.warning("Uploaded file is empty.")
        else:
            st.write(f"Read {len(messages)} messages from the file.")

            results = []
            for i, message in enumerate(messages):
                predicted_label, confidence = predict_spam(message)
                results.append({"Message": message, "Predicted Label": predicted_label, "Confidence": confidence})

            results_df = pd.DataFrame(results)

            st.subheader("Bulk Testing Results")
            st.dataframe(results_df)

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

# --- Model Statistics (Example - Replace with your actual model metrics) ---
st.header("Model Statistics")
st.markdown("""
*   **Model Type:** Tuned Support Vector Machine (SVM)
*   **Vectorizer:** TF-IDF
*   **Preprocessing:** Text Cleaning (lowercase, remove punctuation/stopwords, lemmatization)
*   **Data Imbalance Handling:** SMOTE (applied during training)
""")

# You can add more detailed metrics here if you have them saved or hardcoded
st.subheader("Performance on Test Set (Example Metrics)")
st.write("""
*   Accuracy: 0.9963
*   Precision (Spam): 1.0000
*   Recall (Spam): 0.9925
*   F1-score (Spam): 0.9963
""")
st.info("Note: These are example metrics from the notebook analysis. For a real application, load these from a configuration or results file.")

# --- Instructions ---
st.header("Instructions")
st.markdown("""
1.  **Save the necessary files:** Ensure the `tuned_svm_model.pkl`, `tfidf_vectorizer.pkl`, and this `app.py` file are in the same directory.
2.  **Install libraries:** Make sure you have the required libraries installed (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `nltk`, `imbalanced-learn`, `pickle`). You can install them using pip (see `requirements.txt`).
3.  **Run the app:** Open your terminal or command prompt, navigate to the directory where you saved the files, and run the command: `streamlit run app.py`
4.  **Use the interface:** Enter a message in the text box or upload a file for classification.
""")

# --- Footer ---
st.markdown("---")
st.write("Spam Classifier App")

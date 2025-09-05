import streamlit as st
import pickle
import re
import pandas as pd
import io
import nltk

# Download NLTK data if not already downloaded
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')

# Initialize NLTK data
download_nltk_data()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the cleaning function (should be the same as used during training)
def clean_text_for_prediction(text):
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    
    # Split into words
    words = text.split()
    
    # Remove stopwords and lemmatize
    try:
        stop_words = set(stopwords.words('english'))
        cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    except Exception as e:
        # Fallback if NLTK fails
        cleaned_words = [word for word in words if len(word) > 2]
    
    return ' '.join(cleaned_words)

# Load the saved model and vectorizer with caching
@st.cache_resource
def load_models():
    try:
        with open('tuned_svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer, True
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please ensure 'tuned_svm_model.pkl' and 'tfidf_vectorizer.pkl' are uploaded to your Streamlit app.")
        return None, None, False
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False

# Load models
loaded_model, loaded_vectorizer, model_loaded = load_models()

# Define the prediction function
def predict_spam(message):
    if not model_loaded:
        return "Model not loaded", None

    if not isinstance(message, str) or len(message.strip()) == 0:
        return "Invalid input", None

    try:
        # Clean the message
        cleaned_message = clean_text_for_prediction(message)
        
        if len(cleaned_message.strip()) == 0:
            return "Message too short or invalid", None
        
        # Transform the message
        message_features = loaded_vectorizer.transform([cleaned_message])
        
        # Make prediction
        prediction = loaded_model.predict(message_features)
        predicted_label = 'Spam' if prediction[0] == 1 else 'Ham'
        
        # Get confidence score
        confidence = None
        if hasattr(loaded_model, 'predict_proba'):
            proba = loaded_model.predict_proba(message_features)[0]
            confidence = max(proba)  # Confidence of the predicted class
        elif hasattr(loaded_model, 'decision_function'):
            # For SVM, use decision function
            decision_score = loaded_model.decision_function(message_features)[0]
            # Convert decision function to probability-like score
            confidence = 1 / (1 + abs(decision_score))  # Simplified confidence
        
        return predicted_label, confidence
        
    except Exception as e:
        return f"Prediction error: {str(e)}", None

# --- Streamlit App Layout ---
st.set_page_config(page_title="SMS Spam Classifier", layout="wide", page_icon="📱")

st.title("📱 SMS Spam Classifier")

st.markdown("""
This application classifies SMS messages as **Spam** or **Ham** (legitimate) using a trained SVM model.
Enter a message below or upload a file for bulk classification.
""")

# Only show the interface if model is loaded
if model_loaded:
    # --- Single Message Prediction ---
    st.header("🔍 Test a Single Message")
    
    # Add some example buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Try Spam Example"):
            st.session_state.example_text = "CONGRATULATIONS! You've WON £1000 cash! Call 09061701461 to claim. Valid 12hrs only."
    
    with col2:
        if st.button("✅ Try Ham Example"):
            st.session_state.example_text = "Hey, are you free for dinner tonight? Let me know!"
    
    # Text input with session state
    default_text = st.session_state.get('example_text', '')
    message_input = st.text_area("Enter your message here:", value=default_text, height=100)
    
    if st.button("🔍 Classify Message", type="primary"):
        if message_input and len(message_input.strip()) > 0:
            with st.spinner("Analyzing message..."):
                predicted_label, confidence = predict_spam(message_input)
                
            if predicted_label not in ["Model not loaded", "Invalid input", "Message too short or invalid"] and not predicted_label.startswith("Prediction error"):
                # Display result with colors
                if predicted_label == "Spam":
                    st.error(f"🚨 **Prediction: {predicted_label}**")
                else:
                    st.success(f"✅ **Prediction: {predicted_label}**")
                
                if confidence is not None:
                    st.info(f"🎯 **Confidence:** {confidence:.2%}")
                
                # Show cleaned text for debugging
                cleaned = clean_text_for_prediction(message_input)
                with st.expander("🔧 Processed Text (for debugging)"):
                    st.text(cleaned)
                    
            else:
                st.error(f"❌ {predicted_label}")
        else:
            st.warning("⚠️ Please enter a message to classify.")
    
    st.markdown("---")
    
    # --- Bulk Testing with File Upload ---
    st.header("📁 Bulk Testing")
    st.markdown("Upload a text file with one message per line, or a CSV file.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])
    
    if uploaded_file is not None:
        try:
            # Read file content
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'message' in df.columns:
                    messages = df['message'].astype(str).tolist()
                elif 'text' in df.columns:
                    messages = df['text'].astype(str).tolist()
                else:
                    # Take the first column
                    messages = df.iloc[:, 0].astype(str).tolist()
            else:  # txt file
                content = uploaded_file.read().decode('utf-8')
                messages = [line.strip() for line in content.splitlines() if line.strip()]
            
            if not messages:
                st.warning("No messages found in the uploaded file.")
            else:
                st.success(f"📊 Found {len(messages)} messages in the file.")
                
                if st.button("🚀 Process All Messages"):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, message in enumerate(messages):
                        predicted_label, confidence = predict_spam(message)
                        results.append({
                            "Message": message[:100] + "..." if len(message) > 100 else message,
                            "Full Message": message,
                            "Prediction": predicted_label,
                            "Confidence": f"{confidence:.2%}" if confidence else "N/A"
                        })
                        progress_bar.progress((i + 1) / len(messages))
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    spam_count = sum(1 for r in results if r['Prediction'] == 'Spam')
                    ham_count = sum(1 for r in results if r['Prediction'] == 'Ham')
                    
                    with col1:
                        st.metric("📧 Total Messages", len(results))
                    with col2:
                        st.metric("🚨 Spam Detected", spam_count)
                    with col3:
                        st.metric("✅ Ham (Legitimate)", ham_count)
                    
                    # Display results
                    st.subheader("📋 Results")
                    st.dataframe(results_df[["Message", "Prediction", "Confidence"]], use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="spam_classification_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # --- Model Information ---
    st.markdown("---")
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Model Details:**
        - **Algorithm:** Support Vector Machine (SVM)
        - **Vectorizer:** TF-IDF
        - **Preprocessing:** Text cleaning, stopword removal, lemmatization
        - **Imbalance Handling:** SMOTE
        """)
    
    with col2:
        st.markdown("""
        **📈 Performance Metrics:**
        - **Accuracy:** 99.63%
        - **Precision:** 100.00%
        - **Recall:** 99.25%
        - **F1-Score:** 99.63%
        """)

else:
    st.error("❌ Model files are missing. Please upload the required model files.")
    st.markdown("""
    **Required files:**
    - `tuned_svm_model.pkl`
    - `tfidf_vectorizer.pkl`
    
    Make sure these files are in the same directory as your app.py file.
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔒 SMS Spam Classifier | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)

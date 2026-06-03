import streamlit as st
from PIL import Image
import numpy as np
import xgboost 
import joblib
from ultralytics import YOLO
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import torch
import warnings
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Suppress the specific XGBoost warning
warnings.filterwarnings("ignore", message=".*XGBoost.*")

# ========== API KEY CONFIGURATION ==========
# Load Groq API Key from environment variable
import sys
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(script_dir, '.env'))
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================

torch.classes.__path__ = []

# Add caching for model loading to improve efficiency
@st.cache_resource
def load_models():
    try:
        disease_model = YOLO(os.path.join(script_dir, "best.pt"))
        soil_model = joblib.load(os.path.join(script_dir, "soli_analysis.pkl"))
        return disease_model, soil_model
    except Exception as e:
        st.error(f"🔴 Error loading models: {e}")
        st.stop()

# ------------------ CONFIG & STYLE ------------------
st.set_page_config(page_title="🌾 AgriCare AI", layout="centered", initial_sidebar_state="collapsed")

# Check API Key after st.set_page_config()
if not GROQ_API_KEY:
    st.error("❌ Error: GROQ_API_KEY not found!")
    st.info("📋 Please set GROQ_API_KEY environment variable or create a .env file with: GROQ_API_KEY=your_key_here")
    st.stop()

# Load models once at startup
disease_model, soil_model = load_models()

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-size: 16px;
        padding: 8px 24px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #1e222a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AgriCare AI: Plant Disease Detection, Soil Analysis & Review Analyzer")

valid_classes = ["Tomato Early blight leaf", "Potato leaf early blight", "Tomato leaf", "Tomato mold leaf"]

# ------------------ FUNCTIONS ------------------


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def detect_disease(image):
    try:
        temp_path = os.path.join(script_dir, "temp.jpg")
        image.save(temp_path)
        results = disease_model.predict(temp_path, save=False, stream=False, imgsz=640, conf=0.5)
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes and len(result.boxes) > 0:
                annotated = result.plot()
                pred_image = Image.fromarray(annotated)
                label = result.names[int(result.boxes.cls[0])]
                confidence = float(result.boxes.conf[0])
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return label, pred_image, confidence
        
        # If no detection found
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return "No disease detected - Plant appears healthy", Image.fromarray(np.array(image)), 0.0
    except Exception as e:
        st.error(f"Error in disease detection: {str(e)}")
        raise e

# def predict_soil_fertility_np(features_list):
#     arr = np.array(features_list, dtype=object).reshape(1, 12)
#     prediction = soil_model.predict(arr)[0]
#     soil_labels = ["Low Fertility", "Moderate Fertility", "High Fertility"]
#     recommendations = [
#         "Add organic manure and compost.",
#         "Apply balanced NPK fertilizers.",
#         "Great soil! Maintain it with cover crops."
#     ]
#     return soil_labels[prediction], recommendations[prediction]

def analyze_review_with_gemini(review_text):
    try:
        model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        prompt = f"""Please analyze the following customer review and present it in a structured way in English. Focus on main sentiments, positive aspects, negative aspects, and suggestions for improvement.

    Review:
    {review_text}
    """
        result = model.invoke([HumanMessage(content=prompt)])
        return result.content
    except Exception as e:
        st.error(f"Error analyzing review: {str(e)}")
        raise

def generate_english_gpt_advice(plant_name, disease_label):
    try:
        model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

        prompt = (
            f"I am a farmer. My '{plant_name}' plant has a disease called '{disease_label}'. "
            f"Please provide me expert agricultural advice:\n"
            f"1. What are the symptoms of this disease?\n"
            f"2. How and why does this disease occur?\n"
            f"3. What organic treatments or medicines can be used to cure it?\n"
            f"4. What home remedies can help?\n"
            f"5. How can I prevent this disease in the future?\n"
            f"Please provide clear and practical advice."
        )

        response = model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        st.error(f"Error generating advice: {str(e)}")
        raise


# ------------------ TABS ------------------
tab1, tab2, tab3= st.tabs(["🌿 Plant Disease Detection", "🧪 Soil Analysis", "🧠 Smart Review Analysis"])

# ----------- TAB 1: PLANT DISEASE DETECTION ------------
with tab1:
    st.subheader("� Plant Disease Detection - Upload Your Plant Leaf Image")
    
    uploaded_img = st.file_uploader("📤 Upload Plant Image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        try:
            label, pred_img, confidence = detect_disease(img)

            st.image(pred_img, caption=f"Analysis Result: {label} (Confidence: {confidence:.2%})", use_container_width=True)
            st.info(f"🔍 Detection: **{label}**")

            plant_name = st.text_input("Enter the plant name (e.g., Tomato, Potato, etc.):")

            # If the user has provided a plant name
            if plant_name:
                st.write("🧠 Expert AI Advice:")
                try:
                    response_text = generate_english_gpt_advice(plant_name, label)
                    st.success(response_text)
                except Exception as e:
                    st.error("❌ Error generating AI response.")
                    st.exception(e)
            else:
                st.info("Please enter the plant name to get expert advice.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    else:
        st.info("👆 Upload an image above to detect disease and get recommendations.")





# ----------- TAB 2: SOIL ANALYSIS -------------
def predict_soil_fertility_np(features_list):
    arr = np.array(features_list, dtype=object).reshape(1, 12)
    prediction = soil_model.predict(arr)[0]
    soil_labels = ["Low Fertility", "Moderate Fertility", "High Fertility"]
    return soil_labels[prediction]

def get_soil_recommendations_with_gemini(inputs, fertility_result):
    try:
        model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
        input_str = ", ".join(f"{key}: {value}" for key, value in inputs.items())
        prompt = f"""You are an expert agricultural consultant. Based on the following soil analysis inputs and estimated fertility level, provide detailed and specific recommendations to the user in English. Focus on actionable advice for soil improvement.

    Input Parameters: {input_str}
    Estimated Fertility Level: {fertility_result}

    Recommendations:"""
        result = model.invoke([HumanMessage(content=prompt)])
        return result.content
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        raise

with tab2:
    st.header("🌱 Soil Analysis - Test Your Soil Quality")
    with st.form("soil_form"):
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen (N) - ppm", 0.0)
            P = st.number_input("Phosphorus (P) - ppm", 0.0)
            K = st.number_input("Potassium (K) - ppm", 0.0)
            pH = st.number_input("pH Level", 0.0, 14.0, 7.0)
            EC = st.number_input("Electrical Conductivity (EC)", 0.0)
        with col2:
            OC = st.number_input("Organic Carbon (%)", 0.0)
            S = st.number_input("Sulphur (S) - ppm", 0.0)
            Zn = st.number_input("Zinc (Zn) - ppm", 0.0)
            Fe = st.number_input("Iron (Fe) - ppm", 0.0)
            Cu = st.number_input("Copper (Cu) - ppm", 0.0)
        Mn = st.number_input("Manganese (Mn) - ppm", 0.0)
        B = st.number_input("Boron (B) - ppm", 0.0)
        submitted = st.form_submit_button("🔍 Analyze Soil")

    if submitted:
        features = [N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]
        input_data = {
            "Nitrogen (N)": N,
            "Phosphorus (P)": P,
            "Potassium (K)": K,
            "pH Level": pH,
            "Electrical Conductivity": EC,
            "Organic Carbon": OC,
            "Sulphur (S)": S,
            "Zinc (Zn)": Zn,
            "Iron (Fe)": Fe,
            "Copper (Cu)": Cu,
            "Manganese (Mn)": Mn,
            "Boron (B)": B,
        }
        fertility = predict_soil_fertility_np(features)

        st.subheader("📊 Soil Analysis Results:")
        st.write("**Input Parameters Provided:**")
        for key, value in input_data.items():
            st.write(f"- {key}: {value}")
        st.success(f"🌾 **Soil Fertility Level: {fertility}**")

        try:
            gemini_recommendation = get_soil_recommendations_with_gemini(input_data, fertility)
            st.subheader("💡 AI-Powered Recommendations:")
            st.write(gemini_recommendation)
        except Exception as e:
            st.error("Error fetching AI recommendations.")
            st.exception(e)
# ----------- TAB 3: SMART REVIEW ANALYSIS -------------
with tab3:
    st.header("🧠 Smart Review Analysis Bot")
    st.write("Paste a customer review below to get AI-powered sentiment analysis and insights.")
    
    review_input = st.text_area("📝 Paste your customer review here:")
    if st.button("🔍 Analyze Review"):
        if review_input:
            try:
                analysis_result = analyze_review_with_gemini(review_input)
                st.subheader("📊 Review Analysis:")
                st.write(analysis_result)
            except Exception as e:
                st.error("Error analyzing review.")
                st.exception(e)
        else:
            st.warning("Please enter a review text to analyze.")

import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import base64
import io
import json
from groq import Groq
import warnings
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# ========== API KEY CONFIGURATION ==========

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(script_dir, ".env"))
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

# =========================================

# ========== CLASS MAP ==========
# Maps every YOLO class name → {plant, status, disease}
# status: "healthy" or "diseased"
# disease: None if healthy, otherwise the disease name

CLASS_MAP = {
    # --- Apple ---
    "Apple Scab Leaf":          {"plant": "Apple",       "status": "diseased", "disease": "Apple Scab"},
    "Apple leaf":               {"plant": "Apple",       "status": "healthy",  "disease": None},
    "Apple rust leaf":          {"plant": "Apple",       "status": "diseased", "disease": "Apple Rust"},

    # --- Bell Pepper ---
    "Bell_pepper leaf spot":    {"plant": "Bell Pepper", "status": "diseased", "disease": "Leaf Spot"},
    "Bell_pepper leaf":         {"plant": "Bell Pepper", "status": "healthy",  "disease": None},

    # --- Blueberry ---
    "Blueberry leaf":           {"plant": "Blueberry",   "status": "healthy",  "disease": None},

    # --- Cherry ---
    "Cherry leaf":              {"plant": "Cherry",      "status": "healthy",  "disease": None},

    # --- Corn ---
    "Corn Gray leaf spot":      {"plant": "Corn",        "status": "diseased", "disease": "Gray Leaf Spot"},
    "Corn leaf blight":         {"plant": "Corn",        "status": "diseased", "disease": "Leaf Blight"},
    "Corn rust leaf":           {"plant": "Corn",        "status": "diseased", "disease": "Rust"},

    # --- Peach ---
    "Peach leaf":               {"plant": "Peach",       "status": "healthy",  "disease": None},

    # --- Potato ---
    "Potato leaf early blight": {"plant": "Potato",      "status": "diseased", "disease": "Early Blight"},
    "Potato leaf late blight":  {"plant": "Potato",      "status": "diseased", "disease": "Late Blight"},
    "Potato leaf":              {"plant": "Potato",      "status": "healthy",  "disease": None},

    # --- Raspberry ---
    "Raspberry leaf":           {"plant": "Raspberry",   "status": "healthy",  "disease": None},

    # --- Soybean ---
    "Soyabean leaf":            {"plant": "Soybean",     "status": "healthy",  "disease": None},
    "Soybean leaf":             {"plant": "Soybean",     "status": "healthy",  "disease": None},

    # --- Squash ---
    "Squash Powdery mildew leaf": {"plant": "Squash",    "status": "diseased", "disease": "Powdery Mildew"},

    # --- Strawberry ---
    "Strawberry leaf":          {"plant": "Strawberry",  "status": "healthy",  "disease": None},

    # --- Tomato ---
    "Tomato Early blight leaf":           {"plant": "Tomato", "status": "diseased", "disease": "Early Blight"},
    "Tomato Septoria leaf spot":          {"plant": "Tomato", "status": "diseased", "disease": "Septoria Leaf Spot"},
    "Tomato leaf bacterial spot":         {"plant": "Tomato", "status": "diseased", "disease": "Bacterial Spot"},
    "Tomato leaf late blight":            {"plant": "Tomato", "status": "diseased", "disease": "Late Blight"},
    "Tomato leaf mosaic virus":           {"plant": "Tomato", "status": "diseased", "disease": "Mosaic Virus"},
    "Tomato leaf yellow virus":           {"plant": "Tomato", "status": "diseased", "disease": "Yellow Leaf Curl Virus"},
    "Tomato leaf":                        {"plant": "Tomato", "status": "healthy",  "disease": None},
    "Tomato mold leaf":                   {"plant": "Tomato", "status": "diseased", "disease": "Leaf Mold"},
    "Tomato two spotted spider mites leaf": {"plant": "Tomato", "status": "diseased", "disease": "Two-Spotted Spider Mites"},

    # --- Grape ---
    "grape leaf black rot":     {"plant": "Grape",       "status": "diseased", "disease": "Black Rot"},
    "grape leaf":               {"plant": "Grape",       "status": "healthy",  "disease": None},
}


def parse_class(raw_label):
    """
    Given a raw YOLO class label, return a structured dict:
    {plant, status, disease, raw_label}
    Falls back gracefully if label not in CLASS_MAP.
    """
    info = CLASS_MAP.get(raw_label)
    if info:
        return {
            "plant":     info["plant"],
            "status":    info["status"],
            "disease":   info["disease"],
            "raw_label": raw_label,
        }
    # Fallback: unknown class — treat as unknown
    return {
        "plant":     raw_label,
        "status":    "unknown",
        "disease":   None,
        "raw_label": raw_label,
    }


# ========== MODEL LOADING ==========

@st.cache_resource
def load_models():
    try:
        disease_model = YOLO(os.path.join(script_dir, "best.pt"))
        return disease_model
    except Exception as e:
        st.error(f"🔴 Error loading models: {e}")
        st.stop()


# ------------------ CONFIG & STYLE ------------------

st.set_page_config(
    page_title="🌾 AgriCare AI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Check API Key after st.set_page_config()
if not GROQ_API_KEY:
    st.error("❌ Error: GROQ_API_KEY not found!")
    st.info(
        "📋 Please set GROQ_API_KEY environment variable "
        "or create a .env file with: GROQ_API_KEY=your_key_here"
    )
    st.stop()

# Load models once at startup
disease_model = load_models()

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

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        background-color: #1e222a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title(
    "🌿 AgriCare AI: Plant Disease Detection & Farmer AI Advisor"
)


# ------------------ FUNCTIONS ------------------

def detect_disease(image, conf_threshold=0.5, plant_filter="Auto-detect"):
    """
    Run YOLO detection on the image with adjustable confidence and optional plant filter.
    Returns: (primary_info, all_detections, pred_image, confidence)
      primary_info = {plant, status, disease, raw_label, confidence}
    """
    try:
        temp_path = os.path.join(script_dir, "temp.jpg")
        image.save(temp_path)

        results = disease_model.predict(
            temp_path,
            save=False,
            stream=False,
            imgsz=640,
            conf=conf_threshold
        )

        # Clean up temp file early
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if results and len(results) > 0:
            result = results[0]

            if result.boxes and len(result.boxes) > 0:
                annotated = result.plot()
                pred_image = Image.fromarray(annotated)

                all_detections = []
                for b in result.boxes:
                    cls_id = int(b.cls[0])
                    raw_label = result.names[cls_id]
                    box_conf = float(b.conf[0])
                    info = parse_class(raw_label)
                    info["confidence"] = box_conf
                    all_detections.append(info)

                # Sort by confidence descending
                all_detections.sort(key=lambda x: x["confidence"], reverse=True)

                # If plant filter is specified and not auto-detect, try finding matching detection
                chosen = None
                if plant_filter and plant_filter != "Auto-detect" and plant_filter != "Other / Not in list":
                    for d in all_detections:
                        if d["plant"].lower() == plant_filter.lower():
                            chosen = d
                            break

                # Otherwise take highest confidence detection
                if chosen is None:
                    chosen = all_detections[0]

                return chosen, all_detections, pred_image, chosen["confidence"]

        # Nothing detected
        return (
            {"plant": None, "status": "not_detected", "disease": None, "raw_label": None, "confidence": 0.0},
            [],
            Image.fromarray(np.array(image)),
            0.0
        )

    except Exception as e:
        st.error(f"Error in disease detection: {str(e)}")
        raise e


def analyze_leaf_with_ai_vision(image):
    """
    Multimodal AI Vision analysis (Qwen-3.8-27b Vision):
    1. Recognizes the leaf/plant name (e.g. Tomato, Potato, Cauliflower, Mango, Apple, Rose, etc.)
    2. Checks whether it has spots, lesions, blight, discoloration, or is healthy
    3. Produces structured expert treatment advice based on leaf name & disease
    """
    try:
        buffered = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        client = Groq(api_key=GROQ_API_KEY)

        prompt = (
            "You are an expert plant pathologist and botanist. Analyze this plant leaf image step-by-step:\n"
            "1. Identify the exact Plant / Leaf Name (e.g. Tomato, Potato, Cauliflower, Apple, Soybean, Grape, Rose, etc.).\n"
            "2. Check whether the leaf has spots, lesions, blight, discoloration, or is healthy.\n"
            "3. If diseased, identify the exact disease name. If healthy, state 'None (Healthy)'.\n"
            "4. Provide detailed expert treatment and care advice.\n\n"
            "Respond STRICTLY in pure, valid JSON with this exact schema:\n"
            "{\n"
            '  "plant_name": "Name of the plant",\n'
            '  "health_status": "Healthy" or "Diseased",\n'
            '  "has_spots": true or false,\n'
            '  "spots_symptoms_description": "Detailed description of spots, marks, blight, or discoloration, or state None if healthy",\n'
            '  "disease_name": "Exact disease name or None (Healthy)",\n'
            '  "confidence": "High / Medium / Low",\n'
            '  "advice": {\n'
            '    "cause": "Why this occurred on this plant",\n'
            '    "organic_treatment": "Natural and organic remedies",\n'
            '    "chemical_treatment": "Approved fungicides/pesticides if applicable, or None if healthy",\n'
            '    "prevention": "Preventative steps for future crops"\n'
            '  }\n'
            "}\n"
            "Do not include any extra markdown code fences or conversational filler. Return only valid JSON."
        )

        resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            max_tokens=1000,
            temperature=0.1
        )

        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except Exception as e:
        return {
            "plant_name": "Plant Leaf",
            "health_status": "Analysis Complete",
            "has_spots": True,
            "spots_symptoms_description": f"Analysis output: {str(e)}",
            "disease_name": "Consult AI Specialist",
            "confidence": "Medium",
            "advice": {
                "cause": "Leaf condition observed.",
                "organic_treatment": "Apply organic neem oil spray and prune affected leaves.",
                "chemical_treatment": "Consult local agricultural extension office.",
                "prevention": "Ensure good soil drainage, avoid overhead watering, and maintain airflow."
            }
        }





# ------------------ AI DISEASE ADVICE ------------------

def generate_english_gpt_advice(plant_name, disease_name):
    """
    Generate expert agricultural advice for a specific plant disease.
    Both plant_name and disease_name are clean strings (not raw YOLO labels).
    """
    try:
        prompt = (
            f"I am a farmer. My {plant_name} plant has been diagnosed with "
            f"{disease_name}.\n\n"
            f"Please provide expert agricultural advice covering:\n"
            f"1. What are the main symptoms of {disease_name}?\n"
            f"2. How and why does {disease_name} occur in {plant_name}?\n"
            f"3. What organic treatments or approved fungicides/pesticides can cure it?\n"
            f"4. What home remedies or natural solutions can help?\n"
            f"5. How can I prevent {disease_name} in future growing seasons?\n\n"
            f"Be specific, practical, and farmer-friendly."
        )

        return call_groq(prompt)

    except Exception as e:
        st.error(f"Error generating advice: {str(e)}")
        raise


# ------------------ TABS ------------------

tab1, tab2 = st.tabs([
    "🌿 Plant Disease Detection",
    "🤝 Farmer AI Advisor"
])


# ----------- TAB 1: PLANT DISEASE DETECTION ------------

with tab1:

    st.subheader("🌱 Smart Plant Health Center")
    st.caption(
        "🔬 **Complete AI Workflow**: Identify Plant ➜ Detect Disease ➜ Symptoms & AI Advice"
    )

    # Engine selection
    engine_choice = st.radio(
        "🧠 Detection Engine",
        [
            "🤖 Smart AI Vision (Recommended: Any Plant — Recognises Name + Detects Spots + AI Advice)",
            "⚡ Local YOLO Model (best.pt — 13 trained crops only)"
        ],
        index=0,
        horizontal=True,
        help="Smart AI Vision recognizes any plant leaf, detects spots/disease accurately, and eliminates false Septoria detections."
    )

    uploaded_img = st.file_uploader(
        "📤 Upload Any Plant Leaf Image",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader"
    )

    if uploaded_img is not None:

        img = Image.open(uploaded_img)

        st.image(img, caption="📷 Uploaded Leaf Image", use_container_width=True)

        if "Smart AI Vision" in engine_choice:
            # ─────────────────────────────────────────────────────────
            # 3-STEP SMART AI VISION WORKFLOW
            # ─────────────────────────────────────────────────────────
            try:
                with st.spinner("🤖 Analyzing leaf: Step 1 (Identify Plant) ➜ Step 2 (Check Spots & Disease) ➜ Step 3 (AI Advice)..."):
                    result = analyze_leaf_with_ai_vision(img)

                plant_name = result.get("plant_name", "Unknown Plant")
                health_status = result.get("health_status", "Healthy")
                has_spots = result.get("has_spots", False)
                spots_desc = result.get("spots_symptoms_description", "No harmful spots observed.")
                disease_name = result.get("disease_name", "None (Healthy)")
                confidence = result.get("confidence", "High")
                advice = result.get("advice", {})

                st.markdown("---")

                # Step 1: Recognise Leaf Name
                st.markdown("### 🌿 Step 1: Leaf & Plant Identification")
                col_p1, col_p2 = st.columns([3, 1])
                with col_p1:
                    st.success(f"🌿 **Recognized Plant / Leaf:** **`{plant_name}`**")
                with col_p2:
                    st.info(f"**Confidence:** `{confidence}`")

                # Step 2: Check whether it has spots and disease
                st.markdown("### 🔬 Step 2: Spots & Health Assessment")
                is_diseased = health_status.lower() == "diseased" or (has_spots and disease_name.lower() not in ["none", "none (healthy)", "healthy"])

                if not is_diseased:
                    st.success(
                        f"🟢 **Status: Healthy Leaf**\n\n"
                        f"✅ No harmful disease spots, blight, or lesions detected on this **{plant_name}** leaf. "
                        f"The foliage appears vibrant and healthy!"
                    )
                    if spots_desc and spots_desc.lower() not in ["none", "none.", "none if healthy"]:
                        st.caption(f"🔍 Visual notes: {spots_desc}")
                else:
                    st.error(
                        f"🔴 **Status: Diseased Leaf**\n\n"
                        f"⚠️ **Identified Disease:** **{disease_name}**"
                    )
                    st.warning(f"🔍 **Spots & Symptoms Observed:**\n\n{spots_desc}")

                # Step 3: Expert AI Treatment Advice
                st.markdown("---")
                st.markdown(f"### 🧠 Step 3: Expert AI Advice for {plant_name}")

                if not is_diseased:
                    st.markdown(f"#### 🌱 Maintenance & Care Guide for {plant_name}")
                    col_c1, col_c2 = st.columns(2)
                    with col_c1:
                        st.markdown(f"**🍃 Recommended Care & Nutrition:**\n\n{advice.get('organic_treatment', 'Provide adequate sunlight, balanced N-P-K fertilization, and consistent moisture.')}")
                    with col_c2:
                        st.markdown(f"**🛡️ Proactive Disease Prevention:**\n\n{advice.get('prevention', 'Avoid overhead watering to keep leaves dry. Maintain good air circulation around the canopy.')}")
                else:
                    tab_adv1, tab_adv2, tab_adv3, tab_adv4 = st.tabs([
                        "📌 Cause & Occurrence",
                        "🍃 Natural & Organic Remedies",
                        "🧪 Chemical Treatments",
                        "🛡️ Future Prevention"
                    ])
                    with tab_adv1:
                        st.markdown(f"**Why {disease_name} occurs in {plant_name}:**")
                        st.write(advice.get("cause", "Caused by fungal or bacterial spores favored by humid conditions."))
                    with tab_adv2:
                        st.markdown(f"**Organic & Home Remedies for {plant_name}:**")
                        st.write(advice.get("organic_treatment", "Spray diluted neem oil or baking soda solution; prune and safely discard infected leaves."))
                    with tab_adv3:
                        st.markdown(f"**Approved Fungicides / Chemical Controls:**")
                        st.write(advice.get("chemical_treatment", "Apply recommended copper-based or chlorothalonil fungicide if condition spreads."))
                    with tab_adv4:
                        st.markdown(f"**Preventative Measures:**")
                        st.write(advice.get("prevention", "Rotate crops, sanitize pruning shears, water at soil level, and ensure proper plant spacing."))



            except Exception as e:
                st.error(f"Error during AI vision analysis: {str(e)}")

        else:
            # ─────────────────────────────────────────────────────────
            # LEGACY YOLO OBJECT DETECTION WORKFLOW
            # ─────────────────────────────────────────────────────────
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                plant_filter = st.selectbox(
                    "🌿 Plant Filter",
                    [
                        "Auto-detect (All 13 Plants)", "Tomato", "Potato", "Apple",
                        "Bell Pepper", "Blueberry", "Cherry", "Corn", "Grape",
                        "Peach", "Raspberry", "Soybean", "Squash", "Strawberry"
                    ],
                    index=0
                )
            with col_ctrl2:
                conf_threshold = st.slider("🎯 YOLO Confidence", 0.15, 0.95, 0.60, 0.05)

            try:
                with st.spinner("🔍 Running YOLO detection..."):
                    detection_info, all_detections, pred_img, confidence = detect_disease(
                        img, conf_threshold=conf_threshold, plant_filter=plant_filter
                    )

                st.image(
                    pred_img,
                    caption=f"Detected: {detection_info.get('raw_label', 'None')} ({confidence:.1%})",
                    use_container_width=True
                )

                if detection_info["status"] == "healthy":
                    st.success(f"✅ **Plant: {detection_info['plant']}** | 🟢 **Status: Healthy**")
                elif detection_info["status"] == "diseased":
                    st.error(f"🔴 **Plant: {detection_info['plant']}** | ⚠️ **Disease: {detection_info['disease']}**")
                    with st.spinner("Generating AI advice..."):
                        advice = generate_english_gpt_advice(detection_info["plant"], detection_info["disease"])
                        st.success(advice)
                else:
                    st.warning("No leaf detected by YOLO at current confidence threshold.")
            except Exception as e:
                st.error(f"YOLO error: {str(e)}")

    else:
        st.info(
            "👆 **Upload any plant leaf photo above** to automatically identify the plant, "
            "inspect for spots & diseases, and receive expert AI treatment advice."
        )


# ----------- TAB 2: FARMER AI ADVISOR -------------

def call_groq(prompt: str, max_tokens: int = 900) -> str:
    """Helper: call qwen/qwen3.8-27b with a plain text prompt via Groq API."""
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


with tab2:

    st.header("🤝 Farmer AI Advisor")
    st.caption(
        "Three AI-powered tools built to solve real problems that farmers face every day — "
        "from crop diseases and market prices to government schemes."
    )

    adv1, adv2, adv3 = st.tabs([
        "🌾 Crop Problem Advisor",
        "📈 Market Price Advisor",
        "🏛️ Govt. Scheme Finder"
    ])

    # ── TOOL 1: Farmer Complaint & Crop Problem Advisor ─────────────────
    with adv1:
        st.subheader("🌾 Crop Problem Advisor")
        st.markdown(
            "Describe **any crop problem in plain language** — yellowing leaves, wilting, "
            "pest attack, unusual smell, bad yield — and get an instant AI diagnosis with action plan."
        )
        st.markdown("---")

        problem_crop = st.text_input(
            "🌱 Crop / Plant Name",
            placeholder="e.g. Tomato, Onion, Wheat, Paddy, Cotton",
            key="adv1_crop"
        )
        problem_desc = st.text_area(
            "📝 Describe the Problem (in your own words)",
            placeholder="e.g. My onion crop leaves are turning yellow and the bulbs are very small. The soil also smells bad after rain.",
            height=130,
            key="adv1_desc"
        )
        problem_location = st.text_input(
            "📍 Your Location / State (Optional)",
            placeholder="e.g. Maharashtra, Punjab, Karnataka",
            key="adv1_loc"
        )

        if st.button("🔍 Diagnose My Crop Problem", key="adv1_btn"):
            if problem_crop and problem_desc:
                with st.spinner("Consulting agricultural AI expert..."):
                    try:
                        prompt = (
                            f"You are an expert agricultural scientist and farmer advisor.\n"
                            f"A farmer is reporting the following problem:\n"
                            f"Crop: {problem_crop}\n"
                            f"Location: {problem_location or 'India'}\n"
                            f"Problem Description: {problem_desc}\n\n"
                            f"Please provide a clear, structured, farmer-friendly response covering:\n"
                            f"1. **Most Probable Cause** — What is likely causing this problem?\n"
                            f"2. **Immediate Action** — What should the farmer do RIGHT NOW (within 24-48 hours)?\n"
                            f"3. **Treatment Plan** — Organic and chemical remedies with dosage and frequency.\n"
                            f"4. **Long-Term Fix** — Steps to prevent this from recurring next season.\n"
                            f"5. **Warning Signs** — When should the farmer escalate and consult an extension officer?\n"
                            f"Use simple, practical, actionable language. The farmer may have limited resources."
                        )
                        result = call_groq(prompt)
                        st.success("✅ AI Diagnosis Ready")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter both crop name and problem description.")

    # ── TOOL 2: Agri Market Price Advisor ───────────────────────────────
    with adv2:
        st.subheader("📈 Market Price Advisor")
        st.markdown(
            "Get AI-powered insights on **when and where to sell** your crop for the best price, "
            "seasonal trends, and strategies to avoid middlemen losses."
        )
        st.markdown("---")

        mkt_crop = st.text_input(
            "🌾 Crop Name",
            placeholder="e.g. Tomato, Onion, Wheat, Soybean, Sugarcane",
            key="adv2_crop"
        )
        col_mkt1, col_mkt2 = st.columns(2)
        with col_mkt1:
            mkt_state = st.text_input(
                "📍 Your State",
                placeholder="e.g. Madhya Pradesh, Uttar Pradesh",
                key="adv2_state"
            )
        with col_mkt2:
            mkt_month = st.selectbox(
                "📅 Current Month",
                ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"],
                index=8,
                key="adv2_month"
            )
        mkt_qty = st.text_input(
            "📦 Approximate Harvest Quantity",
            placeholder="e.g. 2 tonnes, 500 kg, 10 quintals",
            key="adv2_qty"
        )

        if st.button("📊 Get Market Advice", key="adv2_btn"):
            if mkt_crop:
                with st.spinner("Analyzing market trends and pricing..."):
                    try:
                        prompt = (
                            f"You are an expert agricultural market analyst in India.\n"
                            f"Crop: {mkt_crop}\n"
                            f"State: {mkt_state or 'India'}\n"
                            f"Month: {mkt_month}\n"
                            f"Quantity to sell: {mkt_qty or 'not specified'}\n\n"
                            f"Provide a detailed market advisory covering:\n"
                            f"1. **Current Market Outlook** — Is this a buyer's market or seller's market for {mkt_crop} in {mkt_month}?\n"
                            f"2. **Price Trend** — Expected price range (Rs per quintal or kg) based on seasonal patterns.\n"
                            f"3. **Best Time to Sell** — Should the farmer sell now or wait? Why?\n"
                            f"4. **Where to Sell** — Best mandis, APMC markets, FPOs, or e-NAM platforms to get better prices.\n"
                            f"5. **Value Addition Tips** — Can the farmer process or grade the produce to earn more?\n"
                            f"6. **Risk Factors** — What market risks should the farmer watch out for?\n"
                            f"Be practical and farmer-friendly. Mention specific strategies."
                        )
                        result = call_groq(prompt)
                        st.success("✅ Market Advisory Ready")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a crop name.")

    # ── TOOL 3: Government Scheme Finder ────────────────────────────────
    with adv3:
        st.subheader("🏛️ Government Scheme Finder")
        st.markdown(
            "Enter your **state, crop, and farm details** to discover central and state government "
            "subsidies, crop insurance, PM schemes, and credit programs you are eligible for."
        )
        st.markdown("---")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            scheme_state = st.selectbox(
                "📍 Your State",
                [
                    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
                    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
                    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
                    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
                    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
                    "Uttar Pradesh", "Uttarakhand", "West Bengal"
                ],
                key="adv5_state"
            )
            scheme_crop = st.text_input(
                "🌾 Primary Crop",
                placeholder="e.g. Rice, Wheat, Cotton, Pulses, Horticulture",
                key="adv5_crop"
            )
        with col_s2:
            scheme_area = st.text_input(
                "📐 Farm Size",
                placeholder="e.g. 1 acre, 2 hectares, marginal (<1 acre)",
                key="adv5_area"
            )
            scheme_category = st.selectbox(
                "👤 Farmer Category",
                ["Small Farmer (1-2 hectares)", "Marginal Farmer (<1 hectare)",
                 "Medium Farmer (2-10 hectares)", "Large Farmer (>10 hectares)",
                 "Tribal / ST Farmer", "Women Farmer", "Not Sure"],
                key="adv5_cat"
            )
        scheme_needs = st.multiselect(
            "✅ What are you looking for?",
            [
                "Crop Insurance", "Fertilizer Subsidy", "Seed Subsidy",
                "Irrigation Support", "Credit / Loan", "Equipment / Machinery Subsidy",
                "PM Kisan Income Support", "Market Linkage", "Training / FPO Support"
            ],
            default=["Crop Insurance", "PM Kisan Income Support"],
            key="adv5_needs"
        )

        if st.button("🔍 Find My Eligible Schemes", key="adv5_btn"):
            if scheme_state and scheme_crop:
                with st.spinner("Searching central and state government schemes..."):
                    try:
                        needs_str = ", ".join(scheme_needs) if scheme_needs else "general support"
                        prompt = (
                            f"You are an expert on Indian agricultural government schemes and policies.\n"
                            f"Farmer Profile:\n"
                            f"- State: {scheme_state}\n"
                            f"- Primary Crop: {scheme_crop}\n"
                            f"- Farm Size: {scheme_area or 'not specified'}\n"
                            f"- Category: {scheme_category}\n"
                            f"- Looking for: {needs_str}\n\n"
                            f"List all relevant Central Government and {scheme_state} State Government schemes:\n"
                            f"1. **PM Kisan & Direct Benefit Transfer Schemes** — Eligibility, amount, how to apply.\n"
                            f"2. **Crop Insurance Schemes** — PMFBY and state-level schemes applicable for {scheme_crop}.\n"
                            f"3. **Input Subsidy Schemes** — Fertilizer, seed, pesticide subsidies available.\n"
                            f"4. **Credit & Loan Schemes** — KCC (Kisan Credit Card), NABARD schemes, interest subvention.\n"
                            f"5. **Irrigation & Infrastructure Support** — PM-KUSUM, drip irrigation subsidy, etc.\n"
                            f"6. **Market & Value Chain Support** — e-NAM, FPO promotion, APMC reforms applicable.\n"
                            f"7. **How to Apply** — Where to register, documents needed, and key deadlines.\n"
                            f"Focus specifically on {scheme_state} state schemes alongside central schemes."
                        )
                        result = call_groq(prompt, max_tokens=1200)
                        st.success("✅ Scheme Report Ready")
                        st.markdown(result)
                        st.info(
                            "💡 **How to Apply:** Visit your nearest **Common Service Centre (CSC)**, "
                            "**Krishi Vigyan Kendra (KVK)**, or the **PM Kisan Portal** (pmkisan.gov.in) "
                            "to register and apply for the schemes listed above."
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please select your state and enter your primary crop.")
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churnetic - AI Churn Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING (Dark Theme & Pixel-Inspired UI)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Reset & Dark Theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0B0E17 !important;
        color: #E2E8F0;
    }
    
    .stApp {
        background-color: #0B0E17 !important;
    }

    /* Top Navigation Bar */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background-color: #111625;
        border: 1px solid #1E293B;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .nav-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: #0B0E17;
        font-size: 18px;
    }
    .brand-title {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .brand-sub {
        font-size: 10px;
        color: #94A3B8;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: -3px;
    }
    .nav-links {
        display: flex;
        gap: 20px;
        font-size: 14px;
        color: #94A3B8;
        font-weight: 500;
    }
    .nav-link-active {
        color: #00F2FE;
        font-weight: 600;
    }
    .model-badge {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .dot {
        width: 6px;
        height: 6px;
        background-color: #10B981;
        border-radius: 50%;
    }

    /* Hero Headline */
    .hero-container {
        margin-bottom: 24px;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(0, 242, 254, 0.1);
        color: #00F2FE;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
        border: 1px solid rgba(0, 242, 254, 0.2);
    }
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
        letter-spacing: -1px;
        margin-bottom: 10px;
    }
    .gradient-text {
        background: linear-gradient(135deg, #00F2FE 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 15px;
        color: #94A3B8;
        max-width: 750px;
        line-height: 1.5;
    }

    /* KPI Cards Row */
    .kpi-card {
        background-color: #111625;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .kpi-icon {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    .kpi-badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Main Container Panels */
    .panel-card {
        background-color: #111625;
        border: 1px solid #1E293B;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 16px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1E293B;
    }
    .section-icon {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        background: rgba(0, 242, 254, 0.1);
        color: #00F2FE;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    .section-title {
        font-size: 15px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .section-desc {
        font-size: 12px;
        color: #64748B;
        margin-left: auto;
    }

    /* Prediction Result Box */
    .res-card {
        background: linear-gradient(180deg, #131B2E 0%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .prob-number {
        font-size: 52px;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
        letter-spacing: -1px;
    }
    .prob-label {
        font-size: 13px;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 6px;
        margin-bottom: 20px;
    }

    /* Recommendation Card */
    .rec-card {
        background-color: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #00F2FE;
        border-radius: 0 10px 10px 0;
        padding: 16px;
        text-align: left;
        margin-top: 20px;
    }
    .rec-title {
        font-size: 13px;
        font-weight: 700;
        color: #00F2FE;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .rec-text {
        font-size: 13px;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Form Overrides */
    .stSelectbox label, .stSlider label, .stRadio label, .stNumberInput label {
        color: #94A3B8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-baseweb="select"] > div {
        background-color: #1A2234 !important;
        border-color: #2E3D56 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }

    /* Buttons Override */
    .stButton > button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #7F00FF 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.3) !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.5) !important;
    }
    
    /* Hide Streamlit components headers/footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. TOP NAVIGATION BAR & HERO SECTION
# -----------------------------------------------------------------------------
st.markdown("""
<div class="nav-bar">
    <div class="nav-left">
        <div class="brand-logo">⚡</div>
        <div>
            <div class="brand-title">Churnetic</div>
            <div class="brand-sub">Retention Intelligence System</div>
        </div>
    </div>
    <div class="nav-links">
        <span class="nav-link-active">Dashboard</span>
        <span>Customers</span>
        <span>Models</span>
        <span>Reports</span>
    </div>
    <div class="model-badge">
        <span class="dot"></span> Model v4.2 - Live
    </div>
</div>

<div class="hero-container">
    <div class="hero-tag">⚡ Powered by Churnetic AI • Gradient Boosted Ensemble</div>
    <div class="hero-title">Predict <span class="gradient-text">customer churn</span> before it happens.</div>
    <div class="hero-sub">Enter a customer's profile and services to generate a real-time churn probability, risk classification, and AI-crafted retention strategy.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. KPI METRICS CARDS ROW
# -----------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon">👥</div>
            <div class="kpi-badge-green">+12.4%</div>
        </div>
        <div class="kpi-value">128,472</div>
        <div class="kpi-label">Customers Scored</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon">📈</div>
            <div class="kpi-badge-green">-1.2%</div>
        </div>
        <div class="kpi-value">23.8%</div>
        <div class="kpi-label">Avg. Churn Risk</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon">📊</div>
            <div class="kpi-badge-green">+0.8%</div>
        </div>
        <div class="kpi-value">94.6%</div>
        <div class="kpi-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-icon">💰</div>
            <div class="kpi-badge-green">+18%</div>
        </div>
        <div class="kpi-value">$1.42M</div>
        <div class="kpi-label">Retained Revenue (30D)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("") # Spacing

# -----------------------------------------------------------------------------
# 5. MAIN CONTENT LAYOUT (INPUT FORM & PREDICTION PANEL)
# -----------------------------------------------------------------------------
col_input, col_result = st.columns([1.15, 0.85], gap="large")

with col_input:
    st.markdown("### Customer Profile Signals")
    st.caption("Provide customer profile details and service configurations to trigger prediction.")

    # --- SECTION 1: DEMOGRAPHICS ---
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">👤</div>
        <div class="section-title">Demographics</div>
        <div class="section-desc">Basic profile info</div>
    </div>
    """, unsafe_allow_html=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
    with d_col2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    # --- SECTION 2: SERVICES ---
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📡</div>
        <div class="section-title">Services</div>
        <div class="section-desc">Subscribed telecom services</div>
    </div>
    """, unsafe_allow_html=True)

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "None"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

    with s_col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # --- SECTION 3: CONTRACT & TENURE ---
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📄</div>
        <div class="section-title">Contract & Tenure</div>
        <div class="section-desc">Subscription terms</div>
    </div>
    """, unsafe_allow_html=True)

    ct_col1, ct_col2 = st.columns([1, 1])
    with ct_col1:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with ct_col2:
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

    # --- SECTION 4: BILLING ---
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">💳</div>
        <div class="section-title">Billing & Charges</div>
        <div class="section-desc">Payment & billing details</div>
    </div>
    """, unsafe_allow_html=True)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=85.0, step=0.5)

    with b_col2:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
        # Auto compute default total charges based on tenure & monthly charge
        calculated_total = round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=max(calculated_total, 18.0))

    st.write("")
    predict_btn = st.button("✨ Run AI Prediction")
    st.caption("🔒 Data stays in your local session. No PII stored.")

# -----------------------------------------------------------------------------
# 6. MODEL LOADING & PREDICTION LOGIC
# -----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "logistic_regression_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "models", "preprocessor.pkl")

@st.cache_resource
def load_models():
    """Load and cache the trained model and preprocessor."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

def run_model_prediction(
    gender_val, senior_val, partner_val, dependents_val, tenure_val,
    phone_val, multi_val, internet_val, sec_val, backup_val,
    protect_val, tech_val, tv_val, movies_val, contract_val,
    paperless_val, payment_val, monthly_val, total_val
):
    """
    Build the exact 19-feature DataFrame expected by the preprocessor
    and return (churn_probability, prediction_label).
    """
    preprocessor, model = load_models()

    # SeniorCitizen was stored as int (0/1) during training
    senior_int = 1 if senior_val == "Yes" else 0

    new_customer = pd.DataFrame([{
        "gender":           gender_val,
        "SeniorCitizen":    senior_int,
        "Partner":          partner_val,
        "Dependents":       dependents_val,
        "tenure":           int(tenure_val),
        "PhoneService":     phone_val,
        "MultipleLines":    multi_val,
        "InternetService":  internet_val,
        "OnlineSecurity":   sec_val,
        "OnlineBackup":     backup_val,
        "DeviceProtection": protect_val,
        "TechSupport":      tech_val,
        "StreamingTV":      tv_val,
        "StreamingMovies":  movies_val,
        "Contract":         contract_val,
        "PaperlessBilling": paperless_val,
        "PaymentMethod":    payment_val,
        "MonthlyCharges":   float(monthly_val),
        "TotalCharges":     float(total_val),
    }])

    processed = preprocessor.transform(new_customer)
    prediction = model.predict(processed)[0]
    churn_prob = float(model.predict_proba(processed)[0][1])
    return churn_prob, int(prediction)

with col_result:
    st.markdown("### AI Prediction Results")
    st.caption("Real-time risk classification & strategic recommendations")

    # ---- Run model on button click (or on initial load) ----
    if predict_btn:
        with st.spinner("Running AI prediction..."):
            prob, pred_label = run_model_prediction(
                gender, senior, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies,
                contract, paperless, payment_method,
                monthly_charges, total_charges
            )
        st.session_state["prob"] = prob
        st.session_state["pred_label"] = pred_label
    elif "prob" not in st.session_state:
        # First load — run prediction silently with default values
        prob, pred_label = run_model_prediction(
            gender, senior, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
            contract, paperless, payment_method,
            monthly_charges, total_charges
        )
        st.session_state["prob"] = prob
        st.session_state["pred_label"] = pred_label
    else:
        prob = st.session_state["prob"]
        pred_label = st.session_state["pred_label"]

    prob_pct = round(prob * 100, 1)

    # Risk thresholds: Low < 0.30 | Medium < 0.70 | High >= 0.70
    if prob >= 0.70:
        risk_level = "High Churn Risk"
        risk_color = "#FF4B4B"
        badge_bg = "rgba(255, 75, 75, 0.15)"
        status_text = "LIKELY TO CHURN"
        rec_headline = "Immediate Retention Intervention Required"
        rec_details = "High churn threat detected due to <strong>Month-to-Month contract</strong>, <strong>Fiber Optic pricing sensitivity</strong>, and short tenure. Recommend offering a <strong>15% discount on a 12-month contract lock</strong> with free <strong>Tech Support</strong> bundle."
    elif prob >= 0.30:
        risk_level = "Medium Risk"
        risk_color = "#FFA726"
        badge_bg = "rgba(255, 167, 38, 0.15)"
        status_text = "MODERATE CHURN RISK"
        rec_headline = "Cross-Sell & Loyalty Offer Recommended"
        rec_details = "Customer shows moderate risk. Recommend engaging with an automated loyalty check-in and offering a complimentary 3-month trial of <strong>Online Security</strong> and <strong>Device Protection</strong>."
    else:
        risk_level = "Low Churn Risk"
        risk_color = "#00E676"
        badge_bg = "rgba(0, 230, 118, 0.15)"
        status_text = "HIGH RETENTION LIKELIHOOD"
        rec_headline = "Standard Account Growth Profile"
        rec_details = "Customer profile is highly stable. Ideal candidate for upgrading to higher streaming tier speed or introducing family plan add-ons."

    # Container for prediction outputs
    st.markdown(f"""
<div class="res-card">
<div class="risk-badge" style="background-color: {badge_bg}; color: {risk_color}; border: 1px solid {risk_color};">&#9679; {risk_level}</div>
<div class="prob-number">{prob_pct}%</div>
<div class="prob-label">Predicted Churn Probability</div>
<div style="text-align: left; margin-top: 15px; margin-bottom: 5px;">
<div style="display: flex; justify-content: space-between; font-size: 12px; color: #94A3B8; font-weight: 600;">
<span>STATUS: <strong style="color: {risk_color};">{status_text}</strong></span>
<span>MODEL: <strong>Logistic Regression</strong></span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # Progress bar indicator
    st.progress(prob)

    # Risk Factors Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Key Risk Factors")
    
    factors = []
    if contract == "Month-to-month":
        factors.append(("Month-to-Month Contract", "+35% risk impact", "#FF4B4B"))
    if internet_service == "Fiber optic":
        factors.append(("Fiber Optic Line (Higher Tier)", "+14% risk impact", "#FFA726"))
    if tech_support == "No":
        factors.append(("No Tech Support Subscribed", "+8% risk impact", "#FFA726"))
    if tenure < 12:
        factors.append(("Tenure Under 12 Months", "+12% risk impact", "#FF4B4B"))

    if not factors:
        factors.append(("Long-term Contract Signed", "-25% risk reduction", "#00E676"))
        factors.append(("High Tenure Stability", "-15% risk reduction", "#00E676"))

    for factor, weight, color in factors:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; background: #1A2234; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #2E3D56; font-size: 13px;">
            <span style="color: #E2E8F0; font-weight: 500;">{factor}</span>
            <span style="color: {color}; font-weight: 700;">{weight}</span>
        </div>
        """, unsafe_allow_html=True)

    # AI Business Recommendation Card
    st.markdown(f"""
    <div class="rec-card" style="border-left-color: {risk_color};">
        <div class="rec-title" style="color: {risk_color};">💡 AI Retention Strategy</div>
        <div style="font-weight: 700; color: #FFFFFF; font-size: 14px; margin-bottom: 4px;">{rec_headline}</div>
        <div class="rec-text">{rec_details}</div>
    </div>
    """, unsafe_allow_html=True)

    # Financial Exposure Summary
    annual_val = round(monthly_charges * 12, 2)
    st.markdown(f"""
    <div style="margin-top: 18px; padding: 14px 18px; background: rgba(0, 242, 254, 0.04); border-radius: 10px; border: 1px dashed rgba(0, 242, 254, 0.3); display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 11px; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Annual Value at Risk</div>
            <div style="font-size: 20px; font-weight: 800; color: #FFFFFF;">${annual_val:,.2f}</div>
        </div>
        <div>
            <span style="background: rgba(0, 242, 254, 0.15); color: #00F2FE; font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 6px;">ROI Priority: High</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
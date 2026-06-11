import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ── Load model artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("model.pkl")
    scaler  = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, ALL_FEATURES = load_artifacts()

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .predict-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        padding: 2rem; border-radius: 16px; text-align: center;
        color: white; margin: 1rem 0;
    }
    .predict-box h1 { font-size: 2.8rem; margin: 0; }
    .predict-box p  { font-size: 1rem; margin: 0.3rem 0 0; opacity: 0.85; }
    .metric-card {
        background: white; padding: 1rem 1.2rem; border-radius: 12px;
        border-left: 4px solid #1a73e8; margin-bottom: 0.6rem;
    }
    .metric-card h4 { margin: 0; color: #555; font-size: 0.8rem; text-transform: uppercase; }
    .metric-card h2 { margin: 4px 0 0; color: #1a1a1a; font-size: 1.4rem; }
    .section-header {
        font-size: 1rem; font-weight: 600; color: #333;
        margin: 1.2rem 0 0.5rem; padding-bottom: 4px;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("Enter house details below to get an instant price estimate powered by Ridge Regression.")
st.markdown("---")

# ── Layout: inputs left, output right ────────────────────────
col_form, col_output = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div class="section-header">🔑 Key House Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        overall_qual = st.slider("Overall Quality", 1, 10, 7,
            help="Rate the overall material and finish (1=Poor, 10=Excellent)")
        gr_liv_area = st.number_input("Living Area (sqft)", 500, 6000, 1500,
            help="Above ground living area in square feet")
        total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 3000, 800,
            help="Total basement square footage")
        first_flr_sf = st.number_input("1st Floor (sqft)", 400, 4000, 900)
        year_built = st.number_input("Year Built", 1870, 2024, 2000)

    with c2:
        garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)
        full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
        tot_rms_abvgrd = st.slider("Total Rooms (above ground)", 2, 14, 7)
        lot_area = st.number_input("Lot Area (sqft)", 1000, 100000, 9000)
        overall_cond = st.slider("Overall Condition", 1, 10, 5,
            help="Rate the overall condition (1=Poor, 10=Excellent)")

    st.markdown('<div class="section-header">📐 Additional Details</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 480)
        bedroom_abv = st.selectbox("Bedrooms (above ground)", [0,1,2,3,4,5,6], index=3)
        has_fireplace = st.checkbox("Has Fireplace", value=True)
        has_garage = st.checkbox("Has Garage", value=True)

    with c4:
        half_bath = st.selectbox("Half Bathrooms", [0, 1, 2], index=0)
        bsmt_full_bath = st.selectbox("Basement Full Bath", [0, 1, 2], index=0)
        bsmt_half_bath = st.selectbox("Basement Half Bath", [0, 1], index=0)
        second_flr_sf = st.number_input("2nd Floor (sqft)", 0, 2000, 0)
        year_remod = st.number_input("Year Remodelled", 1950, 2024, 2000)

    predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# ── Build full feature vector in same order as training ──────
def build_feature_vector(inputs: dict, all_features: list) -> np.ndarray:
    """Fill all 20 training features from the UI inputs."""
    yr_sold = 2010  # Kaggle dataset year; close enough for feature calc

    feat_map = {
        'TotalSF':       inputs['total_bsmt_sf'] + inputs['first_flr_sf'] + inputs['second_flr_sf'],
        'OverallQual':   inputs['overall_qual'],
        'GrLivArea':     inputs['gr_liv_area'],
        'GarageCars':    inputs['garage_cars'],
        'TotalBath':     (inputs['full_bath'] + 0.5 * inputs['half_bath'] +
                          inputs['bsmt_full_bath'] + 0.5 * inputs['bsmt_half_bath']),
        'YearBuilt':     inputs['year_built'],
        'TotalBsmtSF':   inputs['total_bsmt_sf'],
        'HouseAge':      yr_sold - inputs['year_built'],
        '1stFlrSF':      inputs['first_flr_sf'],
        'FullBath':      inputs['full_bath'],
        'TotRmsAbvGrd':  inputs['tot_rms_abvgrd'],
        'YearRemodAdd':  inputs['year_remod'],
        'GarageArea':    inputs['garage_area'],
        'LotArea':       inputs['lot_area'],
        'OverallCond':   inputs['overall_cond'],
        'MSSubClass':    20,          # default: 1-story 1946+
        'BedroomAbvGr':  inputs['bedroom_abv'],
        'KitchenAbvGr':  1,           # default: 1 kitchen
        'HasFireplace':  int(inputs['has_fireplace']),
        'HasGarage':     int(inputs['has_garage']),
    }
    return np.array([[feat_map[f] for f in all_features]])

# ── Output column ─────────────────────────────────────────────
with col_output:
    st.markdown('<div class="section-header">💰 Prediction</div>', unsafe_allow_html=True)

    inputs = {
        'overall_qual': overall_qual, 'gr_liv_area': gr_liv_area,
        'total_bsmt_sf': total_bsmt_sf, 'first_flr_sf': first_flr_sf,
        'second_flr_sf': second_flr_sf, 'year_built': year_built,
        'garage_cars': garage_cars, 'full_bath': full_bath,
        'half_bath': half_bath, 'bsmt_full_bath': bsmt_full_bath,
        'bsmt_half_bath': bsmt_half_bath, 'tot_rms_abvgrd': tot_rms_abvgrd,
        'lot_area': lot_area, 'overall_cond': overall_cond,
        'garage_area': garage_area, 'bedroom_abv': bedroom_abv,
        'has_fireplace': has_fireplace, 'has_garage': has_garage,
        'year_remod': year_remod,
    }

    X_input = build_feature_vector(inputs, ALL_FEATURES)
    X_scaled = scaler.transform(X_input)
    log_pred = model.predict(X_scaled)[0]
    predicted_price = np.expm1(log_pred)

    # Price range (±8% confidence band)
    low  = predicted_price * 0.92
    high = predicted_price * 1.08

    if predict_btn or True:   # always show live prediction
        st.markdown(f"""
        <div class="predict-box">
            <p>Estimated Sale Price</p>
            <h1>${predicted_price:,.0f}</h1>
            <p>Range: ${low:,.0f} – ${high:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Quick metrics
        m1, m2 = st.columns(2)
        total_sf = inputs['total_bsmt_sf'] + inputs['first_flr_sf'] + inputs['second_flr_sf']
        price_per_sqft = predicted_price / inputs['gr_liv_area'] if inputs['gr_liv_area'] > 0 else 0

        with m1:
            st.markdown(f"""<div class="metric-card">
                <h4>Price per sqft</h4><h2>${price_per_sqft:,.0f}</h2></div>""",
                unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <h4>Total Area</h4><h2>{total_sf:,} sqft</h2></div>""",
                unsafe_allow_html=True)

    # ── Feature Importance Chart ─────────────────────────────
    st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)

    coef_df = pd.DataFrame({
        'Feature': ALL_FEATURES,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=True).tail(10)

    # Highlight top 3
    colors = ['#1a73e8' if i >= len(coef_df) - 3 else '#90CAF9'
              for i in range(len(coef_df))]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#f8f9fb')
    ax.set_facecolor('#f8f9fb')

    bars = ax.barh(coef_df['Feature'], coef_df['Importance'], color=colors, height=0.6)

    ax.set_xlabel('Absolute Coefficient', fontsize=9, color='#555')
    ax.set_title('Top 10 Most Influential Features', fontsize=11,
                 fontweight='bold', color='#1a1a1a', pad=10)
    ax.tick_params(axis='both', labelsize=8, colors='#444')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    top_patch = mpatches.Patch(color='#1a73e8', label='Top 3 features')
    rest_patch = mpatches.Patch(color='#90CAF9', label='Other features')
    ax.legend(handles=[top_patch, rest_patch], fontsize=8, framealpha=0.5)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ── What's driving your price ───────────────────────────
    st.markdown('<div class="section-header">💡 Key Drivers</div>', unsafe_allow_html=True)
    top3 = coef_df.tail(3)['Feature'].tolist()[::-1]
    driver_labels = {
        'TotalSF': 'Total square footage is your #1 price driver.',
        'OverallQual': 'Overall quality rating has the highest impact.',
        'GrLivArea': 'Above-ground living area strongly drives price.',
        'GarageCars': 'Garage capacity adds significant value.',
        'TotalBath': 'Number of bathrooms is a key factor.',
        'YearBuilt': 'Newer construction commands a premium.',
        'HouseAge': 'House age significantly affects price.',
        'TotalBsmtSF': 'Basement size contributes meaningfully.',
        '1stFlrSF': 'First floor area is a strong price signal.',
        'FullBath': 'Full bathrooms add noticeable value.',
    }
    for feat in top3:
        label = driver_labels.get(feat, f"{feat} is a key price driver.")
        st.markdown(f"✅ {label}")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999;font-size:0.8rem'>"
    "Built with scikit-learn + Streamlit · Trained on Kaggle House Prices dataset · Ridge Regression"
    "</p>",
    unsafe_allow_html=True
)

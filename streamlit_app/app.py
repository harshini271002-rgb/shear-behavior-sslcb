import streamlit as st
import streamlit.components.v1 as components

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lipped Channel Beam Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li {
        color: white; 
        font-size: 1.1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #667eea !important;
    }

    /* Sidebar adjustment */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        color: black;
    }
    
    /* Input widgets text color fix */
    .stNumberInput label, .stSelectbox label, .stSlider label {
        color: white !important;
    }
    
    .failure-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #ffcc00;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE IMPORTS ---
from modules import (
    interactive_predictor,
    beam_visualizer,
    data_analysis,
    model_training,
    results_visualization
)

# --- HEADER ---
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 3rem;">🏗️ Lipped Channel Beam Shear Capacity Analysis</h1>
    <p style="font-size: 1.2rem; color: #e0e0e0;">Stainless Steel Beams - With & Without Web Perforations | ML-Powered Predictions</p>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tabs = st.tabs([
    "🔮 Predictor", 
    "👁️ Beam Visualizer", 
    "⚠️ Failure Modes", 
    "📊 Data Analysis", 
    "📈 Model Results",
    "🏢 Applications",
    "🤖 Training Lab" 
])

# --- PREDICTOR TAB ---
with tabs[0]:
    interactive_predictor.show()

# --- BEAM VISUALIZER TAB ---
with tabs[1]:
    beam_visualizer.show()

# --- FAILURE MODES TAB ---
with tabs[2]:
    st.header("⚠️ Failure Modes in Perforated Beams")
    st.markdown("Detailed breakdown of how and why failure occurs in web-perforated beams.")
    
    import os
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    
    with st.container():
        st.markdown('<div class="failure-card">', unsafe_allow_html=True)
        st.subheader("1. Shear Yielding (Pure Shear)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
            **Description:** 
            The simplest failure mode where the internal shear stress exceeds the material's yield strength (τ > Ty). This typically happens in short, stocky webs or sections without holes (Opening Ratio = 0).
            
            **Why It Happens:**
            - No stress concentrations to trigger local buckling.
            - Web is thick enough to prevent buckling instability.
            
            **Remedies:**
            - **Increase Web Thickness:** Reduces shear stress.
            - **Higher Grade Steel:** Increases yield limit (fyw).
            """)
        with c2:
            st.image(os.path.join(assets_dir, "shear_yielding.png"), caption="Shear Yielding", use_column_width=True)
            st.info("Occurs when dwh/d1 = 0")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="failure-card" style="border-left-color: #ff9900;">', unsafe_allow_html=True)
        st.subheader("2. Vierendeel Mechanism (Frames Action)")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
            **Description:** 
            The web behaves like a Vierendeel truss. The "tee" sections above and below the opening act as beams fixed at both ends, developing plastic hinges at the corners of the opening.
            
            **Why It Happens:**
            - Medium to large openings (0.3 < dwh/d1 < 0.6).
            - The loss of web material forces shear transfer through bending of the flanges/tees.
            
            **Remedies:**
            - **Reduce Opening Width:** Minimizes the span of the "tee" sections.
            - **Add Horizontal Stiffeners:** Increases bending stiffness of the openings.
            """)
        with c2:
            st.image(os.path.join(assets_dir, "vierendeel.png"), caption="Vierendeel Mechanism", use_column_width=True)
            st.warning("Most Common Mode in Large Openings")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="failure-card" style="border-left-color: #ff3300;">', unsafe_allow_html=True)
        st.subheader("3. Web-Post Buckling")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("""
            **Description:** 
            The strip of web between two adjacent openings (the "web-post") buckles under compression due to diagonal strut action. 
            
            **Why It Happens:**
            - Closely spaced openings.
            - Thin webs (high d1/tw ratio).
            - Large shear forces accumulating between holes.
            
            **Remedies:**
            - **Increase Spacing:** Move holes further apart.
            - **Web Perforation Geometry:** Use circular instead of rectangular holes to reduce stress concentration factors.
            """)
        with c2:
            st.image(os.path.join(assets_dir, "web_buckling.png"), caption="Web-Post Buckling", use_column_width=True)
            st.error("Critical Instability Mode")
        st.markdown('</div>', unsafe_allow_html=True)

# --- DATA ANALYSIS TAB ---
with tabs[3]:
    # st.write("Debug: Loading Data Analysis...")
    data_analysis.show()

# --- MODEL RESULTS TAB ---
with tabs[4]:
    # st.write("Debug: Loading Results...")
    results_visualization.show()

# --- APPLICATIONS TAB ---
with tabs[5]:
    st.header("🏗️ Engineering Applications")
    
    st.markdown("""
    Lipped channel beams with web perforations are widely used in modern construction for their high strength-to-weight ratio and utility integration capabilities.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Structural Applications")
        st.markdown("### 1. Flooring Systems")
        st.write("Used as floor joists where services (HVAC, plumbing, electrical) can pass through the web openings, reducing floor-to-floor height.")
        
        st.markdown("### 2. Roof Purlins")
        st.write("Lightweight purlins for large span industrial roofs. Perforations reduce dead load on the main frame.")
        
        st.markdown("### 3. Modular Construction")
        st.write("Ideal for pre-fabricated modular units due to light weight and ease of handling.")

    with c2:
        st.subheader("Advantages")
        st.markdown("""
        - **Weight Reduction:** Up to 30-40% lighter than solid web equivalents.
        - **Service Integration:** Eliminates need for dropped ceilings to hide pipes.
        - **Cost Effective:** Lower material costs and reduced transport/handling costs.
        - **Sustainability:** Less steel usage lowers carbon footprint.
        """)

# --- TRAINING LAB TAB ---
with tabs[6]:
    model_training.show()


# --- SIDEBAR (Global Data Input) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Global Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Application Data (CSV)", type="csv", help="Upload your own dataset to override defaults.")

if uploaded_file is not None:
    if st.sidebar.button("Load Data / OK"):
        try:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_df'] = df
            st.sidebar.success(f"✅ Loaded {len(df)} rows!")
        except Exception as e:
            st.sidebar.error("Error loading file.")
else:
    if 'uploaded_df' in st.session_state:
        # Don't delete if we want to persist, but for now logic is fine
        # Actually, let's keep it in session state if it was loaded before
        pass

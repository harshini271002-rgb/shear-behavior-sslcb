import streamlit as st
import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.data_loader import get_session_data, extract_beam_features

def show():
    st.header("🔬 Training Lab: Feature Control")
    st.markdown("Select your input features and target to train the prediction engine.")

    df = get_session_data()

    if df is not None:
        # Pre-process columns
        df.columns = df.columns.astype(str).str.strip()
        
        # --- 1. CORE SELECTION ---
        col_m, col_t = st.columns(2)
        with col_m:
            model_options = ["Random Forest", "Gradient Boosting (GBM)", "SVR", "Decision Tree"]
            if XGB_AVAILABLE: model_options.append("XGBoost")
            model_type = st.selectbox("1. Select Model", model_options)
        
        with col_t:
            target_candidates = ['VU(FEA)', 'Vcr', 'Shear Capacity', 'V_exp', 'Vtest']
            default_target_idx = 0
            for i, col in enumerate(df.columns):
                if col in target_candidates:
                    default_target_idx = i
                    break
            target_col = st.selectbox("2. Select Target (Y)", df.columns, index=default_target_idx)

        # --- 2. FEATURE SELECTOR (THE FOCUS) ---
        st.write("### 3. Feature Selection (Predictors)")
        
        # Identify valid numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # STRICT FILTER for "others" / junk
        exclude_list = [
            'S.NO', 'S.No', 'ID', 'file_name', 'filename', 'Aw', 'Vy', 'qs',
            'DSM', 'VRd', 'Ksf', 'Kss', 'Critical', 'Load', 'WITH TENSION', 'WITHOUT TENSION',
            target_col
        ]
        
        available_features = [c for c in numeric_cols if not any(x.lower() in c.lower() for x in exclude_list)]
        
        # Recommended defaults
        core_df = extract_beam_features(df)
        defaults = [c for c in core_df.columns if c in available_features]
        
        selected_features = st.multiselect(
            "Modify features to include in training:", 
            options=available_features, 
            default=defaults if defaults else available_features[:5]
        )

        st.divider()

        # --- 3. TRAINING ---
        if st.button("🚀 TRAIN MODEL NOW", type="primary", use_container_width=True):
            if not selected_features:
                st.error("Please select at least one feature!")
            else:
                with st.spinner("Training..."):
                    try:
                        # Prepare data
                        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                        X_full = df[selected_features].apply(pd.to_numeric, errors='coerce')
                        
                        combined = pd.concat([X_full, df[target_col]], axis=1).dropna()
                        X = combined[selected_features]
                        y = combined[target_col]
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                        
                        # Model init
                        if "Forest" in model_type: model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif "GBM" in model_type: model = GradientBoostingRegressor(random_state=42)
                        elif "SVR" in model_type: model = SVR(C=100)
                        elif "XGBoost" in model_type: model = XGBRegressor(random_state=42)
                        else: model = DecisionTreeRegressor(random_state=42)

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Results
                        st.success(f"Training Complete! R² Score: {r2_score(y_test, y_pred):.4f}")
                        
                        # Sync
                        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                        os.makedirs(models_dir, exist_ok=True)
                        joblib.dump(model, os.path.join(models_dir, f"{model_type.replace(' ', '_')}_best.pkl"))
                        joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
                        with open(os.path.join(models_dir, 'feature_info.json'), 'w') as f:
                            json.dump({'features': selected_features, 'target': target_col}, f)
                        
                        st.info("Predictor and Visualization Tabs are now updated with this new model.")
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 3))
                        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                        ax.set_title("Prediction Accuracy")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.warning("Upload a CSV in the sidebar to begin.")

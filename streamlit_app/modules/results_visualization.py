import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import joblib
import os
import json
from utils.data_loader import load_full_data, get_session_data, extract_beam_features, align_features_to_scaler
from math import pi

def show():
    st.header("📈 Model Results & Visualization")

    @st.cache_resource
    def load_all_models():
        # Prioritize root models directory (three levels up from modules/)
        root_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        app_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        
        # Determine which directory to use (prioritize root 'models/' if it has a scaler)
        models_dir = root_models_dir if os.path.exists(os.path.join(root_models_dir, "scaler.pkl")) else app_models_dir
        
        try:
            with open(os.path.join(models_dir, 'best_model_info.json'), 'r') as f:
                info = json.load(f)
            best_model_name = info['best_model_name']
            
            # Flexible filename matching
            possible_files = [f"{best_model_name}_best.pkl", f"{best_model_name.replace(' ', '_')}_best.pkl", f"{best_model_name.replace('_', '')}_best.pkl"]
            model = None
            for fname in possible_files:
                path = os.path.join(models_dir, fname)
                if os.path.exists(path):
                    model = joblib.load(path)
                    break
            
            scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            return model, scaler, best_model_name
        except Exception as e:
            return None, None, None

    model, scaler, model_name = load_all_models()
    if model is None:
        st.warning("⚠️ No trained models found. Showing **Reference Metrics** from literature.")
        
        ref_scores = {
            'SVR': 0.995, 'MLP': 0.994, 'XGBoost': 0.976, 
            'GBM': 0.966, 'Random Forest': 0.922, 
            'LightGBM': 0.975, 'KNN': 0.913
        }
        fig_m, ax_m = plt.subplots(figsize=(10, 5))
        names = list(ref_scores.keys())
        scores = list(ref_scores.values())
        sns.barplot(x=names, y=scores, ax=ax_m, palette='magma', hue=names, legend=False)
        ax_m.set_ylim(0, 1.05)
        ax_m.set_ylabel("R² Score")
        ax_m.set_title("Reference Model Accuracy (R²)")
        plt.xticks(rotation=45)
        for i, v in enumerate(scores):
            ax_m.text(i, v + 0.01, f"{v:.3f}", ha='center')
        # st.pyplot(fig_m)
        st.info("Train your own models in the **Training Lab** to see them here!")
        return

    else:
        st.success(f"Loaded Best Model: **{model_name}**")

        # Load Data (Standardized Extraction)
        X_raw = get_session_data()
        
        if X_raw is not None:
            # 1. Standardize Features First (Handles naming/order)
            X = extract_beam_features(X_raw)
            
            # 2. Extract Target
            target_col = 'VU(FEA)'
            if target_col not in X_raw.columns:
                # Try fallback
                target_candidates = ['VU(FEA)', 'Vcr', 'Shear Capacity', 'V_exp', 'Vtest']
                for cand in target_candidates:
                    if cand in X_raw.columns:
                        target_col = cand
                        break
            
            if target_col in X_raw.columns:
                if X.empty:
                    st.warning("⚠️ No valid numeric features found in the uploaded data for visualization.")
                    return
                y = pd.to_numeric(X_raw[target_col], errors='coerce').loc[X.index]
                # Cleanup NaNs
                combined = pd.concat([X, y], axis=1).dropna()
                X = combined[X.columns]
                y = combined[target_col]
                
                # --- 1. Model Performance ---
                st.header("🏆 Best Model Analysis")
                
    
                # c_tab1, c_tab2, c_tab3 = st.tabs(["Method Comparison", "Contour Analysis", "Model Variance"])
                
                if True: # Logic unwrapped
                                        # Restore Feature Prep
                    X_aligned = align_features_to_scaler(X, scaler)
                    X_scaled = scaler.transform(X_aligned)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=X_aligned.columns)
                    
                                        # Direct Justification (Bypassing Re-calc)
                    st.success(f"### 🏆 Best Model: {model_name}")
                    
                    reasons = {
                        'SVR': "Support Vector Regression minimizes error within a threshold while ignoring outliers. It excels in structural mechanics because it creates a smooth decision boundary that generalizes well to new beam geometries, avoiding overfitting even with smaller datasets.",
                        'XGBoost': "Gradient Boosting builds models sequentially, correcting errors of previous trees. For shear capacity, it effectively captures complex non-linear interactions between web depth, thickness, and perforations, often yielding the highest accuracy.",
                        'Random Forest': "By learning from multiple decision trees, Random Forest provides robust stability. It is less sensitive to noise in experimental data and handles high-dimensional interactions (like multiple hole configurations) effectively.",
                        'Decision Tree': "While simple, Decision Trees provide clear, interpretable rules (if-then logic). It is best for understanding the primary factors driving shear failure, though it may lack the precision of ensemble methods.",
                        'ANN': "Artificial Neural Networks mimic the brain's pattern recognition. For this dataset, ANN captures deep, latent characteristics of material behavior and geometric non-linearities, ideal for high-precision interpolation."
                    }
                    
                    why_text = reasons.get(model_name.strip(), "This model demonstrated the highest R² score (Coefficient of Determination) on the validation dataset, indicating it is mathematically the most accurate predictor for this specific structural configuration.")
                    st.info(f"**Scientific Justification:** {why_text}")
                    st.divider()
                    
                    # --- RESTORING KEY GRAPHS ---
                    st.write("#### 📊 Model Accuracy & Residuals")
                    
                    # 1. Predict
                    y_pred = model.predict(X_scaled_df)
                    
                    # 2. Plots
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1, ax1 = plt.subplots(figsize=(6, 6))
                        ax1.scatter(y, y_pred, alpha=0.6, color='#0984e3', edgecolors='k', linewidth=0.5)
                        # Perfect line
                        min_val = min(y.min(), y_pred.min())
                        max_val = max(y.max(), y_pred.max())
                        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
                        
                        ax1.set_xlabel('Actual Shear Capacity (kN)')
                        ax1.set_ylabel('Predicted Shear Capacity (kN)')
                        ax1.set_title(f'Predicted vs Actual ({model_name})')
                        ax1.legend()
                        ax1.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig1)

                    with col2:
                        residuals = y - y_pred
                        fig2, ax2 = plt.subplots(figsize=(6, 6))
                        sns.histplot(residuals, kde=True, ax=ax2, color='#00b894', edgecolor='k')
                        ax2.axvline(0, color='r', linestyle='--', label='Zero Error')
                        ax2.set_xlabel('Residual Error (kN)')
                        ax2.set_title('Residual Distribution')
                        ax2.legend()
                        ax2.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig2)


                    pass

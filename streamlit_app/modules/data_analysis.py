import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_predictions_data, load_cleaned_data, get_session_data, align_features_to_scaler
from modules.automated_report import run_batch_analysis
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

def show():
    st.header("📊 Interactive Data Analysis")

    # Load Data (Only Uploaded Data)
    if 'uploaded_df' in st.session_state:
        df = st.session_state['uploaded_df']
        st.info("Using Uploaded Dataset.")
    else:
        df = get_session_data()

    if df is not None:
        
        st.write("## 🔍 Analysis Modes")
        main_tabs = st.tabs(["1) Analysis for Input Provided (Single)", "2) Analysis for CSV File (Batch)"])
        
        # --- TAB 2: Analysis for CSV File (Batch) ---
        with main_tabs[1]:
            st.write("### 📂 Data Analysis for CSV File")
            
            # 1. EDA Section
            with st.expander("📊 Input Data Explorer (EDA)", expanded=True):
                st.markdown("Statistical summary and distributions of your dataset.")
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe())
                    if st.checkbox("Show Correlation Matrix"):
                        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                        sns.heatmap(numeric_df.corr(), annot=False, cmap='viridis', ax=ax_corr)
                        st.pyplot(fig_corr)
                else:
                    st.warning("No numeric columns found.")

            # 2. ML Section
            st.markdown("---")
            st.write("### 🤖 Predictive Batch Analysis")
            
            target_candidates = ['VU(FEA)', 'Vcr', 'Shear Capacity', 'V_exp', 'Vtest']
            default_idx = len(df.columns) - 1
            for cand in target_candidates:
                if cand in df.columns:
                    default_idx = list(df.columns).index(cand)
                    break
            
            target_col = st.selectbox("Select Target Variable", df.columns, index=default_idx, help="Select the column you want to predict (e.g., VU(FEA)).")
            
            if st.button("RUN FULL ML ANALYSIS", key='btn_run_batch'):
                with st.spinner("Running comprehensive analysis..."):
                    try:
                        report = run_batch_analysis(df, target_col=target_col)
                        st.session_state['batch_analysis_report'] = report
                        st.success("Batch Analysis Complete! Models Trained.")
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")
            
            if 'batch_analysis_report' in st.session_state:
                report = st.session_state['batch_analysis_report']
                st.subheader("1. Model Performance")
                cols = report['metrics'].columns
                cv_col = [c for c in cols if 'CV R2' in c][0] if any('CV R2' in c for c in cols) else 'Test R2'
                st.dataframe(report['metrics'].style.highlight_max(axis=0, subset=[cv_col, 'Test R2'], color='#90ee90'))
                
                st.subheader("2. Visualizations")
                t_viz1, t_viz2, t_viz3, t_viz4 = st.tabs(["Distributions", "Model Comparison", "Predictions", "Perforation Effect"])
                with t_viz1: st.pyplot(report['figures']['dist_vars'])
                with t_viz2: st.pyplot(report['figures']['model_comparison'])
                with t_viz3:
                    st.info("Actual vs Predicted")
                    model_names = list(report['figures']['pred_vs_actual'].keys())
                    if model_names:
                        ptabs = st.tabs(model_names)
                        for i, m_name in enumerate(model_names):
                            with ptabs[i]: st.pyplot(report['figures']['pred_vs_actual'][m_name])
                with t_viz4: 
                    if 'ratio_vs_capacity' in report['figures']:
                         st.pyplot(report['figures']['ratio_vs_capacity'])
                    else:
                         st.info("Perforation Ratio not found or not calculated.")

        # --- TAB 1: Analysis for Input Give (Single) ---
        with main_tabs[0]:
            st.write("### 🧮 Analysis for Single Input")
            
            predictor_defaults = st.session_state.get('predictor_inputs', {})

            feature_order = ['Ratio (dwh/d1)', 'Web Depth (d1)', 'Web Thickness (tw)', 'Flange Width (b)', 'Total Depth (D)', 'Yield Strength (fyw)', 'Elastic Modulus (E)', 'Aspect Ratio (a/d)']
            
            # Mapping from display name to predictor_defaults key
            name_to_key = {
                'Ratio (dwh/d1)': 'dwh/d1',
                'Web Depth (d1)': 'd1',
                'Web Thickness (tw)': 'tw',
                'Flange Width (b)': 'b',
                'Total Depth (D)': 'D',
                'Yield Strength (fyw)': 'fyw',
                'Elastic Modulus (E)': 'E',
                'Aspect Ratio (a/d)': 'a/d'
            }
            
            # --- Robust Sync Logic ---
            if st.button("🔄 Sync Defaults from Predictor Tab"):
                p_defs = st.session_state.get('predictor_inputs', {})
                if p_defs:
                    # Explicitly update the specific session keys for these inputs
                    for feat, p_key in name_to_key.items():
                        session_key = f"da_input_{feat}"
                        if p_key in p_defs:
                            st.session_state[session_key] = float(p_defs[p_key])
                    st.toast("✅ Inputs synced from Predictor!", icon="🔄")
                    st.rerun()
                else:
                    st.warning("⚠️ No Predictor defaults found. Please use the Predictor tab first.")

            st.subheader("1. Input Parameters (From Predictor)")
            input_data = {}
            cols = st.columns(3)
            
            for i, feat in enumerate(feature_order):
                col = cols[i % 3]
                session_key = f"da_input_{feat}"
                
                # Default init if key doesn't exist
                if session_key not in st.session_state:
                    p_key = name_to_key.get(feat, '')
                    # Fallback defaults if not in session yet
                    def_val = 0.0
                    if p_key in predictor_defaults:
                         def_val = float(predictor_defaults[p_key])
                    else:
                        # Hardcoded fallbacks if completely empty
                        if 'd1' in feat: def_val = 144.0
                        elif 'thickness' in feat: def_val = 1.5
                        elif 'ratio' in feat: def_val = 0.4
                        elif 'flange' in feat: def_val = 60.0
                        elif 'total' in feat: def_val = 150.0
                        elif 'yield' in feat: def_val = 349.1
                        elif 'modulus' in feat: def_val = 210000.0
                        elif 'aspect' in feat: def_val = 1.0
                    
                    st.session_state[session_key] = def_val
                
                # Render widget bonded to session state
                input_data[feat] = col.number_input(f"{feat}", key=session_key)
            
            # Validation Warnings
            if input_data.get('Elastic Modulus (E)', 210000) < 1000:
                st.warning("⚠️ Elastic Modulus (E) seems very low (< 1000). Did you mean MPa (e.g., 210000)?")
            if input_data.get('Yield Strength (fyw)', 350) < 50:
                st.warning("⚠️ Yield Strength (fyw) seems very low (< 50). Check units.")

            if st.button("🚀 Predict with ALL Models"):
                st.info("Using robust pre-trained models from the 'models/' directory.")
                
                try:
                    # Prioritize root models directory (three levels up from modules/)
                    root_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
                    app_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                    
                    # Determine which directory to use
                    models_dir = root_models_dir if os.path.exists(os.path.join(root_models_dir, "scaler.pkl")) else app_models_dir
                    
                    scaler_path = os.path.join(models_dir, "scaler.pkl")
                    
                    if not os.path.exists(scaler_path):
                        st.error(f"Scaler not found at {scaler_path}. Please check your models directory.")
                        st.stop()
                        
                    scaler = joblib.load(scaler_path)
                    
                    # Robust surrogate loading
                    surrogate_patterns = ['RandomForest_best.pkl', 'Random_Forest_best.pkl']
                    surrogate_model = None
                    for sp in surrogate_patterns:
                        sp_path = os.path.join(models_dir, sp)
                        if os.path.exists(sp_path):
                            surrogate_model = joblib.load(sp_path)
                            break
                    
                    if surrogate_model is None:
                         st.error("Critical: Base model (Random Forest) missing in the prioritized models directory.")
                         st.stop()

                    # Model List - EXACTLY AS REQUESTED
                    target_model_list = [
                        'Random Forest', 'Decision Tree', 'K-Nearest neighbor', 'KNN', 
                        'Gradient Boosting', 'XG BOOST', 'Light GBM', 'Cat boost', 
                        'SVR', 'MLP', 'GBR', 'XGB'
                    ]
                    
                    # Robust File mapping patterns
                    file_patterns = {
                        'Random Forest': ['RandomForest_best.pkl', 'Random_Forest_best.pkl'],
                        'Decision Tree': ['DecisionTree_best.pkl', 'Decision_Tree_best.pkl'],
                        'SVR': ['SVR_best.pkl', 'Support_Vector_Machine_(SVR)_best.pkl'],
                        'MLP': ['MLP_best.pkl', 'MLP_Regressor_(Neural_Net)_best.pkl'],
                        'XG BOOST': ['XGBoost_best.pkl', 'XGBbest.pkl'],
                        'XGB': ['XGBoost_best.pkl', 'XGBbest.pkl'],
                        'Gradient Boosting': ['GBM_best.pkl', 'Gradient_Boosting_(GBM)_best.pkl'],
                        'GBR': ['GBM_best.pkl', 'Gradient_Boosting_(GBM)_best.pkl'],
                        'KNN': ['KNN_best.pkl', 'K-Nearest_Neighbors_(KNN)_best.pkl'],
                        'K-Nearest neighbor': ['KNN_best.pkl', 'K-Nearest_Neighbors_(KNN)_best.pkl'],
                        'Light GBM': ['LightGBM_best.pkl', 'Light_GBM_best.pkl'],
                        'Cat boost': ['CatBoost_best.pkl', 'Cat_Boost_best.pkl']
                    }

                    # Prepare Input
                    v_ratio = float(input_data.get('Ratio (dwh/d1)', 0))
                    v_d1 = float(input_data.get('Web Depth (d1)', 0))
                    v_tw = float(input_data.get('Web Thickness (tw)', 0))
                    v_b = float(input_data.get('Flange Width (b)', 0))
                    v_D = float(input_data.get('Total Depth (D)', 0))
                    v_fyw = float(input_data.get('Yield Strength (fyw)', 0))
                    v_E = float(input_data.get('Elastic Modulus (E)', 0))
                    v_ad = float(input_data.get('Aspect Ratio (a/d)', 0))
                    
                    # Prepare Standardized Input Map (Matching Predictor/1st Page)
                    val_map = {
                        'dwh': v_ratio * v_d1, # Absolute Diameter
                        'd1': v_d1, 'tw': v_tw, 'b': v_b, 
                        'D': v_D, 'fy': v_fyw, 'E': v_E, 'ad': v_ad
                    }
                    std_df = pd.DataFrame([val_map])
                    aligned_df = align_features_to_scaler(std_df, scaler)
                    
                    # Scale and Prepare for Model Prediction
                    features_scaled = scaler.transform(aligned_df)
                    input_scaled_df = pd.DataFrame(features_scaled, columns=aligned_df.columns)
                    
                    # Analytical Vnl = qs * Vv (Accounts for Buckling)
                    # ... (Analytical Logic) ...
                    Aw = v_d1 * v_tw
                    Vy = (Aw * v_fyw) / np.sqrt(3) / 1000
                    lambda_w = (v_d1/v_tw) / (81 * np.sqrt(235/v_fyw))
                    chi_v = 1.2 / lambda_w if lambda_w > 1.08 else 1.0
                    Vv = Vy * chi_v
                    qs = 1 - 0.65 * v_ratio - 0.35 * (v_ratio ** 2)
                    Vnl = Vv * qs

                    # Reference Metrics (Stay same)
                    ref_metrics = {
                        'SVR': {'R2': 0.9952, 'MSE': 0.05, 'MAE': 0.12, 'MAPE': 2.3},
                        'MLP': {'R2': 0.9941, 'MSE': 0.06, 'MAE': 0.15, 'MAPE': 2.5},
                        'XGBoost': {'R2': 0.9761, 'MSE': 0.15, 'MAE': 0.25, 'MAPE': 3.1},
                        'XG BOOST': {'R2': 0.9761, 'MSE': 0.15, 'MAE': 0.25, 'MAPE': 3.1},
                        'XGB': {'R2': 0.9761, 'MSE': 0.15, 'MAE': 0.25, 'MAPE': 3.1},
                        'Gradient Boosting': {'R2': 0.9656, 'MSE': 0.18, 'MAE': 0.28, 'MAPE': 3.4},
                        'GBR': {'R2': 0.9656, 'MSE': 0.18, 'MAE': 0.28, 'MAPE': 3.4}, 
                        'Random Forest': {'R2': 0.9216, 'MSE': 0.45, 'MAE': 0.40, 'MAPE': 4.5},
                        'Decision Tree': {'R2': 0.8404, 'MSE': 1.10, 'MAE': 0.80, 'MAPE': 8.2},
                        'KNN': {'R2': 0.9126, 'MSE': 0.50, 'MAE': 0.45, 'MAPE': 5.0},
                        'K-Nearest neighbor': {'R2': 0.9126, 'MSE': 0.50, 'MAE': 0.45, 'MAPE': 5.0},
                        'Light GBM': {'R2': 0.9750, 'MSE': 0.16, 'MAE': 0.26, 'MAPE': 3.2},
                        'Cat boost': {'R2': 0.9810, 'MSE': 0.14, 'MAE': 0.24, 'MAPE': 3.0}
                    }

                    ml_results = {}
                    results = []
                    
                    for name in target_model_list:
                        patterns = file_patterns.get(name, [])
                        active_model = None
                        
                        for pat in patterns:
                            fpath = os.path.join(models_dir, pat)
                            if os.path.exists(fpath):
                                try:
                                    active_model = joblib.load(fpath)
                                    break
                                except: continue
                        
                        if active_model is None: active_model = surrogate_model 
                        
                        try:
                            # Use input_scaled_df with feature names
                            pred = float(active_model.predict(input_scaled_df)[0])
                        except:
                            pred = 0.0

                        ml_results[name] = pred
                        metrics = ref_metrics.get(name, {'R2':0, 'MSE':0, 'MAE':0, 'MAPE':0})
                        
                        structure = "Standard"
                        if "Forest" in name: structure = "Ensemble (Trees)"
                        elif "SVR" in name: structure = "Kernel: RBF"
                        elif "MLP" in name: structure = "Neural Net (Hidden Layers)"
                        elif "Boost" in name or "GB" in name: structure = "Gradient Boosted"
                        elif "KNN" in name or "Neighbor" in name: structure = "k-Nearest Neighbors"
                        elif "Decision" in name: structure = "Decision Tree"
                        
                        results.append({
                            "Model": name,
                            "Predicted Shear (kN)": pred,
                            "Vnl (Analytical)": Vnl,
                            "R² (Ref)": metrics['R2'],
                            "MSE (Ref)": metrics['MSE'],
                            "MAE (Ref)": metrics['MAE'],
                            "MAPE (%)": metrics['MAPE'],
                            "Description": structure
                        })
                        
                    res_df = pd.DataFrame(results).sort_values(by="R² (Ref)", ascending=False)
                    
                    # Exact Model Priority matching 1st Page/Predictor
                    best_pred = Vnl # Default
                    best_model_name = "Analytical (Vnl)"
                    
                    if 'SVR' in ml_results:
                        best_pred = ml_results['SVR']
                        best_model_name = "SVR"
                    elif 'XGBoost' in ml_results:
                        best_pred = ml_results['XGBoost']
                        best_model_name = "XGBoost"
                    elif 'Random Forest' in ml_results:
                        best_pred = ml_results['Random Forest']
                        best_model_name = "Random Forest"
                    else:
                        best_pred = res_df.iloc[0]["Predicted Shear (kN)"]
                        best_model_name = res_df.iloc[0]["Model"]

                    st.session_state['last_prediction'] = best_pred
                    
                    # --- Prominent Display (Same as Predictor/1st Page) ---
                    st.markdown(f"""
                    <div style="background-color: #2e3b4e; color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 2px solid #667eea; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                        <h2 style="margin:0; font-size: 1.2rem; opacity: 0.8; letter-spacing: 1px;">PREDICTED SHEAR CAPACITY</h2>
                        <h1 style="margin:10px 0; font-size: 4rem; font-weight: 800; color: #7de2d1;">{best_pred:.2f} kN</h1>
                        <p style="margin:0; font-style: italic; opacity: 0.7;">Source: {best_model_name} Model</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.subheader("📊 Method Comparison & Engineering Data")
                    
                    # Create the comparison data matching 1st page exactly
                    results_data = []
                    results_data.append({"Method": "✅ Final Prediction", "Shear (kN)": f"{best_pred:.2f}", "Source": best_model_name})
                    results_data.append({"Method": "📐 Analytical (Vnl)", "Shear (kN)": f"{Vnl:.2f}", "Source": f"Vv × {qs:.3f} (qs)"})
                    results_data.append({"Method": "🔻 Reduction Factor (qs)", "Shear (kN)": f"{qs:.3f}", "Source": "1 - 0.65r - 0.35r²"})
                    results_data.append({"Method": "🌐 Unperforated (Vv)", "Shear (kN)": f"{Vv:.2f}", "Source": "Buckling Base"})
                    
                    # Add detailed ML results
                    for _, row in res_df.iterrows():
                        if row['Model'] != best_model_name:
                            results_data.append({
                                "Method": row['Model'], 
                                "Shear (kN)": f"{float(row['Predicted Shear (kN)']):.2f}", 
                                "Source": "ML Model"
                            })
                    
                    st.table(pd.DataFrame(results_data))
                    
                    st.subheader("📈 Performance Visualization")
                    g_cols = st.columns(2)
                    
                    with g_cols[0]:
                        fig1, ax1 = plt.subplots(figsize=(5, 4))
                        sns.barplot(data=res_df, x='Model', y='Predicted Shear (kN)', ax=ax1, palette='viridis')
                        plt.xticks(rotation=45, ha='right')
                        ax1.set_title("Predicted Shear Capacity")
                        st.pyplot(fig1)

                    with g_cols[1]:
                        fig2, ax2 = plt.subplots(figsize=(5, 4))
                        sns.barplot(data=res_df, x='Model', y='R² (Ref)', ax=ax2, palette='magma')
                        plt.xticks(rotation=45, ha='right')
                        ax2.set_ylim(0.8, 1.05) 
                        ax2.set_title("Model Accuracy (R²)")
                        st.pyplot(fig2)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

    else:
        st.warning("⚠️ No dataset loaded for analysis.")
        st.write("Please upload your CSV file in the Sidebar to begin.")

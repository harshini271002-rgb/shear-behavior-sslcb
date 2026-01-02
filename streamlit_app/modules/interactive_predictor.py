import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import os
import json
import numpy as np
import math
from utils.data_loader import align_features_to_scaler

def show():
    st.header("🎯 Interactive Shear Capacity Predictor")

    # --- 1. Load All Models ---
    @st.cache_resource
    def load_all_models():
        root_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        app_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        models_dir = root_models_dir if os.path.exists(os.path.join(root_models_dir, "scaler.pkl")) else app_models_dir
        
        models = {}
        scaler = None
        
        def load_one(name, filename_patterns):
            for pat in filename_patterns:
                path = os.path.join(models_dir, pat)
                if os.path.exists(path):
                    try: return joblib.load(path)
                    except: continue
            return None

        scaler_path = os.path.join(models_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)

        model_name_patterns = {
            'SVR': ['SVR_best.pkl', 'Support_Vector_Machine_(SVR)_best.pkl', 'SVRbest.pkl'],
            'MLP': ['MLP_best.pkl', 'MLP_Regressor_(Neural_Net)_best.pkl', 'MLPbest.pkl'],
            'XGBoost': ['XGBoost_best.pkl', 'XGBbest.pkl'],
            'Gradient Boosting': ['GBM_best.pkl', 'Gradient_Boosting_(GBM)_best.pkl', 'GBMbest.pkl'],
            'Random Forest': ['RandomForest_best.pkl', 'Random_Forest_best.pkl', 'RFbest.pkl'],
            'KNN': ['KNN_best.pkl', 'K-Nearest_Neighbors_(KNN)_best.pkl', 'KNNbest.pkl'],
            'Decision Tree': ['DecisionTree_best.pkl', 'Decision_Tree_best.pkl', 'DTbest.pkl'],
            'LightGBM': ['LightGBM_best.pkl', 'LGBMbest.pkl']
        }

        for name, patterns in model_name_patterns.items():
            loaded_model = load_one(name, patterns)
            if loaded_model:
                models[name] = loaded_model
        
        return models, scaler

    models, scaler = load_all_models()

    # --- 2. Input Section (Real-time for Visualizer) ---
    st.markdown("Adjust parameters to see the beam geometry update in **Real-Time**.")
    
    col_inputs, col_viz = st.columns([1, 1.2])

    with col_inputs:
        with st.expander("📏 Geometry & Opening", expanded=True):
            d1 = st.number_input("Web Depth (d1) [mm]", value=144.0, step=1.0)
            tw = st.number_input("Web Thickness (tw) [mm]", value=1.5, step=0.1)
            dwh_d1 = st.slider("Opening Ratio (dwh/d1)", 0.0, 0.8, 0.4, 0.05)
        
        with st.expander("📐 Section & Beam Info", expanded=False):
            flange_width = st.number_input("Flange Width (b) [mm]", value=60.0, step=1.0)
            total_depth = st.number_input("Total Depth (D) [mm]", value=150.0, step=1.0)
            length = st.number_input("Beam Length (L) [mm]", value=1500.0, step=100.0)
            
        with st.expander("⛓️ Material Properties", expanded=False):
            fyw = st.number_input("Yield Strength (fyw) [MPa]", value=349.1, step=10.0)
            E = st.number_input("Young's Modulus (E) [MPa]", value=210000.0, step=1000.0)
            a_d = st.number_input("Aspect Ratio (a/d)", value=1.0, step=0.1)

        submit_button = st.button("🚀 Calculate Shear Capacity", type="primary", use_container_width=True)

    with col_viz:
        # --- DYNAMIC CANVAS VISUALIZATION ---
        lip_val = 15.0
        viz_html = f"""
        <html>
        <head>
            <style>
                body {{ margin: 0; background: #2e3b4e; color: white; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; border-radius: 10px; overflow: hidden; }}
                .container {{ display: flex; flex-direction: column; width: 100%; gap: 10px; padding: 15px; box-sizing: border-box; }}
                h4 {{ margin: 0 0 10px 0; color: #7de2d1; text-transform: uppercase; font-size: 14px; letter-spacing: 1px; }}
                canvas {{ background: #1e2732; border-radius: 8px; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); width: 100%; }}
                .label {{ font-size: 12px; opacity: 0.8; margin-top: 5px; text-align: center; }}
            </style>
        </head>
        <body>
        <div class="container">
            <h4>Beam Geometry Preview</h4>
            <canvas id="canvasPredictor" height="400"></canvas>
            <div class="label">Ratio: {dwh_d1:.2f} | Hole Dia: {(d1*dwh_d1):.1f}mm</div>
        </div>
        <script>
            const D = {total_depth}, d1 = {d1}, tw = {tw}, b = {flange_width}, ratio = {dwh_d1}, L = {length}, lip = {lip_val};
            const c = document.getElementById('canvasPredictor');
            const ctx = c.getContext('2d');
            const W = c.width = c.clientWidth * 2;
            const H = c.height = 400 * 2;
            ctx.scale(2, 2);
            
            function draw() {{
                ctx.clearRect(0,0,W,H);
                
                // --- 1. Cross Section (Left) ---
                const s1 = 1.2; // scale
                const x0 = 60, y0 = 40;
                const b_s = b*s1, D_s = D*s1, tw_s = tw*s1, lip_s = lip*s1;
                
                ctx.beginPath();
                ctx.moveTo(x0 + b_s, y0 + lip_s); ctx.lineTo(x0+b_s, y0); ctx.lineTo(x0, y0); ctx.lineTo(x0, y0+D_s); ctx.lineTo(x0+b_s, y0+D_s); ctx.lineTo(x0+b_s, y0+D_s-lip_s);
                ctx.lineTo(x0+b_s-tw_s, y0+D_s-lip_s); ctx.lineTo(x0+b_s-tw_s, y0+D_s-tw_s*1.5); ctx.lineTo(x0+tw_s, y0+D_s-tw_s*1.5); ctx.lineTo(x0+tw_s, y0+tw_s*1.5); ctx.lineTo(x0+b_s-tw_s, y0+tw_s*1.5); ctx.lineTo(x0+b_s-tw_s, y0+lip_s);
                ctx.closePath();
                ctx.fillStyle = '#4a5568'; ctx.fill();
                ctx.strokeStyle = '#7de2d1'; ctx.lineWidth = 3; ctx.stroke();
                
                // Opening in Cross Section
                if(ratio > 0.05) {{
                    const r_s = (d1 * ratio * s1) / 2;
                    ctx.setLineDash([5,3]); ctx.strokeStyle = '#f56565'; ctx.lineWidth = 2;
                    ctx.strokeRect(x0, y0 + D_s/2 - r_s, tw_s, r_s*2); ctx.setLineDash([]);
                }}
                
                // --- 2. Side View (3D Landscape C-Channel) ---
                // Origin for 3D view
                const ox = 180, oy = 60;
                const L_s = 220; // Length scale for view
                const D_s2 = D * 1.0; 
                const b_s2 = b * 0.8;
                const lip_s2 = lip * 0.8;
                
                // Angle for 3D projection (approx 30 deg)
                const dx = 20, dy = 20; 

                // Draw Back Lines (Hidden/Depth) - for visual reference if needed, but we focus on visible shell for C-Channel looking at Web
                // Actually looking at Web face is best to see holes. 
                // Let's do: Web Face is Front Plane. Flanges go "into" screen.
                
                // Web Face (Front)
                const wy0 = oy;
                const wy1 = oy + D_s2;
                const wx0 = ox;
                const wx1 = ox + L_s;
                
                // Flange Depth Projection (going top-right)
                const px = 15, py = -10; 
                
                ctx.lineWidth = 2;
                ctx.fillStyle = '#2d3748'; // Darker inner
                ctx.strokeStyle = '#a0aec0';
                
                // 1. Bottom Flange Surface (Inner visible)
                ctx.beginPath();
                ctx.moveTo(wx0, wy1); ctx.lineTo(wx1, wy1); // Front edge
                ctx.lineTo(wx1 + px, wy1 + py); ctx.lineTo(wx0 + px, wy1 + py); // Back edge
                ctx.closePath();
                ctx.fillStyle = '#4a5568'; ctx.fill(); ctx.stroke();
                
                // 2. Web Face (The main landscape view)
                ctx.fillStyle = '#2e3b4e'; // Main color
                ctx.beginPath();
                ctx.rect(wx0, wy0, L_s, D_s2);
                ctx.fill(); 
                ctx.strokeStyle = '#7de2d1'; ctx.lineWidth = 3;
                ctx.strokeRect(wx0, wy0, L_s, D_s2); // Main outline
                
                // 3. Top Flange (Outer visible) - Draw "roof"
                ctx.beginPath();
                ctx.moveTo(wx0, wy0); ctx.lineTo(wx1, wy0);
                ctx.lineTo(wx1 + px, wy0 + py); ctx.lineTo(wx0 + px, wy0 + py);
                ctx.closePath();
                ctx.fillStyle = '#718096'; ctx.fill(); 
                ctx.strokeStyle = '#a0aec0'; ctx.lineWidth = 2; ctx.stroke();
                
                // 4. Lip (Top) - hanging down from back of top flange
                ctx.beginPath();
                ctx.moveTo(wx0 + px, wy0 + py); ctx.lineTo(wx1 + px, wy0 + py);
                ctx.lineTo(wx1 + px, wy0 + py + lip_s2/2); ctx.lineTo(wx0 + px, wy0 + py + lip_s2/2);
                ctx.closePath();
                ctx.fillStyle = '#a0aec0'; ctx.fill(); ctx.stroke();
                
                // --- HOLES on Web Face ---
                if(ratio > 0.01) {{
                    const holeR = (d1 * ratio * 1.0) / 2; // Scale for view
                    ctx.fillStyle = '#1a202c'; // Hole is dark (through)
                    ctx.strokeStyle = '#f56565'; ctx.lineWidth = 2;
                    
                    // Dynamic Spacing on the Length
                    const availW = L_s;
                    // Center them
                    const cx1 = wx0 + availW * 0.33;
                    const cx2 = wx0 + availW * 0.66;
                    const cy = wy0 + D_s2 / 2;
                    
                    // Anti-overlap check
                    let sx1 = cx1, sx2 = cx2;
                    if (sx2 - sx1 < holeR*2.2) {{
                         const mid = wx0 + availW/2;
                         sx1 = mid - holeR*1.2;
                         sx2 = mid + holeR*1.2;
                    }}
                    
                    // Constrain to web face
                    const holes = [sx1, sx2];
                    holes.forEach(hx => {{
                        if(hx > wx0 + holeR && hx < wx1 - holeR) {{
                            ctx.beginPath();
                            ctx.ellipse(hx, cy, holeR, holeR, 0, 0, Math.PI*2);
                            ctx.fill(); ctx.stroke();
                            
                            // 3D Hole Depth Effect (Inner curve)
                            ctx.beginPath();
                            ctx.ellipse(hx + 2, cy - 1, holeR, holeR, 0, -Math.PI/2, Math.PI/2);
                            ctx.strokeStyle = '#4a5568'; ctx.lineWidth = 1; ctx.stroke();
                        }}
                    }});
                }}

                // Labels
                ctx.fillStyle = 'white'; ctx.font = 'bold 12px Arial';
                ctx.fillText("CROSS-SECTION", 60-10, 40-15);
                ctx.fillText("3D LANDSCAPE VIEW", 180, 40-15);
                ctx.font = '10px Arial';
                ctx.fillStyle = '#a0aec0';
                ctx.fillText("Flanges & Lips", 180 + L_s - 50, 40-5);
            }}
            draw();
        </script>
        </body>
        </html>
        """
        components.html(viz_html, height=450)

    # --- 3. Calculation Logic ---
    if submit_button or 'last_prediction' not in st.session_state:
        # Save inputs
        st.session_state['predictor_inputs'] = {
            'dwh/d1': dwh_d1, 'd1': d1, 'tw': tw, 'b': flange_width, 'D': total_depth, 'fyw': fyw, 'E': E, 'a/d': a_d
        }

        # Calculations
        Aw = d1 * tw
        Vy = (Aw * fyw) / math.sqrt(3) / 1000 
        lambda_w = (d1/tw) / (81 * math.sqrt(235/fyw)) if tw > 0 else 1.0
        chi_v = 1.0
        if lambda_w > 1.08: chi_v = 1.2 / lambda_w
        Vv = Vy * chi_v 
        qs = 1 - 0.65 * dwh_d1 - 0.35 * (dwh_d1 ** 2)
        Vnl = Vv * qs 
        
        ml_results = {}
        best_model_val = Vnl 
        best_model_name = "Analytical (Vnl)"
        
        if scaler:
            try:
                dwh_absolute = dwh_d1 * d1
                val_map = {'dwh': dwh_absolute, 'd1': d1, 'tw': tw, 'b': flange_width, 'D': total_depth, 'fy': fyw, 'E': E, 'ad': a_d}
                std_df = pd.DataFrame([val_map])
                aligned_df = align_features_to_scaler(std_df, scaler)
                features_scaled = scaler.transform(aligned_df)
                X_input = pd.DataFrame(features_scaled, columns=aligned_df.columns)
                
                for name, model in models.items():
                    try:
                        pred = model.predict(X_input)[0]
                        ml_results[name] = float(pred)
                    except: ml_results[name] = None
                
                if 'SVR' in ml_results and ml_results['SVR'] is not None:
                    best_model_val, best_model_name = ml_results['SVR'], "SVR"
                elif 'XGBoost' in ml_results and ml_results['XGBoost'] is not None:
                    best_model_val, best_model_name = ml_results['XGBoost'], "XGBoost"
                elif 'Random Forest' in ml_results and ml_results['Random Forest'] is not None:
                    best_model_val, best_model_name = ml_results['Random Forest'], "Random Forest"
                elif ml_results:
                    for name, val in ml_results.items():
                        if val is not None:
                            best_model_val, best_model_name = val, name
                            break
            except Exception as e:
                st.caption(f"Note: ML Prediction unavailable ({e})")

        st.session_state['last_prediction'] = best_model_val
        st.session_state['best_model_name'] = best_model_name
        st.session_state['Vnl_val'] = Vnl
        st.session_state['Vv_val'] = Vv
        st.session_state['ml_results'] = ml_results

    # --- 4. Display Results ---
    if 'last_prediction' in st.session_state:
        pred_val = st.session_state['last_prediction']
        model_name = st.session_state['best_model_name']
        
        st.markdown(f"""
        <div style="background-color: #2e3b4e; color: white; padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 2px solid #667eea; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <h2 style="margin:0; font-size: 1.2rem; opacity: 0.8; letter-spacing: 1px;">PREDICTED SHEAR CAPACITY</h2>
            <h1 style="margin:10px 0; font-size: 4rem; font-weight: 800; color: #7de2d1;">{pred_val:.2f} kN</h1>
            <p style="margin:0; font-style: italic; opacity: 0.7;">Source: {model_name} Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📊 Method Comparison & Engineering Data", expanded=True):
            results_data = [
                {"Method": "✅ Final Prediction", "Shear (kN)": f"{pred_val:.2f}", "Source": model_name},
                {"Method": "📐 Analytical (Vnl)", "Shear (kN)": f"{st.session_state['Vnl_val']:.2f}", "Source": "qs * Vv"},
                {"Method": "🌐 Unperforated (Vv)", "Shear (kN)": f"{st.session_state['Vv_val']:.2f}", "Source": "Buckling Base"}
            ]
            for name, val in st.session_state.get('ml_results', {}).items():
                if val is not None and name != model_name:
                    results_data.append({"Method": name, "Shear (kN)": f"{val:.2f}", "Source": "ML Model"})
            st.table(pd.DataFrame(results_data))

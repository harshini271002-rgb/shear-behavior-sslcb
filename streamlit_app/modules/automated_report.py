
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib
import io

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from utils.data_loader import extract_beam_features

def run_batch_analysis(df, target_col='VU(FEA)'):
    """
    Runs a comprehensive batch analysis on the uploaded dataset.
    Returns a dictionary of results and figures.
    """
    results = {}
    
    # 1. Robust Data Processing
    # Ensure target exists and is numeric
    df.columns = df.columns.astype(str).str.strip()
    target_col = target_col.strip()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Use standardized feature extractor (Only picks the 8 core variables)
    X = extract_beam_features(df)
    y = df.loc[X.index, target_col]
    
    # Safety Check
    if len(X) < 5:
         raise ValueError(f"No valid data found after filtering features. Required columns: [d1, tw, b, D, fy, E, ad, dwh/Ratio]. Found rows: {len(X)}")
    
    numeric_cols = X.columns.tolist()
    
    # Drop rows with NaN
    df_clean = df[numeric_cols + [target_col]].dropna()
    
    if len(df_clean) < 5:
         raise ValueError(f"Not enough clean data ({len(df_clean)} rows). Columns with NaNs: {df[numeric_cols].isna().sum().to_dict()}")

    X = df_clean[numeric_cols]
    y = df_clean[target_col]
    
    # Dynamic CV splits based on sample size
    n_splits = min(10, len(df_clean))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
    
    # 2. Model Definitions
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'MLP': MLPRegressor(max_iter=500, random_state=42),
        'KNN': KNeighborsRegressor(),
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
    if LGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(iterations=100, verbose=0, random_state=42)
        
    # 3. Train & Evaluate (10-Fold CV)
    model_metrics = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    total_models = len(models)
    
    for i, (name, model) in enumerate(models.items()):
        # Quick Train Test Split for Detailed Metrics & SHAP
        try:
             X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
        except ValueError:
             X_train, X_test, y_train, y_test = X_scaled_df, X_scaled_df, y, y
             
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred_test = model.predict(X_test)
        
        # CV Score (R2)
        try:
            cv_scores = cross_val_score(model, X_scaled_df, y, cv=kf, scoring='r2')
            cv_r2 = cv_scores.mean()
        except:
            cv_r2 = np.nan
        
        # Test Metrics
        if len(y_test) > 0:
            r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        else:
            r2, rmse, mae, mape = 0,0,0,0
        
        model_metrics.append({
            'Model': name,
            f'CV R2 ({n_splits}-Fold)': cv_r2,
            'Test R2': r2,
            'Test RMSE': rmse,
            'Test MAE': mae,
            'Test MAPE (%)': mape
        })
        
        progress_bar.progress((i + 1) / total_models)
        
    results['metrics'] = pd.DataFrame(model_metrics)
    results['trained_models'] = trained_models
    results['X'] = X # Original X (for SHAP)
    results['X_scaled'] = X_scaled_df
    results['y'] = y
    results['scaler'] = scaler
    
    # 4. Generate Specific Plots
    figures = {}
    
    # A. Correlation Matrix (Shear Capacity)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    if target_col in corr.columns:
        sns.heatmap(corr[[target_col]].sort_values(by=target_col, ascending=False), annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title(f"Correlation with {target_col}")
    figures['correlation_target'] = fig_corr
    
    # B. Full Correlation Matrix
    fig_corr_full, ax_corr_full = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='viridis', ax=ax_corr_full)
    ax_corr_full.set_title("Full Correlation Matrix")
    figures['correlation_full'] = fig_corr_full
    
    # C. Distribution of Variables
    fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
    df[numeric_cols].boxplot(ax=ax_dist, rot=90)
    ax_dist.set_title("Distribution of Input Variables")
    figures['dist_vars'] = fig_dist

    # --- NEW: Perforation Ratio vs Shear Capacity ---
    # Find the ratio column (either original or calculated)
    ratio_cols = [c for c in df.columns if 'ratio' in c.lower() or 'dwh/d1' in c.lower()]
    if ratio_cols:
        r_col = ratio_cols[0] # Take the first found
        fig_r, ax_r = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df[r_col], y=df[target_col], ax=ax_r, color='purple', s=80, alpha=0.7)
        sns.regplot(x=df[r_col], y=df[target_col], ax=ax_r, scatter=False, color='red', line_kws={'linestyle':'--'})
        ax_r.set_title(f"{target_col} vs Perforation Ratio")
        ax_r.set_xlabel("Perforation Ratio (dwh/d1)")
        ax_r.set_ylabel(target_col)
        ax_r.grid(True, alpha=0.3)
        figures['ratio_vs_capacity'] = fig_r
    
    # D. Model Comparison (Bar Chart)
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    metrics_df = results['metrics']
    sns.barplot(data=metrics_df, x='Model', y='Test R2', ax=ax_comp, palette='viridis', hue='Model', legend=False)
    ax_comp.set_ylim(0, 1.1)
    ax_comp.set_title("Model Accuracy Comparison (Test R²)")
    plt.xticks(rotation=45)
    figures['model_comparison'] = fig_comp
    
    # E. SHAP (Tree-based models)
    shap_model_name = None
    shap_model = None
    
    for m_name in ['CatBoost', 'XGBoost', 'LightGBM', 'GBM', 'Random Forest']:
        if m_name in trained_models:
            shap_model_name = m_name
            shap_model = trained_models[m_name]
            break
            
    if shap_model:
        try:
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer(X_scaled_df)
            
            fig_shap_sum, ax_ss = plt.subplots()
            shap.summary_plot(shap_values, X_scaled_df, show=False)
            figures['shap_summary'] = fig_shap_sum
            
            vals = np.abs(shap_values.values).mean(0)
            feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            top_feature = feature_importance.iloc[0]['col_name']
            
            fig_shap_dep, ax_sd = plt.subplots()
            shap.dependence_plot(top_feature, shap_values.values, X_scaled_df, ax=ax_sd, show=False)
            figures['shap_dependence'] = fig_shap_dep
            results['shap_target_model'] = shap_model_name
        except Exception as e:
            print(f"SHAP failed: {e}")

    # F. Actual vs Predicted (ALL Models)
    figures['pred_vs_actual'] = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_scaled_df)
        
        fig_res, ax_res = plt.subplots(figsize=(6, 6))
        ax_res.scatter(y, y_pred, alpha=0.6, edgecolors='k', s=40)
        
        d_min, d_max = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax_res.plot([d_min, d_max], [d_min, d_max], 'r--', lw=2)
        
        ax_res.set_xlabel(f"Actual {target_col}")
        ax_res.set_ylabel(f"Predicted by {name}")
        ax_res.set_title(f"{name}: Actual vs Predicted")
        ax_res.grid(True, alpha=0.3)
        
        figures['pred_vs_actual'][name] = fig_res
        
    results['figures'] = figures
    
    return results

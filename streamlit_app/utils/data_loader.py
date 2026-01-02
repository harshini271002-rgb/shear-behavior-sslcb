import pandas as pd
import os
import streamlit as st

# Constants for data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
FULL_DATA_PATH = os.path.join(DATA_DIR, "Input csv.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "final_predictions.csv")

@st.cache_data
def load_cleaned_data():
    """Load the cleaned dataset used for training (Reference Study)."""
    if not os.path.exists(CLEANED_DATA_PATH):
        return None
    # Header is on row 3 (index 2) in the original/cleaned excel-derived CSVs
    return pd.read_csv(CLEANED_DATA_PATH, header=2)

def get_session_data():
    """Helper to get data from session state, falling back to default data."""
    if 'uploaded_df' in st.session_state and st.session_state['uploaded_df'] is not None:
        return st.session_state['uploaded_df']
    
    # Fallback to default data so tabs aren't empty
    return load_cleaned_data()

@st.cache_data
def load_full_data():
    """Load the original full dataset with comparison columns."""
    if not os.path.exists(FULL_DATA_PATH):
        return None
    # Replicating the logic from visualize_results.py
    # Skip first 2 rows as header seems to be on row 3 (index 2) based on previous code
    return pd.read_csv(FULL_DATA_PATH, header=2)

@st.cache_data
def load_predictions_data():
    """Load the data with predictions if available."""
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    return pd.read_csv(PREDICTIONS_PATH)

def extract_beam_features(df):
    """
    Strictly extracts the 8 core independent features for beam analysis.
    Ensures consistent names and order: [dwh, d1, tw, b, D, fy, E, ad]
    """
    # Clean column names
    df.columns = df.columns.astype(str).str.strip()
    
    mapping = {
        'dwh': ['dwh', 'hole diameter', 'perforation size', 'opening diameter', 'diameter'],
        'd1': ['d1', 'web depth'],
        'tw': ['tw(mm)', 'tw', 'web thickness', 'thickness'],
        'b': ['flange width(mm)', 'flange width', 'b'],
        'D': ['total depth D (mm)', 'total depth', 'D'],
        'fy': ['fyw', 'yield strength', 'fy'],
        'E': ['E', 'elastic modulus', 'youngs modulus'],
        'ad': ['a/d', 'aspect ratio']
    }
    
    extracted = pd.DataFrame()
    for key, variants in mapping.items():
        found = False
        for col in df.columns:
            if col.lower() in variants or any(v in col.lower() for v in variants):
                extracted[key] = pd.to_numeric(df[col], errors='coerce')
                found = True
                break
        if not found:
            # Fallback for the first column (dwh) - check for ratios and convert TO absolute
            if key == 'dwh':
                ratio_variants = ['ratio', 'dwh/d1', 'dwh / d1', 'opening ratio']
                d1_col = extracted['d1'] if 'd1' in extracted.columns else None
                
                for col in df.columns:
                    if any(v in col.lower() for v in ratio_variants):
                        ratio_vals = pd.to_numeric(df[col], errors='coerce')
                        if d1_col is not None:
                             st.info("💡 Converting Perforation Ratio to Absolute Diameter for ML Alignment.")
                             extracted['dwh'] = ratio_vals * d1_col
                             found = True
                             break
            
            if not found:
                extracted[key] = 0.0 # Default fallback
                
    return extracted.dropna()

def align_features_to_scaler(df_standard, scaler):
    """
    Renames and reorders a standardized DataFrame (from extract_beam_features) 
    to match the feature names expected by the scaler.
    """
    if not hasattr(scaler, "feature_names_in_"):
        return df_standard
        
    expected = list(scaler.feature_names_in_)
    rename_map = {}
    
    # Mapping of standard keys to common descriptive names (seen in fit time)
    # Order matters: check for 'dwh' (perforation) before 'd1' (depth)
    mapping_logic = [
        ('dwh', ['dwh/d1', 'depth of web opening', 'opening ratio', 'perforation ratio', 'dwh']),
        ('d1', ['d1', 'web depth']),
        ('tw', ['tw', 'thickness']),
        ('b', ['flange width', 'b']),
        ('D', ['total depth', 'd']),
        ('fy', ['fy', 'yield']),
        ('E', ['e', 'modulus', 'youngs']),
        ('ad', ['a/d', 'aspect', 'ad'])
    ]
    
    # Track which expected columns have been filled to avoid overwriting or duplicates
    filled_expected = set()
    
    for std_key, variations in mapping_logic:
        if std_key not in df_standard.columns:
            continue
            
        # Try to find the BEST match in 'expected' for this 'std_key'
        for exp_name in expected:
            if exp_name in filled_expected:
                continue
                
            low_exp = exp_name.lower()
            
            # Special logic for d1 vs dwh/d1
            if std_key == 'd1':
                if 'd1' in low_exp and 'dwh' not in low_exp:
                    rename_map[std_key] = exp_name
                    filled_expected.add(exp_name)
                    break
            else:
                if any(v in low_exp for v in variations):
                    rename_map[std_key] = exp_name
                    filled_expected.add(exp_name)
                    break
                
    aligned = df_standard.rename(columns=rename_map)
    
    # Only keep columns expected by scaler, in correct order
    # If a column is missing, add it as 0.0 to prevent crash
    for exp_name in expected:
        if exp_name not in aligned.columns:
            # Check if we can find it by rename_map (should have been handled by rename)
            if exp_name not in aligned.columns:
                aligned[exp_name] = 0.0
            
    return aligned[expected]

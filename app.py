"""
CreditLens — Credit Risk Intelligence Platform
================================================
Streamlit UI — loads pre-trained PKL models only.
Run model_train.py first to generate models/
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, warnings, os
import sklearn
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CreditLens — Credit Risk Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — original look preserved, sidebar text brightened
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

:root {
    --bg:     #0D1117;
    --bg2:    #161B22;
    --bg3:    #1C2330;
    --border: #30363D;
    --text:   #E6EDF3;
    --muted:  #7D8590;
    --teal:   #00D4AA;
    --blue:   #58A6FF;
    --orange: #F0883E;
    --red:    #FF7B72;
    --green:  #3FB950;
    --purple: #BC8CFF;
    --yellow: #E3B341;
}

html, body {
    font-family: 'Inter', sans-serif !important;
    background-color: #0D1117 !important;
}
/* Only set font on css-scoped divs — do NOT inherit color here;
   Streamlit's var(--text) resolution can override explicit white labels */
[class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #0D1117 !important; }

/* ── Widget labels (number inputs, sliders, selects, etc.) ── */
/* Target every Streamlit widget label directly so the html/body
   inheritance chain cannot win over var(--text) */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label,
[data-testid="stMarkdownContainer"] p,
.stNumberInput > label,
.stSlider     > label,
.stSelectbox  > label,
.stMultiSelect > label,
.stRadio      > label,
.stCheckbox   > label,
.stTextInput  > label,
.stTextArea   > label,
div[data-baseweb="form-control"] label,
div[data-baseweb="form-control"] p {
    color: #E6EDF3 !important;
    font-family: 'Inter', sans-serif !important;
}
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #0A0F1A 100%) !important;
    border-right: 1px solid #30363D !important;
}
/* Sidebar radio — brighter visible text */
[data-testid="stSidebar"] .stRadio > div > label {
    color: #CDD5E0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 2px 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    color: #E6EDF3 !important;
}
/* Selected radio item */
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked ~ div label {
    color: #00D4AA !important;
    font-weight: 600 !important;
}
/* Radio button circle */
[data-testid="stSidebar"] [data-baseweb="radio"] div[data-checked="true"] div {
    border-color: #00D4AA !important;
    background: #00D4AA !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div div {
    border-color: #A8B4C0 !important;
}
/* Sidebar general text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #CDD5E0 !important;
}
[data-testid="stSidebar"] b {
    color: #E6EDF3 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
}
[data-testid="metric-container"] label {
    color: #A8B4C0 !important;
    font-size: 12px !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E6EDF3 !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Cards ── */
.card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.card-sm {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ── Typography ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4AA 0%, #58A6FF 50%, #BC8CFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 12px;
}
.hero-sub {
    font-size: 18px;
    color: #A8B4C0;
    font-weight: 300;
    margin-bottom: 40px;
    letter-spacing: 0.3px;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #E6EDF3;
    margin-bottom: 8px;
    padding-bottom: 12px;
    border-bottom: 1px solid #30363D;
}
.section-sub {
    font-size: 14px;
    color: #A8B4C0;
    margin-bottom: 24px;
    margin-top: -4px;
}
.label-sm {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #A8B4C0;
    margin-bottom: 6px;
}
.formula {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: #00D4AA;
    background: #0D1117;
    border: 1px solid #30363D;
    border-left: 3px solid #00D4AA;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin: 10px 0;
    line-height: 1.6;
}
.gloss-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.gloss-term  { font-size: 16px; font-weight: 600; margin-bottom: 6px; }
.gloss-def   { font-size: 14px; color: #B0BAC6; line-height: 1.7; margin-bottom: 8px; }
.gloss-example {
    font-size: 13px;
    color: #A8B4C0;
    font-style: italic;
    background: #0D1117;
    border-radius: 6px;
    padding: 8px 12px;
    margin-top: 8px;
}
.pill { display:inline-block; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:500; }
.pill-teal   { background:rgba(0,212,170,.12);  color:#00D4AA; border:1px solid rgba(0,212,170,.3); }
.pill-red    { background:rgba(255,123,114,.12); color:#FF7B72; border:1px solid rgba(255,123,114,.3); }
.pill-blue   { background:rgba(88,166,255,.12);  color:#58A6FF; border:1px solid rgba(88,166,255,.3); }
.pill-orange { background:rgba(240,136,62,.12);  color:#F0883E; border:1px solid rgba(240,136,62,.3); }
.pill-purple { background:rgba(188,140,255,.12); color:#BC8CFF; border:1px solid rgba(188,140,255,.3); }
.pill-green  { background:rgba(63,185,80,.12);   color:#3FB950; border:1px solid rgba(63,185,80,.3); }

.insight {
    background: rgba(0,212,170,.05);
    border: 1px solid rgba(0,212,170,.2);
    border-radius: 10px; padding: 14px 18px;
    font-size: 14px; color: #B0E0D8; margin: 12px 0; line-height: 1.6;
}
.insight-warn {
    background: rgba(240,136,62,.06);
    border: 1px solid rgba(240,136,62,.25);
    border-radius: 10px; padding: 14px 18px;
    font-size: 14px; color: #E8C4A0; margin: 12px 0; line-height: 1.6;
}
.insight-red {
    background: rgba(255,123,114,.06);
    border: 1px solid rgba(255,123,114,.25);
    border-radius: 10px; padding: 14px 18px;
    font-size: 14px; color: #F0B8B4; margin: 12px 0; line-height: 1.6;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22 !important; border-radius: 10px !important;
    padding: 4px !important; border: 1px solid #30363D !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #B0BAC6 !important; font-weight: 500 !important;
    font-size: 13px !important; border-radius: 8px !important; padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #0D1117 !important; color: #E6EDF3 !important;
    border: 1px solid #30363D !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4AA, #0097A7) !important;
    color: #0D1117 !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

/* ── Form elements ── */
.stSelectbox [data-baseweb="select"] {
    background: #161B22 !important; border: 1px solid #30363D !important;
    border-radius: 8px !important; color: #E6EDF3 !important;
}
.stNumberInput input {
    background: #161B22 !important; border: 1px solid #30363D !important;
    color: #E6EDF3 !important; border-radius: 8px !important;
}
/* Stepper +/− buttons */
.stNumberInput button {
    background: #1C2128 !important; border-color: #30363D !important;
    color: #E6EDF3 !important;
}
.stNumberInput button:hover {
    background: #30363D !important; color: #FFFFFF !important;
}
/* Slider current-value label */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] .st-emotion-cache-1vzeuhh,
[data-testid="stSlider"] p {
    color: #E6EDF3 !important;
}
.streamlit-expanderHeader {
    background: #161B22 !important; border: 1px solid #30363D !important;
    border-radius: 8px !important; color: #E6EDF3 !important; font-weight: 500 !important;
}
[data-testid="stDataFrame"] { border: 1px solid #30363D; border-radius: 10px; overflow: hidden; }
.stProgress > div > div { background: linear-gradient(90deg, #00D4AA, #58A6FF) !important; }
hr { border-color: #30363D !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME (shared)
# ══════════════════════════════════════════════════════════════════════════════
PT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,27,34,0.8)',
    font=dict(family='Inter', color='#B0BAC6', size=12),
    title_font=dict(family='Inter', color='#E6EDF3', size=16),
    xaxis=dict(gridcolor='#30363D', linecolor='#30363D', zerolinecolor='#30363D'),
    yaxis=dict(gridcolor='#30363D', linecolor='#30363D', zerolinecolor='#30363D'),
)
# margin is intentionally excluded from PT — every update_layout() call passes its own
COLORS_MODEL = ['#00D4AA', '#4DB8FF', '#F5A623', '#BC8CFF']

def L(x):
    """Convert any array-like to a plain Python list so Plotly never receives
    numpy arrays or pandas Series (which newer Plotly serializes as binary
    bdata dicts, causing 'undefined' values in the browser chart)."""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return list(x)

def pxdf(df_in):
    """Return a copy of a DataFrame with all numeric columns cast to Python float
    so Plotly Express never serializes them as binary bdata arrays."""
    d = df_in.copy()
    for col in d.select_dtypes(include='number').columns:
        d[col] = d[col].astype(float)
    for col in d.select_dtypes(include='category').columns:
        d[col] = d[col].astype(str)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# DATA + MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """
    Load Credit_Card_Default.csv and perform preprocessing.
    
    Returns:
        pd.DataFrame: Preprocessed dataset with 30,000 rows and enhanced columns
        
    Columns Added:
        - SEX_LABEL: Human-readable gender labels
        - EDU_LABEL: Human-readable education labels
        - MAR_LABEL: Human-readable marriage labels
        - DEFAULT_LABEL: Human-readable default labels
        - UTIL_RATE: Credit utilization rate (0-2)
        - AVG_PAY_STATUS: Average payment delay across 6 months
        - TOTAL_BILL: Sum of all 6 bill amounts
        - TOTAL_PAY: Sum of all 6 payment amounts
        - PAY_RATIO: Payment to bill ratio (0-5)
        - AGE_GROUP: Categorical age bins
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, 'Credit_Card_Default.csv')
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Rename columns
        df = df.rename(columns={
            'default.payment.next.month': 'DEFAULT',
            'SEX': 'GENDER'
        })
        
        # Drop ID column
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        
        # Create human-readable labels for categorical features
        df['SEX_LABEL'] = df['GENDER'].map({1: 'Male', 2: 'Female'})
        df['EDU_LABEL'] = df['EDUCATION'].map({
            1: 'Graduate School',
            2: 'University',
            3: 'High School',
            4: 'Others',
            5: 'Unknown',
            6: 'Unknown'
        }).fillna('Others')
        df['MAR_LABEL'] = df['MARRIAGE'].map({
            1: 'Married',
            2: 'Single',
            3: 'Others'
        }).fillna('Others')
        df['DEFAULT_LABEL'] = df['DEFAULT'].map({0: 'No Default', 1: 'Default'})
        
        # Calculate derived features with error handling for division by zero
        df['UTIL_RATE'] = (df['BILL_AMT1'] / df['LIMIT_BAL'].replace(0, np.nan)).clip(0, 2).fillna(0)
        df['AVG_PAY_STATUS'] = df[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].mean(axis=1)
        df['TOTAL_BILL'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].sum(axis=1)
        df['TOTAL_PAY'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].sum(axis=1)
        df['PAY_RATIO'] = (df['TOTAL_PAY'] / df['TOTAL_BILL'].replace(0, np.nan)).clip(0, 5).fillna(0)
        
        # Create age group bins
        df['AGE_GROUP'] = pd.cut(
            df['AGE'],
            bins=[20, 25, 30, 35, 40, 45, 50, 60, 80],
            labels=['21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-60', '60+']
        )
        
        return df
        
    except FileNotFoundError:
        st.error(f"⚠️ Dataset file not found. Expected: {csv_path}")
        st.info("Please ensure Credit_Card_Default.csv is in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {str(e)}")
        st.stop()

def display_version_mismatch_guidance(expected_version, current_version):
    """
    Display structured error panel with version mismatch information and resolution guidance.
    
    This function creates a user-friendly error display using Streamlit components to guide
    users through resolving sklearn version mismatches. It presents two clear resolution paths:
    installing the compatible sklearn version or retraining models with the current version.
    
    Args:
        expected_version (str): The sklearn version used to train the models (e.g., "1.2.0")
        current_version (str): The currently installed sklearn version (e.g., "1.3.0")
        
    Returns:
        None: Displays error panel directly in Streamlit UI
        
    Examples:
        >>> display_version_mismatch_guidance("1.2.0", "1.3.0")
        # Displays error panel with version comparison and resolution options
        
    Notes:
        - Uses st.error() for the main alert message
        - Uses st.columns() to display version comparison side-by-side
        - Uses st.expander() for detailed resolution steps
        - Provides copy-pasteable commands for both resolution paths
        - Validates: Requirements 2.2, 2.3
    """
    # Main error alert
    st.error("⚠️ Sklearn Version Mismatch Detected")
    
    # Display version comparison using columns for side-by-side view
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models Trained With", f"sklearn {expected_version}")
    with col2:
        st.metric("Currently Installed", f"sklearn {current_version}")
    
    # Resolution guidance header
    st.info("**Resolution Options:**")
    
    # Option 1: Install compatible sklearn version (recommended)
    with st.expander("✅ Option 1: Install Compatible Sklearn Version (Recommended)", expanded=True):
        st.markdown("This will allow you to use the existing trained models immediately without retraining.")
        st.code(f"pip install scikit-learn=={expected_version}", language="bash")
        st.markdown("**Steps:**")
        st.markdown(f"1. Copy the command above")
        st.markdown(f"2. Run it in your terminal or command prompt")
        st.markdown(f"3. Restart the Streamlit app")
        st.markdown(f"4. Models should load successfully")
    
    # Option 2: Retrain models with current version
    with st.expander("🔄 Option 2: Retrain Models with Current Version"):
        st.markdown(f"If you prefer to use the current sklearn version ({current_version}), you can retrain the models.")
        st.markdown("**⚠️ Warning:** This may take several minutes and will overwrite existing model files.")
        
        # Check if training script exists
        if os.path.exists("model_train.py"):
            st.code("python model_train.py", language="bash")
            st.markdown("**Steps:**")
            st.markdown("1. Copy the command above")
            st.markdown("2. Run it in your terminal (this will take several minutes)")
            st.markdown("3. Wait for training to complete")
            st.markdown("4. Restart the Streamlit app")
        elif os.path.exists("CreditLens_Final.ipynb"):
            st.markdown("**Steps:**")
            st.markdown("1. Open `CreditLens_Final.ipynb` in Jupyter Notebook")
            st.markdown("2. Run all cells to retrain the models")
            st.markdown("3. Wait for training to complete (may take several minutes)")
            st.markdown("4. Restart the Streamlit app")
        else:
            st.markdown("**Steps:**")
            st.markdown("1. Locate your model training script or notebook")
            st.markdown("2. Run the training process to regenerate model files")
            st.markdown("3. Ensure models are saved to the `models/` directory")
            st.markdown("4. Restart the Streamlit app")
    
    # Additional troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("**If Option 1 doesn't work:**")
        st.markdown("- Ensure you're in the correct virtual environment")
        st.markdown("- Try uninstalling sklearn first: `pip uninstall scikit-learn`")
        st.markdown(f"- Then reinstall: `pip install scikit-learn=={expected_version}`")
        st.markdown("- Check for conflicting packages that may pin sklearn to a different version")
        
        st.markdown("**If Option 2 doesn't work:**")
        st.markdown("- Ensure you have the training data file (`Credit_Card_Default.csv`)")
        st.markdown("- Check that you have sufficient disk space for model files")
        st.markdown("- Verify all required dependencies are installed (`pip install -r requirements.txt`)")
        st.markdown("- Check the training script for any errors or missing dependencies")

def extract_sklearn_version_from_pickle(file_path):
    """
    Extract sklearn version metadata from a pickled model file without fully loading it.
    
    This function attempts to extract the scikit-learn version used to train a model
    by examining the pickle file's metadata or by parsing exception messages from
    attempted loads. This allows version mismatch detection before full model loading.
    
    Args:
        file_path (str): Path to the pickled model file (.pkl)
        
    Returns:
        tuple: (success: bool, version: str, error_msg: str)
            - success: True if version was successfully extracted, False otherwise
            - version: The sklearn version string (e.g., "1.2.0") or None if extraction failed
            - error_msg: Error message if extraction failed, empty string otherwise
            
    Examples:
        >>> success, version, error = extract_sklearn_version_from_pickle("model.pkl")
        >>> if success:
        ...     print(f"Model trained with sklearn {version}")
        ... else:
        ...     print(f"Could not extract version: {error}")
    
    Notes:
        - Handles cases where version metadata is not available (older joblib versions)
        - Does not fully load the model to avoid memory overhead
        - Falls back to parsing exception messages if direct metadata access fails
    """
    import pickle
    import io
    
    try:
        # Approach 1: Try to extract version from joblib metadata
        # Joblib stores sklearn version information in the pickle stream
        with open(file_path, 'rb') as f:
            # Read the file content
            file_content = f.read()
            
        # Try to find sklearn version in the pickle stream
        # Joblib often embeds version info as strings in the pickle
        file_str = str(file_content)
        
        # Look for sklearn version patterns in the pickle bytes
        # Common patterns: "sklearn_version", "__version__", version tuples
        import re
        
        # Pattern 1: Look for explicit sklearn version strings (e.g., "1.2.0", "1.3.1")
        # Use word boundary to avoid matching partial numbers
        version_pattern = r'(?:sklearn|scikit-learn).*?\\x00(\d+\.\d+\.\d+)'
        matches = re.findall(version_pattern, file_str)
        if matches:
            # Return the first version found
            return (True, matches[0], "")
        
        # Pattern 2: Look for version in common pickle metadata locations
        # Try to find version strings that look like "1.2.0" preceded by sklearn-related text
        broader_pattern = r'sklearn.*?(\d+\.\d+\.\d+)'
        matches = re.findall(broader_pattern, file_str)
        if matches:
            # Filter out unlikely versions (e.g., starting with 0 or very high numbers)
            for match in matches:
                parts = match.split('.')
                major = int(parts[0])
                if 0 < major <= 2:  # sklearn versions are typically 0.x, 1.x, or 2.x
                    return (True, match, "")
        
        # Pattern 3: Look for version tuples in pickle metadata
        # Joblib may store version as (1, 2, 0) tuple
        tuple_pattern = r'\((\d+),\s*(\d+),\s*(\d+)\)'
        tuple_matches = re.findall(tuple_pattern, file_str)
        if tuple_matches:
            # Check if this looks like a version tuple (reasonable version numbers)
            for match in tuple_matches:
                major, minor, patch = match
                if 0 < int(major) <= 2 and int(minor) <= 20:  # Reasonable sklearn version bounds
                    version_str = f"{major}.{minor}.{patch}"
                    return (True, version_str, "")
        
        # Approach 2: Try to load and catch the exception message
        # sklearn version mismatch exceptions often contain version information
        try:
            with open(file_path, 'rb') as f:
                joblib.load(f)
            # If load succeeds, we can't extract version this way
            # Return the current sklearn version as it's compatible
            return (True, sklearn.__version__, "")
        except Exception as load_error:
            error_str = str(load_error)
            
            # Parse exception message for version information
            # Common patterns in sklearn/joblib exceptions:
            # "sklearn version 1.2.0 is required"
            # "module 'sklearn' has no attribute..." (version mismatch)
            # "cannot import name..." (version mismatch)
            
            version_in_error = re.findall(r'version\s+(\d+\.\d+\.\d+)', error_str)
            if version_in_error:
                return (True, version_in_error[0], "")
            
            # If we can't extract version from error, return failure
            return (False, None, f"Could not extract version from pickle or error message: {error_str[:100]}")
            
    except FileNotFoundError:
        return (False, None, f"File not found: {file_path}")
    except Exception as e:
        return (False, None, f"Error reading pickle file: {str(e)[:100]}")

@st.cache_resource
def load_models():
    """
    Load pre-trained models from models/ directory.
    
    Returns:
        Tuple containing:
        - pkls: Dict mapping model names to loaded model objects (or mock objects if loading fails)
        - results: Dict containing performance metrics from results_summary.json
        - comp_df: DataFrame with comparison metrics
        
    Models Loaded:
        All PKL files found in models/ directory including:
        - Baseline models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
        - SMOTE models (Logistic Regression, Random Forest, XGBoost)
        - Weight models (Logistic Regression, Random Forest, XGBoost)
        - Tuned models (SMOTE XGBoost, Weight XGBoost)
    """
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    pkls = {}
    
    # Define all expected model files with their display names
    model_files = [
        ('baseline_Logistic_Regression.pkl', 'Baseline — Logistic Regression'),
        ('baseline_Decision_Tree.pkl', 'Baseline — Decision Tree'),
        ('baseline_Random_Forest.pkl', 'Baseline — Random Forest'),
        ('baseline_Gradient_Boosting.pkl', 'Baseline — Gradient Boosting'),
        ('baseline_XGBoost.pkl', 'Baseline — XGBoost'),
        ('SMOTE_Logistic_Regression.pkl', 'SMOTE — Logistic Regression'),
        ('SMOTE_Random_Forest.pkl', 'SMOTE — Random Forest'),
        ('SMOTE_XGBoost.pkl', 'SMOTE — XGBoost'),
        ('Weight_Logistic_Regression.pkl', 'Weight — Logistic Regression'),
        ('Weight_Random_Forest.pkl', 'Weight — Random Forest'),
        ('Weight_XGBoost.pkl', 'Weight — XGBoost'),
        ('tuned_smote_xgb.pkl', 'Tuned SMOTE XGB'),
        ('tuned_weight_xgb.pkl', 'Tuned Weight XGB'),
    ]
    
    # Load each model file
    for fname, display_name in model_files:
        model_path = os.path.join(base, fname)
        if os.path.exists(model_path):
            # Extract expected sklearn version from model metadata before attempting to load
            success, expected_version, version_error = extract_sklearn_version_from_pickle(model_path)
            
            try:
                loaded = joblib.load(model_path)
                # Models are saved as dicts with {'model': ..., 'scaler': ..., 'features': ..., 'scaled': ...}
                # Store the entire dict
                pkls[display_name] = loaded
            except (ModuleNotFoundError, AttributeError, ImportError) as e:
                # Version mismatch or module compatibility error
                # Store detailed version information for error reporting
                current_version = sklearn.__version__
                
                pkls[display_name] = {
                    'model': None,
                    'scaler': None,
                    'features': [],
                    'scaled': False,
                    '_load_error': str(e),
                    '_expected_sklearn_version': expected_version if success else 'unknown',
                    '_current_sklearn_version': current_version,
                    '_error_type': 'version_mismatch'
                }
            except Exception as e:
                # Other loading errors (not version-related)
                current_version = sklearn.__version__
                
                pkls[display_name] = {
                    'model': None,
                    'scaler': None,
                    'features': [],
                    'scaled': False,
                    '_load_error': str(e),
                    '_expected_sklearn_version': expected_version if success else 'unknown',
                    '_current_sklearn_version': current_version,
                    '_error_type': 'other'
                }
        else:
            st.warning(f"⚠️ Model file not found: {fname}")
    
    # Load results_summary.json for performance metrics
    results_path = os.path.join(base, 'results_summary.json')
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            # Debug: Print loaded model names
            if results:
                print(f"✅ Loaded {len(results)} models from results_summary.json")
                print(f"Model names: {list(results.keys())[:3]}...")
            else:
                print("⚠️ results_summary.json is empty")
        except json.JSONDecodeError as e:
            st.warning(f"⚠️ Failed to parse results_summary.json: {str(e)}")
        except Exception as e:
            st.warning(f"⚠️ Error loading results_summary.json: {str(e)}")
    else:
        st.warning("⚠️ results_summary.json not found in models/ directory")
    
    # Load comparison_table.csv for additional metrics
    comp_path = os.path.join(base, 'comparison_table.csv')
    comp_df = None
    if os.path.exists(comp_path):
        try:
            comp_df = pd.read_csv(comp_path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load comparison_table.csv: {str(e)}")
    
    return pkls, results, comp_df

df = load_data()
pkls, results_json, comp_df = load_models()

# Check if we have any data to display (either models or metrics)
if not pkls and not results_json and comp_df is None:
    st.error("⚠️ No model data found in ./models/ directory. Please ensure model files exist.")
    st.info("Debug info: Check that results_summary.json and comparison_table.csv exist in models/ folder")
    st.stop()

# Show warning if ANY model failed to load (not just when all fail)
failed_pkls = {k: v for k, v in pkls.items() if v.get('_load_error')}
if failed_pkls:
    first_failed = next(
        (v for v in failed_pkls.values() if v.get('_expected_sklearn_version')),
        list(failed_pkls.values())[0]
    )
    expected_version = first_failed.get('_expected_sklearn_version', 'unknown')
    current_version  = sklearn.__version__

    if expected_version != current_version:
        display_version_mismatch_guidance(expected_version, current_version)
    else:
        st.warning(f"⚠️ {len(failed_pkls)} model(s) could not be loaded.")
        with st.expander("Show load errors"):
            for name, pkg in failed_pkls.items():
                st.code(f"{name}: {pkg.get('_load_error', 'unknown error')}")

    n_ok = len(pkls) - len(failed_pkls)
    st.info(
        f"📊 **Note:** {n_ok}/{len(pkls)} models loaded successfully. "
        "Metrics visualization is still available. "
        "Run `python model_train.py` to regenerate compatible model files."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 30px;'>
        <div style='font-size:36px; margin-bottom:8px;'>🔬</div>
        <div style='font-family:Playfair Display,serif; font-size:20px; font-weight:700;
                    background:linear-gradient(135deg,#00D4AA,#58A6FF);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            CreditLens
        </div>
        <div style='font-size:11px; color:#A8B4C0 !important; letter-spacing:1.5px; margin-top:4px;'>
            RISK INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:11px; color:#A8B4C0; letter-spacing:1px; "
        "padding:0 4px 8px; text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Overview & Glossary",
        "📊  Exploratory Analysis",
        "⚙️  Feature Engineering",
        "🤖  Model Building",
        "📈  Model Comparison",
        "🎯  Live Predictor",
    ], label_visibility="collapsed")

    st.markdown("---")
    
    # Determine model loading status for success indicator
    models_loaded_successfully = False
    if pkls:
        # Check if at least one model loaded successfully (no _load_error key)
        models_loaded_successfully = any(not pkg.get('_load_error') for pkg in pkls.values())
    
    # Build sidebar info with dynamic model status
    sidebar_info = """
    <div style='font-size:12px; line-height:2.0;'>
        <div>📁 <b>30,000</b> records</div>
        <div>🔢 <b>25</b> variables</div>
        <div>⚠️ <b>22.1%</b> default rate</div>
        <div>📅 Apr – Sep 2005</div>
        <div style='margin-top:8px;'>🇹🇼 Taiwan credit market</div>
    """
    
    # Add success indicator if models loaded successfully
    if models_loaded_successfully:
        sklearn_version = sklearn.__version__
        sidebar_info += f"<div style='margin-top:8px; color:#00D4AA;'>✅ Models loaded successfully with sklearn {sklearn_version}</div>"
    else:
        # Show generic message if models failed to load
        sidebar_info += "<div style='margin-top:8px; color:#FF6B6B;'>⚠️ Model loading issues detected</div>"
    
    sidebar_info += """
    </div>
    """
    
    st.markdown(sidebar_info, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GLOSSARY HELPERS — pre-build HTML to avoid ternary-in-fstring rendering bug
# ══════════════════════════════════════════════════════════════════════════════
def _gcard(term, color, defn, example, formula=None):
    formula_html = (
        "<div class='formula'>" + formula + "</div>"
    ) if formula else ""
    html = (
        "<div class='gloss-card' style='border-left-color:" + color + "'>"
        "<div class='gloss-term' style='color:" + color + "'>" + term + "</div>"
        "<div class='gloss-def'>" + defn + "</div>"
        + formula_html +
        "<div class='gloss-example'>📌 Example: " + example + "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

def _gcard_noex(term, color, defn, note):
    html = (
        "<div class='gloss-card' style='border-left-color:" + color + "'>"
        "<div class='gloss-term' style='color:" + color + "'>" + term + "</div>"
        "<div class='gloss-def'>" + defn + "</div>"
        "<div class='gloss-example'>📌 " + note + "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & GLOSSARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview & Glossary":

    st.markdown(
        "<div class='hero-title'>Credit Risk<br>Intelligence</div>"
        "<div class='hero-sub'>A complete analytical journey from raw data to predictive models — built for everyone.</div>",
        unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Customers","30,000",help="Total records in dataset")
    with c2: st.metric("Default Rate","22.1%",delta="-77.9% safe",delta_color="inverse")
    with c3: st.metric("Avg Credit Limit","NT$167K")
    with c4: st.metric("Features","24")
    with c5: st.metric("Time Period","6 Months")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── INTERACTIVE RISK FLOW ────────────────────────────────────────────────
    st.markdown("<div class='section-title'>The Credit Risk Journey</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Click each stage to understand what happens at every step from healthy credit to default</div>", unsafe_allow_html=True)

    st.components.v1.html("""
<!DOCTYPE html><html><head><style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0D1117; font-family:'Inter',-apple-system,sans-serif; padding:20px 20px 16px; }

/* Row: stages + connectors, vertically centered at node mid */
.row { display:flex; align-items:flex-start; width:100%; }
.stage { flex:1; display:flex; flex-direction:column; align-items:center;
         cursor:pointer; padding:0 4px; min-width:0; }
/* Connector: fixed height matching node so line centres perfectly */
.conn { flex:0 0 18px; display:flex; align-items:center; height:46px; }
.conn-line { width:100%; height:2px; border-radius:1px; }

.node { width:46px; height:46px; border-radius:50%; border:2px solid;
        display:flex; align-items:center; justify-content:center;
        font-size:15px; font-weight:700; transition:transform .3s, box-shadow .3s;
        position:relative; z-index:2; flex-shrink:0; }
.node:hover { transform:scale(1.18); }
.node.active { transform:scale(1.26); box-shadow:0 0 0 6px rgba(255,255,255,.04), 0 0 18px currentColor; }

.slbl { font-size:10px; text-align:center; margin-top:8px; line-height:1.3;
        font-weight:500; color:#A8B4C0; transition:color .25s; max-width:80px; }
.slbl.active { color:#E6EDF3; }

.detail { border-radius:14px; border:1px solid; padding:22px 26px;
          margin-top:18px; transition:all .4s; min-height:200px; }
.d-header { display:flex; align-items:center; gap:14px; margin-bottom:16px; }
.d-icon { font-size:28px; line-height:1; flex-shrink:0; }
.d-title { font-size:18px; font-weight:600; }
.d-days { font-size:12px; color:#A8B4C0; margin-top:2px; }
.stats-row { display:flex; gap:20px; margin-bottom:16px; flex-wrap:wrap; }
.stat { background:#0D1117; border-radius:10px; padding:12px 16px; min-width:100px; }
.stat-lbl { font-size:10px; color:#A8B4C0; letter-spacing:.8px; text-transform:uppercase; margin-bottom:4px; }
.stat-val { font-size:22px; font-weight:600; line-height:1; }
.stat-bench { font-size:10px; color:#A8B4C0; margin-top:3px; }
.d-desc { font-size:13px; color:#C0CAD4; line-height:1.75; margin-bottom:14px; }
.sec-lbl { font-size:10px; color:#A8B4C0; letter-spacing:.8px; text-transform:uppercase; margin-bottom:8px; margin-top:12px; }
.tag-row { display:flex; flex-wrap:wrap; gap:6px; }
.tag { font-size:11px; padding:3px 10px; border-radius:16px; border:1px solid; }
.rbar { margin-top:18px; }
.rlbl { display:flex; justify-content:space-between; font-size:11px; color:#A8B4C0; margin-bottom:6px; }
.rtrack { height:8px; background:#1C2330; border-radius:4px; overflow:hidden; }
.rfill { height:100%; border-radius:4px; transition:width .8s cubic-bezier(.4,0,.2,1); }
</style></head><body>
<div class="row" id="stages"></div>
<div class="detail" id="detail"></div>
<script>
const S=[
  {icon:'✦',label:'Normal Use',days:'Current',color:'#00D4AA',
   util:'18%',score:'780',pd:'0.4%',riskPct:8,
   desc:'The borrower is current on all payments. Balance is well within limits. Banks consider this a healthy asset generating steady interest income. Eligible for limit increases and lower interest rates.',
   tags:['On-time payments','Low utilization','Full pay monthly','Positive history'],
   bankTags:['Routine monitoring','Limit increase eligible','Low provision needed'],
   tC:'rgba(0,212,170,.15)',tB:'rgba(0,212,170,.3)',tF:'#00D4AA'},
  {icon:'▲',label:'Rising Utilization',days:'0 DPD',color:'#4DB8FF',
   util:'52%',score:'710',pd:'1.8%',riskPct:22,
   desc:'Still current but paying only the minimum. Balances growing. Bank behavioral scoring silently flags this — score declines even without late payments.',
   tags:['Minimum payments only','Balance growing','Possible income stress','Spending > repaying'],
   bankTags:['Behavioral flag raised','Risk score downgraded','Limit freeze review'],
   tC:'rgba(77,184,255,.12)',tB:'rgba(77,184,255,.3)',tF:'#4DB8FF'},
  {icon:'!',label:'Payment Delay',days:'1–30 DPD',color:'#F5A623',
   util:'71%',score:'648',pd:'5.2%',riskPct:40,
   desc:'First missed payment. Late fee immediately charged. No bureau report yet (begins at 30 DPD). Risk score drops sharply. One on-time payment now has outsized positive impact.',
   tags:['First missed payment','Late fee applied','Near credit maximum','Cash flow constraint'],
   bankTags:['Internal alert raised','Automated reminders sent','Account under review'],
   tC:'rgba(245,166,35,.12)',tB:'rgba(245,166,35,.3)',tF:'#F5A623'},
  {icon:'!!',label:'Delinquency',days:'30–60 DPD',color:'#E8732A',
   util:'84%',score:'581',pd:'14%',riskPct:58,
   desc:'Two consecutive missed payments. Credit bureau now notified — permanent mark. Compound interest rapidly grows balance. Bank begins active outreach. Score damage is severe.',
   tags:['Bureau notification','Compounding interest','Income shortfall','Borrowing from others'],
   bankTags:['Collections queue','Payment plan offered','Limit suspended'],
   tC:'rgba(232,115,42,.12)',tB:'rgba(232,115,42,.3)',tF:'#E8732A'},
  {icon:'⚠',label:'Serious Delinquency',days:'60–90 DPD',color:'#D94F1E',
   util:'96%',score:'502',pd:'38%',riskPct:76,
   desc:'Collections department owns the account. Bank calculates Expected Loss and reserves funds. Probability of full default crosses critical threshold. Account may be sold to collectors.',
   tags:['Collections dept. active','Job loss possible','Legal advice sought','Multiple creditor default'],
   bankTags:['Loss provisioning','Settlement discussions','Debt restructuring'],
   tC:'rgba(217,79,30,.12)',tB:'rgba(217,79,30,.3)',tF:'#D94F1E'},
  {icon:'✕',label:'Default',days:'90+ DPD',color:'#FF4D4F',
   util:'100%',score:'430',pd:'82%',riskPct:95,
   desc:'Account charged off. Loss permanently recorded. Debt sold to recovery agency or pursued legally. Seven-year credit record impact. Bank writes outstanding amount as a loss against revenue.',
   tags:['Charge-off recorded','Legal action possible','Credit destroyed 7yr','Wage garnishment risk'],
   bankTags:['Loss recognized','Debt sale / recovery','Regulatory reporting'],
   tC:'rgba(255,77,79,.12)',tB:'rgba(255,77,79,.3)',tF:'#FF4D4F'},
];
let active=0;
function render(){
  const wrap=document.getElementById('stages');
  let html='';
  S.forEach((s,i)=>{
    const on=i===active;
    html+=`<div class="stage" onclick="sel(${i})">
      <div class="node${on?' active':''}" style="border-color:${s.color};color:${s.color};background:${on?s.color+'22':'transparent'}">${s.icon}</div>
      <div class="slbl${on?' active':''}" style="${on?'color:'+s.color:''}">${s.label}<br>
        <span style="color:#A8B4C0;font-weight:400;font-size:9px">${s.days}</span>
      </div>
    </div>`;
    if(i<S.length-1) html+=`<div class="conn"><div class="conn-line" style="background:linear-gradient(90deg,${s.color},${S[i+1].color});opacity:.4"></div></div>`;
  });
  wrap.innerHTML=html;
  const s=S[active];
  const det=document.getElementById('detail');
  det.style.borderColor=s.color+'40'; det.style.background=s.color+'06';
  det.innerHTML=`
    <div class="d-header">
      <div class="d-icon">${s.icon}</div>
      <div><div class="d-title" style="color:${s.color}">${s.label}</div>
        <div class="d-days">${s.days} — Days Past Due</div></div>
    </div>
    <div class="stats-row">
      ${[['Utilization',s.util,'of credit limit used'],['Credit Score',s.score,'CIBIL / bureau est.'],['Prob. of Default',s.pd,"bank's internal model"]].map(([l,v,b])=>`
      <div class="stat"><div class="stat-lbl">${l}</div><div class="stat-val" style="color:${s.color}">${v}</div><div class="stat-bench">${b}</div></div>`).join('')}
    </div>
    <div class="d-desc">${s.desc}</div>
    <div class="sec-lbl">Borrower signals</div>
    <div class="tag-row">${s.tags.map(t=>`<span class="tag" style="background:${s.tC};border-color:${s.tB};color:${s.tF}">${t}</span>`).join('')}</div>
    <div class="sec-lbl">Bank responses</div>
    <div class="tag-row">${s.bankTags.map(t=>`<span class="tag" style="background:rgba(88,166,255,.1);border-color:rgba(88,166,255,.25);color:#58A6FF">${t}</span>`).join('')}</div>
    <div class="rbar">
      <div class="rlbl"><span>Bank risk exposure</span><span style="color:${s.color}">${s.riskPct}% — ${['Very Low','Low','Moderate','Elevated','High','Critical'][active]}</span></div>
      <div class="rtrack"><div class="rfill" style="width:${s.riskPct}%;background:linear-gradient(90deg,#00D4AA,${s.color})"></div></div>
    </div>`;
}
function sel(i){active=i;render();}
render();
</script></body></html>
""", height=600)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── GLOSSARY ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📖 Complete Glossary — Plain English</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Every technical term — explained simply, with real examples and formulas</div>", unsafe_allow_html=True)

    tabs = st.tabs(["💳 Credit Basics","📐 Calculations & Formulas","🤖 ML Concepts","📊 Metrics & Stats","📁 Dataset Variables"])

    with tabs[0]:
        _gcard("Credit Limit (LIMIT_BAL)","#00D4AA",
               "The maximum amount a bank is willing to lend you on a credit card. Think of it as the ceiling of your borrowing power.",
               "If your limit is NT$100,000, you cannot charge more than that to the card.")
        _gcard("Credit Utilization","#4DB8FF",
               "The percentage of your available credit you are currently using. One of the most powerful signals banks use — higher utilization signals financial stress.",
               "If you owe NT$60,000 on a NT$100,000 limit, your utilization is 60%.",
               "Utilization = (Balance Owed ÷ Credit Limit) × 100")
        _gcard("Repayment Status (PAY_0 to PAY_6)","#F5A623",
               "How many months behind you are on payments. -1 means paid on time (or early). 0 means minimum payment. 1 means 1 month late, 2 means 2 months late, etc.",
               "PAY_0 = 2 means the customer was 2 months late on the September 2005 payment.")
        _gcard("Bill Amount (BILL_AMT1–6)","#BC8CFF",
               "The total outstanding balance on your credit card statement for a given month. This includes all charges, unpaid balances rolled over, and interest.",
               "BILL_AMT1 = NT$29,000 means there was NT$29,000 owed in the September statement.")
        _gcard("Payment Amount (PAY_AMT1–6)","#3FB950",
               "The actual cash amount the customer paid toward their bill in a given month. This is what they sent to the bank — not what they owed.",
               "PAY_AMT1 = NT$1,500 means the customer sent NT$1,500 in September regardless of how much they owed.")
        _gcard("Default","#FF4D4F",
               "Failure to repay debt per the agreed schedule. In this dataset, 'default' means the customer did NOT pay their minimum balance for the following month.",
               "DEFAULT = 1 → this person will fail to pay next month. This is what we are trying to predict.")
        _gcard("Charge-off","#FF7B72",
               "When a bank officially declares a loan as a loss on its books — typically after 180 days of non-payment. The debt still exists legally, but the bank has given up on recovering it through normal means.",
               "After 6 months of non-payment, the bank records a charge-off and sells the debt to a collections agency.")
        _gcard("Delinquency","#F0883E",
               "Being overdue on a payment by any amount of time. Technically starts at 1 day past due. More severe stages: 30-DPD, 60-DPD, 90-DPD (days past due).",
               "A customer who missed their August payment is 30 days delinquent by September.")

    with tabs[1]:
        _gcard("Credit Utilization Rate","#00D4AA",
               "Measures how much of your available credit you are using. Above 30% raises concern; above 75% is a strong default signal.",
               "Customer A: BILL_AMT1 = NT$80,000, LIMIT_BAL = NT$100,000 → Utilization = 80% (High Risk)",
               "Utilization = BILL_AMT1 ÷ LIMIT_BAL × 100")
        _gcard("Payment-to-Bill Ratio","#4DB8FF",
               "How much of the monthly bill the customer actually paid. A ratio of 1.0 = paid in full. Below 0.1 = barely paying the minimum — a strong stress signal.",
               "Bill = NT$20,000, Payment = NT$2,000 → Ratio = 0.10 (paying only 10%)",
               "Pay Ratio = PAY_AMT1 ÷ BILL_AMT1")
        _gcard("Average Payment Status","#F5A623",
               "The average repayment delay across all 6 months. Negative values are good (paid early). Positive values indicate months of delay.",
               "Values of [0, 1, 2, 1, 0, -1] → Avg = 0.5 (occasional delay)",
               "Avg Status = (PAY_0 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6) ÷ 6")
        _gcard("Total Outstanding Debt","#BC8CFF",
               "Sum of all 6 months of bill statements. Gives a full picture of cumulative debt exposure across the analysis period.",
               "Six bills of NT$30K each → Total = NT$180,000 total debt exposure",
               "Total Bill = BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6")
        _gcard("Gini Coefficient (Model Quality)","#3FB950",
               "Measures how well a model separates defaulters from non-defaulters. 0 = random guessing, 100 = perfect. Industry standard for credit scoring models.",
               "AUC = 0.82 → Gini = 0.64 → Good discriminatory power",
               "Gini = 2 × AUC − 1")
        _gcard("Log Loss","#FF4D4F",
               "Measures the accuracy of probability predictions. Lower is better. A model that says 99% confidence and is wrong gets severely penalized.",
               "Predicted 90% default, actual = default → Low loss. Predicted 90% default, actual = no default → High loss",
               "Log Loss = −[y × log(p) + (1−y) × log(1−p)]")

    with tabs[2]:
        for term, color, defn, ex in [
            ("Classification Model","#00D4AA",
             "A model that predicts which category something belongs to. In our case: 'Will this person default? YES or NO?' The model outputs a probability (0 to 1), and we pick a threshold (e.g. 0.5) to decide.",
             "If model outputs 0.73, we interpret: 73% chance this person defaults next month."),
            ("Training vs Testing Split","#4DB8FF",
             "We divide data into training (80%) — what the model learns from — and testing (20%) — data the model has never seen, used to measure real-world performance.",
             "30,000 rows → 24,000 training rows + 6,000 test rows"),
            ("Logistic Regression","#F5A623",
             "The simplest classification model. It finds a mathematical boundary that separates defaulters from non-defaulters using a sigmoid curve. Highly interpretable — you can see exactly which variable pushed the decision.",
             "Output: probability = 1 / (1 + e^(-z)) where z is a weighted sum of your features"),
            ("Random Forest","#BC8CFF",
             "An ensemble of hundreds of decision trees, each trained on slightly different data. Each tree votes, and the majority wins. Handles non-linear patterns naturally.",
             "500 trees each say YES or NO → 340 say YES → prediction = YES (68% confidence)"),
            ("XGBoost","#3FB950",
             "Gradient Boosted trees — each new tree specifically learns from the mistakes of the previous one. State-of-the-art for tabular data. Often wins machine learning competitions.",
             "Tree 1 gets 70% right → Tree 2 focuses on the 30% errors → combined = 92%"),
            ("SMOTE","#58A6FF",
             "Synthetic Minority Over-sampling Technique. Creates synthetic minority-class examples by interpolating between existing ones. Handles class imbalance — applied to training set only.",
             "22% defaulters vs 78% non-defaulters → SMOTE → 50/50 balanced training set"),
            ("Class Imbalance","#F0883E",
             "When one outcome is much rarer than the other. Here, only 22% defaulted vs 78% didn't. A lazy model can achieve 78% accuracy just by predicting 'no default' — but that's useless!",
             "22% defaulters vs 78% non-defaulters → model needs to be smart, not lazy"),
            ("ROC-AUC","#FF4D4F",
             "Receiver Operating Characteristic — Area Under Curve. Measures model quality independent of threshold. 0.5 = random coin flip. 0.8 = good. 0.9+ = excellent. The single best number to compare models.",
             "AUC=0.82 means: if you pick one defaulter and one non-defaulter at random, the model ranks the defaulter higher 82% of the time"),
        ]:
            _gcard_noex(term, color, defn, ex)

    with tabs[3]:
        for term, color, defn, ex in [
            ("Precision","#00D4AA",
             "Of all the people your model predicted WILL default, how many actually did? High precision = fewer false alarms. Banks care about this because investigating false alarms wastes money.",
             "Model flagged 100 people → 78 actually defaulted → Precision = 78%"),
            ("Recall (Sensitivity)","#F5A623",
             "Of all the people who actually DID default, how many did your model correctly catch? High recall = fewer missed defaults. Banks care about this even more — missing a defaulter costs real money.",
             "200 real defaulters in test set → model caught 160 → Recall = 80%"),
            ("F1 Score","#4DB8FF",
             "The harmonic mean of Precision and Recall. Gives a single number that balances both. Especially useful when the dataset is imbalanced.",
             "F1 = 2 × (0.78 × 0.80) / (0.78 + 0.80) = 0.79"),
            ("Confusion Matrix","#BC8CFF",
             "A 2×2 table showing all four outcomes: True Positive (correctly predicted default), True Negative (correctly predicted no default), False Positive (wrongly flagged), False Negative (missed a real default).",
             "TP=160, TN=1800, FP=44, FN=40 → reading the full picture"),
            ("Correlation","#3FB950",
             "How strongly two variables move together. Ranges from -1 (perfectly opposite) to +1 (perfectly aligned). 0 = no relationship.",
             "PAY_0 and DEFAULT have correlation +0.32 → strong positive → late payments predict default"),
            ("Standard Deviation","#FF4D4F",
             "Measures how spread out the values are. Small std = everyone has similar values. Large std = wide range.",
             "Average age = 35.5 years, std = 9.2 years → most customers between 26–45 years old"),
            ("P-value","#F0883E",
             "In statistical testing: the probability of seeing this result by chance. Below 0.05 (5%) = the pattern is real, not random.",
             "p-value = 0.001 for PAY_0 → extremely unlikely to see this correlation by chance"),
        ]:
            _gcard_noex(term, color, defn, ex)

    with tabs[4]:
        st.markdown("<div class='insight'>This dataset covers <b>30,000 credit card clients in Taiwan</b> from April to September 2005. Each row is one customer. The target variable tells us: did they default on their payment the following month?</div>", unsafe_allow_html=True)
        var_data = {
            "Variable": ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1–6","PAY_AMT1–6","DEFAULT"],
            "Type": ["ID","Numeric","Categorical","Categorical","Categorical","Numeric","Ordinal","Ordinal","Ordinal","Ordinal","Ordinal","Ordinal","Numeric","Numeric","Target"],
            "Description": ["Unique customer identifier (not used in modeling)","Credit limit in NT dollars — individual + supplementary","1 = Male, 2 = Female","1=Graduate, 2=University, 3=High School, 4=Others","1=Married, 2=Single, 3=Others","Customer age in years","Repayment status September 2005 (most recent)","Repayment status August 2005","Repayment status July 2005","Repayment status June 2005","Repayment status May 2005","Repayment status April 2005 (oldest)","Bill statement amount for each month (Sep → Apr)","Previous payment amount for each month (Sep → Apr)","1 = Defaulted next month, 0 = Did not default"],
            "Values / Range": ["1 – 30,000","NT$10K – NT$1M","1, 2","1, 2, 3, 4, 5, 6","1, 2, 3","21 – 79 years","-2, -1, 0, 1, 2 … 9","-2, -1, 0, 1, 2 … 8","-2, -1, 0, 1, 2 … 8","-2, -1, 0, 1, 2 … 8","-2, -1, 0, 1, 2 … 8","-2, -1, 0, 1, 2 … 8","NT$0 – NT$1M+","NT$0 – NT$500K+","0, 1"]
        }
        st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)
        st.markdown("<div class='insight-warn' style='margin-top:16px'>⚠️ <b>PAY column note:</b> PAY_0 is September (most recent), PAY_2 is August. There is no PAY_1 — this is a quirk of the original dataset. Values of -2 indicate no consumption that month, -1 means paid in full, 0 means revolving credit, and positive integers indicate months of delay.</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Exploratory Analysis":
    st.markdown("<div class='hero-title' style='font-size:38px'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Understanding the data before building any model — the most important step</div>", unsafe_allow_html=True)

    COLORS = {'Default':'#FF4D4F','No Default':'#00D4AA'}
    eda_tabs = st.tabs(["🔍 Data Profile","🎯 Target Analysis","📈 Distributions","🔗 Correlations","💡 Key Insights"])

    with eda_tabs[0]:
        st.markdown("<div class='section-title'>Dataset Profile</div>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Rows",f"{df.shape[0]:,}")
        with c2: st.metric("Columns",f"{df.shape[1]}")
        with c3: st.metric("Missing Values","0",delta="Clean dataset")
        with c4: st.metric("Duplicate IDs","0")
        st.markdown("<br><div class='section-title'>Statistical Summary</div>", unsafe_allow_html=True)
        nc = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
        s = df[nc].describe().T
        s.columns = ['Count','Mean','Std Dev','Min','25th %ile','Median','75th %ile','Max']
        st.dataframe(s.round(0).astype(int), use_container_width=True)
        st.markdown("<div class='insight'>All columns are numeric with no missing values. The dataset is clean and ready for analysis. BILL_AMT and PAY_AMT columns have wide ranges — some customers have zero bills (no usage) while others carry NT$1M+ in balances.</div>", unsafe_allow_html=True)

    with eda_tabs[1]:
        st.markdown("<div class='section-title'>Target Variable — Default Distribution</div>", unsafe_allow_html=True)
        col1,col2 = st.columns([1,2])
        with col1:
            cnts = df['DEFAULT_LABEL'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=L(cnts.index), values=L(cnts.values),
                hole=0.65, marker=dict(colors=['#00D4AA','#FF4D4F'], line=dict(color='#0D1117',width=3)),
                textfont=dict(size=13,family='Inter'))])
            fig.add_annotation(text="22.1%<br>defaulted",x=0.5,y=0.5,
                font=dict(size=16,color='#FF4D4F',family='Inter'),showarrow=False)
            fig.update_layout(**PT,height=320,showlegend=True,
                legend=dict(x=0.3,y=-0.05,orientation='h'),margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            demo_groups = {'By Sex':('SEX_LABEL','sex'),'By Education':('EDU_LABEL','education'),'By Marriage':('MAR_LABEL','marital status')}
            sel_demo = st.selectbox("Break down default rate by:",list(demo_groups.keys()))
            cn, cl = demo_groups[sel_demo]
            dr = df.groupby(cn)['DEFAULT'].agg(['mean','count']).reset_index()
            dr['mean'] = dr['mean']*100
            dr.columns = [cl,'Default Rate (%)','Count']
            fig2 = px.bar(pxdf(dr),x=cl,y='Default Rate (%)',
                text=dr['Default Rate (%)'].round(1).astype(str)+'%',
                color='Default Rate (%)',
                color_continuous_scale=[[0,'#00D4AA'],[0.5,'#F0883E'],[1,'#FF4D4F']])
            fig2.update_traces(textposition='outside')
            fig2.update_layout(**PT,height=320,showlegend=False,
                coloraxis_showscale=False,margin=dict(t=20,b=10,l=10,r=10))
            st.plotly_chart(fig2,use_container_width=True)
        st.markdown("<div class='section-title'>Default Rate by Age Group</div>", unsafe_allow_html=True)
        age_dr = df.groupby('AGE_GROUP',observed=True)['DEFAULT'].agg(['mean','count']).reset_index()
        age_dr['mean'] = (age_dr['mean']*100).round(1)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=L(age_dr['AGE_GROUP'].astype(str)),y=L(age_dr['mean']),
            marker_color=['#FF4D4F' if v>25 else '#F0883E' if v>20 else '#00D4AA' for v in L(age_dr['mean'])],
            text=[str(v)+'%' for v in L(age_dr['mean'])],textposition='outside'))
        fig3.update_layout(**PT,height=300,xaxis_title='Age Group',yaxis_title='Default Rate (%)',margin=dict(t=20,b=10,l=10,r=10))
        st.plotly_chart(fig3,use_container_width=True)
        st.markdown("<div class='insight'>Younger customers (21–25) show the highest default rate — likely due to lower financial stability and less credit experience. Risk decreases with age up to ~45, then stabilizes.</div>", unsafe_allow_html=True)

    with eda_tabs[2]:
        st.markdown("<div class='section-title'>Variable Distributions</div>", unsafe_allow_html=True)
        sel_col = st.selectbox("Select variable to explore:",
            ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','UTIL_RATE','PAY_RATIO'])
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(pxdf(df),x=sel_col,color='DEFAULT_LABEL',color_discrete_map=COLORS,
                nbins=50,barmode='overlay',opacity=0.75,labels={'DEFAULT_LABEL':'Status'})
            fig.update_layout(**PT,height=320,title=f'Distribution of {sel_col} by Default Status',margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2 = px.box(pxdf(df),x='DEFAULT_LABEL',y=sel_col,color='DEFAULT_LABEL',color_discrete_map=COLORS,points=False)
            fig2.update_layout(**PT,height=320,title=f'{sel_col} — Box Plot by Default Status',margin=dict(t=40,b=10,l=10,r=10),showlegend=False)
            st.plotly_chart(fig2,use_container_width=True)
        st.markdown("<div class='section-title'>Repayment Status Distribution (PAY_0)</div>", unsafe_allow_html=True)
        pd0 = df['PAY_0'].value_counts().sort_index().reset_index()
        pd0.columns=['PAY_Status','Count']
        pd0['Label']=pd0['PAY_Status'].map({-2:'-2 (No consumption)',-1:'-1 (Paid on time)',0:'0 (Revolving)',1:'1 (1 month late)',2:'2 months late',3:'3 months late',4:'4 months late',5:'5 months late',6:'6 months late',7:'7 months late',8:'8+ months late'}).fillna(pd0['PAY_Status'].astype(str))
        fig3=px.bar(pd0.assign(Count=L(pd0['Count'])),x='Label',y='Count',color='Count',color_continuous_scale=[[0,'#00D4AA'],[0.5,'#F0883E'],[1,'#FF4D4F']])
        fig3.update_layout(**PT,height=300,coloraxis_showscale=False,xaxis_tickangle=-30,margin=dict(t=20,b=80,l=10,r=10))
        st.plotly_chart(fig3,use_container_width=True)

    with eda_tabs[3]:
        st.markdown("<div class='section-title'>Correlation Matrix</div>", unsafe_allow_html=True)
        cc=['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','DEFAULT']
        cm=df[cc].corr()
        fig=px.imshow(cm.round(3),color_continuous_scale='RdBu_r',zmin=-1,zmax=1,aspect='auto',text_auto='.2f')
        fig.update_traces(textfont=dict(size=9))
        fig.update_layout(**PT,height=520,coloraxis_colorbar=dict(title='Correlation'),margin=dict(t=20,b=10,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
        ct=df[cc].corr()['DEFAULT'].drop('DEFAULT').sort_values(key=abs,ascending=True)
        fig2=go.Figure(go.Bar(x=L(ct.values),y=L(ct.index),orientation='h',
            marker_color=['#FF4D4F' if v>0 else '#00D4AA' for v in L(ct.values)],
            text=[f'{v:.3f}' for v in L(ct.values)],textposition='outside'))
        fig2.update_layout(**PT,height=400,xaxis_title='Correlation with DEFAULT',margin=dict(t=20,b=10,l=10,r=30))
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown("<div class='insight'>PAY_0 (September payment status) has the highest positive correlation with default at +0.32. LIMIT_BAL has a negative correlation (−0.15) — higher credit limits are extended to more creditworthy customers who are less likely to default. Bill amounts show moderate positive correlation.</div>", unsafe_allow_html=True)

    with eda_tabs[4]:
        st.markdown("<div class='section-title'>💡 Key Data Insights</div>", unsafe_allow_html=True)
        for title, color, text, cls in [
            ("🎯 Strong Class Imbalance","#F0883E","78% of customers did NOT default vs 22% who did. A model that always predicts 'safe' gets 78% accuracy — but catches zero actual defaulters. We must use AUC-ROC and F1 Score as our primary metrics, not raw accuracy.","insight-warn"),
            ("🔑 PAY_0 is King","#00D4AA","The most recent payment status (September 2005) has the highest correlation with default at +0.32. This single feature alone separates defaulters from non-defaulters better than any other variable.","insight"),
            ("💰 Credit Limit as a Proxy for Creditworthiness","#4DB8FF","Higher LIMIT_BAL customers default less (-0.15 correlation). Banks already screen applicants before granting high limits — so high-limit customers are pre-selected for better credit behavior.","insight"),
            ("📈 Bill Amounts are Highly Correlated With Each Other","#BC8CFF","BILL_AMT1 through BILL_AMT6 correlate at 0.8+ with each other. We engineer TOTAL_BILL and UTIL_RATE instead of using all 6 separately.","insight"),
            ("👤 Demographics Matter Less Than Behavior","#3FB950","Gender, education, and marital status show ±3–5% default rate differences. Behavioral features (payment history, utilization) are 5–10× more predictive than who someone is.","insight"),
            ("⚠️ The 'Never-Used' Problem","#FF4D4F","Many customers have BILL_AMT = 0 for several months — they have the card but don't use it. Some still default (annual fees, small charges). We handle zero-bill records carefully in feature engineering.","insight-red"),
        ]:
            st.markdown(f"<div class='{cls}'><b style='color:{color}'>{title}</b><br>{text}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Feature Engineering":
    st.markdown("<div class='hero-title' style='font-size:38px'>Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Transforming raw data into powerful signals that models can learn from</div>", unsafe_allow_html=True)
    st.markdown("<div class='insight'>Raw columns tell the model <i>what happened</i>. Engineered features tell it <i>what it means</i>. Combining and transforming columns creates richer signals.</div>", unsafe_allow_html=True)

    COLORS = {'Default':'#FF4D4F','No Default':'#00D4AA'}
    fe_tabs = st.tabs(["🔧 Engineered Features","📊 Feature Importance Preview","🔄 Preprocessing Pipeline"])

    with fe_tabs[0]:
        features = [
            ("UTIL_RATE","#00D4AA","Credit Utilization Rate","BILL_AMT1 ÷ LIMIT_BAL",
             "How much of the credit limit is being used. Values above 0.75 signal high stress.",df['UTIL_RATE'].describe()),
            ("PAY_RATIO","#4DB8FF","Payment Coverage Ratio","TOTAL_PAY ÷ TOTAL_BILL",
             "What fraction of total bill was actually paid back. Below 0.1 means barely making minimums.",df['PAY_RATIO'].describe()),
            ("AVG_PAY_STATUS","#F5A623","Average Repayment Delay","(PAY_0 + PAY_2 + ... + PAY_6) ÷ 6",
             "Mean months of delay across all 6 months. Positive = consistently late.",df['AVG_PAY_STATUS'].describe()),
            ("TOTAL_BILL","#BC8CFF","Total 6-Month Bill Exposure","Sum of BILL_AMT1 through BILL_AMT6",
             "Captures cumulative debt load across the entire observation period.",df['TOTAL_BILL'].describe()),
            ("TOTAL_PAY","#3FB950","Total 6-Month Payments Made","Sum of PAY_AMT1 through PAY_AMT6",
             "How much the customer actually sent to the bank over 6 months.",df['TOTAL_PAY'].describe()),
        ]
        for feat, color, name, formula, desc, stats in features:
            with st.expander(f"**{feat}** — {name}", expanded=False):
                c1,c2 = st.columns([1.5,1])
                with c1:
                    st.markdown(f"<div class='gloss-def'>{desc}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='formula'>{formula}</div>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:12px'>
                        <span class='pill pill-teal'>Mean: {stats['mean']:.2f}</span>
                        <span class='pill pill-blue'>Median: {stats['50%']:.2f}</span>
                        <span class='pill pill-orange'>Std: {stats['std']:.2f}</span>
                        <span class='pill pill-red'>Max: {stats['max']:.2f}</span>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    fig = px.histogram(pxdf(df), x=feat, color='DEFAULT_LABEL',
                        color_discrete_map=COLORS, nbins=40, barmode='overlay', opacity=0.75, height=200)
                    fig.update_layout(**PT, showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
                    st.plotly_chart(fig, use_container_width=True)

    with fe_tabs[1]:
        st.markdown("<div class='section-title'>Feature Correlation with Default (All Features)</div>", unsafe_allow_html=True)
        fc = ['LIMIT_BAL','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
              'BILL_AMT1','PAY_AMT1','UTIL_RATE','PAY_RATIO','AVG_PAY_STATUS','TOTAL_BILL','TOTAL_PAY']
        corr = df[fc+['DEFAULT']].corr()['DEFAULT'].drop('DEFAULT').sort_values(key=abs,ascending=False)
        fig = go.Figure(go.Bar(x=L(corr.index),y=L(corr.values),
            marker_color=['#FF4D4F' if v>0 else '#00D4AA' for v in L(corr.values)],
            text=[f'{v:.3f}' for v in L(corr.values)],textposition='outside'))
        fig.update_layout(**PT,height=380,yaxis_title='Correlation with DEFAULT',
            xaxis_tickangle=-30,margin=dict(t=20,b=80,l=10,r=10))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("<div class='insight'><b>AVG_PAY_STATUS</b> and <b>PAY_0</b> are the strongest predictors. UTIL_RATE and PAY_RATIO add signal beyond the raw columns by normalizing relative to limit and total exposure.</div>", unsafe_allow_html=True)

    with fe_tabs[2]:
        st.markdown("<div class='section-title'>Preprocessing Pipeline</div>", unsafe_allow_html=True)
        for step, label, desc, formula in [
            ("1","Drop ID","Remove the customer ID column — it is just a row number and carries zero predictive signal.",
             "X = df.drop(columns=['ID', 'DEFAULT', label_cols...])"),
            ("2","Encode Categoricals","SEX, EDUCATION, and MARRIAGE are already numeric (1,2,3...) in the raw data. We keep them as-is since the model can interpret ordinal relationships.",
             "No transformation needed — already numeric"),
            ("3","Feature Scaling","Logistic Regression is sensitive to scale — a column ranging 0–1,000,000 will dominate columns ranging 0–1. We apply StandardScaler. Tree-based models do NOT need scaling.",
             "X_scaled = (X - mean) ÷ std_deviation"),
            ("4","Handle Class Imbalance","With 78% non-defaulters, models are biased toward predicting 'no default'. We use SMOTE on the training set to balance classes to 50/50.",
             "SMOTE: x_new = xi + λ × (xj − xi)   where λ ~ Uniform(0,1)"),
            ("5","Train / Test Split (80/20)","We reserve 20% of data (6,000 rows) as a test set the model never sees during training. Stratified split preserves the 22% default rate in both halves.",
             "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)"),
        ]:
            st.markdown(
                "<div class='card'>"
                f"<div class='label-sm'>Step {step} — {label}</div>"
                f"<div class='gloss-def' style='margin-top:6px'>{desc}</div>"
                f"<div class='formula'>{formula}</div>"
                "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL BUILDING (loads PKL, inspect any model)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Building":
    st.markdown("<div class='hero-title' style='font-size:38px'>Model Building</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>All models are pre-trained with SMOTE + hyperparameter tuning — loaded instantly from PKL files</div>", unsafe_allow_html=True)
    st.markdown("<div class='insight'>Models were trained offline using <code>model_train.py</code> with SMOTE balancing and GridSearchCV. Select any model below to inspect its performance and feature importance.</div>", unsafe_allow_html=True)

    col1,col2 = st.columns([1,2])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label-sm'>Select model to inspect</div>", unsafe_allow_html=True)
        sel_model = st.selectbox("",list(pkls.keys()),label_visibility="collapsed")
        r = results_json.get(sel_model,{})
        if r:
            st.markdown("<div class='label-sm' style='margin-top:16px'>Best hyperparameters</div>", unsafe_allow_html=True)
            for k,v in r.get('best_params',{}).items():
                st.markdown(
                    f"<div style='font-size:13px;padding:4px 0;color:#B0BAC6'>"
                    f"<span style='color:#E6EDF3;font-weight:500'>{k}</span> = "
                    f"<span style='color:#00D4AA;font-family:JetBrains Mono,monospace'>{v}</span></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='label-sm' style='margin-top:16px'>Training details</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:12px;color:#B0BAC6;line-height:2'>"
            "SMOTE applied: ✅<br>Class balanced: ✅<br>CV folds: 5 (LR) / 3 (others)<br>"
            "Scoring: AUC-ROC<br>Test set: 6,000 rows (unseen)</div>",
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if r:
            # Calculate metrics from confusion matrix if report is missing
            rep = r.get('report', {})
            cm = np.array(r.get('cm', [[0,0],[0,0]]))
            
            # Get metrics from report if available, otherwise calculate from confusion matrix
            if rep and '1' in rep:
                precision = rep.get('1', {}).get('precision', 0)
                recall = rep.get('1', {}).get('recall', 0)
                f1_score = rep.get('1', {}).get('f1-score', 0)
            elif cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1_score = 0
            
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("AUC-ROC",f"{r.get('auc',0):.4f}")
            with c2: st.metric("Precision",f"{precision:.3f}")
            with c3: st.metric("Recall",f"{recall:.3f}")
            with c4: st.metric("F1 Score",f"{f1_score:.3f}")

            t1,t2,t3 = st.tabs(["ROC Curve","Confusion Matrix","Feature Importance"])

            with t1:
                fpr=r.get('fpr',[]); tpr=r.get('tpr',[])
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=L(fpr),y=L(tpr),fill='tozeroy',
                    fillcolor='rgba(0,212,170,0.1)',line=dict(color='#00D4AA',width=2.5),
                    name=f"AUC={r.get('auc',0):.4f}"))
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],
                    line=dict(color='#A8B4C0',dash='dash'),name='Random (AUC=0.5)'))
                fig.update_layout(**PT,height=380,
                    xaxis_title='False Positive Rate',yaxis_title='True Positive Rate',
                    title=f'ROC Curve — {sel_model}', margin=dict(t=40,b=20,l=20,r=20))
                st.plotly_chart(fig,use_container_width=True)

            with t2:
                cm = np.array(r.get('cm',[[0,0],[0,0]]))
                labels=['No Default','Default']
                fig=px.imshow([[int(v) for v in row] for row in cm],text_auto=True,x=labels,y=labels,
                    color_continuous_scale=[[0,'#1C2330'],[0.5,'#0A3A5C'],[1,'#00D4AA']],
                    labels=dict(x='Predicted',y='Actual'))
                fig.update_layout(**PT,height=380,title='Confusion Matrix',coloraxis_showscale=False,margin=dict(t=40,b=20,l=20,r=20))
                st.plotly_chart(fig,use_container_width=True)
                if cm.size==4:
                    tn,fp,fn,tp = cm.ravel()
                    st.markdown(f"""
                    <div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:8px'>
                        <span class='pill pill-teal'>✓ True Negatives (Correctly safe): {tn:,}</span>
                        <span class='pill pill-teal'>✓ True Positives (Caught defaulters): {tp:,}</span>
                        <span class='pill pill-orange'>✗ False Positives (False alarms): {fp:,}</span>
                        <span class='pill pill-red'>✗ False Negatives (Missed defaulters): {fn:,}</span>
                    </div>""", unsafe_allow_html=True)

            with t3:
                fi = r.get('feature_importances',{})
                if fi:
                    fi_s = pd.Series(fi).sort_values(ascending=False).head(15)
                    fig=go.Figure(go.Bar(x=L(fi_s.values),y=L(fi_s.index),orientation='h',
                        marker=dict(color=L(fi_s.values),colorscale=[[0,'#1C4A6E'],[0.5,'#4DB8FF'],[1,'#00D4AA']]),
                        text=[f'{v:.4f}' for v in L(fi_s.values)],textposition='outside'))
                    fig.update_layout(**PT,height=400,title='Top 15 Feature Importances',
                        xaxis_title='Importance Score',margin=dict(t=40,b=20,l=20,r=60))
                    st.plotly_chart(fig,use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#161B22;border:1px dashed #30363D;border-radius:16px;
                        height:350px;display:flex;align-items:center;justify-content:center;
                        flex-direction:column;gap:12px'>
                <div style='font-size:40px'>🤖</div>
                <div style='color:#A8B4C0;font-size:14px'>Model results not found — run model_train.py first</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Comparison":
    st.markdown("<div class='hero-title' style='font-size:38px'>Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>All models head-to-head — pick the winner with confidence</div>", unsafe_allow_html=True)

    if not results_json and comp_df is None:
        st.error("No results found. Run model_train.py first.")
        st.stop()

    # Leaderboard - use comparison_table.csv if available, otherwise calculate from results_json
    st.markdown("<div class='section-title'>🏆 Model Leaderboard</div>", unsafe_allow_html=True)
    
    if comp_df is not None:
        # Use comparison_table.csv data
        ldf = comp_df.copy()
        ldf = ldf.sort_values('AUC', ascending=False)
        
        # Add caught defaulters column from confusion matrices
        rows_with_caught = []
        for _, row in ldf.iterrows():
            model_name = row['Model']
            # Find matching model in results_json
            if model_name in results_json:
                cm = np.array(results_json[model_name].get('cm', [[0,0],[0,0]]))
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    row['Caught Defaulters'] = f"{tp:,} / {tp+fn:,}"
                    row['False Alarms'] = f"{fp:,}"
                else:
                    row['Caught Defaulters'] = "N/A"
                    row['False Alarms'] = "N/A"
            else:
                row['Caught Defaulters'] = "N/A"
                row['False Alarms'] = "N/A"
            rows_with_caught.append(row)
        
        ldf = pd.DataFrame(rows_with_caught)
        
        # Display table with formatted columns
        display_cols = ['Model', 'AUC', 'F1 Score', 'Precision', 'Recall', 'Test Accuracy', 'Caught Defaulters', 'False Alarms']
        st.dataframe(ldf[display_cols], use_container_width=True, hide_index=True)
        
    else:
        # Fallback: calculate from results_json
        rows = []
        for name, r in results_json.items():
            cm = np.array(r.get('cm', [[0,0],[0,0]]))
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Calculate precision and recall from confusion matrix
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            rows.append({
                'Model': name,
                'AUC-ROC': round(r.get('auc', 0), 4),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'F1 Score': round(f1, 3),
                'Accuracy': round(accuracy, 3),
                'Caught Defaulters': f"{tp:,} / {tp+fn:,}",
                'False Alarms': f"{fp:,}"
            })
        ldf = pd.DataFrame(rows).sort_values('AUC-ROC', ascending=False)
        st.dataframe(ldf, use_container_width=True, hide_index=True)
    
    best = ldf.iloc[0]['Model']
    best_auc = ldf.iloc[0]['AUC'] if 'AUC' in ldf.columns else ldf.iloc[0]['AUC-ROC']
    st.markdown(f"<div class='insight'>🏆 <b>{best}</b> achieves the highest AUC score of {best_auc:.4f}, making it the best overall discriminator between defaulters and non-defaulters. AUC is our primary metric because it measures ranking quality independent of threshold — crucial for credit scoring.</div>", unsafe_allow_html=True)

    comp_tabs = st.tabs(["📈 ROC Curves","📉 Precision-Recall","🔢 Confusion Matrices","🎯 Feature Importance"])

    with comp_tabs[0]:
        st.markdown("<div class='section-sub'>ROC curves show the trade-off between True Positive Rate and False Positive Rate</div>", unsafe_allow_html=True)
        
        # Check if we have fpr/tpr data in results_json
        has_roc_data = any('fpr' in r and 'tpr' in r for r in results_json.values())
        
        if has_roc_data:
            fig = go.Figure()
            for i, (name, r) in enumerate(results_json.items()):
                if 'fpr' in r and 'tpr' in r:
                    fig.add_trace(go.Scatter(
                        x=L(r.get('fpr', [])),
                        y=L(r.get('tpr', [])),
                        name=f"{name} (AUC={r.get('auc', 0):.4f})",
                        line=dict(color=COLORS_MODEL[i % 4], width=2.5)
                    ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                line=dict(color='#A8B4C0', dash='dash'),
                name='Random Classifier'
            ))
            fig.update_layout(
                **PT, height=450,
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                title='ROC Curves — All Models',
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight'>The ROC curve shows how well each model trades off catching defaulters (True Positive Rate) against false alarms (False Positive Rate). A curve closer to the top-left corner is better. The diagonal line represents a model no better than random chance.</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ ROC curve data (fpr/tpr) not found in results_summary.json. Please regenerate models with ROC curve data.")
            st.info("💡 To fix this, update your model training script to save fpr and tpr values from sklearn.metrics.roc_curve()")

    with comp_tabs[1]:
        st.markdown("<div class='section-sub'>Precision-Recall curves are more informative for imbalanced datasets</div>", unsafe_allow_html=True)
        
        # Check if we have precision/recall curve data
        has_pr_data = any('rec' in r and 'prec' in r for r in results_json.values())
        
        if has_pr_data:
            fig = go.Figure()
            for i, (name, r) in enumerate(results_json.items()):
                if 'rec' in r and 'prec' in r:
                    fig.add_trace(go.Scatter(
                        x=L(r.get('rec', [])),
                        y=L(r.get('prec', [])),
                        name=name,
                        line=dict(color=COLORS_MODEL[i % 4], width=2.5),
                        fill='tozeroy' if i == 0 else None,
                        fillcolor='rgba(0,212,170,0.05)' if i == 0 else None
                    ))
            fig.update_layout(
                **PT, height=450,
                xaxis_title='Recall (% of defaulters caught)',
                yaxis_title='Precision (% of predictions correct)',
                title='Precision-Recall Curves — All Models',
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='insight'>The precision-recall curve is more informative than ROC for imbalanced datasets. A bank wants high recall (catch most defaulters) but not at the cost of too many false alarms.</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Precision-Recall curve data not found in results_summary.json. Please regenerate models with precision-recall curve data.")
            st.info("💡 To fix this, update your model training script to save precision and recall arrays from sklearn.metrics.precision_recall_curve()")

    with comp_tabs[2]:
        st.markdown("<div class='section-sub'>Confusion matrices show the breakdown of predictions vs actual outcomes</div>", unsafe_allow_html=True)
        
        # Display confusion matrices for all models
        num_models = len(results_json)
        cols_per_row = 3
        
        for i in range(0, num_models, cols_per_row):
            cols = st.columns(min(cols_per_row, num_models - i))
            for col, (name, r) in zip(cols, list(results_json.items())[i:i+cols_per_row]):
                with col:
                    cm = np.array(r.get('cm', [[0,0],[0,0]]))
                    labels = ['Safe', 'Default']
                    fig = px.imshow(
                        [[int(v) for v in row] for row in cm],
                        text_auto=True,
                        x=labels,
                        y=labels,
                        color_continuous_scale=[[0,'#1C2330'],[1,'#00D4AA']],
                        labels=dict(x='Predicted', y='Actual')
                    )
                    fig.update_layout(
                        **PT, height=280,
                        title=name,
                        coloraxis_showscale=False,
                        margin=dict(t=40, b=10, l=10, r=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                        st.markdown(f"""
                        <div style='font-size:11px;color:#A8B4C0;line-height:2'>
                        Caught: <b style='color:#00D4AA'>{tp:,}</b> defaulters<br>
                        Missed: <b style='color:#FF4D4F'>{fn:,}</b> defaulters<br>
                        False alarms: <b style='color:#F0883E'>{fp:,}</b>
                        </div>""", unsafe_allow_html=True)

    with comp_tabs[3]:
        st.markdown("<div class='section-sub'>Feature importance shows which variables drive predictions</div>", unsafe_allow_html=True)
        
        model_sel = st.selectbox("Select model to view feature importance:", list(results_json.keys()))
        fi = results_json[model_sel].get('feature_importances', {})
        
        if fi:
            top15 = pd.Series(fi).sort_values(ascending=False).head(15)
            fig = go.Figure(go.Bar(
                x=L(top15.values),
                y=L(top15.index),
                orientation='h',
                marker=dict(
                    color=L(top15.values),
                    colorscale=[[0,'#1A3C5E'],[0.5,'#4DB8FF'],[1,'#00D4AA']]
                ),
                text=[f'{v:.4f}' for v in L(top15.values)],
                textposition='outside'
            ))
            fig.update_layout(
                **PT, height=420,
                title=f'Feature Importance — {model_sel}',
                xaxis_title='Importance',
                margin=dict(t=40, b=20, l=20, r=60)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ Feature importance data not available for {model_sel}")
            st.info("💡 Feature importance is typically available for tree-based models (Random Forest, Gradient Boosting, XGBoost)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Live Predictor":
    st.markdown("<div class='hero-title' style='font-size:38px'>Live Default Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Enter a customer's details and get an instant default risk assessment with full explanation</div>", unsafe_allow_html=True)

    # Model selector
    pred_model_name = st.selectbox("Choose model for prediction:", list(pkls.keys()))
    pkg = pkls[pred_model_name]
    
    # Extract model components from the dictionary
    # Models are saved as {'model': ..., 'scaler': ..., 'features': ..., 'scaled': ...}
    pred_model = pkg.get('model')
    pred_scaler = pkg.get('scaler')
    feature_cols = pkg.get('features', [])
    use_scaled = pkg.get('scaled', False)
    
    if pred_model is None:
        # Check if this is a version mismatch error with version information
        if '_load_error' in pkg and '_expected_sklearn_version' in pkg and '_current_sklearn_version' in pkg:
            expected_version = pkg['_expected_sklearn_version']
            current_version = pkg['_current_sklearn_version']
            
            st.error(f"⚠️ Model requires sklearn {expected_version} but you have {current_version} installed")
            
            # Display resolution guidance
            display_version_mismatch_guidance(expected_version, current_version)
        else:
            # Generic error for other loading failures
            st.error(f"⚠️ Model '{pred_model_name}' could not be loaded properly. Please check the model file.")
        
        st.stop()

    st.markdown("<div class='section-title'>Customer Profile</div>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("<div class='label-sm'>Demographics</div>", unsafe_allow_html=True)
        limit_bal  = st.number_input("Credit Limit (NT$)",10000,1000000,100000,step=10000)
        age        = st.slider("Age",21,79,35)
        sex        = st.selectbox("Sex",[1,2],format_func=lambda x:"Male" if x==1 else "Female")
        education  = st.selectbox("Education",[1,2,3,4],format_func=lambda x:{1:'Graduate School',2:'University',3:'High School',4:'Others'}[x])
        marriage   = st.selectbox("Marital Status",[1,2,3],format_func=lambda x:{1:'Married',2:'Single',3:'Others'}[x])

    with col2:
        st.markdown("<div class='label-sm'>Repayment History (months of delay; -1=on time, 0=revolving)</div>", unsafe_allow_html=True)
        pay_0 = st.slider("September (PAY_0)",-2,8,0)
        pay_2 = st.slider("August (PAY_2)",-2,8,0)
        pay_3 = st.slider("July (PAY_3)",-2,8,0)
        pay_4 = st.slider("June (PAY_4)",-2,8,0)
        pay_5 = st.slider("May (PAY_5)",-2,8,0)
        pay_6 = st.slider("April (PAY_6)",-2,8,0)

    with col3:
        st.markdown("<div class='label-sm'>Bill Amounts (NT$) - Last 6 Months</div>", unsafe_allow_html=True)
        bill1  = st.number_input("Bill Sep (BILL_AMT1)", 0, 1000000, 20000, step=1000)
        bill2  = st.number_input("Bill Aug (BILL_AMT2)", 0, 1000000, 19000, step=1000)
        bill3  = st.number_input("Bill Jul (BILL_AMT3)", 0, 1000000, 18000, step=1000)
        bill4  = st.number_input("Bill Jun (BILL_AMT4)", 0, 1000000, 17000, step=1000)
        bill5  = st.number_input("Bill May (BILL_AMT5)", 0, 1000000, 16000, step=1000)
        bill6  = st.number_input("Bill Apr (BILL_AMT6)", 0, 1000000, 15000, step=1000)
        
        st.markdown("<div class='label-sm' style='margin-top:16px'>Payment Amounts (NT$) - Last 6 Months</div>", unsafe_allow_html=True)
        pay_a1 = st.number_input("Payment Sep (PAY_AMT1)", 0, 500000, 2000, step=500)
        pay_a2 = st.number_input("Payment Aug (PAY_AMT2)", 0, 500000, 2000, step=500)
        pay_a3 = st.number_input("Payment Jul (PAY_AMT3)", 0, 500000, 2000, step=500)
        pay_a4 = st.number_input("Payment Jun (PAY_AMT4)", 0, 500000, 2000, step=500)
        pay_a5 = st.number_input("Payment May (PAY_AMT5)", 0, 500000, 2000, step=500)
        pay_a6 = st.number_input("Payment Apr (PAY_AMT6)", 0, 500000, 2000, step=500)

    if st.button("⚡ Predict Default Risk"):
        # Calculate ALL engineered features exactly as in the notebook
        
        # UTIL_RATE: BILL_AMT1 / LIMIT_BAL, clipped to [-1, 2]
        util_rate = (bill1 / limit_bal) if limit_bal > 0 else 0
        util_rate = max(-1, min(2, util_rate))
        
        # AVG_PAY_STATUS: mean of all PAY_* columns
        avg_pay = np.mean([pay_0, pay_2, pay_3, pay_4, pay_5, pay_6])
        
        # TOTAL_BILL: sum of all 6 BILL_AMT columns
        total_bill = bill1 + bill2 + bill3 + bill4 + bill5 + bill6
        
        # TOTAL_PAY: sum of all 6 PAY_AMT columns
        total_pay = pay_a1 + pay_a2 + pay_a3 + pay_a4 + pay_a5 + pay_a6
        
        # PAY_RATIO: TOTAL_PAY / TOTAL_BILL, clipped to [-2, 5]
        pay_ratio = (total_pay / total_bill) if total_bill > 0 else 0
        pay_ratio = max(-2, min(5, pay_ratio))
        
        # BILL_TREND: BILL_AMT1 - BILL_AMT6
        bill_trend = bill1 - bill6  # Using actual bill6
        
        # PAY_TREND: PAY_AMT1 - PAY_AMT6
        pay_trend = pay_a1 - pay_a6  # Using actual pay_a6
        
        # MAX_PAY_DELAY: maximum of all PAY_* columns
        max_pay_delay = max([pay_0, pay_2, pay_3, pay_4, pay_5, pay_6])
        
        # CONSEC_LATE: count of PAY_* columns > 0
        consec_late = sum([1 for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6] if p > 0])
        
        # CREDIT_USAGE_RATIO: Similar to UTIL_RATE but different calculation
        credit_usage_ratio = util_rate  # Simplified - same as UTIL_RATE

        # Build input DataFrame with ALL known features.
        # We then slice to exactly the columns the loaded model was trained on
        # (stored in feature_cols from the bundle) so no extra/missing column errors occur.
        _all_input_data = {
            'LIMIT_BAL': limit_bal, 'GENDER': sex, 'EDUCATION': education,
            'MARRIAGE': marriage, 'AGE': age,
            'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3,
            'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
            'BILL_AMT1': bill1, 'BILL_AMT2': bill2, 'BILL_AMT3': bill3,
            'BILL_AMT4': bill4, 'BILL_AMT5': bill5, 'BILL_AMT6': bill6,
            'PAY_AMT1': pay_a1, 'PAY_AMT2': pay_a2, 'PAY_AMT3': pay_a3,
            'PAY_AMT4': pay_a4, 'PAY_AMT5': pay_a5, 'PAY_AMT6': pay_a6,
            'UTIL_RATE': util_rate, 'PAY_RATIO': pay_ratio,
            'AVG_PAY_STATUS': avg_pay, 'TOTAL_BILL': total_bill,
            'TOTAL_PAY': total_pay, 'BILL_TREND': bill_trend,
            'PAY_TREND': pay_trend, 'MAX_PAY_DELAY': max_pay_delay,
            'CONSEC_LATE': consec_late, 'CREDIT_USAGE_RATIO': credit_usage_ratio,
        }
        # Use the model's stored feature list if available; otherwise fall back to
        # the 31-column training set (excludes CREDIT_USAGE_RATIO).
        _default_features = [
            'LIMIT_BAL', 'GENDER', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            'UTIL_RATE', 'PAY_RATIO', 'AVG_PAY_STATUS', 'TOTAL_BILL', 'TOTAL_PAY',
            'BILL_TREND', 'PAY_TREND', 'MAX_PAY_DELAY', 'CONSEC_LATE',
        ]
        _cols = feature_cols if feature_cols else _default_features
        X_input = pd.DataFrame(
            [[_all_input_data.get(c, 0) for c in _cols]],
            columns=_cols
        )

        try:
            # The model IS a Pipeline that includes preprocessing
            # For linear models: Pipeline([('prep', ColumnTransformer with PowerTransformer/OneHotEncoder), ('model', LogisticRegression)])
            # For tree models: Pipeline([('prep', ColumnTransformer with OneHotEncoder), ('model', RandomForest/XGBoost)])
            # So we just pass raw data and the pipeline handles:
            # 1. OneHotEncoding of categorical variables (GENDER, EDUCATION, MARRIAGE)
            # 2. PowerTransformer for BILL_AMT columns (linear models only)
            # 3. Log1p for PAY_AMT columns (linear models only)
            # 4. StandardScaler for other numeric columns (linear models only)
            
            # Make prediction - the pipeline handles all preprocessing internally
            prob = pred_model.predict_proba(X_input)[0][1]
            pct  = prob * 100

            if pct < 15:
                tier,color,icon,verdict = "LOW RISK","#00D4AA","✅","This customer profile is unlikely to default. The bank can confidently extend or maintain credit."
            elif pct < 35:
                tier,color,icon,verdict = "MODERATE RISK","#F0883E","⚠️","This customer shows some risk signals. The bank should monitor closely and may consider limit restrictions."
            elif pct < 60:
                tier,color,icon,verdict = "HIGH RISK","#FF7B72","🚨","Significant default risk. The bank should initiate outreach, reduce limit, and begin early intervention."
            else:
                tier,color,icon,verdict = "CRITICAL RISK","#FF4D4F","🔴","Very high probability of default. Immediate action required — collections outreach, account freeze, or settlement offer."

            st.markdown(f"""
            <div style='background:rgba(22,27,34,0.9);border:2px solid {color};border-radius:20px;
                        padding:32px 36px;margin:20px 0;text-align:center'>
                <div style='font-size:56px;margin-bottom:12px'>{icon}</div>
                <div style='font-size:13px;letter-spacing:2px;color:{color};font-weight:600;margin-bottom:8px'>{tier}</div>
                <div style='font-size:64px;font-weight:700;color:{color};font-family:Inter;line-height:1'>{pct:.1f}%</div>
                <div style='font-size:14px;color:#A8B4C0;margin-top:8px'>Probability of Default Next Month</div>
                <div style='font-size:13px;color:#B0BAC6;margin-top:8px'>Model: {pred_model_name}</div>
                <div style='font-size:14px;color:#B0BAC6;margin-top:20px;max-width:600px;margin-left:auto;margin-right:auto;line-height:1.7'>
                    {verdict}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Why this prediction? — Factor Analysis</div>", unsafe_allow_html=True)
            for fname,risk_val,display,level,fcolor in [
                ("Credit Utilization",util_rate*100,f"{util_rate*100:.1f}%",
                 "HIGH" if util_rate>0.75 else "MODERATE" if util_rate>0.3 else "LOW",
                 "#FF4D4F" if util_rate>0.75 else "#F0883E" if util_rate>0.3 else "#00D4AA"),
                ("Recent Payment Status (Sep)",max(0,pay_0)/8*100,
                 f"{pay_0} months delay" if pay_0>0 else "On time ✓",
                 "HIGH" if pay_0>2 else "MODERATE" if pay_0>0 else "LOW",
                 "#FF4D4F" if pay_0>2 else "#F0883E" if pay_0>0 else "#00D4AA"),
                ("Avg Payment Delay (6 months)",max(0,avg_pay)/8*100,
                 f"{avg_pay:.1f} avg months late",
                 "HIGH" if avg_pay>2 else "MODERATE" if avg_pay>0 else "LOW",
                 "#FF4D4F" if avg_pay>2 else "#F0883E" if avg_pay>0 else "#00D4AA"),
                ("Payment Coverage",max(0,1-pay_ratio)*100,
                 f"{pay_ratio:.1%} of bills paid",
                 "HIGH" if pay_ratio<0.1 else "MODERATE" if pay_ratio<0.3 else "LOW",
                 "#FF4D4F" if pay_ratio<0.1 else "#F0883E" if pay_ratio<0.3 else "#00D4AA"),
                ("Credit Limit (Higher = safer)",max(0,1-limit_bal/1000000)*100,
                 f"NT${limit_bal:,}",
                 "LOW" if limit_bal>200000 else "MODERATE" if limit_bal>80000 else "HIGH",
                 "#00D4AA" if limit_bal>200000 else "#F0883E" if limit_bal>80000 else "#FF4D4F"),
            ]:
                bar_pct = min(100, risk_val)
                st.markdown(f"""
                <div style='margin-bottom:14px'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:6px'>
                        <span style='font-size:13px;color:#E6EDF3'>{fname}</span>
                        <span style='font-size:12px;color:{fcolor};font-weight:500'>{display} &nbsp;
                            <span style='font-size:10px;background:{fcolor}22;border:1px solid {fcolor}44;
                                  padding:2px 8px;border-radius:10px'>{level}</span>
                        </span>
                    </div>
                    <div style='height:6px;background:#1C2330;border-radius:3px;overflow:hidden'>
                        <div style='height:100%;width:{bar_pct:.0f}%;background:{fcolor};border-radius:3px'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pct,
                number={'suffix':'%','font':{'size':36,'color':color,'family':'Inter'}},
                delta={'reference':22.1,'suffix':'%','font':{'size':14}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#A8B4C0','tickfont':{'color':'#A8B4C0'}},
                    'bar':{'color':color,'thickness':0.25},
                    'bgcolor':'#1C2330','bordercolor':'#30363D',
                    'steps':[
                        {'range':[0,15],'color':'rgba(0,212,170,0.12)'},
                        {'range':[15,35],'color':'rgba(240,136,62,0.12)'},
                        {'range':[35,60],'color':'rgba(255,123,114,0.12)'},
                        {'range':[60,100],'color':'rgba(255,77,79,0.15)'},
                    ],
                    'threshold':{'line':{'color':'#E6EDF3','width':2},'value':22.1}
                }))
            fig_gauge.update_layout(**PT,height=300,
                title=dict(text='Default Probability vs Dataset Average (22.1%)',font=dict(color='#A8B4C0',size=12)),
                margin=dict(t=50,b=10,l=30,r=30))
            st.plotly_chart(fig_gauge,use_container_width=True)
            st.markdown(f"""
            <div class='insight'>
            <b>How to read this:</b> The gauge shows this customer's estimated default probability ({pct:.1f}%)
        compared to the dataset average of 22.1% (white line).
        The delta ({pct-22.1:+.1f}%) tells you how much above or below average this customer is.
        </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")
            st.info("💡 This may be due to a mismatch between the model's expected features and the input provided. Please check the model file.")
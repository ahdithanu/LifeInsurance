import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from anthropic import Anthropic
import json

# Page Configuration
st.set_page_config(
    page_title="Life Insurance Insights | Syntex Data",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Syntex Data Brand Colors
BURNT_ORANGE = "#B75400"
CHARCOAL_GRAY = "#5B3333"
LIGHT_GRAY = "#E9E2E2"

# Custom CSS for Syntex Data Branding
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    .main {{
        background-color: #EBFFF7;
    }}
    
    .header-container {{
        background: linear-gradient(135deg, {BURNT_ORANGE} 0%, #9A4600 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .header-title {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }}
    
    .header-tagline {{
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }}
    
    .stButton>button {{
        background-color: {BURNT_ORANGE};
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
    }}
    
    .stButton>button:hover {{
        background-color: #9A4600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .insight-box {{
        background: linear-gradient(135deg, {BURNT_ORANGE} 0%, #9A4600 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 2rem;
        font-weight: 600;
        color: {CHARCOAL_GRAY};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {BURNT_ORANGE};
        color: white;
        border-radius: 8px;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    h1, h2, h3 {{
        color: {CHARCOAL_GRAY};
    }}
    </style>
""", unsafe_allow_html=True)

# Data persistence functions
DATA_FILE = 'data/persistent_insurance_data.csv'

def save_data(df):
    """Save uploaded data to persistent storage"""
    os.makedirs('data', exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    return True

def load_persistent_data():
    """Load previously saved data"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

# Initialize Anthropic client
@st.cache_resource
@st.cache_resource
def get_claude_client():
    """Get Claude API client with proper error handling"""
    try:
        # Try to get API key from secrets first, then environment
        api_key = None
        
        # Method 1: Streamlit secrets (most common in Streamlit Cloud)
        if hasattr(st, 'secrets') and "ANTHROPIC_API_KEY" in st.secrets:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        
        # Method 2: Environment variable (for local development)
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # If no API key found, return None
        if not api_key or api_key == "":
            return None
        
        # Create and return client
        return Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Claude client: {str(e)}")
        return None

@st.cache_data
def load_data(uploaded_file=None):
    """Load and process life insurance data"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Save for persistence
        save_data(df)
    else:
        # Try to load persistent data first
        df = load_persistent_data()
        if df is None:
            return None
    
    # Convert date fields
    if 'date_field' in df.columns:
        df['date_field'] = pd.to_datetime(df['date_field'], errors='coerce')
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365
    
    # Convert numeric fields
    for col in ['insurance_face_amount', 'height', 'weight']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate BMI
    if 'height' in df.columns and 'weight' in df.columns:
        df['bmi'] = df.apply(lambda x: calculate_bmi(x['height'], x['weight']), axis=1)
    
    # Identify risk factors and opportunities
    df['risk_factors'] = df.apply(identify_risk_factors, axis=1)
    df['risk_count'] = df['risk_factors'].apply(len)
    df['upsell_opportunities'] = df.apply(identify_upsell_opportunities, axis=1)
    df['upsell_count'] = df['upsell_opportunities'].apply(len)
    
    return df

def calculate_bmi(height, weight):
    """Calculate BMI from height (inches) and weight (lbs)"""
    if pd.isna(height) or pd.isna(weight) or height <= 0 or weight <= 0:
        return None
    height_m = height * 0.0254
    weight_kg = weight * 0.453592
    return weight_kg / (height_m ** 2)

def identify_risk_factors(row):
    """Identify risk factors for a customer"""
    risks = []
    
    if str(row.get('tobacco_use', '')).lower() in ['yes', 'true', '1']:
        risks.append("Tobacco Use")
    
    if str(row.get('has_medical_conditions', '')).lower() in ['yes', 'true', '1']:
        risks.append("Pre-existing Medical Conditions")
    
    if str(row.get('hospitalization_history', '')).lower() in ['yes', 'true', '1']:
        risks.append("Hospitalization History")
    
    if 'bmi' in row and not pd.isna(row['bmi']):
        if row['bmi'] > 30:
            risks.append("High BMI")
        elif row['bmi'] < 18.5:
            risks.append("Low BMI")
    
    if 'age' in row and not pd.isna(row['age']):
        if row['age'] > 65:
            risks.append("Advanced Age")
    
    return risks

def identify_upsell_opportunities(row):
    """Identify upsell opportunities for a customer"""
    opportunities = []
    
    if 'insurance_face_amount' in row and not pd.isna(row['insurance_face_amount']):
        if row['insurance_face_amount'] < 100000:
            opportunities.append("Low Coverage - Consider Increase")
    
    if str(row.get('has_medicaid_eligibility', '')).lower() in ['yes', 'true', '1']:
        opportunities.append("Supplemental Coverage Opportunity")
    
    if 'policy_type' in row:
        policy = str(row.get('policy_type', '')).lower()
        if 'term' in policy or 'basic' in policy:
            opportunities.append("Upgrade to Whole Life/Universal")
    
    if 'age' in row and not pd.isna(row['age']):
        tobacco = str(row.get('tobacco_use', '')).lower() in ['yes', 'true', '1']
        medical = str(row.get('has_medical_conditions', '')).lower() in ['yes', 'true', '1']
        if row['age'] < 40 and not tobacco and not medical:
            opportunities.append("Long-term Investment Products")
    
    if str(row.get('has_preauth_payments', '')).lower() in ['yes', 'true', '1']:
        opportunities.append("Cross-sell Additional Products")
    
    return opportunities

def generate_insights_with_ai(df, claude_client):
    """Generate AI insights using Claude"""
    if claude_client is None:
        return "AI insights unavailable. Please configure ANTHROPIC_API_KEY in Streamlit secrets."
    
    summary_stats = {
        "total_customers": len(df),
        "avg_age": df['age'].mean() if 'age' in df else None,
        "avg_coverage": df['insurance_face_amount'].mean() if 'insurance_face_amount' in df else None,
        "tobacco_users": (df['tobacco_use'] == 'Yes').sum() if 'tobacco_use' in df else None,
        "medical_conditions": (df['has_medical_conditions'] == 'Yes').sum() if 'has_medical_conditions' in df else None,
    }
    
    prompt = f"""Analyze this life insurance customer portfolio and provide 3-5 key actionable insights focused on:
1. Upsell opportunities
2. At-risk customers requiring attention
3. Portfolio optimization recommendations

Data Summary:
- Total Customers: {summary_stats['total_customers']}
- Average Age: {summary_stats['avg_age']:.1f} years
- Average Coverage: ${summary_stats['avg_coverage']:,.0f}
- Tobacco Users: {summary_stats['tobacco_users']}
- Customers with Medical Conditions: {summary_stats['medical_conditions']}

Be specific and data-driven. Format as clear bullet points with actionable recommendations."""
    
    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

# Header
st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">Life Insurance Customer Insights</h1>
        <p class="header-tagline">Unlock the Power of AI. From Data to Deployment.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/B75400/FFFFFF?text=SYNTEX+Data")
    
    st.markdown("### Upload Data")
    uploaded_file = st.file_uploader("Upload Life Insurance CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.success("[OK] New data uploaded and saved!")
    elif load_persistent_data() is not None:
        st.info("📂 Using previously uploaded data")
    
    st.markdown("---")
    
    # Clear data option
    if st.button("🗑️ Clear Saved Data"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.cache_data.clear()
            st.success("Data cleared! Please upload new file.")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard analyzes life insurance customer data to identify:
    - Upsell opportunities
    - At-risk customers
    - Portfolio optimization insights
    - Geographic distribution
    - Trends over time
    
    **Built by Syntex Data**
    """)
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load data
df = load_data(uploaded_file)

if df is None:
    st.warning("No data loaded. Please upload a CSV file.")
    st.info("**Tip:** Upload a CSV with customer life insurance data to get started.")
    st.stop()

# Store in session state
if 'df' not in st.session_state:
    st.session_state['df'] = df
else:
    df = st.session_state['df']

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    avg_coverage = df['insurance_face_amount'].mean() if 'insurance_face_amount' in df else 0
    st.metric("Avg Coverage", f"${avg_coverage:,.0f}")

with col3:
    high_risk = len(df[df['risk_count'] >= 3])
    st.metric("High Risk", f"{high_risk:,}")

with col4:
    upsell_candidates = len(df[df['upsell_count'] >= 2])
    st.metric("Upsell Opportunities", f"{upsell_candidates:,}")

# GLOBAL FILTERS (Apply to all tabs)
st.markdown("### Filters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'date_field' in df.columns:
        date_options = ["All Time", "Last Month", "Last 3 Months", "Last 6 Months", "Last Year"]
        selected_date = st.selectbox("Time Period", date_options, key="global_date")
    else:
        selected_date = "All Time"

with col2:
    if 'policy_type' in df.columns:
        policy_options = ["All Policies"] + sorted([p for p in df['policy_type'].unique() if pd.notna(p)])
        selected_policy_filter = st.selectbox("Policy Type", policy_options, key="global_policy")
    else:
        selected_policy_filter = "All Policies"

with col3:
    if 'state' in df.columns:
        state_options = ["All States"] + sorted(df['state'].unique().tolist())
        selected_state_filter = st.selectbox("State", state_options, key="global_state")
    else:
        selected_state_filter = "All States"

with col4:
    risk_options = ["All Risk Levels", "Low Risk (0-1)", "Medium Risk (2)", "High Risk (3+)"]
    selected_risk_filter = st.selectbox("Risk Level", risk_options, key="global_risk")

# APPLY FILTERS TO CREATE FILTERED DATAFRAME (used by all tabs)
filtered_df = df.copy()

# Filter by policy type
if selected_policy_filter != "All Policies":
    filtered_df = filtered_df[filtered_df['policy_type'] == selected_policy_filter]

# Filter by state
if selected_state_filter != "All States":
    filtered_df = filtered_df[filtered_df['state'] == selected_state_filter]

# Filter by risk level
if selected_risk_filter == "Low Risk (0-1)":
    filtered_df = filtered_df[filtered_df['risk_count'] <= 1]
elif selected_risk_filter == "Medium Risk (2)":
    filtered_df = filtered_df[filtered_df['risk_count'] == 2]
elif selected_risk_filter == "High Risk (3+)":
    filtered_df = filtered_df[filtered_df['risk_count'] >= 3]

# Filter by time period
if selected_date != "All Time" and 'date_field' in filtered_df.columns:
    from datetime import datetime, timedelta
    cutoff_date = datetime.now()
    
    if selected_date == "Last Month":
        cutoff_date = cutoff_date - timedelta(days=30)
    elif selected_date == "Last 3 Months":
        cutoff_date = cutoff_date - timedelta(days=90)
    elif selected_date == "Last 6 Months":
        cutoff_date = cutoff_date - timedelta(days=180)
    elif selected_date == "Last Year":
        cutoff_date = cutoff_date - timedelta(days=365)
    
    filtered_df = filtered_df[filtered_df['date_field'] >= cutoff_date]

# Show filter summary if filters are active
active_filters = []
if selected_policy_filter != "All Policies":
    active_filters.append(f"{selected_policy_filter}")
if selected_state_filter != "All States":
    active_filters.append(f"{selected_state_filter}")
if selected_risk_filter != "All Risk Levels":
    active_filters.append(f"{selected_risk_filter}")
if selected_date != "All Time":
    active_filters.append(f"{selected_date}")

if active_filters:
    st.info(f"**Active Filters:** {' | '.join(active_filters)} | Showing {len(filtered_df):,} of {len(df):,} customers")

st.markdown("---")

# Navigation - Now with 6 tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Dashboard", 
    "Geographic Heatmap",
    "Portfolio Analysis", 
    "Upsell Opportunities", 
    "Risk Analysis", 
    "AI Chat"
])

# TAB 1: EXECUTIVE DASHBOARD (REDESIGNED)
with tab1:
    st.markdown("## Executive Dashboard")
    
    # AI INSIGHTS BOX (Prominent at top)
    claude_client = get_claude_client()
    
    if claude_client and 'overview_insights' not in st.session_state:
        with st.spinner("Analyzing your portfolio trends..."):
            # Calculate key metrics for AI (USING FILTERED DATA)
            total_customers = len(filtered_df)
            total_portfolio = filtered_df['insurance_face_amount'].sum()
            avg_coverage = filtered_df['insurance_face_amount'].mean()
            avg_age = filtered_df['age'].mean() if 'age' in filtered_df else 0
            high_risk_count = len(filtered_df[filtered_df['risk_count'] >= 3])
            high_risk_pct = (high_risk_count / total_customers * 100) if total_customers > 0 else 0
            
            # Get top policy
            if 'policy_type' in filtered_df.columns:
                top_policy = filtered_df[filtered_df['policy_type'].notna()]['policy_type'].mode()[0] if len(filtered_df[filtered_df['policy_type'].notna()]) > 0 else "N/A"
                top_policy_pct = (filtered_df['policy_type'] == top_policy).sum() / len(filtered_df) * 100
            else:
                top_policy = "N/A"
                top_policy_pct = 0
            
            # Upsell opportunities
            upsell_count = len(filtered_df[filtered_df['upsell_count'] >= 2])
            low_coverage = len(filtered_df[filtered_df['insurance_face_amount'] < 100000])
            
            prompt = f"""You are analyzing a life insurance portfolio. Provide exactly 3 clear, actionable insights.

Portfolio Metrics:
- Total Customers: {total_customers:,}
- Total Portfolio Value: ${total_portfolio:,.0f}
- Average Coverage: ${avg_coverage:,.0f}
- Average Customer Age: {avg_age:.1f} years
- High Risk Customers: {high_risk_count:,} ({high_risk_pct:.1f}%)
- Top Policy Type: {top_policy} ({top_policy_pct:.1f}%)
- Upsell Candidates: {upsell_count:,} customers
- Low Coverage (<$100K): {low_coverage:,} customers

Provide EXACTLY 3 insights in this format:

INSIGHT 1 | TYPE: [Revenue/Risk/Growth]
TITLE: [5-7 word headline]
FINDING: [One sentence with specific numbers]
ACTION: [One sentence - what to do]

INSIGHT 2 | TYPE: [Revenue/Risk/Growth]
TITLE: [5-7 word headline]
FINDING: [One sentence with specific numbers]
ACTION: [One sentence - what to do]

INSIGHT 3 | TYPE: [Revenue/Risk/Growth]
TITLE: [5-7 word headline]
FINDING: [One sentence with specific numbers]
ACTION: [One sentence - what to do]

Rules:
- TYPE must be exactly one of: Revenue, Risk, or Growth
- TITLE must be 5-7 words maximum
- FINDING must include specific numbers from the metrics
- ACTION must be one clear sentence
- Keep each sentence under 25 words
- Be direct and concise"""
            
            try:
                message = claude_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                st.session_state['overview_insights'] = message.content[0].text
            except Exception as e:
                st.session_state['overview_insights'] = f"Unable to generate insights: {str(e)}"
    
    # Display AI insights prominently
    if 'overview_insights' in st.session_state:
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("⟳", key="refresh_overview", help="Refresh insights"):
                del st.session_state['overview_insights']
                st.rerun()
        
        # Parse insights into structured format
        insights_text = st.session_state['overview_insights']
        
        # Try to parse structured insights
        try:
            insights = []
            current_insight = {}
            
            for line in insights_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('INSIGHT') and '|' in line and 'TYPE:' in line:
                    if current_insight:
                        insights.append(current_insight)
                    # Extract type
                    type_part = line.split('TYPE:')[1].strip()
                    current_insight = {'type': type_part}
                elif line.startswith('TITLE:'):
                    current_insight['title'] = line.replace('TITLE:', '').strip()
                elif line.startswith('FINDING:'):
                    current_insight['finding'] = line.replace('FINDING:', '').strip()
                elif line.startswith('ACTION:'):
                    current_insight['action'] = line.replace('ACTION:', '').strip()
            
            # Add last insight
            if current_insight and 'title' in current_insight:
                insights.append(current_insight)
            
            # Display insights in separate cards if parsed successfully
            if len(insights) >= 2:
                st.markdown("### AI-Powered Insights")
                
                for idx, insight in enumerate(insights):
                    insight_type = insight.get('type', 'Growth').lower()
                    
                    # Color scheme based on type - professional colors
                    if 'revenue' in insight_type or 'upsell' in insight_type or 'opportunity' in insight_type:
                        bg_color = '#2E7D32'  # Dark green
                        type_label = 'REVENUE OPPORTUNITY'
                    elif 'risk' in insight_type or 'alert' in insight_type:
                        bg_color = '#C62828'  # Dark red
                        type_label = 'RISK ALERT'
                    else:  # Growth or other
                        bg_color = '#1565C0'  # Dark blue
                        type_label = 'GROWTH INSIGHT'
                    
                    title = insight.get('title', 'Insight')
                    finding = insight.get('finding', '')
                    action = insight.get('action', '')
                    
                    st.markdown(f"""
                    <div style='background: {bg_color}; 
                                color: white; 
                                padding: 1.5rem; 
                                border-radius: 8px; 
                                margin-bottom: 1rem;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='font-size: 0.7rem; 
                                    font-weight: 700; 
                                    letter-spacing: 0.1em; 
                                    margin-bottom: 0.75rem; 
                                    opacity: 0.95;
                                    text-transform: uppercase;'>
                            {type_label}
                        </div>
                        <h4 style='color: white; 
                                   margin: 0 0 0.75rem 0; 
                                   font-size: 1.15rem; 
                                   font-weight: 600;'>{title}</h4>
                        <p style='margin: 0 0 1rem 0; 
                                  font-size: 0.95rem; 
                                  line-height: 1.6; 
                                  opacity: 0.95;'>{finding}</p>
                        <div style='background: rgba(255,255,255,0.2); 
                                    padding: 0.875rem; 
                                    border-radius: 6px; 
                                    border-left: 3px solid rgba(255,255,255,0.6);'>
                            <strong style='font-size: 0.85rem; 
                                          letter-spacing: 0.05em;'>RECOMMENDED ACTION:</strong> 
                            <span style='font-size: 0.9rem; 
                                        margin-left: 0.5rem;'>{action}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Fallback to original display if parsing fails
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {BURNT_ORANGE} 0%, #9A4600 100%); 
                                color: white; 
                                padding: 1.5rem; 
                                border-radius: 10px; 
                                margin-bottom: 2rem;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h3 style="color: white; margin-top: 0; margin-bottom: 1rem;">AI-Powered Insights</h3>
                        <div style="line-height: 1.8; font-size: 0.95rem;">
                        {st.session_state['overview_insights'].replace(chr(10), '<br>')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            # Fallback display on any error
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {BURNT_ORANGE} 0%, #9A4600 100%); 
                            color: white; 
                            padding: 1.5rem; 
                            border-radius: 10px; 
                            margin-bottom: 2rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h3 style="color: white; margin-top: 0; margin-bottom: 1rem;">AI-Powered Insights</h3>
                    <div style="line-height: 1.8; font-size: 0.95rem;">
                    {st.session_state['overview_insights'].replace(chr(10), '<br>')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    elif not claude_client:
        # Show helpful message about adding API key
        st.markdown("""
            <div style='background: #EBF8FF; 
                        padding: 1.5rem; 
                        border-radius: 10px; 
                        margin-bottom: 2rem;
                        border-left: 4px solid #4299E1;'>
                <h4 style="margin-top: 0; color: #2D3748;">AI Insights Available</h4>
                <p style="margin-bottom: 1rem; color: #4A5568;">Enable AI-powered portfolio analysis by adding your Anthropic API key.</p>
                <p style="margin-bottom: 0.5rem; color: #2D3748;"><strong>How to add your API key:</strong></p>
                <ol style="margin-left: 1rem; color: #4A5568;">
                    <li>Click <strong>"Manage app"</strong> (bottom right of your Streamlit app)</li>
                    <li>Go to <strong>Settings → Secrets</strong></li>
                    <li>Add: <code>ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"</code></li>
                    <li>Click <strong>Save</strong></li>
                    <li>Refresh this page</li>
                </ol>
                <p style="margin-top: 1rem; color: #4A5568; font-size: 0.9rem;">
                    Get your API key at: <a href="https://console.anthropic.com/" target="_blank" style="color: #4299E1;">console.anthropic.com</a>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Add debug expander
        with st.expander("Troubleshooting API Key"):
            st.markdown("**Checking API key configuration...**")
            
            # Check if secrets exist
            has_secrets = hasattr(st, 'secrets')
            st.write(f"[OK] Streamlit secrets accessible: {has_secrets}")
            
            if has_secrets:
                has_key = "ANTHROPIC_API_KEY" in st.secrets
                st.write(f"{'[OK]' if has_key else '[ERROR]'} ANTHROPIC_API_KEY in secrets: {has_key}")
                
                if has_key:
                    key_value = st.secrets["ANTHROPIC_API_KEY"]
                    key_length = len(key_value) if key_value else 0
                    key_starts_correctly = key_value.startswith("sk-ant-") if key_value else False
                    
                    st.write(f"[OK] Key length: {key_length} characters")
                    st.write(f"{'[OK]' if key_starts_correctly else '[ERROR]'} Key starts with 'sk-ant-': {key_starts_correctly}")
                    
                    if not key_starts_correctly:
                        st.error("[ERROR] Your API key should start with 'sk-ant-api03-'")
                    elif key_length < 50:
                        st.error("[ERROR] Your API key seems too short")
                    else:
                        st.success("[OK] API key format looks correct!")
                        st.info("If insights still don't work, try clicking the Refresh button above or refreshing the page.")
            else:
                st.error("[ERROR] Cannot access Streamlit secrets")
            
            # Check environment variable
            env_key = os.getenv("ANTHROPIC_API_KEY")
            has_env = env_key is not None and env_key != ""
            st.write(f"{'[OK]' if has_env else '[ERROR]'} Environment variable set: {has_env}")
    
    # Big metric cards with trends (USING GLOBAL FILTERED DATA)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #666; font-size: 0.9rem; margin: 0;'>Total Customers</p>
            <h2 style='color: #2D3748; font-size: 2.5rem; margin: 0.5rem 0;'>{:,}</h2>
            <p style='color: #48BB78; font-size: 0.85rem; margin: 0;'>↑ Active portfolio</p>
        </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    with col2:
        avg_coverage = filtered_df['insurance_face_amount'].mean() if 'insurance_face_amount' in filtered_df else 0
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #666; font-size: 0.9rem; margin: 0;'>Avg Coverage</p>
            <h2 style='color: #2D3748; font-size: 2.5rem; margin: 0.5rem 0;'>${:,.0f}</h2>
            <p style='color: #48BB78; font-size: 0.85rem; margin: 0;'>Per customer</p>
        </div>
        """.format(avg_coverage), unsafe_allow_html=True)
    
    with col3:
        total_portfolio = filtered_df['insurance_face_amount'].sum() if 'insurance_face_amount' in filtered_df else 0
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #666; font-size: 0.9rem; margin: 0;'>Total Portfolio Value</p>
            <h2 style='color: #2D3748; font-size: 2.5rem; margin: 0.5rem 0;'>${:,.0f}</h2>
            <p style='color: #48BB78; font-size: 0.85rem; margin: 0;'>Total coverage</p>
        </div>
        """.format(total_portfolio), unsafe_allow_html=True)
    
    with col4:
        high_risk = len(filtered_df[filtered_df['risk_count'] >= 3])
        risk_pct = (high_risk / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <p style='color: #666; font-size: 0.9rem; margin: 0;'>High Risk</p>
            <h2 style='color: #2D3748; font-size: 2.5rem; margin: 0.5rem 0;'>{:,}</h2>
            <p style='color: #F56565; font-size: 0.85rem; margin: 0;'>{:.1f}% of portfolio</p>
        </div>
        """.format(high_risk, risk_pct), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Portfolio by Policy Type")
        if 'policy_type' in filtered_df.columns:
            policy_counts = filtered_df[filtered_df['policy_type'].notna()]['policy_type'].value_counts()
            fig = px.pie(
                values=policy_counts.values, 
                names=policy_counts.index,
                hole=0.5,
                color_discrete_sequence=['#4299E1', '#48BB78', '#ED8936', '#9F7AEA', '#F56565']
            )
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(size=12)
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Coverage Distribution")
        if 'insurance_face_amount' in filtered_df.columns:
            coverage_data = filtered_df[filtered_df['insurance_face_amount'] > 0]['insurance_face_amount']
            
            # Create bins
            bins = [0, 50000, 100000, 200000, 300000, 500000, 1000000, float('inf')]
            labels = ['<$50K', '$50-100K', '$100-200K', '$200-300K', '$300-500K', '$500K-$1M', '>$1M']
            coverage_binned = pd.cut(coverage_data, bins=bins, labels=labels)
            coverage_counts = coverage_binned.value_counts().sort_index()
            
            fig = px.bar(
                x=coverage_counts.index,
                y=coverage_counts.values,
                color_discrete_sequence=['#4299E1']
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Coverage Range",
                yaxis_title="Customers",
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Geographic Distribution")
        if 'state' in filtered_df.columns:
            state_counts = filtered_df['state'].value_counts().head(10)
            
            fig = px.bar(
                x=state_counts.values,
                y=state_counts.index,
                orientation='h',
                color_discrete_sequence=['#48BB78']
            )
            fig.update_layout(
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Customers",
                yaxis_title="",
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Distribution")
        risk_labels = ['Low (0-1)', 'Medium (2)', 'High (3+)']
        risk_counts = [
            len(filtered_df[filtered_df['risk_count'] <= 1]),
            len(filtered_df[filtered_df['risk_count'] == 2]),
            len(filtered_df[filtered_df['risk_count'] >= 3])
        ]
        
        fig = px.bar(
            x=risk_labels,
            y=risk_counts,
            color=risk_labels,
            color_discrete_map={
                'Low (0-1)': '#48BB78',
                'Medium (2)': '#ED8936',
                'High (3+)': '#F56565'
            }
        )
        fig.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Risk Level",
            yaxis_title="Customers",
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key insights section
    st.markdown("### Key Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        if 'age' in filtered_df.columns:
            avg_age = filtered_df['age'].mean()
            st.markdown(f"""
            <div style='background: #EBF8FF; padding: 1rem; border-radius: 8px; border-left: 4px solid #4299E1;'>
                <p style='margin: 0; font-weight: 600; color: #2D3748;'>Average Customer Age</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #4299E1;'>{avg_age:.1f} years</p>
            </div>
            """, unsafe_allow_html=True)
    
    with insight_col2:
        if 'policy_type' in filtered_df.columns:
            top_policy = filtered_df[filtered_df['policy_type'].notna()]['policy_type'].mode()[0] if len(filtered_df[filtered_df['policy_type'].notna()]) > 0 else "N/A"
            top_policy_pct = (filtered_df['policy_type'] == top_policy).sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            st.markdown(f"""
            <div style='background: #F0FFF4; padding: 1rem; border-radius: 8px; border-left: 4px solid #48BB78;'>
                <p style='margin: 0; font-weight: 600; color: #2D3748;'>Most Popular Policy</p>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #48BB78;'>{top_policy} ({top_policy_pct:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with insight_col3:
        upsell_candidates = len(filtered_df[filtered_df['upsell_count'] >= 2])
        st.markdown(f"""
        <div style='background: #FFFAF0; padding: 1rem; border-radius: 8px; border-left: 4px solid #ED8936;'>
            <p style='margin: 0; font-weight: 600; color: #2D3748;'>Upsell Opportunities</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #ED8936;'>{upsell_candidates:,} customers</p>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: GEOGRAPHIC HEATMAP (NEW)
with tab2:
    st.markdown("## Geographic Distribution & Performance")
    
    if 'state' in filtered_df.columns:
        # State-level aggregation
        state_stats = filtered_df.groupby('state').agg({
            'primary_full_name': 'count',
            'insurance_face_amount': ['sum', 'mean'],
            'risk_count': 'mean',
            'upsell_count': 'mean'
        }).reset_index()
        
        state_stats.columns = ['state', 'Customer Count', 'Total Coverage', 'Avg Coverage', 'Avg Risk', 'Avg Upsell Opportunities']
        
        # Calculate additional metrics for hover
        state_stats['Coverage per Customer'] = state_stats['Total Coverage'] / state_stats['Customer Count']
        state_stats['Market Share %'] = (state_stats['Customer Count'] / state_stats['Customer Count'].sum() * 100).round(2)
        
        # Create hover text template
        def create_hover_text(row):
            return (
                f"<b>{row['state']}</b><br>"
                f"Customers: {row['Customer Count']:,}<br>"
                f"Market Share: {row['Market Share %']:.1f}%<br>"
                f"Total Coverage: ${row['Total Coverage']:,.0f}<br>"
                f"Avg Coverage: ${row['Avg Coverage']:,.0f}<br>"
                f"Risk Score: {row['Avg Risk']:.2f}<br>"
                f"Upsell Opps: {row['Avg Upsell Opportunities']:.2f}"
            )
        
        state_stats['hover_text'] = state_stats.apply(create_hover_text, axis=1)
        
        # Map selector - one view at a time
        st.markdown("### Select View")
        map_view = st.selectbox(
            "Choose which metric to visualize:",
            ["Customer Distribution", "Total Portfolio Value", "Average Coverage per Customer", "Risk Distribution"],
            key="map_selector"
        )
        
        st.markdown("---")
        
        # Display selected map - LARGE SIZE
        if map_view == "Customer Distribution":
            st.markdown("### 🇺🇸 Customer Distribution by State")
            st.caption("Hover over states for detailed metrics • Darker colors = more customers")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Customer Count'],
                locationmode='USA-states',
                colorscale='Blues',
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                marker_line_color='white',
                marker_line_width=2
            ))
            
            fig.update_layout(
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=True,
                    lakecolor='#e6f2ff',
                    bgcolor='#f8f9fa',
                    landcolor='#f0f0f0',
                    coastlinecolor='#cccccc',
                    coastlinewidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family='Poppins, sans-serif', size=13)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics for this view
            col1, col2, col3 = st.columns(3)
            with col1:
                top_state = state_stats.nlargest(1, 'Customer Count')['state'].values[0]
                top_count = state_stats.nlargest(1, 'Customer Count')['Customer Count'].values[0]
                st.metric("🏆 Top State", top_state, f"{top_count:,} customers")
            with col2:
                total_customers = state_stats['Customer Count'].sum()
                st.metric("👥 Total Customers", f"{total_customers:,}")
            with col3:
                states_with_customers = len(state_stats)
                st.metric("States with Customers", f"{states_with_customers}")
        
        elif map_view == "Total Portfolio Value":
            st.markdown("### Total Portfolio Value by State")
            st.caption("Total insurance coverage amount per state • Green intensity shows market value")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Total Coverage'],
                locationmode='USA-states',
                colorscale='Greens',
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                marker_line_color='white',
                marker_line_width=2
            ))
            
            fig.update_layout(
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=True,
                    lakecolor='#e6f2ff',
                    bgcolor='#f8f9fa',
                    landcolor='#f0f0f0',
                    coastlinecolor='#cccccc',
                    coastlinewidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family='Poppins, sans-serif', size=13)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics for this view
            col1, col2, col3 = st.columns(3)
            with col1:
                highest_value_state = state_stats.nlargest(1, 'Total Coverage')['state'].values[0]
                highest_value = state_stats.nlargest(1, 'Total Coverage')['Total Coverage'].values[0]
                st.metric("🏆 Highest Value State", highest_value_state, f"${highest_value:,.0f}")
            with col2:
                total_portfolio = state_stats['Total Coverage'].sum()
                st.metric("💼 Total Portfolio", f"${total_portfolio:,.0f}")
            with col3:
                avg_state_value = state_stats['Total Coverage'].mean()
                st.metric("Avg State Value", f"${avg_state_value:,.0f}")
        
        elif map_view == "Average Coverage per Customer":
            st.markdown("### Average Coverage per Customer")
            st.caption("Average policy value by state • Orange intensity shows premium customers")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Avg Coverage'],
                locationmode='USA-states',
                colorscale='Oranges',
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                marker_line_color='white',
                marker_line_width=2
            ))
            
            fig.update_layout(
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=True,
                    lakecolor='#e6f2ff',
                    bgcolor='#f8f9fa',
                    landcolor='#f0f0f0',
                    coastlinecolor='#cccccc',
                    coastlinewidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family='Poppins, sans-serif', size=13)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics for this view
            col1, col2, col3 = st.columns(3)
            with col1:
                highest_avg_state = state_stats.nlargest(1, 'Avg Coverage')['state'].values[0]
                highest_avg = state_stats.nlargest(1, 'Avg Coverage')['Avg Coverage'].values[0]
                st.metric("🏆 Premium Market", highest_avg_state, f"${highest_avg:,.0f} avg")
            with col2:
                overall_avg = state_stats['Avg Coverage'].mean()
                st.metric("National Average", f"${overall_avg:,.0f}")
            with col3:
                premium_states = len(state_stats[state_stats['Avg Coverage'] > overall_avg])
                st.metric("⭐ Above Average States", f"{premium_states}")
        
        elif map_view == "Risk Distribution":
            st.markdown("### Risk Distribution by State")
            st.caption("Average risk score per state • Red = high risk, Yellow = medium, Green = low risk")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Avg Risk'],
                locationmode='USA-states',
                colorscale='RdYlGn_r',
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                marker_line_color='white',
                marker_line_width=2,
                zmid=state_stats['Avg Risk'].median()
            ))
            
            fig.update_layout(
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=True,
                    lakecolor='#e6f2ff',
                    bgcolor='#f8f9fa',
                    landcolor='#f0f0f0',
                    coastlinecolor='#cccccc',
                    coastlinewidth=1
                ),
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=700,
                margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family='Poppins, sans-serif', size=13)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics for this view
            col1, col2, col3 = st.columns(3)
            with col1:
                highest_risk_state = state_stats.nlargest(1, 'Avg Risk')['state'].values[0]
                highest_risk = state_stats.nlargest(1, 'Avg Risk')['Avg Risk'].values[0]
                st.metric("Highest Risk State", highest_risk_state, f"{highest_risk:.2f} risk score")
            with col2:
                avg_risk = state_stats['Avg Risk'].mean()
                st.metric("National Avg Risk", f"{avg_risk:.2f}")
            with col3:
                high_risk_states = len(state_stats[state_stats['Avg Risk'] >= 2.5])
                st.metric("🚨 High Risk States", f"{high_risk_states}")
        
        # Summary Table - always show below the map
        st.markdown("---")
        st.markdown("### State Performance Summary")
        
        # Create a sortable, formatted table
        summary_table = state_stats[['state', 'Customer Count', 'Market Share %', 'Total Coverage', 
                                     'Avg Coverage', 'Avg Risk', 'Avg Upsell Opportunities']].copy()
        summary_table = summary_table.sort_values('Customer Count', ascending=False)
        
        # Format the columns
        summary_table['Customer Count'] = summary_table['Customer Count'].apply(lambda x: f"{x:,}")
        summary_table['Market Share %'] = summary_table['Market Share %'].apply(lambda x: f"{x:.1f}%")
        summary_table['Total Coverage'] = summary_table['Total Coverage'].apply(lambda x: f"${x:,.0f}")
        summary_table['Avg Coverage'] = summary_table['Avg Coverage'].apply(lambda x: f"${x:,.0f}")
        summary_table['Avg Risk'] = summary_table['Avg Risk'].apply(lambda x: f"{x:.2f}")
        summary_table['Avg Upsell Opportunities'] = summary_table['Avg Upsell Opportunities'].apply(lambda x: f"{x:.2f}")
        
        # Rename columns for display
        summary_table.columns = ['State', 'Customers', 'Market Share', 'Total Portfolio Value', 
                                'Avg Policy Value', 'Risk Score', 'Upsell Opportunities']
        
        st.dataframe(
            summary_table,
            use_container_width=True, 
            hide_index=True,
            height=400
        )

# TAB 3: PORTFOLIO ANALYSIS (Original Overview)
with tab3:
    st.markdown("## AI-Generated Insights")
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Regenerate", key="regenerate_insights"):
            with st.spinner("Generating insights..."):
                claude_client = get_claude_client()
                insights = generate_insights_with_ai(filtered_df, claude_client)
                st.session_state['insights'] = insights
    
    if 'insights' in st.session_state:
        st.markdown(f"""
            <div class="insight-box">
                {st.session_state['insights'].replace(chr(10), '<br>')}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click 'Regenerate' to generate AI insights for your portfolio.")
    
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Coverage Distribution")
        if 'insurance_face_amount' in df.columns:
            fig = px.histogram(df, x='insurance_face_amount', nbins=30,
                              color_discrete_sequence=[BURNT_ORANGE])
            fig.update_layout(showlegend=False, xaxis_title="Coverage Amount", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Age Distribution")
        if 'age' in df.columns:
            fig = px.histogram(df, x='age', nbins=30,
                              color_discrete_sequence=[BURNT_ORANGE])
            fig.update_layout(showlegend=False, xaxis_title="Age", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Policy Types")
        if 'policy_type' in filtered_df.columns:
            policy_counts = filtered_df['policy_type'].value_counts()
            fig = px.pie(values=policy_counts.values, names=policy_counts.index,
                        color_discrete_sequence=[BURNT_ORANGE, CHARCOAL_GRAY, LIGHT_GRAY, "#9A4600"])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Distribution")
        risk_dist = filtered_df['risk_count'].value_counts().sort_index()
        fig = px.bar(x=risk_dist.index, y=risk_dist.values,
                    labels={'x': 'Risk Factors', 'y': 'Customer Count'},
                    color_discrete_sequence=[BURNT_ORANGE])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: UPSELL OPPORTUNITIES
with tab4:
    st.markdown("## Upsell Opportunities")
    
    upsell_df = filtered_df[filtered_df['upsell_count'] >= 2].sort_values('upsell_count', ascending=False)
    
    st.markdown(f"### Found {len(upsell_df)} customers with 2+ upsell opportunities")
    
    if len(upsell_df) > 0:
        all_opportunities = []
        for opps in upsell_df['upsell_opportunities']:
            all_opportunities.extend(opps)
        
        opp_counts = pd.Series(all_opportunities).value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Opportunity Breakdown")
            for opp, count in opp_counts.items():
                st.markdown(f"**{opp}**: {count} customers")
        
        with col2:
            fig = px.bar(x=opp_counts.values, y=opp_counts.index, orientation='h',
                        labels={'x': 'Number of Customers', 'y': 'Opportunity Type'},
                        color_discrete_sequence=[BURNT_ORANGE])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Top Upsell Candidates")
        
        display_cols = ['primary_full_name', 'age', 'insurance_face_amount', 'policy_type', 
                       'upsell_count', 'upsell_opportunities']
        available_cols = [col for col in display_cols if col in upsell_df.columns]
        
        st.dataframe(upsell_df[available_cols].head(20), use_container_width=True, hide_index=True)

# TAB 5: RISK ANALYSIS
with tab5:
    st.markdown("## Risk Analysis")
    
    high_risk = filtered_df[filtered_df['risk_count'] >= 3]
    medium_risk = filtered_df[filtered_df['risk_count'] == 2]
    low_risk = filtered_df[filtered_df['risk_count'] <= 1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk", f"{len(high_risk):,}", help="3+ risk factors")
    
    with col2:
        st.metric("Medium Risk", f"{len(medium_risk):,}", help="2 risk factors")
    
    with col3:
        st.metric("Low Risk", f"{len(low_risk):,}", help="0-1 risk factors")
    
    st.markdown("---")
    
    all_risks = []
    for risks in filtered_df['risk_factors']:
        all_risks.extend(risks)
    
    risk_counts = pd.Series(all_risks).value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Risk Factor Prevalence")
        for risk, count in risk_counts.items():
            pct = (count / len(filtered_df)) * 100
            st.markdown(f"**{risk}**: {count} ({pct:.1f}%)")
    
    with col2:
        fig = px.bar(x=risk_counts.values, y=risk_counts.index, orientation='h',
                    labels={'x': 'Number of Customers', 'y': 'Risk Factor'},
                    color_discrete_sequence=['#d32f2f'])
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### High Risk Customers Requiring Attention")
    
    if len(high_risk) > 0:
        display_cols = ['primary_full_name', 'age', 'insurance_face_amount', 
                       'risk_count', 'risk_factors']
        available_cols = [col for col in display_cols if col in high_risk.columns]
        
        st.dataframe(high_risk[available_cols].head(20), use_container_width=True, hide_index=True)
    else:
        st.success("[OK] No high-risk customers found!")

# TAB 6: AI CHAT
with tab6:
    st.markdown("## AI Assistant")
    st.markdown("Ask questions about your life insurance portfolio and get AI-powered insights.")
    
    claude_client = get_claude_client()
    
    if claude_client is None:
        st.warning("AI chat unavailable. Please configure ANTHROPIC_API_KEY in Streamlit secrets.")
    else:
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.markdown(f"""
                    <div style='background: {LIGHT_GRAY}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                        <strong>You:</strong> {message['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="insight-box">
                        <strong>AI:</strong> {message['content']}
                    </div>
                """, unsafe_allow_html=True)
        
        user_query = st.text_input("Ask a question:", key="chat_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send", key="send_chat"):
                if user_query:
                    st.session_state['chat_history'].append({'role': 'user', 'content': user_query})
                    
                    with st.spinner("Thinking..."):
                        try:
                            messages = [{"role": msg['role'], "content": msg['content']} 
                                      for msg in st.session_state['chat_history']]
                            
                            response = claude_client.messages.create(
                                model="claude-sonnet-4-20250514",
                                max_tokens=1500,
                                messages=messages
                            )
                            
                            ai_response = response.content[0].text
                            st.session_state['chat_history'].append({'role': 'assistant', 'content': ai_response})
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state['chat_history'] = []
                st.rerun()
        
        if len(st.session_state['chat_history']) == 0:
            st.markdown("---")
            st.markdown("#### Suggested Questions")
            
            suggestions = [
                "Which customers should we prioritize for upselling?",
                "What are the common characteristics of high-risk customers?",
                "Which states have the highest average coverage amounts?",
                "How can we reduce churn risk in our portfolio?",
            ]
            
            col1, col2 = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with col1 if i % 2 == 0 else col2:
                    if st.button(suggestion, key=f"suggest_{i}"):
                        st.session_state['chat_history'].append({'role': 'user', 'content': suggestion})
                        
                        with st.spinner("Thinking..."):
                            try:
                                messages = [{"role": msg['role'], "content": msg['content']} 
                                          for msg in st.session_state['chat_history']]
                                
                                response = claude_client.messages.create(
                                    model="claude-sonnet-4-20250514",
                                    max_tokens=1500,
                                    messages=messages
                                )
                                
                                ai_response = response.content[0].text
                                st.session_state['chat_history'].append({'role': 'assistant', 'content': ai_response})
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                        
                        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #5B3333; padding: 1rem;'>
        <p><strong>SYNTEX Data</strong> | Unlock the Power of AI. From Data to Deployment.</p>
        <p style='font-size: 0.9rem;'>syntex-data.com</p>
    </div>
""", unsafe_allow_html=True)

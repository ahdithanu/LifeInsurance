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
def get_claude_client():
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)

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
    
    st.markdown("### 📊 Upload Data")
    uploaded_file = st.file_uploader("Upload Life Insurance CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.success("✅ New data uploaded and saved!")
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
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load data
df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ No data loaded. Please upload a CSV file.")
    st.info("💡 **Tip:** Upload a CSV with customer life insurance data to get started.")
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

st.markdown("---")

# Navigation - Now with 6 tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview & Trends", 
    "🗺️ Geographic Heatmap",
    "📈 Portfolio Analysis", 
    "🎯 Upsell Opportunities", 
    "⚠️ Risk Analysis", 
    "🤖 AI Chat"
])

# TAB 1: OVERVIEW & TRENDS (NEW)
with tab1:
    st.markdown("## 📊 Executive Summary & Key Insights")
    
    # Generate AI insights for the overview
    claude_client = get_claude_client()
    
    if claude_client and 'overview_insights' not in st.session_state:
        with st.spinner("🤖 Analyzing your portfolio trends..."):
            # Calculate trend metrics
            if 'date_field' in df.columns:
                df_sorted = df.sort_values('date_field')
                df['month'] = df['date_field'].dt.to_period('M').astype(str)
                monthly_stats = df.groupby('month').agg({
                    'primary_full_name': 'count',
                    'insurance_face_amount': 'mean',
                    'risk_count': 'mean'
                }).reset_index()
                
                # Calculate trends
                customer_trend = monthly_stats['primary_full_name'].pct_change().mean() * 100
                coverage_trend = monthly_stats['insurance_face_amount'].pct_change().mean() * 100
                risk_trend = monthly_stats['risk_count'].diff().mean()
                
                recent_month = monthly_stats.iloc[-1]
                prev_month = monthly_stats.iloc[-2] if len(monthly_stats) > 1 else recent_month
                
                prompt = f"""You are analyzing a life insurance portfolio. Provide a clear executive summary with actionable insights.

Portfolio Metrics:
- Total Customers: {len(df):,}
- Total Portfolio Value: ${df['insurance_face_amount'].sum():,.0f}
- Average Coverage: ${df['insurance_face_amount'].mean():,.0f}
- Average Customer Age: {df['age'].mean():.1f} years

Trends (Month-over-Month):
- Customer Growth: {customer_trend:+.1f}% average monthly change
- Coverage Trend: {coverage_trend:+.1f}% average monthly change  
- Risk Score Trend: {risk_trend:+.2f} change in risk score

Recent Performance:
- Last Month: {recent_month['primary_full_name']:.0f} customers, ${recent_month['insurance_face_amount']:,.0f} avg coverage
- Previous Month: {prev_month['primary_full_name']:.0f} customers, ${prev_month['insurance_face_amount']:,.0f} avg coverage

Top Policy Type: {df['policy_type'].mode()[0]} ({(df['policy_type'] == df['policy_type'].mode()[0]).sum() / len(df) * 100:.1f}%)

Provide 4-5 bullet points that:
1. Start with the most important finding (what matters most right now)
2. Explain trends in plain English (avoid jargon)
3. Give specific, actionable recommendations
4. Highlight risks or opportunities
5. Answer "so what?" for each insight

Format as clear bullets starting with bold action words like "⚠️ Address:", "💡 Opportunity:", "📈 Trend:", etc."""
                
                try:
                    message = claude_client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.session_state['overview_insights'] = message.content[0].text
                except Exception as e:
                    st.session_state['overview_insights'] = "Unable to generate insights at this time."
    
    # Display AI insights prominently
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_overview"):
            if 'overview_insights' in st.session_state:
                del st.session_state['overview_insights']
            st.rerun()
    
    if 'overview_insights' in st.session_state:
        st.markdown(f"""
            <div class="insight-box">
                <h3 style="color: white; margin-top: 0;">🎯 What You Need to Know</h3>
                {st.session_state['overview_insights'].replace(chr(10), '<br>')}
            </div>
        """, unsafe_allow_html=True)
    elif not claude_client:
        st.info("💡 **Quick Insights:** Connect your API key to get AI-powered executive summaries of your portfolio trends.")
    
    st.markdown("---")
    
    # Time-based metrics with narrative
    if 'date_field' in df.columns:
        df_sorted = df.sort_values('date_field')
        
        # Monthly trends
        df['month'] = df['date_field'].dt.to_period('M').astype(str)
        monthly_stats = df.groupby('month').agg({
            'primary_full_name': 'count',
            'insurance_face_amount': 'mean',
            'risk_count': 'mean'
        }).reset_index()
        monthly_stats.columns = ['Month', 'New Customers', 'Avg Coverage', 'Avg Risk Score']
        
        # Calculate trend direction
        customer_change = 0
        coverage_change = 0
        risk_change = 0
        
        if len(monthly_stats) >= 2:
            customer_change = ((monthly_stats['New Customers'].iloc[-1] - monthly_stats['New Customers'].iloc[0]) / 
                             monthly_stats['New Customers'].iloc[0] * 100)
            coverage_change = ((monthly_stats['Avg Coverage'].iloc[-1] - monthly_stats['Avg Coverage'].iloc[0]) / 
                             monthly_stats['Avg Coverage'].iloc[0] * 100)
            risk_change = monthly_stats['Avg Risk Score'].iloc[-1] - monthly_stats['Avg Risk Score'].iloc[0]
        
        st.markdown("### 📈 Trend Analysis")
        
        # Customer Acquisition with insight
        if len(monthly_stats) >= 2:
            st.markdown(f"""
            **Customer Growth Trajectory**
            {f"📈 Up {customer_change:.1f}% from baseline" if customer_change > 0 else f"📉 Down {abs(customer_change):.1f}% from baseline"}
            """)
        else:
            st.markdown("**Customer Growth Trajectory**")
            st.caption("Need at least 2 months of data to show trends")
        
        fig = px.line(monthly_stats, x='Month', y='New Customers',
                     markers=True, color_discrete_sequence=[BURNT_ORANGE])
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(monthly_stats) >= 2:
                st.markdown(f"""
                **Coverage Trends**
                {f"💰 Average coverage {'increased' if coverage_change > 0 else 'decreased'} by {abs(coverage_change):.1f}%"}
                """)
            else:
                st.markdown("**Coverage Trends**")
                st.caption("Need at least 2 months of data to show trends")
            
            fig = px.line(monthly_stats, x='Month', y='Avg Coverage',
                         markers=True, color_discrete_sequence=[CHARCOAL_GRAY])
            fig.update_layout(showlegend=False, yaxis_tickprefix="$", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(monthly_stats) >= 2:
                st.markdown(f"""
                **Risk Profile Changes**
                {f"⚠️ Risk score {'increased' if risk_change > 0 else 'decreased'} by {abs(risk_change):.2f} points"}
                """)
            else:
                st.markdown("**Risk Profile Changes**")
                st.caption("Need at least 2 months of data to show trends")
            
            fig = px.line(monthly_stats, x='Month', y='Avg Risk Score',
                         markers=True, color_discrete_sequence=['#d32f2f'])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Policy mix analysis
        if 'policy_type' in df.columns and len(monthly_stats) >= 1:
            st.markdown("---")
            st.markdown("### 📊 Portfolio Composition")
            
            policy_monthly = df.groupby(['month', 'policy_type']).size().reset_index(name='count')
            
            # Calculate policy mix insights
            if len(policy_monthly) > 0:
                latest_month = policy_monthly[policy_monthly['month'] == policy_monthly['month'].max()]
                if len(latest_month) > 0:
                    policy_leader = latest_month.nlargest(1, 'count')['policy_type'].values[0]
                    policy_leader_pct = latest_month.nlargest(1, 'count')['count'].values[0] / latest_month['count'].sum() * 100
                    
                    st.markdown(f"""
                    **Current Mix:** {policy_leader} dominates at {policy_leader_pct:.1f}% of new policies
                    """)
                    
                    fig = px.area(policy_monthly, x='month', y='count', color='policy_type',
                                color_discrete_sequence=[BURNT_ORANGE, CHARCOAL_GRAY, LIGHT_GRAY, "#9A4600"])
                    fig.update_layout(xaxis_title="Month", yaxis_title="Count", height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics with context
    st.markdown("---")
    st.markdown("### 🎯 Portfolio Snapshot")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            age_segment = "Young" if avg_age < 35 else "Mid-Career" if avg_age < 55 else "Pre-Retirement"
            st.metric("Avg Customer Age", f"{avg_age:.1f} years", f"{age_segment} Focus")
    
    with col2:
        total_coverage = df['insurance_face_amount'].sum() if 'insurance_face_amount' in df else 0
        avg_per_customer = total_coverage / len(df)
        st.metric("Total Portfolio Value", f"${total_coverage:,.0f}", f"${avg_per_customer:,.0f}/customer")
        
    with col3:
        if 'policy_type' in df.columns:
            top_policy = df['policy_type'].mode()[0]
            pct = (df['policy_type'] == top_policy).sum() / len(df) * 100
            st.metric("Leading Policy Type", top_policy, f"{pct:.1f}% share")
    
    with col4:
        high_value = len(df[df['insurance_face_amount'] > df['insurance_face_amount'].median()])
        pct_high_value = high_value / len(df) * 100
        st.metric("High-Value Customers", f"{high_value:,}", f"{pct_high_value:.1f}% of base")

# TAB 2: GEOGRAPHIC HEATMAP (NEW)
with tab2:
    st.markdown("## 🗺️ Geographic Distribution & Performance")
    
    if 'state' in df.columns:
        # State-level aggregation
        state_stats = df.groupby('state').agg({
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
                colorscale=[
                    [0, '#deebf7'],      # Very light blue
                    [0.2, '#9ecae1'],    # Light blue
                    [0.4, '#4292c6'],    # Medium blue
                    [0.6, '#2171b5'],    # Blue
                    [0.8, '#08519c'],    # Dark blue
                    [1, '#08306b']       # Very dark blue
                ],
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                colorbar=dict(
                    title="<b>Customers</b>",
                    thickness=25,
                    len=0.8,
                    bgcolor='rgba(255,255,255,0.9)',
                    tickfont=dict(size=13),
                    titlefont=dict(size=14, family='Poppins, sans-serif')
                ),
                marker=dict(
                    line=dict(
                        color='#ffffff',
                        width=2
                    )
                )
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
                st.metric("📍 States with Customers", f"{states_with_customers}")
        
        elif map_view == "Total Portfolio Value":
            st.markdown("### 💰 Total Portfolio Value by State")
            st.caption("Total insurance coverage amount per state • Green intensity shows market value")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Total Coverage'],
                locationmode='USA-states',
                colorscale=[
                    [0, '#f7fcf5'],      # Very light green
                    [0.2, '#c7e9c0'],    # Light green
                    [0.4, '#74c476'],    # Medium light green
                    [0.6, '#31a354'],    # Medium green
                    [0.8, '#006d2c'],    # Dark green
                    [1, '#00441b']       # Very dark green
                ],
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                colorbar=dict(
                    title="<b>Total Value</b>",
                    thickness=25,
                    len=0.8,
                    bgcolor='rgba(255,255,255,0.9)',
                    tickformat='$,.0f',
                    tickfont=dict(size=13),
                    titlefont=dict(size=14, family='Poppins, sans-serif')
                ),
                marker=dict(
                    line=dict(
                        color='#ffffff',
                        width=2
                    )
                )
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
                st.metric("📊 Avg State Value", f"${avg_state_value:,.0f}")
        
        elif map_view == "Average Coverage per Customer":
            st.markdown("### 📊 Average Coverage per Customer")
            st.caption("Average policy value by state • Orange intensity shows premium customers")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Avg Coverage'],
                locationmode='USA-states',
                colorscale=[
                    [0, '#fff5eb'],      # Very light orange
                    [0.2, '#fee6ce'],    # Light orange
                    [0.4, '#fdd0a2'],    # Light medium orange
                    [0.6, '#fdae6b'],    # Medium orange
                    [0.8, '#e6550d'],    # Dark orange (Syntex brand)
                    [1, '#a63603']       # Very dark orange
                ],
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                colorbar=dict(
                    title="<b>Avg Coverage</b>",
                    thickness=25,
                    len=0.8,
                    bgcolor='rgba(255,255,255,0.9)',
                    tickformat='$,.0f',
                    tickfont=dict(size=13),
                    titlefont=dict(size=14, family='Poppins, sans-serif')
                ),
                marker=dict(
                    line=dict(
                        color='#ffffff',
                        width=2
                    )
                )
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
                st.metric("📊 National Average", f"${overall_avg:,.0f}")
            with col3:
                premium_states = len(state_stats[state_stats['Avg Coverage'] > overall_avg])
                st.metric("⭐ Above Average States", f"{premium_states}")
        
        elif map_view == "Risk Distribution":
            st.markdown("### ⚠️ Risk Distribution by State")
            st.caption("Average risk score per state • Red = high risk, Yellow = medium, Green = low risk")
            
            fig = go.Figure(data=go.Choropleth(
                locations=state_stats['state'],
                z=state_stats['Avg Risk'],
                locationmode='USA-states',
                colorscale=[
                    [0, '#1a9850'],      # Dark green (low risk)
                    [0.25, '#91cf60'],   # Light green
                    [0.5, '#ffffbf'],    # Yellow (medium risk)
                    [0.75, '#fc8d59'],   # Orange
                    [1, '#d73027']       # Red (high risk)
                ],
                text=state_stats['state'],
                hovertext=state_stats['hover_text'],
                hoverinfo='text',
                colorbar=dict(
                    title="<b>Risk Score</b>",
                    thickness=25,
                    len=0.8,
                    bgcolor='rgba(255,255,255,0.9)',
                    tickfont=dict(size=13),
                    titlefont=dict(size=14, family='Poppins, sans-serif')
                ),
                marker=dict(
                    line=dict(
                        color='#ffffff',
                        width=2
                    )
                ),
                zmid=state_stats['Avg Risk'].median()  # Center the color scale at median
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
                st.metric("⚠️ Highest Risk State", highest_risk_state, f"{highest_risk:.2f} risk score")
            with col2:
                avg_risk = state_stats['Avg Risk'].mean()
                st.metric("📊 National Avg Risk", f"{avg_risk:.2f}")
            with col3:
                high_risk_states = len(state_stats[state_stats['Avg Risk'] >= 2.5])
                st.metric("🚨 High Risk States", f"{high_risk_states}")
        
        # Summary Table - always show below the map
        st.markdown("---")
        st.markdown("### 📈 State Performance Summary")
        
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
    st.markdown("## 🤖 AI-Generated Insights")
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Regenerate", key="regenerate_insights"):
            with st.spinner("Generating insights..."):
                claude_client = get_claude_client()
                insights = generate_insights_with_ai(df, claude_client)
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
        if 'policy_type' in df.columns:
            policy_counts = df['policy_type'].value_counts()
            fig = px.pie(values=policy_counts.values, names=policy_counts.index,
                        color_discrete_sequence=[BURNT_ORANGE, CHARCOAL_GRAY, LIGHT_GRAY, "#9A4600"])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Risk Distribution")
        risk_dist = df['risk_count'].value_counts().sort_index()
        fig = px.bar(x=risk_dist.index, y=risk_dist.values,
                    labels={'x': 'Risk Factors', 'y': 'Customer Count'},
                    color_discrete_sequence=[BURNT_ORANGE])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: UPSELL OPPORTUNITIES
with tab4:
    st.markdown("## 🎯 Upsell Opportunities")
    
    upsell_df = df[df['upsell_count'] >= 2].sort_values('upsell_count', ascending=False)
    
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
    st.markdown("## ⚠️ Risk Analysis")
    
    high_risk = df[df['risk_count'] >= 3]
    medium_risk = df[df['risk_count'] == 2]
    low_risk = df[df['risk_count'] <= 1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk", f"{len(high_risk):,}", help="3+ risk factors")
    
    with col2:
        st.metric("Medium Risk", f"{len(medium_risk):,}", help="2 risk factors")
    
    with col3:
        st.metric("Low Risk", f"{len(low_risk):,}", help="0-1 risk factors")
    
    st.markdown("---")
    
    all_risks = []
    for risks in df['risk_factors']:
        all_risks.extend(risks)
    
    risk_counts = pd.Series(all_risks).value_counts()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Risk Factor Prevalence")
        for risk, count in risk_counts.items():
            pct = (count / len(df)) * 100
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
        st.success("✅ No high-risk customers found!")

# TAB 6: AI CHAT
with tab6:
    st.markdown("## 🤖 AI Assistant")
    st.markdown("Ask questions about your life insurance portfolio and get AI-powered insights.")
    
    claude_client = get_claude_client()
    
    if claude_client is None:
        st.warning("⚠️ AI chat unavailable. Please configure ANTHROPIC_API_KEY in Streamlit secrets.")
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
            st.markdown("#### 💡 Suggested Questions")
            
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

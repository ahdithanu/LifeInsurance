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
    st.markdown("## 📊 Portfolio Overview & Trends")
    
    # Time-based metrics
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Customer Acquisition Trend")
            fig = px.line(monthly_stats, x='Month', y='New Customers',
                         markers=True, color_discrete_sequence=[BURNT_ORANGE])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Average Coverage Over Time")
            fig = px.line(monthly_stats, x='Month', y='Avg Coverage',
                         markers=True, color_discrete_sequence=[CHARCOAL_GRAY])
            fig.update_layout(showlegend=False, yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚠️ Risk Score Trend")
            fig = px.line(monthly_stats, x='Month', y='Avg Risk Score',
                         markers=True, color_discrete_sequence=['#d32f2f'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Policy Type Distribution Over Time")
            if 'policy_type' in df.columns:
                policy_monthly = df.groupby(['month', 'policy_type']).size().reset_index(name='count')
                fig = px.area(policy_monthly, x='month', y='count', color='policy_type',
                            color_discrete_sequence=[BURNT_ORANGE, CHARCOAL_GRAY, LIGHT_GRAY, "#9A4600"])
                fig.update_layout(xaxis_title="Month", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
    
    # Key insights summary
    st.markdown("---")
    st.markdown("### 🔑 Key Portfolio Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'age' in df.columns:
            avg_age = df['age'].mean()
            st.metric("Average Customer Age", f"{avg_age:.1f} years")
            age_trend = "↗️ Increasing" if df.groupby('month')['age'].mean().diff().mean() > 0 else "↘️ Decreasing"
            st.caption(f"Trend: {age_trend}")
    
    with col2:
        total_coverage = df['insurance_face_amount'].sum() if 'insurance_face_amount' in df else 0
        st.metric("Total Portfolio Value", f"${total_coverage:,.0f}")
        
    with col3:
        if 'policy_type' in df.columns:
            top_policy = df['policy_type'].mode()[0]
            st.metric("Most Popular Policy", top_policy)
            pct = (df['policy_type'] == top_policy).sum() / len(df) * 100
            st.caption(f"{pct:.1f}% of portfolio")

# TAB 2: GEOGRAPHIC HEATMAP (NEW)
with tab2:
    st.markdown("## 🗺️ Geographic Distribution")
    
    if 'state' in df.columns:
        # State-level aggregation
        state_stats = df.groupby('state').agg({
            'primary_full_name': 'count',
            'insurance_face_amount': 'mean',
            'risk_count': 'mean',
            'upsell_count': 'mean'
        }).reset_index()
        state_stats.columns = ['state', 'Customer Count', 'Avg Coverage', 'Avg Risk', 'Avg Upsell Opportunities']
        
        # USA Choropleth Map
        st.markdown("### 🇺🇸 Customer Distribution by State")
        
        fig = go.Figure(data=go.Choropleth(
            locations=state_stats['state'],
            z=state_stats['Customer Count'],
            locationmode='USA-states',
            colorscale=[[0, '#EBFFF7'], [0.5, LIGHT_GRAY], [1, BURNT_ORANGE]],
            colorbar_title="Customers",
            marker_line_color='white',
            marker_line_width=1.5
        ))
        
        fig.update_layout(
            geo=dict(
                scope='usa',
                projection=go.layout.geo.Projection(type='albers usa'),
                showlakes=True,
                lakecolor='rgb(255, 255, 255)'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Coverage Heatmap
        st.markdown("### 💰 Average Coverage by State")
        
        fig = go.Figure(data=go.Choropleth(
            locations=state_stats['state'],
            z=state_stats['Avg Coverage'],
            locationmode='USA-states',
            colorscale=[[0, '#EBFFF7'], [0.5, '#FFD700'], [1, '#228B22']],
            colorbar_title="Avg Coverage ($)",
            marker_line_color='white',
            marker_line_width=1.5
        ))
        
        fig.update_layout(
            geo=dict(
                scope='usa',
                projection=go.layout.geo.Projection(type='albers usa'),
                showlakes=True,
                lakecolor='rgb(255, 255, 255)'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Heatmap
        st.markdown("### ⚠️ Risk Distribution by State")
        
        fig = go.Figure(data=go.Choropleth(
            locations=state_stats['state'],
            z=state_stats['Avg Risk'],
            locationmode='USA-states',
            colorscale=[[0, '#90EE90'], [0.5, '#FFD700'], [1, '#d32f2f']],
            colorbar_title="Avg Risk Score",
            marker_line_color='white',
            marker_line_width=1.5
        ))
        
        fig.update_layout(
            geo=dict(
                scope='usa',
                projection=go.layout.geo.Projection(type='albers usa'),
                showlakes=True,
                lakecolor='rgb(255, 255, 255)'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top States Table
        st.markdown("---")
        st.markdown("### 🏆 Top States by Customer Count")
        
        top_states = state_stats.nlargest(10, 'Customer Count')
        top_states['Avg Coverage'] = top_states['Avg Coverage'].apply(lambda x: f"${x:,.0f}")
        top_states['Avg Risk'] = top_states['Avg Risk'].apply(lambda x: f"{x:.2f}")
        top_states['Avg Upsell Opportunities'] = top_states['Avg Upsell Opportunities'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(top_states, use_container_width=True, hide_index=True)

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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from anthropic import Anthropic

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
    
    /* Header styling */
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
    
    /* Metric cards */
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {BURNT_ORANGE};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {BURNT_ORANGE};
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: #9A4600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Insight box */
    .insight-box {{
        background: linear-gradient(135deg, {BURNT_ORANGE} 0%, #9A4600 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Tab styling */
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
        background-color: transparent;
        border-radius: 8px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {BURNT_ORANGE};
        color: white;
    }}
    
    /* Risk badges */
    .risk-high {{
        color: #d32f2f;
        background-color: rgba(211, 47, 47, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }}
    
    .risk-medium {{
        color: #f57c00;
        background-color: rgba(245, 124, 0, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }}
    
    .risk-low {{
        color: #388e3c;
        background-color: rgba(56, 142, 60, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    h1, h2, h3 {{
        color: {CHARCOAL_GRAY};
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize Anthropic client
@st.cache_resource
def get_claude_client():
    api_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)

# Data loading and processing functions
@st.cache_data
def load_data(uploaded_file=None):
    """Load and process life insurance data"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Try to load default data
        try:
            df = pd.read_csv('data/life_insurance_data.csv')
        except:
            return None
    
    # Convert date fields
    if 'date_field' in df.columns:
        df['date_field'] = pd.to_datetime(df['date_field'], errors='coerce')
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365
    
    # Convert numeric fields
    numeric_cols = ['insurance_face_amount', 'height', 'weight']
    for col in numeric_cols:
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
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard analyzes life insurance customer data to identify:
    - Upsell opportunities
    - At-risk customers
    - Portfolio optimization insights
    
    **Built by Syntex Data**
    """)
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load data
df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ No data loaded. Please upload a CSV file or ensure data/life_insurance_data.csv exists.")
    st.stop()

# Store in session state
if 'df' not in st.session_state:
    st.session_state['df'] = df
else:
    df = st.session_state['df']

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Customers",
        value=f"{len(df):,}",
        delta=None
    )

with col2:
    avg_coverage = df['insurance_face_amount'].mean() if 'insurance_face_amount' in df else 0
    st.metric(
        label="Avg Coverage",
        value=f"${avg_coverage:,.0f}",
        delta=None
    )

with col3:
    high_risk = len(df[df['risk_count'] >= 3])
    st.metric(
        label="High Risk",
        value=f"{high_risk:,}",
        delta=None,
        delta_color="inverse"
    )

with col4:
    upsell_candidates = len(df[df['upsell_count'] >= 2])
    st.metric(
        label="Upsell Opportunities",
        value=f"{upsell_candidates:,}",
        delta=None,
        delta_color="normal"
    )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎯 Upsell Opportunities", "⚠️ Risk Analysis", "🤖 AI Chat"])

with tab1:
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

with tab2:
    st.markdown("## 🎯 Upsell Opportunities")
    
    upsell_df = df[df['upsell_count'] >= 2].sort_values('upsell_count', ascending=False)
    
    st.markdown(f"### Found {len(upsell_df)} customers with 2+ upsell opportunities")
    
    if len(upsell_df) > 0:
        # Opportunity breakdown
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
        
        # Display table
        display_cols = ['primary_full_name', 'age', 'insurance_face_amount', 'policy_type', 
                       'upsell_count', 'upsell_opportunities']
        available_cols = [col for col in display_cols if col in upsell_df.columns]
        
        st.dataframe(
            upsell_df[available_cols].head(20),
            use_container_width=True,
            hide_index=True
        )

with tab3:
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
    
    # Risk factor breakdown
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
        
        st.dataframe(
            high_risk[available_cols].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ No high-risk customers found!")

with tab4:
    st.markdown("## 🤖 AI Assistant")
    st.markdown("Ask questions about your life insurance portfolio and get AI-powered insights.")
    
    claude_client = get_claude_client()
    
    if claude_client is None:
        st.warning("⚠️ AI chat unavailable. Please configure ANTHROPIC_API_KEY in Streamlit secrets.")
    else:
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Display chat history
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
        
        # Chat input
        user_query = st.text_input("Ask a question:", key="chat_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Send", key="send_chat"):
                if user_query:
                    st.session_state['chat_history'].append({'role': 'user', 'content': user_query})
                    
                    # Generate AI response
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
        
        # Suggested questions
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

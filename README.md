# Life Insurance Insights Dashboard | Syntex Data

![Syntex Data](https://img.shields.io/badge/Built%20by-Syntex%20Data-B75400?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge)

**Unlock the Power of AI. From Data to Deployment.**

AI-powered dashboard for analyzing life insurance customer data with beautiful Syntex Data branding.

## 🎯 Features

- **📊 Interactive Dashboard** - Real-time analytics with Plotly charts
- **🎯 Upsell Detection** - Automatically identifies customer opportunities
- **⚠️ Risk Analysis** - Flags high-risk customers
- **🤖 AI Chat** - Ask questions about your data (powered by Claude)
- **📁 File Upload** - Drag & drop CSV files
- **🎨 Syntex Branding** - Burnt orange & charcoal gray theme

## 🚀 Quick Start Options

### Option 1: Deploy from Jupyter Notebook (Easiest!)

1. Open `DEPLOY_FROM_JUPYTER.ipynb`
2. Run each cell in order
3. Follow the instructions
4. Done!

### Option 2: Deploy Manually

```bash
# 1. Generate sample data
python generate_sample_data.py

# 2. Initialize git
git init
git add .
git commit -m "Initial commit"

# 3. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main

# 4. Deploy on https://share.streamlit.io
```

## 📊 Using Your Own Data

### Method 1: Upload Through UI
- Run the app
- Click "Upload Life Insurance CSV" in sidebar
- Select your file

### Method 2: Replace Sample Data
```bash
cp your-data.csv data/life_insurance_data.csv
```

### Required CSV Columns

Your CSV must include:
- `primary_full_name` - Customer name
- `dob` - Date of birth (YYYY-MM-DD)
- `insurance_face_amount` - Coverage amount (numeric)
- `policy_type` - Type of policy
- `tobacco_use` - Yes/No
- `has_medical_conditions` - Yes/No
- `hospitalization_history` - Yes/No
- `height` - Height in inches
- `weight` - Weight in pounds
- `state` - State code
- Plus more (see full list below)

<details>
<summary>Click to see all required columns</summary>

- file_name
- primary_full_name
- date_field
- city
- state
- has_medicaid_eligibility
- has_preauth_payments
- dob
- height
- weight
- sex
- birthplace
- occupation
- owner_address_line
- owner_city
- phone_home
- phone_cell
- insurance_face_amount
- policy_type
- payment_method
- tobacco_use
- has_medical_conditions
- hospitalization_history
- us_citizen

</details>

## 🔐 Setup Secrets

### For Local Testing:
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

### For Streamlit Cloud:
1. Go to app settings
2. Click "Secrets"
3. Add:
   ```toml
   ANTHROPIC_API_KEY = "your-api-key-here"
   ```

Get your API key at: https://console.anthropic.com

## 📁 Project Structure

```
life-insurance-streamlit/
├── streamlit_app.py                    # Main application
├── requirements.txt                    # Dependencies
├── generate_sample_data.py            # Sample data generator
├── DEPLOY_FROM_JUPYTER.ipynb          # Deployment notebook
├── .streamlit/
│   ├── config.toml                    # Syntex Data theme
│   └── secrets.toml.example           # Secrets template
├── data/
│   └── life_insurance_data.csv        # Your data goes here
└── README.md                          # This file
```

## 🎨 Customization

### Change Colors

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#B75400"  # Your brand color
backgroundColor = "#EBFFF7"
```

### Change Company Name

Edit `streamlit_app.py` line ~85:
```python
st.markdown("<h1>Your Company Name</h1>")
```

### Add Custom Risk Factors

Edit the `identify_risk_factors()` function in `streamlit_app.py`.

## 🐛 Troubleshooting

### "No data loaded"
```bash
python generate_sample_data.py
```

### "AI not working"
- Check `.streamlit/secrets.toml` exists
- Verify API key is correct
- Ensure you have API credits

### Dependencies error
```bash
pip install -r requirements.txt
```

### Git authentication error
- Use Personal Access Token instead of password
- Get one at: https://github.com/settings/tokens
- Check "repo" scope
- Use token as password

## 📚 Documentation

- **DEPLOY_FROM_JUPYTER.ipynb** - Step-by-step deployment
- **Streamlit Docs** - https://docs.streamlit.io
- **Anthropic API** - https://docs.anthropic.com

## 🚀 Deployment Checklist

- [ ] Downloaded all files
- [ ] Generated sample data
- [ ] Tested locally with `streamlit run streamlit_app.py`
- [ ] Created GitHub repository
- [ ] Pushed code to GitHub
- [ ] Deployed to Streamlit Cloud
- [ ] Added API key in secrets
- [ ] Tested live app

## 💡 Features Breakdown

### Overview Tab
- Portfolio summary metrics
- AI-generated insights
- Interactive charts (coverage, age, policy types, risk)

### Upsell Opportunities Tab
- Opportunity type breakdown
- Top candidate recommendations
- Sortable customer table

### Risk Analysis Tab
- Risk level categorization
- Risk factor prevalence
- High-risk customer alerts

### AI Chat Tab
- Natural language queries
- Conversation history
- Suggested questions
- Context-aware responses

## 🆘 Support

Questions or issues?
- Check this README
- Review the Jupyter notebook
- Visit Streamlit Community Forum
- Email: support@syntex-data.com

## 📄 License

Proprietary - Syntex Data

---

**Built with ❤️ by Syntex Data**

*Unlock the Power of AI. From Data to Deployment.*

🌐 syntex-data.com

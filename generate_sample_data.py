import pandas as pd
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
random.seed(42)

# Configuration
NUM_CUSTOMERS = 500

# Data generation
data = []

policy_types = ['Term Life', 'Whole Life', 'Universal Life', 'Variable Life']
payment_methods = ['Monthly', 'Quarterly', 'Annual']
occupations = [
    'Software Engineer', 'Teacher', 'Accountant', 'Manager', 
    'Sales Representative', 'Doctor', 'Lawyer', 'Consultant', 
    'Entrepreneur', 'Retired', 'Nurse', 'Engineer', 'Analyst'
]
states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 
          'AZ', 'WA', 'MA', 'TN', 'IN']
cities = {
    'CA': ['Los Angeles', 'San Francisco', 'San Diego', 'San Jose'],
    'TX': ['Houston', 'Dallas', 'Austin', 'San Antonio'],
    'FL': ['Miami', 'Tampa', 'Orlando', 'Jacksonville'],
    'NY': ['New York', 'Buffalo', 'Rochester', 'Albany'],
    'PA': ['Philadelphia', 'Pittsburgh', 'Allentown'],
}

first_names = [
    'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
    'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara',
    'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty',
    'Margaret', 'Sandra', 'Ashley', 'Kimberly', 'Emily', 'Donna'
]

last_names = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
    'Lee', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark'
]

def generate_phone():
    return f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"

for i in range(NUM_CUSTOMERS):
    # Demographics
    age = random.randint(20, 80)
    dob = (datetime.now() - timedelta(days=age*365 + random.randint(0, 364))).strftime('%Y-%m-%d')
    
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    full_name = f"{first_name} {last_name}"
    
    sex = random.choice(['M', 'F'])
    
    # Location
    state = random.choice(states)
    city = random.choice(cities.get(state, ['Springfield']))
    birthplace = random.choice([c for sublist in cities.values() for c in sublist])
    
    # Physical attributes
    height = random.randint(58, 78)  # inches
    if sex == 'M':
        base_weight = (height - 58) * 4 + 140
    else:
        base_weight = (height - 58) * 3.5 + 115
    weight = base_weight + random.randint(-30, 50)
    
    # Risk factors (correlated with age)
    tobacco_use = random.random() < 0.15
    has_medical_conditions = random.random() < (0.15 + age * 0.004)  # Increases with age
    hospitalization_history = random.random() < (0.10 + age * 0.003)
    
    # Coverage and policy details
    if age < 35:
        face_amount = random.choice([50000, 75000, 100000, 150000, 200000])
    elif age < 50:
        face_amount = random.choice([100000, 150000, 200000, 250000, 300000, 500000])
    elif age < 65:
        face_amount = random.choice([100000, 150000, 200000, 250000])
    else:
        face_amount = random.choice([50000, 75000, 100000, 150000])
    
    policy_type = random.choice(policy_types)
    payment_method = random.choice(payment_methods)
    
    # Engagement indicators
    has_preauth_payments = random.random() < 0.65
    has_medicaid_eligibility = random.random() < 0.10
    
    # Create record
    record = {
        'file_name': f'policy_{i+1:04d}.pdf',
        'primary_full_name': full_name,
        'date_field': (datetime.now() - timedelta(days=random.randint(30, 3650))).strftime('%Y-%m-%d'),
        'city': city,
        'state': state,
        'has_medicaid_eligibility': 'Yes' if has_medicaid_eligibility else 'No',
        'has_preauth_payments': 'Yes' if has_preauth_payments else 'No',
        'dob': dob,
        'height': height,
        'weight': weight,
        'sex': sex,
        'birthplace': birthplace,
        'occupation': random.choice(occupations),
        'owner_address_line': f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Pine'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr', 'Ln'])}",
        'owner_city': city,
        'phone_home': generate_phone(),
        'phone_cell': generate_phone(),
        'insurance_face_amount': face_amount,
        'policy_type': policy_type,
        'payment_method': payment_method,
        'tobacco_use': 'Yes' if tobacco_use else 'No',
        'has_medical_conditions': 'Yes' if has_medical_conditions else 'No',
        'hospitalization_history': 'Yes' if hospitalization_history else 'No',
        'us_citizen': 'Yes' if random.random() < 0.95 else 'No'
    }
    
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Create data directory if it doesn't exist
os.makedirs('backend/data', exist_ok=True)

# Save to CSV
df.to_csv('backend/data/life_insurance_data.csv', index=False)

print(f"✅ Generated {NUM_CUSTOMERS} sample customer records")
print(f"📊 File saved: backend/data/life_insurance_data.csv")
print(f"\nSample statistics:")
print(f"- Average age: {df['dob'].apply(lambda x: (datetime.now() - pd.to_datetime(x)).days // 365).mean():.1f} years")
print(f"- Average coverage: ${df['insurance_face_amount'].mean():,.0f}")
print(f"- Tobacco users: {(df['tobacco_use'] == 'Yes').sum()} ({(df['tobacco_use'] == 'Yes').sum()/len(df)*100:.1f}%)")
print(f"- Medical conditions: {(df['has_medical_conditions'] == 'Yes').sum()} ({(df['has_medical_conditions'] == 'Yes').sum()/len(df)*100:.1f}%)")
print(f"\nStates represented: {', '.join(df['state'].unique())}")
print(f"Policy types: {', '.join(df['policy_type'].unique())}")

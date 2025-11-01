
# Now create actual synthetic dataset generator for Indian students after 12th

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define realistic distributions for Indian students

# 1. Demographics
states = ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh', 
          'West Bengal', 'Gujarat', 'Rajasthan', 'Kerala', 'Punjab', 
          'Telangana', 'Andhra Pradesh', 'Madhya Pradesh', 'Bihar', 'Haryana']

genders = ['Male', 'Female', 'Other']
urban_rural = ['Urban', 'Semi-Urban', 'Rural']
income_brackets = ['<5 Lakhs', '5-10 Lakhs', '10-20 Lakhs', '20-50 Lakhs', '>50 Lakhs']

# 2. Academic streams
streams_12th = ['Science-PCM', 'Science-PCB', 'Science-PCMB', 'Commerce', 'Arts']
boards = ['CBSE', 'ICSE', 'State Board']
school_tiers = ['Tier 1', 'Tier 2', 'Tier 3']

# 3. Career paths with realistic probabilities based on stream
career_paths_all = {
    'Science-PCM': [
        'Computer Science Engineering', 'Mechanical Engineering', 'Electrical Engineering',
        'Electronics Engineering', 'Civil Engineering', 'Data Science', 'AI/ML Engineering',
        'Information Technology', 'Aerospace Engineering', 'B.Sc Mathematics', 'B.Sc Physics'
    ],
    'Science-PCB': [
        'MBBS', 'BDS', 'B.Pharm', 'Nursing', 'Physiotherapy', 'BAMS', 'BHMS',
        'Biotechnology', 'B.Sc Biology', 'Veterinary Science', 'Medical Lab Technology'
    ],
    'Science-PCMB': [
        'Computer Science Engineering', 'Biotechnology', 'MBBS', 'B.Pharm',
        'Data Science', 'Bio-informatics', 'Environmental Science'
    ],
    'Commerce': [
        'B.Com', 'BBA', 'CA', 'CS', 'CMA', 'Economics', 'Finance', 
        'Marketing', 'Actuarial Science', 'B.Com + CA'
    ],
    'Arts': [
        'BA Psychology', 'BA Journalism', 'Law (BA LLB)', 'Mass Communication',
        'Design', 'Fine Arts', 'Hotel Management', 'Social Work', 'Literature'
    ]
}

def generate_student_profile(student_id):
    """Generate a single synthetic student profile"""
    
    profile = {
        'student_id': f'STU_{student_id:06d}',
        
        # Demographics
        'age': random.randint(16, 18),
        'gender': np.random.choice(genders, p=[0.52, 0.47, 0.01]),
        'state': np.random.choice(states),
        'urban_rural': np.random.choice(urban_rural, p=[0.4, 0.35, 0.25]),
        'family_income': np.random.choice(income_brackets, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
        
        # Academic Background
        '12th_stream': np.random.choice(streams_12th, p=[0.35, 0.25, 0.10, 0.20, 0.10]),
        '10th_percentage': round(np.random.normal(75, 10), 2),
        '12th_percentage': round(np.random.normal(78, 9), 2),
        'school_board': np.random.choice(boards, p=[0.45, 0.15, 0.40]),
        'school_tier': np.random.choice(school_tiers, p=[0.25, 0.45, 0.30])
    }
    
    # Clip percentages to realistic range
    profile['10th_percentage'] = np.clip(profile['10th_percentage'], 60, 98)
    profile['12th_percentage'] = np.clip(profile['12th_percentage'], 60, 98)
    
    # Competitive Exam Scores (based on stream and percentages)
    stream = profile['12th_stream']
    
    if 'Science' in stream:
        # JEE scores (40% of PCM students appear)
        if 'PCM' in stream or 'PCMB' in stream:
            if random.random() < 0.40:
                # JEE Main percentile (correlated with 12th marks)
                base_percentile = profile['12th_percentage'] * 0.7 + random.uniform(-10, 15)
                profile['JEE_Main_percentile'] = round(np.clip(base_percentile, 45, 99.99), 2)
                
                # JEE Advanced (only top 2.5 lakh qualify ~20% of JEE Main takers)
                if profile['JEE_Main_percentile'] > 87 and random.random() < 0.20:
                    # Rank inversely proportional to percentile
                    profile['JEE_Advanced_rank'] = int(np.clip(
                        (100 - profile['JEE_Main_percentile']) * 2500, 1, 250000
                    ))
                else:
                    profile['JEE_Advanced_rank'] = None
            else:
                profile['JEE_Main_percentile'] = None
                profile['JEE_Advanced_rank'] = None
        else:
            profile['JEE_Main_percentile'] = None
            profile['JEE_Advanced_rank'] = None
        
        # NEET scores (60% of PCB students appear)
        if 'PCB' in stream or 'PCMB' in stream:
            if random.random() < 0.60:
                base_percentile = profile['12th_percentage'] * 0.75 + random.uniform(-12, 12)
                profile['NEET_percentile'] = round(np.clip(base_percentile, 50, 99.99), 2)
                
                # NEET rank (out of ~20 lakh students)
                profile['NEET_rank'] = int((100 - profile['NEET_percentile']) * 20000)
            else:
                profile['NEET_percentile'] = None
                profile['NEET_rank'] = None
        else:
            profile['NEET_percentile'] = None
            profile['NEET_rank'] = None
    else:
        profile['JEE_Main_percentile'] = None
        profile['JEE_Advanced_rank'] = None
        profile['NEET_percentile'] = None
        profile['NEET_rank'] = None
    
    # CUET score (for central universities - 30% appear)
    if random.random() < 0.30:
        profile['CUET_score'] = int(np.random.normal(500, 120))
        profile['CUET_score'] = np.clip(profile['CUET_score'], 200, 800)
    else:
        profile['CUET_score'] = None
    
    # Aptitude Scores (1-10 scale)
    profile['logical_reasoning'] = round(np.random.normal(6, 1.5), 1)
    profile['quantitative_ability'] = round(np.random.normal(6.5, 1.8), 1)
    profile['verbal_ability'] = round(np.random.normal(6.2, 1.6), 1)
    profile['abstract_reasoning'] = round(np.random.normal(6, 1.7), 1)
    profile['spatial_reasoning'] = round(np.random.normal(5.8, 1.5), 1)
    
    # Clip aptitude scores
    for key in ['logical_reasoning', 'quantitative_ability', 'verbal_ability', 
                'abstract_reasoning', 'spatial_reasoning']:
        profile[key] = np.clip(profile[key], 1, 10)
    
    # Interest Areas (1-5 scale)
    profile['interest_technology'] = round(np.random.normal(3.5, 1), 1)
    profile['interest_healthcare'] = round(np.random.normal(3, 1.2), 1)
    profile['interest_business'] = round(np.random.normal(3.2, 1.1), 1)
    profile['interest_creative_arts'] = round(np.random.normal(2.8, 1.3), 1)
    profile['interest_social_service'] = round(np.random.normal(2.5, 1.2), 1)
    profile['interest_research'] = round(np.random.normal(2.7, 1.1), 1)
    
    # Clip interest scores
    for key in ['interest_technology', 'interest_healthcare', 'interest_business',
                'interest_creative_arts', 'interest_social_service', 'interest_research']:
        profile[key] = np.clip(profile[key], 1, 5)
    
    # Personality Traits (1-5 scale)
    profile['leadership'] = round(np.random.normal(3, 0.9), 1)
    profile['teamwork'] = round(np.random.normal(3.5, 0.8), 1)
    profile['creativity'] = round(np.random.normal(3.2, 1), 1)
    profile['analytical_thinking'] = round(np.random.normal(3.3, 1), 1)
    profile['communication'] = round(np.random.normal(3.4, 0.9), 1)
    
    # Clip personality scores
    for key in ['leadership', 'teamwork', 'creativity', 'analytical_thinking', 'communication']:
        profile[key] = np.clip(profile[key], 1, 5)
    
    # Preferences
    profile['preferred_location'] = np.random.choice(
        ['Home State', 'Nearby States', 'Pan India', 'Abroad'], 
        p=[0.45, 0.25, 0.25, 0.05]
    )
    profile['budget_constraint_lakhs'] = int(np.random.choice(
        [2, 5, 10, 15, 20, 30], 
        p=[0.25, 0.30, 0.20, 0.10, 0.10, 0.05]
    ))
    profile['career_goal_timeline'] = np.random.choice(
        ['4 years', '5 years', '6+ years'], 
        p=[0.50, 0.35, 0.15]
    )
    profile['work_preference'] = np.random.choice(
        ['Job', 'Business', 'Research', 'Government'], 
        p=[0.50, 0.20, 0.15, 0.15]
    )
    profile['risk_tolerance'] = np.random.choice(
        ['Low', 'Medium', 'High'], 
        p=[0.40, 0.45, 0.15]
    )
    
    # Extracurricular
    profile['has_sports'] = random.choice([True, False])
    profile['has_cultural'] = random.choice([True, False])
    profile['volunteering_hours'] = int(np.random.choice([0, 10, 20, 50, 100], p=[0.40, 0.30, 0.20, 0.08, 0.02]))
    profile['num_certifications'] = int(np.random.choice([0, 1, 2, 3, 5], p=[0.35, 0.30, 0.20, 0.10, 0.05]))
    profile['num_projects'] = int(np.random.choice([0, 1, 2, 3, 5], p=[0.30, 0.35, 0.20, 0.10, 0.05]))
    
    return profile

# Generate synthetic dataset
num_students = 2000  # Start with 2000 students

print("\nGenerating synthetic dataset for {} Indian students...".format(num_students))
print("This may take a moment...\n")

students = []
for i in range(num_students):
    students.append(generate_student_profile(i))
    if (i + 1) % 500 == 0:
        print(f"Generated {i+1} student profiles...")

df_students = pd.DataFrame(students)

print(f"\n✓ Successfully generated {len(df_students)} student profiles!")
print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Total Students: {len(df_students)}")
print(f"Total Features: {len(df_students.columns)}")
print(f"\nFeature List:")
print(df_students.columns.tolist())

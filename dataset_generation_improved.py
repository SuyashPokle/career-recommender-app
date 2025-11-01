# dataset_generation_improved.py
import pandas as pd
import numpy as np
import random
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("GENERATING IMPROVED DATASET - 10000 Indian Student Profiles")
print("=" * 80)

# 1. DEFINE REALISTIC DISTRIBUTIONS
# Demographics
STATES = [
    'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh',
    'West Bengal', 'Gujarat', 'Rajasthan', 'Kerala', 'Punjab',
    'Telangana', 'Andhra Pradesh', 'Madhya Pradesh', 'Bihar', 'Goa', 'Haryana',
    'Odisha', 'Chhattisgarh', 'Jharkhand', 'Assam', 'Uttarakhand', 'Himachal Pradesh', 'Jammu & Kashmir',
    'Tripura', 'Meghalaya', 'Manipur', 'Nagaland', 'Arunachal Pradesh', 'Mizoram', 'Sikkim'
]

GENDERS = ['Male', 'Female', 'Other']
URBAN_RURAL = ['Urban', 'Semi-Urban', 'Rural']
INCOME_BRACKETS = ['<5 Lakhs', '5-10 Lakhs', '10-20 Lakhs', '20-50 Lakhs', '>50 Lakhs']

# Academic
STREAMS_12TH = ['Science-PCM', 'Science-PCB', 'Science-PCMB', 'Commerce', 'Arts']
BOARDS = ['CBSE', 'ICSE', 'State Board']
SCHOOL_TIERS = ['Tier 1', 'Tier 2', 'Tier 3']

# Career paths by stream (EXPANDED for more diversity)
CAREER_PATHS = {
    'Science-PCM': [
        'Computer Science Engineering', 'Data Science', 'AI/ML Engineering',
        'Mechanical Engineering', 'Electrical Engineering', 'Information Technology',
        'Civil Engineering', 'Chemical Engineering', 'Aerospace Engineering',
        'Robotics Engineering', 'Embedded Systems Engineer', 'Cybersecurity Engineer'
    ],
    'Science-PCB': [
        'MBBS', 'BDS', 'B.Pharm', 'Nursing', 'Physiotherapy',
        'BAMS', 'BHMS', 'B.Sc Microbiology', 'Veterinary Science',
        'Biomedical Engineering', 'Biotechnology', 'Genetic Counselor'
    ],
    'Science-PCMB': [
        'Computer Science Engineering', 'Biotechnology', 'MBBS', 'B.Pharm',
        'Data Science', 'AI/ML Engineering', 'BDS', 'Biomedical Engineering'
    ],
    'Commerce': [
        'CA', 'BBA', 'B.Com', 'Economics', 'Finance', 'CMA',
        'CS', 'Actuarial Science', 'Investment Banking', 'Tax Consultant'
    ],
    'Arts': [
        'Law (BA LLB)', 'BA Psychology', 'Mass Communication', 'Journalism',
        'Design', 'BA History', 'BA Philosophy', 'Social Work',
        'Public Administration', 'Political Science'
    ]
}

# Career progressions (more realistic with various paths)
CAREER_PROGRESSIONS = {
    'Computer Science Engineering': [
        'Software Engineer', 'Senior Software Engineer', 'Tech Lead',
        'Engineering Manager', 'Director of Engineering'
    ],
    'Data Science': [
        'Data Analyst', 'Data Scientist', 'Senior Data Scientist',
        'Data Science Lead', 'Director of Data Science'
    ],
    'AI/ML Engineering': [
        'ML Engineer', 'Senior ML Engineer', 'ML Researcher',
        'AI Lead', 'Chief AI Officer'
    ],
    'MBBS': [
        'Junior Doctor', 'Senior Resident', 'Consultant', 'Specialist', 'Head of Department'
    ],
    'CA': [
        'Junior CA', 'Senior CA', 'Partner', 'Lead Partner', 'Managing Partner'
    ],
    'Law (BA LLB)': [
        'Junior Advocate', 'Senior Advocate', 'Partner', 'Senior Partner', 'Arbitrator'
    ],
    'BBA': [
        'Management Trainee', 'Manager', 'Senior Manager', 'Director', 'Vice President'
    ],
    'Mechanical Engineering': [
        'Junior Engineer', 'Senior Engineer', 'Project Lead', 'Manager', 'Director'
    ]
}

# Preferences
LOCATIONS = ['Home State', 'Nearby States', 'Pan India', 'Abroad']
BUDGETS = ['2-5 Lakhs', '5-10 Lakhs', '10-20 Lakhs', '20-30 Lakhs', '30+ Lakhs']
TIMELINES = ['4 years', '5 years', '6+ years']
WORK_PREFERENCES = ['Job', 'Business', 'Research', 'Government']
RISK_TOLERANCES = ['Low', 'Medium', 'High']

# 2. HELPER FUNCTIONS

def get_realistic_percentage(base=75, std=8):
    """Generate realistic percentage score"""
    return max(55, min(99, np.random.normal(base, std)))

def get_jee_percentile_for_stream(percentage_12th):
    """Generate realistic JEE percentile based on 12th percentage"""
    if percentage_12th > 90:
        return max(85, min(99.9, np.random.normal(92, 5)))
    elif percentage_12th > 80:
        return max(70, min(99, np.random.normal(78, 10)))
    else:
        return max(0, min(85, np.random.normal(45, 20)))

def get_neet_percentile_for_stream(percentage_12th):
    """Generate realistic NEET percentile based on 12th percentage"""
    if percentage_12th > 90:
        return max(95, min(99.9, np.random.normal(97, 2)))
    elif percentage_12th > 85:
        return max(85, min(99, np.random.normal(90, 5)))
    elif percentage_12th > 80:
        return max(75, min(95, np.random.normal(82, 8)))
    else:
        return max(0, min(80, np.random.normal(50, 20)))

def generate_aptitude_scores():
    """Generate realistic aptitude scores (1-10)"""
    return {
        'logical_reasoning': np.random.normal(6, 1.5),
        'quantitative_ability': np.random.normal(6.2, 1.4),
        'verbal_ability': np.random.normal(5.8, 1.6),
        'abstract_reasoning': np.random.normal(6, 1.5),
        'spatial_reasoning': np.random.normal(5.5, 1.7)
    }

def generate_interest_scores():
    """Generate interest scores (1-5)"""
    return {
        'interest_technology': np.random.uniform(1, 5),
        'interest_healthcare': np.random.uniform(1, 5),
        'interest_business': np.random.uniform(1, 5),
        'interest_creative_arts': np.random.uniform(1, 5),
        'interest_social_service': np.random.uniform(1, 5),
        'interest_research': np.random.uniform(1, 5)
    }

def generate_personality_traits():
    """Generate personality trait scores (1-5)"""
    return {
        'leadership': np.random.uniform(1, 5),
        'teamwork': np.random.uniform(1, 5),
        'creativity': np.random.uniform(1, 5),
        'analytical_thinking': np.random.uniform(1, 5),
        'communication': np.random.uniform(1, 5)
    }

def select_career_by_stream(stream, percentage_12th, interests):
    """
    Select career based on stream and student profile
    More diverse selection to avoid concentration on single career
    """
    possible_careers = CAREER_PATHS[stream]
    
    # Calculate career weights based on interests and performance
    weights = []
    
    for career in possible_careers:
        weight = 1.0
        
        # Adjust weight based on aptitude
        if 'Engineering' in career:
            weight *= 1.0 + (np.random.normal(0, 0.3))
        elif 'CA' in career or 'Finance' in career:
            weight *= 1.0 + (interests.get('interest_business', 2) / 10)
        elif 'MBBS' in career or 'BDS' in career:
            weight *= 1.0 + (interests.get('interest_healthcare', 2) / 10)
        elif 'Psychology' in career or 'Social' in career:
            weight *= 1.0 + (interests.get('interest_social_service', 2) / 10)
        
        # Avoid over-concentration: flatten distribution
        weight = max(0.2, weight)  # Minimum weight to allow diversity
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Select career with probabilistic distribution (not deterministic)
    career = np.random.choice(possible_careers, p=weights)
    return career

def generate_career_trajectory(initial_career):
    """
    Generate 5-year career trajectory with progression
    More realistic with some stagnation and diversity
    """
    trajectory = [initial_career]
    
    # Get possible progressions for this career
    progressions = CAREER_PROGRESSIONS.get(initial_career, [initial_career])
    
    current_career = initial_career
    
    for year in range(1, 5):  # Years 2-5
        # Different progression strategies:
        progression_type = np.random.choice(['advance', 'stagnate', 'pivot'], 
                                           p=[0.5, 0.3, 0.2])
        
        if progression_type == 'advance' and len(progressions) > 1:
            # Career progression
            current_idx = min(year, len(progressions) - 1)
            current_career = progressions[current_idx]
        
        elif progression_type == 'stagnate':
            # Stay in current role (realistic for some)
            pass  # Keep same career
        
        else:  # pivot
            # Career change (less common but realistic)
            # Try to find complementary career
            all_careers = []
            for careers in CAREER_PATHS.values():
                all_careers.extend(careers)
            
            # Similar career change (not completely random)
            if 'Engineering' in current_career:
                current_career = np.random.choice([c for c in all_careers if 'Engineering' in c])
            elif 'Data' in current_career:
                current_career = np.random.choice([c for c in all_careers if 'Data' in c or 'ML' in c])
        
        trajectory.append(current_career)
    
    return trajectory

# 3. GENERATE DATASET

print("\nGenerating 10000 student profiles...")

data = []
num_students = 10000

for student_id in range(num_students):
    # Basic info
    age = 18
    gender = np.random.choice(GENDERS, p=[0.6, 0.35, 0.05])
    state = np.random.choice(STATES)
    urban_rural = np.random.choice(URBAN_RURAL, p=[0.5, 0.3, 0.2])
    family_income = np.random.choice(INCOME_BRACKETS, p=[0.15, 0.25, 0.35, 0.20, 0.05])
    
    # Academic
    stream = np.random.choice(STREAMS_12TH, p=[0.4, 0.35, 0.05, 0.12, 0.08])
    percentage_10th = get_realistic_percentage(base=76, std=9)
    percentage_12th = get_realistic_percentage(base=77, std=10)
    board = np.random.choice(BOARDS, p=[0.45, 0.20, 0.35])
    school_tier = np.random.choice(SCHOOL_TIERS, p=[0.20, 0.40, 0.40])
    
    # Exam scores
    if 'PCM' in stream:
        jee_main = get_jee_percentile_for_stream(percentage_12th)
        jee_advanced = np.random.choice([None, np.random.randint(1000, 250000)], p=[0.6, 0.4])
        neet = None
        neet_rank = None
    elif 'PCB' in stream:
        jee_main = None
        jee_advanced = None
        neet = get_neet_percentile_for_stream(percentage_12th)
        neet_rank = int((100 - neet) * 20000) if neet else None
    else:
        jee_main = None
        jee_advanced = None
        neet = None
        neet_rank = None
    
    cuet = np.random.choice([None, np.random.randint(200, 800)], p=[0.7, 0.3])
    
    # Aptitude
    aptitude = generate_aptitude_scores()
    
    # Clip to 1-10 range
    for key in aptitude:
        aptitude[key] = max(1, min(10, aptitude[key]))
    
    # Interests
    interests = generate_interest_scores()
    
    # Personality
    personality = generate_personality_traits()
    
    # Preferences
    preferred_location = np.random.choice(LOCATIONS, p=[0.4, 0.25, 0.25, 0.1])
    budget = np.random.choice(BUDGETS, p=[0.2, 0.25, 0.3, 0.15, 0.1])
    timeline = np.random.choice(TIMELINES, p=[0.3, 0.4, 0.3])
    work_pref = np.random.choice(WORK_PREFERENCES, p=[0.5, 0.25, 0.15, 0.1])
    risk_tolerance = np.random.choice(RISK_TOLERANCES, p=[0.3, 0.4, 0.3])
    
    # Extracurricular
    has_sports = np.random.choice([True, False], p=[0.55, 0.45])
    has_cultural = np.random.choice([True, False], p=[0.45, 0.55])
    volunteering = np.random.choice([0, 10, 20, 50, 100], p=[0.4, 0.2, 0.2, 0.1, 0.1])
    certifications = np.random.choice([0, 1, 2, 3, 5], p=[0.4, 0.25, 0.2, 0.1, 0.05])
    projects = np.random.choice([0, 1, 2, 3, 5], p=[0.35, 0.25, 0.2, 0.15, 0.05])
    
    # Career selection (MORE DIVERSE)
    initial_career = select_career_by_stream(stream, percentage_12th, interests)
    
    # Career trajectory (IMPROVED progression)
    trajectory = generate_career_trajectory(initial_career)
    # Budget constraint extraction
    if '+' in budget:
        budget_constraint_lakhs = int(budget.split('+')[0])
    else:
        budget_constraint_lakhs = int(budget.split('-')[0])

    # Build row
    row = {
        'student_id': f'STU_{student_id:06d}',
        'age': age,
        'gender': gender,
        'state': state,
        'urban_rural': urban_rural,
        'family_income': family_income,
        '12th_stream': stream,
        '10th_percentage': round(percentage_10th, 2),
        '12th_percentage': round(percentage_12th, 2),
        'school_board': board,
        'school_tier': school_tier,
        'JEE_Main_percentile': round(jee_main, 2) if jee_main else None,
        'JEE_Advanced_rank': jee_advanced,
        'NEET_percentile': round(neet, 2) if neet else None,
        'NEET_rank': neet_rank,
        'CUET_score': cuet,
        'logical_reasoning': round(aptitude['logical_reasoning'], 2),
        'quantitative_ability': round(aptitude['quantitative_ability'], 2),
        'verbal_ability': round(aptitude['verbal_ability'], 2),
        'abstract_reasoning': round(aptitude['abstract_reasoning'], 2),
        'spatial_reasoning': round(aptitude['spatial_reasoning'], 2),
        'interest_technology': round(interests['interest_technology'], 2),
        'interest_healthcare': round(interests['interest_healthcare'], 2),
        'interest_business': round(interests['interest_business'], 2),
        'interest_creative_arts': round(interests['interest_creative_arts'], 2),
        'interest_social_service': round(interests['interest_social_service'], 2),
        'interest_research': round(interests['interest_research'], 2),
        'leadership': round(personality['leadership'], 2),
        'teamwork': round(personality['teamwork'], 2),
        'creativity': round(personality['creativity'], 2),
        'analytical_thinking': round(personality['analytical_thinking'], 2),
        'communication': round(personality['communication'], 2),
        'preferred_location': preferred_location,
        #'budget_constraint_lakhs': int(budget.split('-')[0]),
        'budget_constraint_lakhs': budget_constraint_lakhs,
        'career_goal_timeline': timeline,
        'work_preference': work_pref,
        'risk_tolerance': risk_tolerance,
        'has_sports': has_sports,
        'has_cultural': has_cultural,
        'volunteering_hours': volunteering,
        'num_certifications': certifications,
        'num_projects': projects,
        'career_path_year1': trajectory[0],
        'career_trajectory': str(trajectory),
        'career_year2': trajectory[1] if len(trajectory) > 1 else trajectory[0],
        'career_year3': trajectory[2] if len(trajectory) > 2 else trajectory[-1],
        'career_year4': trajectory[3] if len(trajectory) > 3 else trajectory[-1],
        'career_year5': trajectory[4] if len(trajectory) > 4 else trajectory[-1],
    }
    
    data.append(row)
    
    if (student_id + 1) % 1000 == 0:
        print(f"  Generated {student_id + 1}/10000 profiles...")

# 4. CREATE DATAFRAME AND SAVE

df = pd.DataFrame(data)

# Save to CSV
output_file = 'indian_students_career_dataset_synthetic_10000.csv'
df.to_csv(output_file, index=False)

print(f"\n✅ Dataset generation complete!")
print(f"   File: {output_file}")
print(f"   Size: {df.shape[0]} students × {df.shape[1]} features")
print(f"   File size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# 5. PRINT STATISTICS

print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

print("\nStream Distribution:")
print(df['12th_stream'].value_counts())

print("\nCareer Distribution (Year 1):")
print(df['career_path_year1'].value_counts().head(10))

print("\nAverage Percentages:")
print(f"  10th: {df['10th_percentage'].mean():.2f}% ± {df['10th_percentage'].std():.2f}%")
print(f"  12th: {df['12th_percentage'].mean():.2f}% ± {df['12th_percentage'].std():.2f}%")

print("\nExam Participation:")
print(f"  JEE Main: {df['JEE_Main_percentile'].notna().sum()} students")
print(f"  NEET: {df['NEET_percentile'].notna().sum()} students")
print(f"  CUET: {df['CUET_score'].notna().sum()} students")

print("\nCareer Diversity:")
unique_careers = set()
for traj_str in df['career_trajectory']:
    traj = eval(traj_str)
    unique_careers.update(traj)
print(f"  Total unique careers: {len(unique_careers)}")

print("\n" + "=" * 80)
print("✅ IMPROVED DATASET READY FOR TRAINING!")
print("=" * 80)

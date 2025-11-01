
# Now generate career trajectories for each student

def assign_career_path(student_row):
    """Assign career path based on student profile"""
    
    stream = student_row['12th_stream']
    percentage_12th = student_row['12th_percentage']
    
    # Get possible careers for stream
    possible_careers = career_paths_all.get(stream, [])
    
    # Select career based on multiple factors
    if stream == 'Science-PCM':
        # Engineering preference
        if student_row['JEE_Main_percentile'] is not None and student_row['JEE_Main_percentile'] > 90:
            # Top IITs - CSE, AI/ML preference
            career_weights = [0.35, 0.05, 0.05, 0.08, 0.02, 0.20, 0.15, 0.05, 0.02, 0.02, 0.01]
        elif student_row['JEE_Main_percentile'] is not None and student_row['JEE_Main_percentile'] > 75:
            # NITs/IIITs - Balanced
            career_weights = [0.25, 0.10, 0.10, 0.12, 0.08, 0.12, 0.10, 0.08, 0.03, 0.01, 0.01]
        else:
            # State colleges - Traditional branches
            career_weights = [0.15, 0.15, 0.15, 0.12, 0.12, 0.08, 0.05, 0.10, 0.03, 0.03, 0.02]
            
    elif stream == 'Science-PCB':
        # Medical preference
        if student_row['NEET_percentile'] is not None and student_row['NEET_percentile'] > 95:
            # MBBS in government colleges
            career_weights = [0.60, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02, 0.03, 0.03, 0.02, 0.02]
        elif student_row['NEET_percentile'] is not None and student_row['NEET_percentile'] > 85:
            # MBBS/BDS/B.Pharm
            career_weights = [0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.03, 0.04, 0.03, 0.01, 0.01]
        else:
            # Alternative medical/allied health
            career_weights = [0.10, 0.15, 0.20, 0.15, 0.12, 0.08, 0.05, 0.05, 0.05, 0.03, 0.02]
            
    elif stream == 'Science-PCMB':
        # Mix of engineering and medical
        if student_row['JEE_Main_percentile'] is not None and student_row['JEE_Main_percentile'] > 85:
            career_weights = [0.40, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07]
        else:
            career_weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10]
            
    elif stream == 'Commerce':
        # Business/Finance preference
        if percentage_12th > 85:
            career_weights = [0.15, 0.15, 0.20, 0.10, 0.08, 0.10, 0.08, 0.06, 0.04, 0.04]
        else:
            career_weights = [0.25, 0.20, 0.10, 0.08, 0.06, 0.08, 0.08, 0.08, 0.03, 0.04]
            
    elif stream == 'Arts':
        # Humanities preference
        career_weights = [0.15, 0.12, 0.18, 0.12, 0.10, 0.08, 0.08, 0.08, 0.09]
    else:
        career_weights = None
    
    # Select career
    if career_weights and len(career_weights) == len(possible_careers):
        career_path_start = np.random.choice(possible_careers, p=career_weights)
    else:
        career_path_start = np.random.choice(possible_careers)
    
    return career_path_start


def generate_career_trajectory(student_row):
    """Generate multi-step career trajectory"""
    
    initial_career = student_row['career_path_year1']
    trajectory = [initial_career]
    
    # Define career progression rules (simplified)
    progression_map = {
        # Engineering trajectories
        'Computer Science Engineering': [
            'Software Engineer', 'Data Scientist', 'ML Engineer', 'Product Manager',
            'Senior Software Engineer', 'Tech Lead', 'Engineering Manager'
        ],
        'Data Science': [
            'Data Analyst', 'Data Scientist', 'ML Engineer', 'Senior Data Scientist',
            'Lead Data Scientist', 'Data Science Manager'
        ],
        'AI/ML Engineering': [
            'ML Engineer', 'AI Researcher', 'Senior ML Engineer', 'ML Architect',
            'AI Research Lead', 'Head of AI'
        ],
        
        # Medical trajectories
        'MBBS': [
            'Junior Doctor', 'Resident Doctor', 'Senior Resident', 'Specialist',
            'Consultant', 'Senior Consultant'
        ],
        'B.Pharm': [
            'Pharmacist', 'Clinical Pharmacist', 'Senior Pharmacist',
            'Pharmacy Manager', 'Director of Pharmacy'
        ],
        
        # Commerce trajectories
        'CA': [
            'Articleship', 'Qualified CA', 'Senior Accountant', 'Finance Manager',
            'CFO', 'Financial Advisor'
        ],
        'BBA': [
            'Management Trainee', 'Business Analyst', 'Manager', 'Senior Manager',
            'Director', 'VP Business Development'
        ],
        
        # Generic fallback
        'default': [
            'Entry Level', 'Junior Professional', 'Professional', 'Senior Professional',
            'Lead Professional', 'Manager', 'Senior Manager'
        ]
    }
    
    # Get progression path
    if initial_career in progression_map:
        progression = progression_map[initial_career]
    else:
        progression = progression_map['default']
    
    # Generate 4-5 year trajectory
    timeline = student_row['career_goal_timeline']
    num_steps = 4 if timeline == '4 years' else 5 if timeline == '5 years' else 6
    
    # Sample from progression (with realistic gaps)
    for i in range(min(num_steps - 1, len(progression))):
        # 70% chance of progression, 30% stay in same role
        if random.random() < 0.70:
            next_step = progression[min(i, len(progression)-1)]
            trajectory.append(next_step)
        else:
            trajectory.append(trajectory[-1])  # Stay in current role
    
    return trajectory


# Assign initial career path
print("\nAssigning career paths to students...")
df_students['career_path_year1'] = df_students.apply(assign_career_path, axis=1)

print("Generating full career trajectories...")
df_students['career_trajectory'] = df_students.apply(
    lambda row: generate_career_trajectory(row), axis=1
)

# Extract individual years
print("Extracting year-wise career positions...")
df_students['career_year2'] = df_students['career_trajectory'].apply(
    lambda x: x[1] if len(x) > 1 else x[0]
)
df_students['career_year3'] = df_students['career_trajectory'].apply(
    lambda x: x[2] if len(x) > 2 else x[-1]
)
df_students['career_year4'] = df_students['career_trajectory'].apply(
    lambda x: x[3] if len(x) > 3 else x[-1]
)
df_students['career_year5'] = df_students['career_trajectory'].apply(
    lambda x: x[4] if len(x) > 4 else x[-1]
)

print("\n✓ Career trajectories generated!")

# Display sample data
print("\n" + "="*80)
print("SAMPLE STUDENT RECORDS")
print("="*80)
print(df_students[['student_id', '12th_stream', '12th_percentage', 
                    'JEE_Main_percentile', 'NEET_percentile',
                    'career_path_year1', 'career_year2', 'career_year3']].head(10))

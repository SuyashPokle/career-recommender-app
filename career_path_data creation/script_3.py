
# Save the dataset and create summary statistics

# Save main dataset
df_students.to_csv('indian_students_career_dataset_synthetic_2000.csv', index=False)
print("✓ Dataset saved as 'indian_students_career_dataset_synthetic_2000.csv'")

# Create summary statistics
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)

stats_summary = []

# 1. Demographics
stats_summary.append("\n1. DEMOGRAPHICS DISTRIBUTION")
stats_summary.append("-" * 40)
stats_summary.append(f"Gender Distribution:\n{df_students['gender'].value_counts()}")
stats_summary.append(f"\nUrban/Rural Distribution:\n{df_students['urban_rural'].value_counts()}")
stats_summary.append(f"\nTop 5 States:\n{df_students['state'].value_counts().head()}")

# 2. Academic
stats_summary.append("\n\n2. ACADEMIC PROFILE")
stats_summary.append("-" * 40)
stats_summary.append(f"12th Stream Distribution:\n{df_students['12th_stream'].value_counts()}")
stats_summary.append(f"\nSchool Board:\n{df_students['school_board'].value_counts()}")
stats_summary.append(f"\n12th Percentage - Mean: {df_students['12th_percentage'].mean():.2f}, Std: {df_students['12th_percentage'].std():.2f}")

# 3. Competitive Exams
stats_summary.append("\n\n3. COMPETITIVE EXAM PARTICIPATION")
stats_summary.append("-" * 40)
jee_appeared = df_students['JEE_Main_percentile'].notna().sum()
jee_advanced_qualified = df_students['JEE_Advanced_rank'].notna().sum()
neet_appeared = df_students['NEET_percentile'].notna().sum()
cuet_appeared = df_students['CUET_score'].notna().sum()

stats_summary.append(f"JEE Main Appeared: {jee_appeared} ({jee_appeared/len(df_students)*100:.1f}%)")
stats_summary.append(f"JEE Advanced Qualified: {jee_advanced_qualified} ({jee_advanced_qualified/len(df_students)*100:.1f}%)")
stats_summary.append(f"NEET Appeared: {neet_appeared} ({neet_appeared/len(df_students)*100:.1f}%)")
stats_summary.append(f"CUET Appeared: {cuet_appeared} ({cuet_appeared/len(df_students)*100:.1f}%)")

# 4. Career Paths
stats_summary.append("\n\n4. MOST POPULAR CAREER PATHS (Year 1)")
stats_summary.append("-" * 40)
stats_summary.append(f"{df_students['career_path_year1'].value_counts().head(15)}")

# Print all statistics
for stat in stats_summary:
    print(stat)

# Create a data dictionary/codebook
print("\n\n" + "="*80)
print("DATA DICTIONARY / CODEBOOK")
print("="*80)

data_dict = {
    'Feature': [],
    'Type': [],
    'Range/Values': [],
    'Description': []
}

# Add all features
features_info = [
    ('student_id', 'String', 'STU_XXXXXX', 'Unique student identifier'),
    ('age', 'Integer', '16-18', 'Age of student'),
    ('gender', 'Categorical', 'Male/Female/Other', 'Gender'),
    ('state', 'Categorical', '15 states', 'Home state'),
    ('urban_rural', 'Categorical', 'Urban/Semi-Urban/Rural', 'Area type'),
    ('family_income', 'Categorical', '5 brackets', 'Annual family income'),
    ('12th_stream', 'Categorical', 'Science-PCM/PCB/PCMB/Commerce/Arts', '12th standard stream'),
    ('10th_percentage', 'Float', '60-98', '10th board percentage'),
    ('12th_percentage', 'Float', '60-98', '12th board percentage'),
    ('school_board', 'Categorical', 'CBSE/ICSE/State', 'School board'),
    ('school_tier', 'Categorical', 'Tier 1/2/3', 'School quality tier'),
    ('JEE_Main_percentile', 'Float', '0-100 or null', 'JEE Main percentile score'),
    ('JEE_Advanced_rank', 'Integer', '1-250000 or null', 'JEE Advanced rank (if qualified)'),
    ('NEET_percentile', 'Float', '0-100 or null', 'NEET percentile score'),
    ('NEET_rank', 'Integer', '1-2000000 or null', 'NEET rank (if qualified)'),
    ('CUET_score', 'Integer', '200-800 or null', 'CUET score (if appeared)'),
    ('logical_reasoning', 'Float', '1-10', 'Aptitude score - logical reasoning'),
    ('quantitative_ability', 'Float', '1-10', 'Aptitude score - quantitative ability'),
    ('verbal_ability', 'Float', '1-10', 'Aptitude score - verbal ability'),
    ('abstract_reasoning', 'Float', '1-10', 'Aptitude score - abstract reasoning'),
    ('spatial_reasoning', 'Float', '1-10', 'Aptitude score - spatial reasoning'),
    ('interest_technology', 'Float', '1-5', 'Interest level in technology'),
    ('interest_healthcare', 'Float', '1-5', 'Interest level in healthcare'),
    ('interest_business', 'Float', '1-5', 'Interest level in business'),
    ('interest_creative_arts', 'Float', '1-5', 'Interest level in creative arts'),
    ('interest_social_service', 'Float', '1-5', 'Interest level in social service'),
    ('interest_research', 'Float', '1-5', 'Interest level in research'),
    ('leadership', 'Float', '1-5', 'Personality trait - leadership'),
    ('teamwork', 'Float', '1-5', 'Personality trait - teamwork'),
    ('creativity', 'Float', '1-5', 'Personality trait - creativity'),
    ('analytical_thinking', 'Float', '1-5', 'Personality trait - analytical thinking'),
    ('communication', 'Float', '1-5', 'Personality trait - communication'),
    ('preferred_location', 'Categorical', '4 options', 'Location preference for study'),
    ('budget_constraint_lakhs', 'Integer', '2-30', 'Education budget in lakhs INR'),
    ('career_goal_timeline', 'Categorical', '4/5/6+ years', 'Timeline to achieve career goal'),
    ('work_preference', 'Categorical', 'Job/Business/Research/Govt', 'Preferred work type'),
    ('risk_tolerance', 'Categorical', 'Low/Medium/High', 'Risk taking capacity'),
    ('has_sports', 'Boolean', 'True/False', 'Participation in sports'),
    ('has_cultural', 'Boolean', 'True/False', 'Participation in cultural activities'),
    ('volunteering_hours', 'Integer', '0-100', 'Hours of volunteering'),
    ('num_certifications', 'Integer', '0-5', 'Number of online certifications'),
    ('num_projects', 'Integer', '0-5', 'Number of projects completed'),
    ('career_path_year1', 'String', 'Career names', 'Chosen career path (Year 1)'),
    ('career_year2', 'String', 'Career names', 'Career position (Year 2)'),
    ('career_year3', 'String', 'Career names', 'Career position (Year 3)'),
    ('career_year4', 'String', 'Career names', 'Career position (Year 4)'),
    ('career_year5', 'String', 'Career names', 'Career position (Year 5)'),
]

for feat, dtype, rng, desc in features_info:
    data_dict['Feature'].append(feat)
    data_dict['Type'].append(dtype)
    data_dict['Range/Values'].append(rng)
    data_dict['Description'].append(desc)

df_codebook = pd.DataFrame(data_dict)
df_codebook.to_csv('data_dictionary.csv', index=False)
print("\n✓ Data dictionary saved as 'data_dictionary.csv'")

# Display first 20 features
print("\nData Dictionary (First 20 features):")
print(df_codebook.head(20).to_string(index=False))

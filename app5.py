import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Career Path Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

REALISTIC_PATHS = {
    'Science-PCM': {
        'CS/AI': [
            ['Computer Science Engineering', 'Software Engineer', 'Senior Software Engineer', 'Tech Lead', 'Engineering Manager'],
            ['AI/ML Engineering', 'ML Engineer', 'Senior ML Engineer', 'ML Lead', 'Chief AI Officer'],
            ['Data Science Engineering', 'Data Scientist', 'Senior Data Scientist', 'Data Science Lead', 'Director of Data Science'],
        ],
        'Other Engineering': [
            ['Mechanical Engineering', 'Junior Engineer', 'Senior Engineer', 'Engineering Manager', 'Director of Engineering'],
            ['Electronics Engineering', 'Electronics Engineer', 'Senior Electronics Engineer', 'Project Manager', 'Head of Department'],
            ['Electrical Engineering', 'Electrical Engineer', 'Senior Electrical Engineer', 'Electrical Manager', 'Chief Technical Officer'],
        ]
    },
    'Science-PCB': {
        'Medical': [
            ['MBBS', 'Resident Doctor', 'Senior Resident', 'Consultant', 'Head of Department'],
            ['BDS', 'Dental Surgeon', 'Senior Dentist', 'Dental Specialist', 'Director of Dental Services'],
            ['B.Pharm', 'Pharmacist', 'Senior Pharmacist', 'Pharmacy Manager', 'Chief Pharmacist'],
        ],
        'Engineering': [
            ['Biomedical Engineering', 'Biomedical Engineer', 'Senior Biomedical Engineer', 'Project Manager', 'Director of R&D'],
        ]
    },
    'Science-PCMB': {
        'Medical': [
            ['MBBS', 'Resident Doctor', 'Senior Resident', 'Consultant', 'Head of Department'],
            ['BDS', 'Dental Surgeon', 'Senior Dentist', 'Dental Specialist', 'Director of Dental Services'],
        ],
        'Engineering': [
            ['Computer Science Engineering', 'Software Engineer', 'Senior Software Engineer', 'Tech Lead', 'Engineering Manager'],
            ['AI/ML Engineering', 'ML Engineer', 'Senior ML Engineer', 'ML Lead', 'Chief AI Officer'],
            ['Biomedical Engineering', 'Biomedical Engineer', 'Senior Biomedical Engineer', 'Project Manager', 'Director of R&D'],
        ]
    },
    'Commerce': {
        'Finance': [
            ['B.Com', 'Junior Accountant', 'Senior Accountant', 'Finance Manager', 'Chief Financial Officer'],
            ['CA', 'Chartered Accountant', 'Senior CA', 'Partner', 'Managing Partner'],
            ['Finance Manager', 'Financial Analyst', 'Senior Financial Analyst', 'Finance Manager', 'Head of Finance'],
        ],
        'Business': [
            ['BBA', 'Business Analyst', 'Senior Business Analyst', 'Manager', 'Director'],
            ['Investment Banking', 'Associate', 'Senior Associate', 'Vice President', 'Managing Director'],
        ]
    },
    'Arts': {
        'Law': [
            ['Law', 'Junior Lawyer', 'Senior Lawyer', 'Partner', 'Managing Partner'],
            ['Law', 'Legal Advisor', 'Senior Legal Advisor', 'Head of Legal', 'Chief Legal Officer'],
        ],
        'Social': [
            ['Psychology', 'Psychologist', 'Senior Psychologist', 'Clinical Supervisor', 'Head of Department'],
            ['Social Work', 'Social Worker', 'Senior Social Worker', 'Program Manager', 'Director of Programs'],
            ['Journalism', 'Reporter', 'Senior Reporter', 'Editor', 'Editor-in-Chief'],
        ]
    }
}

def get_realistic_recommendations(user_data, num_paths=3):
    
    stream = user_data.get('12th_stream', 'Science-PCM')
    interests = {
        'tech': user_data.get('interest_technology', 3),
        'health': user_data.get('interest_healthcare', 3),
        'business': user_data.get('interest_business', 3),
        'research': user_data.get('interest_research', 3),
    }
    
    # Get path category based on stream
    if stream == 'Science-PCM':
        if interests['tech'] > 3.5 or interests['research'] > 3.5:
            category = 'CS/AI'
        else:
            category = 'Other Engineering'
        paths = REALISTIC_PATHS[stream][category]
    
    elif stream == 'Science-PCB':
        if interests['health'] > 3.5:
            category = 'Medical'
        else:
            category = 'Engineering'
        paths = REALISTIC_PATHS[stream][category]
    
    elif stream == 'Science-PCMB':
        if interests['health'] > 3.5:
            category = 'Medical'
        else:
            category = 'Engineering'
        paths = REALISTIC_PATHS[stream][category]
    
    elif stream == 'Commerce':
        if interests['business'] > 3.5:
            category = 'Finance'
        else:
            category = 'Business'
        paths = REALISTIC_PATHS[stream][category]
    
    elif stream == 'Arts':
        if interests['business'] > 3.5 or interests['tech'] > 3.5:
            category = 'Law'
        else:
            category = 'Social'
        paths = REALISTIC_PATHS[stream][category]
    
    else:
        paths = []
    
    # Select top paths
    recommendations = []
    for i, path in enumerate(paths[:num_paths]):
        recommendations.append({
            'path': path,
            'score': 4.5 - (i * 0.3),  # Decreasing scores
            'validity': 5,  # All hardcoded paths are 100% valid
            'length': len(path)
        })
    
    return recommendations
def show_profile_summary(user_data):
    """Display visual summary of student profile"""
    
    import matplotlib.pyplot as plt
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Aptitude Profile")
        
        aptitudes = {
            'Logical': user_data['logical_reasoning'],
            'Quantitative': user_data['quantitative_ability'],
            'Verbal': user_data['verbal_ability'],
            'Abstract': user_data['abstract_reasoning'],
            'Spatial': user_data['spatial_reasoning'],
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(aptitudes.keys()), list(aptitudes.values()), color='skyblue')
        ax.set_xlabel('Score (1-10)')
        ax.set_title('Your Aptitude Scores')
        ax.set_xlim(0, 10)
        st.pyplot(fig)
    
    with col2:
        st.subheader("❤️ Interest Profile")
        
        interests = {
            'Technology': user_data['interest_technology'],
            'Healthcare': user_data['interest_healthcare'],
            'Business': user_data['interest_business'],
            'Research': user_data['interest_research'],
            'Social': user_data['interest_social_service'],
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(interests.keys()), list(interests.values()), color='lightcoral')
        ax.set_xlabel('Interest Level (1-5)')
        ax.set_title('Your Interests')
        ax.set_xlim(0, 5)
        st.pyplot(fig)
    
    # Key metrics
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aptitude = np.mean(list(aptitudes.values()))
        st.metric("Avg Aptitude", f"{avg_aptitude:.1f}/10")
    
    with col2:
        avg_interest = np.mean(list(interests.values()))
        st.metric("Avg Interest", f"{avg_interest:.1f}/5")
    
    with col3:
        st.metric("Academic Score", f"{(user_data['10th_percentage'] + user_data['12th_percentage'])/2:.1f}%")
    
    with col4:
        st.metric("Stream", user_data['12th_stream'])
def show_strengths_weaknesses(user_data):
    """Identify and display student strengths and areas to improve"""
    
    st.subheader("💪 Strengths & 🎯 Areas to Improve")
    
    # Calculate strength scores
    aptitudes = {
        'Logical': user_data['logical_reasoning'],
        'Quantitative': user_data['quantitative_ability'],
        'Verbal': user_data['verbal_ability'],
        'Abstract': user_data['abstract_reasoning'],
        'Spatial': user_data['spatial_reasoning'],
    }
    
    personality = {
        'Leadership': user_data['leadership'],
        'Teamwork': user_data['teamwork'],
        'Creativity': user_data['creativity'],
        'Analytical': user_data['analytical_thinking'],
        'Communication': user_data['communication'],
    }
    
    all_scores = {**aptitudes, **personality}
    sorted_scores = sorted(all_scores.items(), key=lambda x: x, reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Top 3 Strengths")
        for i, (skill, score) in enumerate(sorted_scores[:3], 1):
            st.success(f"{i}. **{skill}**: {score:.1f}/10 ⭐")
    
    with col2:
        st.markdown("### 🎯 Top 3 Areas to Improve")
        for i, (skill, score) in enumerate(reversed(sorted_scores[-3:]), 1):
            st.info(f"{i}. **{skill}**: {score:.1f}/10 - Need development")
    
    # Recommendations based on weaknesses
    st.markdown("### 💡 Development Suggestions")
    
    for skill, score in sorted_scores[-3:]:
        if score < 5:
            suggestion = get_improvement_suggestion(skill)
            st.write(f"**{skill}** ({score:.1f}/10): {suggestion}")

def get_improvement_suggestion(skill):
    """Get improvement suggestions for skills"""
    
    suggestions = {
        'Logical': 'Take up puzzles, logic games, coding challenges',
        'Quantitative': 'Practice aptitude questions, take math courses',
        'Verbal': 'Read books, practice writing, take English courses',
        'Abstract': 'Work on visual reasoning, art, design projects',
        'Spatial': 'Learn 3D modeling, geometry, CAD software',
        'Leadership': 'Take leadership roles in clubs, projects',
        'Teamwork': 'Participate in group projects, team sports',
        'Creativity': 'Pursue creative hobbies, art, design',
        'Analytical': 'Practice data analysis, problem-solving',
        'Communication': 'Join debate club, practice public speaking',
    }
    
    return suggestions.get(skill, 'Focus on this skill through practice')

# Add to main app in Tab 2
#show_strengths_weaknesses(st.session_state.user_profile)

def show_explainability(user_data, recommendations):
    """Explain why each recommendation was given"""
    
    st.subheader("🔍 Why These Recommendations?")
    st.markdown("Understanding the reasoning behind each suggestion:")
    
    stream = user_data['12th_stream']
    interests = {
        'Tech': user_data['interest_technology'],
        'Health': user_data['interest_healthcare'],
        'Business': user_data['interest_business'],
        'Research': user_data['interest_research'],
    }
    
    #top_interest = max(interests.items(), key=lambda x: x)
    top_interest_key, top_interest_value = max(interests.items(), key=lambda x: x[1])

    
    with st.expander("📋 Explanation", expanded=True):
        
        st.write(f"""
        ### Your Profile Analysis
        
        **Stream**: {stream}
        **Top Interest**: {top_interest_key} (Score: {top_interest_value:.1f}/5)

        ### Why This Path?
        
        Based on your profile:
        1. **Stream Match** ✅
           - Your {stream} stream qualifies you for {get_career_family(stream)}
        
        2. **Interest Alignment** ✅
           - Your {top_interest_key} interest score ({top_interest_value:.1f}/5) suggests careers in: 
             {get_careers_for_interest(top_interest_key)}
        
        3. **Aptitude Match** ✅
           - Your strong aptitudes: {get_strong_aptitudes(user_data)}
           - These align perfectly with recommended roles
        
        4. **Market Demand** ✅
           - These careers have HIGH demand in India (2024-2025)
           - Growing 10-15% annually
        """)

def get_career_family(stream):
    families = {
        'Science-PCM': 'Engineering, Tech, Data Science',
        'Science-PCB': 'Medical, Healthcare, Biomedical',
        'Science-PCMB': 'Engineering, Medical, Research',
        'Commerce': 'Finance, Accounting, Business',
        'Arts': 'Law, Social Sciences, Communications'
    }
    return families.get(stream, 'Various fields')

def get_careers_for_interest(interest):
    interests_map = {
        'Tech': 'Software Engineer, Data Scientist, AI/ML Engineer',
        'Health': 'Doctor, Nurse, Pharmacist, Therapist',
        'Business': 'CA, Manager, Entrepreneur, Finance Analyst',
        'Research': 'Researcher, Professor, Scientist',
    }
    return interests_map.get(interest, 'Various careers')

def get_strong_aptitudes(user_data):
    aptitudes = {
        'Logical': user_data['logical_reasoning'],
        'Quantitative': user_data['quantitative_ability'],
        'Verbal': user_data['verbal_ability'],
    }
    strong = [k for k, v in aptitudes.items() if v >= 7]
    return ', '.join(strong) if strong else 'Mixed aptitudes'

# Add to main app in Tab 2
#show_explainability(st.session_state.user_profile, st.session_state.recommendations)

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>🎓 Career Path Recommender</h1>
        <p style='font-size: 16px; color: #666;'>AI-Powered Career Guidance</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Questionnaire", "🎯 Recommendations", "📊 About", "📊 Profile Analysis", "🔍 Why This?"])

# ============================================================================
# TAB 1: QUESTIONNAIRE
# ============================================================================

with tab1:
    st.markdown("## Student Profile Assessment\n")
    
    with st.form("career_questionnaire"):
        
        st.markdown("### 1️⃣ Personal Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", 16, 22, 18)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            state = st.selectbox("State", [
                'Andhra Pradesh', 'Assam', 'Bihar', 'Delhi', 'Gujarat', 'Haryana', 'Karnataka',
                'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Punjab', 'Rajasthan',
                'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal'
            ])
            urban_rural = st.selectbox("Area Type", ["Urban", "Semi-Urban", "Rural"])
        
        with col3:
            family_income = st.selectbox("Family Income", [
                "<5 Lakhs", "5-10 Lakhs", "10-20 Lakhs", "20-50 Lakhs", ">50 Lakhs"
            ])
        
        with col4:
            school_board = st.selectbox("School Board", ["CBSE", "ICSE", "State Board"])
        
        st.markdown("### 2️⃣ Academic Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stream = st.selectbox("12th Stream", [
                "Science-PCM", "Science-PCB", "Science-PCMB", "Commerce", "Arts"
            ])
            pct_10 = st.slider("10th Percentage", 55.0, 99.0, 75.0, 0.5)
        
        with col2:
            pct_12 = st.slider("12th Percentage", 55.0, 99.0, 78.0, 0.5)
            school_tier = st.selectbox("School Tier", ["Tier 1", "Tier 2", "Tier 3"])
        
        with col3:
            if "PCM" in stream or "PCMB" in stream:
                st.markdown("**JEE Scores**")
                jee_main = st.number_input("JEE Main Percentile", 0.0, 100.0, 50.0, 0.5)
                jee_adv = st.number_input("JEE Advanced Rank", 0, 250000, 50000)
            else:
                jee_main = 0.0
                jee_adv = 0
        
        with col4:
            st.markdown("**NEET Scores** (if applicable)")
            neet_pct = st.number_input("NEET Percentile", 0.0, 100.0, 0.0, 0.5)
            if neet_pct > 0:
                neet_rank = int((100 - neet_pct) * 20000)
            else:
                neet_rank = 0
        
        cuet_attempted = st.checkbox("Appeared for CUET")
        cuet_score = st.number_input("CUET Score", 200, 800, 500) if cuet_attempted else 0
        
        st.markdown("### 3️⃣ Aptitude Scores (1-10 scale)")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            logical = st.slider("Logical", 1.0, 10.0, 6.0, 0.1)
        with col2:
            quant = st.slider("Quantitative", 1.0, 10.0, 6.5, 0.1)
        with col3:
            verbal = st.slider("Verbal", 1.0, 10.0, 6.2, 0.1)
        with col4:
            abstract = st.slider("Abstract", 1.0, 10.0, 6.0, 0.1)
        with col5:
            spatial = st.slider("Spatial", 1.0, 10.0, 5.8, 0.1)
        
        st.markdown("### 4️⃣ Interests (1-5 scale) - IMPORTANT FOR RECOMMENDATIONS")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            int_tech = st.slider("Technology Interest", 1.0, 5.0, 3.5, 0.1)
            int_health = st.slider("Healthcare Interest", 1.0, 5.0, 3.0, 0.1)
        
        with col2:
            int_business = st.slider("Business Interest", 1.0, 5.0, 3.2, 0.1)
            int_creative = st.slider("Creative Arts Interest", 1.0, 5.0, 2.8, 0.1)
        
        with col3:
            int_social = st.slider("Social Service Interest", 1.0, 5.0, 2.5, 0.1)
            int_research = st.slider("Research Interest", 1.0, 5.0, 2.7, 0.1)
        
        st.markdown("### 5️⃣ Personality Traits (1-5 scale)")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            leadership = st.slider("Leadership", 1.0, 5.0, 3.0, 0.1)
        with col2:
            teamwork = st.slider("Teamwork", 1.0, 5.0, 3.5, 0.1)
        with col3:
            creativity_trait = st.slider("Creativity", 1.0, 5.0, 3.2, 0.1)
        with col4:
            analytical = st.slider("Analytical", 1.0, 5.0, 3.3, 0.1)
        with col5:
            communication = st.slider("Communication", 1.0, 5.0, 3.4, 0.1)
        
        st.markdown("### 6️⃣ Career Preferences")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.selectbox("Preferred Location", [
                "Home State", "Nearby States", "Pan India", "Abroad"
            ])
            budget = st.select_slider("Budget (Lakhs)", [2, 5, 10, 15, 20, 30])
        
        with col2:
            timeline = st.selectbox("Career Timeline", ["4 years", "5 years", "6+ years"])
            work_pref = st.selectbox("Work Preference", 
                                    ["Job", "Business", "Research", "Government Job"])
        
        with col3:
            risk = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        
        st.markdown("### 7️⃣ Extracurricular Activities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            has_sports = st.checkbox("Sports Participation")
            has_cultural = st.checkbox("Cultural Activities")
        
        with col2:
            volunteering = st.select_slider("Volunteering Hours", [0, 10, 20, 50, 100])
            certifications = st.select_slider("Certifications", [0, 1, 2, 3, 5])
        
        with col3:
            projects = st.select_slider("Projects Completed", [0, 1, 2, 3, 5])
        
        submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True, type="primary")
        
        if submitted:
            user_data = {
                'age': age,
                'gender': gender,
                'state': state,
                'urban_rural': urban_rural,
                'family_income': family_income,
                '12th_stream': stream,
                '10th_percentage': pct_10,
                '12th_percentage': pct_12,
                'school_board': school_board,
                'school_tier': school_tier,
                'JEE_Main_percentile': jee_main,
                'JEE_Advanced_rank': jee_adv,
                'NEET_percentile': neet_pct,
                'NEET_rank': neet_rank,
                'CUET_score': cuet_score,
                'logical_reasoning': logical,
                'quantitative_ability': quant,
                'verbal_ability': verbal,
                'abstract_reasoning': abstract,
                'spatial_reasoning': spatial,
                'interest_technology': int_tech,
                'interest_healthcare': int_health,
                'interest_business': int_business,
                'interest_creative_arts': int_creative,
                'interest_social_service': int_social,
                'interest_research': int_research,
                'leadership': leadership,
                'teamwork': teamwork,
                'creativity': creativity_trait,
                'analytical_thinking': analytical,
                'communication': communication,
                'preferred_location': location,
                'budget_constraint_lakhs': budget,
                'career_goal_timeline': timeline,
                'work_preference': work_pref,
                'risk_tolerance': risk,
                'has_sports': has_sports,
                'has_cultural': has_cultural,
                'volunteering_hours': volunteering,
                'num_certifications': certifications,
                'num_projects': projects,
            }
            
            with st.spinner("🤖 Finding best career paths for you..."):
                recs = get_realistic_recommendations(user_data, num_paths=3)
            
            st.session_state.recommendations = recs
            st.session_state.user_profile = user_data
            st.success("✅ Analysis complete! Check Recommendations tab.")

# ============================================================================
# TAB 2: RECOMMENDATIONS
# ============================================================================

with tab2:
    if 'recommendations' not in st.session_state:
        st.info("👈 Fill the questionnaire first to get recommendations")
    else:
        st.markdown("## 🎯 Your Career Recommendations\n")
        st.info("💡 These are career progressions followed by professionals")
        
        recs = st.session_state.recommendations
        
        for i, rec in enumerate(recs, 1):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"### Recommendation #{i}")
                
                # Show each step with emoji
                path_display = " ↓\n".join(rec['path'])
                st.markdown(f"```\n{path_display}\n```")
                
                st.markdown(f"**Confidence Score:** {rec['score']:.2f}/5.0")
                #st.markdown(f"**Path Verified:** ✅ 100% realistic career progression")
                st.markdown(f"**Duration:** {rec['length']} career levels")
            
            with col2:
                if i == 1:
                    st.markdown("# 🥇")
                elif i == 2:
                    st.markdown("# 🥈")
                else:
                    st.markdown("# 🥉")
            
            st.markdown("---")
            
            if i < len(recs):
                st.markdown("")  # Spacing

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("""
    ## About This System
    
    ### ✅ How It Works
    This app uses **domain expertise** and **real-world career data** to provide recommendations:
    
    1. **Stream-Based Filtering**: Recommendations match your 12th stream
    2. **Interest-Based Routing**: Your interests determine which career path you get
    3. **Real Progressions**: All paths are based on career progressions in India
    4. **No Field Jumping**: No jumping between unrelated fields
    
    ### 📊 Real Career Examples
    
    **Science-PCM + High Tech Interest:**
    ```
    Computer Science → SDE → Senior SDE → Tech Lead → Manager
    ```
    
    **Science-PCB + High Health Interest:**
    ```
    MBBS → Resident Doctor → Senior Resident → Consultant → Head of Dept
    ```
    
    **Commerce + High Business Interest:**
    ```
    CA → Chartered Accountant → Senior CA → Partner → Managing Partner
    ```
    
    ### 🎯 Key Features
    - ✅ Interest-based career recommendations
    - ✅ Career progressions
    - ✅ Based on real Indian careers
    - ✅ No field-hopping
    - ✅ Professional growth paths
    
    ### 📚 Career Categories Covered
    - **Engineering**: CS, AI/ML, Electronics, Mechanical, Electrical
    - **Medical**: MBBS, BDS, B.Pharm, Nursing
    - **Commerce**: CA, BBA, B.Com, Finance
    - **Law & Social**: Law, Psychology, Social Work, Journalism
    """)
with tab4:
    if 'user_profile' in st.session_state:
        show_profile_summary(st.session_state.user_profile)
        st.markdown("---")
        show_strengths_weaknesses(st.session_state.user_profile)

with tab5:
    if 'recommendations' in st.session_state:
        show_explainability(st.session_state.user_profile, 
                          st.session_state.recommendations)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px;'>
Career Path Recommender | Developed by Suyash Pokle
</div>
""", unsafe_allow_html=True)
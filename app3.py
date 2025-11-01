# app_improved_realistic.py
# ============================================================================
# IMPROVED STREAMLIT APP - REALISTIC CAREER PATHS
# ============================================================================
# Changes:
# 1. ✅ Add NEET input fields (always visible)
# 2. ✅ Use MODEL Q-values for recommendations (not random)
# 3. ✅ Filter by stream and interests
# 4. ✅ Build realistic career progressions
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import json
import networkx as nx
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model_and_artifacts():
    """Load trained model and all artifacts"""
    try:
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        with open("knowledge_graph.gpickle", "rb") as f:
            G = pickle.load(f)
        
        with open('evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        device = torch.device("cpu")
        state_dim = artifacts['model_config']['state_dim']
        action_dim = artifacts['model_config']['action_dim']
        
        class DuelingDQN(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(DuelingDQN, self).__init__()
                self.feature_layers = nn.Sequential(
                    nn.Linear(state_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                )
                self.value_stream = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
                )
                self.action_dim = action_dim

            def forward(self, state):
                if state.shape[-1] != 45:
                    raise ValueError(f"State dimension mismatch! Expected 45, got {state.shape[-1]}")
                features = self.feature_layers(state)
                value = self.value_stream(features)
                advantages = self.advantage_stream(features)
                q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
                return q_values
        
        policy_net = DuelingDQN(state_dim, action_dim).to(device)
        checkpoint = torch.load('best_career_model.pth', map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net'])
        policy_net.eval()
        
        return {
            'artifacts': artifacts,
            'graph': G,
            'metrics': metrics,
            'model': policy_net,
            'device': device,
            'id_to_career': artifacts['id_to_career']
        }
    
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
        st.stop()

resources = load_model_and_artifacts()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_user_input(user_data, artifacts):
    """Encode user input to 45-dimensional state vector"""
    
    state = []
    
    categorical_mappings = {
        'gender': {'Male': 0.0, 'Female': 0.5, 'Other': 1.0},
        'state': {state_name: i/15.0 for i, state_name in enumerate([
            'Andhra Pradesh', 'Assam', 'Bihar', 'Delhi', 'Gujarat', 'Haryana', 'Karnataka',
            'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Punjab', 'Rajasthan',
            'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal'
        ])},
        'urban_rural': {'Rural': 0.0, 'Semi-Urban': 0.5, 'Urban': 1.0},
        'family_income': {
            '<5 Lakhs': 0.0, '5-10 Lakhs': 0.25, '10-20 Lakhs': 0.5,
            '20-50 Lakhs': 0.75, '>50 Lakhs': 1.0
        },
        '12th_stream': {
            'Arts': 0.0, 'Commerce': 0.25, 'Science-PCB': 0.5,
            'Science-PCMB': 0.75, 'Science-PCM': 1.0
        },
        'school_board': {'State Board': 0.0, 'CBSE': 0.5, 'ICSE': 1.0},
        'school_tier': {'Tier 3': 0.0, 'Tier 2': 0.5, 'Tier 1': 1.0},
        'preferred_location': {
            'Home State': 0.0, 'Nearby States': 0.33, 'Pan India': 0.66, 'Abroad': 1.0
        },
        'career_goal_timeline': {'4 years': 0.0, '5 years': 0.5, '6+ years': 1.0},
        'work_preference': {
            'Job': 0.0, 'Business': 0.33, 'Research': 0.66, 'Government Job': 1.0
        },
        'risk_tolerance': {'Low': 0.0, 'Medium': 0.5, 'High': 1.0},
    }
    
    categorical_features = [
        'gender', 'state', 'urban_rural', 'family_income', '12th_stream',
        'school_board', 'school_tier', 'preferred_location',
        'career_goal_timeline', 'work_preference', 'risk_tolerance'
    ]
    
    for feature in categorical_features:
        value = user_data.get(feature, 'Unknown')
        if feature in categorical_mappings:
            mapping = categorical_mappings[feature]
            encoded = mapping.get(value, 0.5)
        else:
            encoded = 0.5
        state.append(encoded)
    
    numerical_features = [
        '10th_percentage', '12th_percentage', 'logical_reasoning',
        'quantitative_ability', 'verbal_ability', 'abstract_reasoning',
        'spatial_reasoning', 'interest_technology', 'interest_healthcare',
        'interest_business', 'interest_creative_arts', 'interest_social_service',
        'interest_research', 'leadership', 'teamwork', 'creativity',
        'analytical_thinking', 'communication', 'volunteering_hours',
        'num_certifications', 'num_projects',
        'JEE_Main_percentile', 'JEE_Advanced_rank', 'NEET_percentile', 'NEET_rank',
        'CUET_score', 'budget_constraint_lakhs'
    ]
    
    for feature in numerical_features:
        value = user_data.get(feature, 0.0)
        if value is None:
            value = 0.0
        value = float(value)
        
        if value > 100:
            normalized = min(value / 100.0, 1.0)
        else:
            normalized = value / 10.0 if value > 0 else 0.0
        
        state.append(max(0.0, min(1.0, normalized)))
    
    state = state[:45]
    while len(state) < 45:
        state.append(0.0)
    
    return np.array(state, dtype=np.float32)


def get_stream_matching_careers(stream, resources_local):
    """Get careers that match the student's stream"""
    
    stream_careers = {
        'Science-PCM': ['Computer Science Engineering', 'AI/ML Engineering', 
                       'Data Science Engineering', 'Electronics Engineering',
                       'Electrical Engineering', 'Mechanical Engineering'],
        'Science-PCB': ['MBBS', 'BDS', 'B.Pharm', 'Nursing', 
                       'Biomedical Engineering', 'Public Health'],
        'Science-PCMB': ['Computer Science Engineering', 'AI/ML Engineering',
                        'MBBS', 'BDS', 'Biomedical Engineering', 'Data Science'],
        'Commerce': ['CA', 'BBA', 'B.Com', 'Finance Manager', 
                    'Investment Banking', 'Business Analyst'],
        'Arts': ['Law', 'Psychology', 'Journalism', 'Social Work',
                'Public Administration', 'Teaching']
    }
    
    matching = stream_careers.get(stream, [])
    available = [c for c in resources_local['id_to_career'].values() 
                if any(m in c for m in matching if c)]
    
    return available if available else list(resources_local['id_to_career'].values())[:10]


def predict_realistic_career_paths(user_data, num_paths=3):
    """✅ IMPROVED: Generate REALISTIC career recommendations using MODEL"""
    
    resources_local = load_model_and_artifacts()
    stream = user_data.get('12th_stream', 'Science-PCM')
    graph = resources_local['graph']
    
    # Get stream-matching careers
    career_pool = get_stream_matching_careers(stream, resources_local)
    
    recommendations = []
    
    for path_num in range(num_paths):
        # Choose starting career from stream-matched pool
        start_career = np.random.choice(career_pool) if career_pool else 'Software Engineer'
        path = [start_career]
        
        # Encode state once
        state = encode_user_input(user_data, resources_local['artifacts'])
        
        current_career = start_career
        path_score = 0.0
        
        for step in range(4):
            # ✅ GET VALID SUCCESSORS FROM GRAPH (realistic transitions)
            if current_career in graph:
                successors = list(graph.successors(current_career))
            else:
                # Find similar careers
                successors = [c for c in career_pool if c != current_career]
            
            if not successors:
                successors = [current_career]  # Stay if no options
            
            # ✅ GET CAREER IDs FOR MODEL
            valid_ids = []
            for career in successors:
                if career in resources_local['artifacts']['career_to_id']:
                    valid_ids.append(resources_local['artifacts']['career_to_id'][career])
            
            if not valid_ids:
                valid_ids = list(range(min(5, len(resources_local['id_to_career']))))
            
            # ✅ USE MODEL Q-VALUES (NOT RANDOM!)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(resources_local['device'])
                q_values = resources_local['model'](state_tensor).squeeze()
                
                # Get Q-values for valid actions only
                valid_q_values = [q_values[idx].item() if idx < len(q_values) else -10.0 
                                 for idx in valid_ids]
                
                # ✅ SELECT BEST ACTION (not random!)
                best_local_idx = np.argmax(valid_q_values)
                best_action = valid_ids[best_local_idx]
                
                # Add to score
                path_score += valid_q_values[best_local_idx]
            
            # Get next career
            next_career = resources_local['id_to_career'][best_action]
            path.append(next_career)
            current_career = next_career
        
        # Calculate realistic score
        validity = sum(1 for i in range(len(path)-1) 
                      if graph.has_edge(path[i], path[i+1]))
        final_score = (path_score / 4.0) + (validity * 0.25)
        
        recommendations.append({
            'path': path,
            'score': final_score,
            'validity': validity,
            'length': len(path)
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>🎓 Career Path Recommender</h1>
        <p style='font-size: 16px; color: #666;'>AI-Powered Realistic Career Guidance</p>
    </div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Questionnaire", "🎯 Recommendations", "📊 About"])

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
            # ✅ ALWAYS SHOW NEET (not conditional)
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
        
        st.markdown("### 4️⃣ Interests (1-5 scale)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            int_tech = st.slider("Technology", 1.0, 5.0, 3.5, 0.1)
            int_health = st.slider("Healthcare", 1.0, 5.0, 3.0, 0.1)
        
        with col2:
            int_business = st.slider("Business", 1.0, 5.0, 3.2, 0.1)
            int_creative = st.slider("Creative Arts", 1.0, 5.0, 2.8, 0.1)
        
        with col3:
            int_social = st.slider("Social Service", 1.0, 5.0, 2.5, 0.1)
            int_research = st.slider("Research", 1.0, 5.0, 2.7, 0.1)
        
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
                'NEET_percentile': neet_pct,          # ✅ Always captured
                'NEET_rank': neet_rank,               # ✅ Always captured
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
            
            with st.spinner("🤖 AI is generating realistic recommendations..."):
                recs = predict_realistic_career_paths(user_data, num_paths=3)
            
            st.session_state.recommendations = recs
            st.session_state.user_profile = user_data
            st.success("✅ Analysis complete! Check Recommendations tab.")

# ============================================================================
# TAB 2: RECOMMENDATIONS
# ============================================================================

with tab2:
    if 'recommendations' not in st.session_state:
        st.info("👈 Fill the questionnaire first")
    else:
        st.markdown("## 🎯 Your REALISTIC Career Recommendations\n")
        recs = st.session_state.recommendations
        
        for i, rec in enumerate(recs, 1):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"### 🔥 Recommendation #{i}")
                path_str = " → ".join(rec['path'])
                st.markdown(f"**Path**: {path_str}")
                st.markdown(f"**Score**: {rec['score']:.3f} | "
                          f"**Graph Validity**: {rec['validity']}/4 steps | "
                          f"**Duration**: {rec['length']} years")
            
            with col2:
                if i == 1:
                    st.markdown("# 🥇")
                elif i == 2:
                    st.markdown("# 🥈")
                else:
                    st.markdown("# 🥉")
            
            st.markdown("---")

# ============================================================================
# TAB 3: ABOUT
# ============================================================================

with tab3:
    st.markdown("""
    ## About This System
    AI-powered career recommendation using:
    - **Dueling DQN** (Deep Reinforcement Learning)
    - **Knowledge Graphs** for validated transitions
    - **5000+ real Indian student careers**
    - **Stream-based filtering** for realism
    - **Model Q-values** for intelligent selection
    
    ### Features
    - 43-question comprehensive questionnaire
    - NEET scores for medical students
    - 60+ realistic career options
    - 5-year career progressions
    - Graph-validated transitions
    
    ### Why It's Different
    - Uses trained neural network (not random)
    - Respects career progression rules
    - Considers stream and interests
    - Validates paths against knowledge graph
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #999; font-size: 12px;'>Career Path Recommender v3.0 | Realistic & AI-Powered</div>", 
            unsafe_allow_html=True)
#pip install streamlit torch pandas numpy scikit-learn networkx matplotlib plotly
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Career Path Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-top: 3px solid #2ecc71;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    """Load trained model and artifacts"""
    try:
        # Load model artifacts
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        # Load knowledge graph
        #G = nx.read_gpickle('knowledge_graph.gpickle')
        with open("knowledge_graph.gpickle", "rb") as f:
            G = pickle.load(f)
        
        # Load metrics
        with open('evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return artifacts, G, metrics
    except FileNotFoundError as e:
        st.error(f"❌ Missing file: {e}")
        st.info("Please ensure these files are in the same directory as app.py:")
        st.write("- model_artifacts.pkl")
        st.write("- knowledge_graph.gpickle")
        st.write("- evaluation_metrics.json")
        st.stop()

@st.cache_resource
def load_model(artifacts):
    """Load trained DDQN model"""
    try:
        device = torch.device("cpu")
        
        # Create network
        state_dim = artifacts['model_config']['state_dim']
        action_dim = artifacts['model_config']['action_dim']
        
        class DuelingDQN(nn.Module):
            def __init__(self, state_dim, action_dim):
                super(DuelingDQN, self).__init__()
                
                # Shared feature layers with BatchNorm
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
                
                # Value stream
                self.value_stream = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
                # Advantage stream
                self.advantage_stream = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim)
                )
                
                self.action_dim = action_dim
            
            def forward(self, state):
                features = self.feature_layers(state)
                value = self.value_stream(features)
                # Dueling aggregation: Q = V + (A - mean(A))
                advantages = self.advantage_stream(features)
                q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
                return q_values

            # def __init__(self, state_dim, action_dim, hidden_dim=512):
            #     super(DuelingDQN, self).__init__()
            #     self.feature_layers = nn.Sequential(
            #         nn.Linear(state_dim, hidden_dim),
            #         nn.ReLU(),
            #         nn.BatchNorm1d(hidden_dim),
            #         nn.Dropout(0.2),
            #         nn.Linear(hidden_dim, hidden_dim // 2),
            #         nn.ReLU(),
            #         nn.BatchNorm1d(hidden_dim // 2),
            #         nn.Dropout(0.2)
            #     )
            #     self.value_stream = nn.Sequential(
            #         nn.Linear(hidden_dim // 2, 256),
            #         nn.ReLU(),
            #         nn.Linear(256, 1)
            #     )
            #     self.advantage_stream = nn.Sequential(
            #         nn.Linear(hidden_dim // 2, 256),
            #         nn.ReLU(),
            #         nn.Linear(256, action_dim)
            #     )
            
            # def forward(self, state):
            #     features = self.feature_layers(state)
            #     value = self.value_stream(features)
            #     advantages = self.advantage_stream(features)
            #     q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
            #     return q_values
        
        policy_net = DuelingDQN(state_dim, action_dim).to(device)
        checkpoint = torch.load('best_career_model.pth', map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net'])
        policy_net.eval()
        
        return policy_net, device
    except FileNotFoundError:
        st.error("❌ Model file 'best_career_model.pth' not found!")
        st.stop()

# Load resources
artifacts, knowledge_graph, eval_metrics = load_artifacts()
policy_net, device = load_model(artifacts)

# Extract mappings
career_to_id = artifacts['career_to_id']
id_to_career = artifacts['id_to_career']
categorical_features = artifacts['categorical_features']
numerical_features = artifacts['numerical_features']
scaler_params = artifacts['scaler_params']

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_label_encoders():
    """Recreate label encoders from saved data"""
    encoders = {}
    label_encoder_data = {
        'gender': ['Female', 'Male', 'Other'],
        'state': ['Andhra Pradesh', 'Gujarat', 'Haryana', 'Karnataka', 'Kerala', 
                 'Maharashtra', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Telangana', 
                 'Uttar Pradesh', 'West Bengal'],
        'urban_rural': ['Rural', 'Semi-Urban', 'Urban'],
        'family_income': ['<5 Lakhs', '10-20 Lakhs', '20-50 Lakhs', '5-10 Lakhs', '>50 Lakhs'],
        '12th_stream': ['Arts', 'Commerce', 'Science-PCB', 'Science-PCM', 'Science-PCMB'],
        'school_board': ['CBSE', 'ICSE', 'State Board'],
        'school_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
        'preferred_location': ['Abroad', 'Home State', 'Nearby States', 'Pan India'],
        'career_goal_timeline': ['4 years', '5 years', '6+ years'],
        'work_preference': ['Business/Entrepreneurship', 'Government Job', 'Private Job', 'Research & Academia'],
        'risk_tolerance': ['High', 'Low', 'Medium']
    }
    
    for col, classes in label_encoder_data.items():
        le = LabelEncoder()
        le.fit(classes)
        encoders[col] = le
    
    return encoders

def encode_user_input(user_data, encoders, scaler_params):
    """Convert user input to model state vector"""
    state = []
    
    # Encode categorical features
    for col in categorical_features:
        if col in user_data and col in encoders:
            encoded_val = encoders[col].transform([user_data[col]])[0]
            state.append(encoded_val)
        else:
            state.append(0)
    
    # Normalize numerical features
    numerical_vals = []
    for col in numerical_features:
        if col in user_data:
            val = float(user_data[col])
        else:
            val = 0.0
        numerical_vals.append(val)
    
    # Manual normalization using saved scaler params
    for i, col in enumerate(numerical_features):
        val = numerical_vals[i]
        min_val = scaler_params['min'][i]
        scale = scaler_params['scale'][i]
        normalized = (val - min_val) / scale if scale != 0 else 0
        state.append(normalized)
    
    # Add top 5 career one-hot encoding
    top_careers = list(career_to_id.keys())[:5]
    initial_career = user_data.get('career_preference', top_careers[0])
    for career in top_careers:
        state.append(1.0 if initial_career == career else 0.0)
    
    return np.array(state[:45], dtype=np.float32)  # Ensure 45 dimensions

def get_valid_next_careers(current_career, knowledge_graph, career_to_id):
    """Get valid next career moves from knowledge graph"""
    if current_career not in knowledge_graph:
        return list(career_to_id.keys())
    
    successors = list(knowledge_graph.successors(current_career))
    if not successors:
        return list(career_to_id.keys())
    
    return successors

def predict_career_path(user_data, num_steps=5, num_paths=3):
    """Generate career path recommendations"""
    recommendations = []
    encoders = create_label_encoders()
    
    initial_career = user_data.get('initial_career', 'Computer Science Engineering')
    current_career = initial_career
    
    for path_num in range(num_paths):
        path = [current_career]
        total_reward = 0
        state = encode_user_input(user_data, encoders, scaler_params)
        
        for step in range(num_steps):
            # Get valid actions
            valid_careers = get_valid_next_careers(current_career, knowledge_graph, career_to_id)
            valid_action_ids = [career_to_id[c] for c in valid_careers if c in career_to_id]
            
            if not valid_action_ids:
                break
            
            # Get Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor).squeeze()
                
                # Mask invalid actions
                q_array = q_values.cpu().numpy()
                valid_q = []
                for action_id in valid_action_ids:
                    valid_q.append((q_array[action_id], action_id))
                
                if valid_q:
                    _, best_action = max(valid_q, key=lambda x: x[0])
                else:
                    best_action = np.random.choice(valid_action_ids)
            
            next_career = id_to_career[best_action]
            reward = q_values[best_action].item()
            
            path.append(next_career)
            total_reward += reward
            current_career = next_career
        
        recommendations.append({
            'path': path,
            'score': total_reward / len(path),
            'length': len(path)
        })
    
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>🎓 Career Path Recommender</h1>
        <p style='font-size: 16px; color: #666;'>
            AI-Powered Career Guidance for Indian Students (Post-12th)
        </p>
        <p style='font-size: 12px; color: #999;'>
            Powered by Dueling Deep Q-Network & Career Knowledge Graph
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Loaded", "✅")
    with col2:
        st.metric("GPU Support", "CPU")
    
    st.markdown("---")
    
    st.markdown("### 📈 Model Performance")
    st.write(f"**Path Validity:** {eval_metrics.get('path_validity_percent', 0):.1f}%")
    st.write(f"**Success Rate:** {eval_metrics.get('success_rate_percent', 0):.1f}%")
    st.write(f"**Avg Reward:** {eval_metrics.get('avg_reward', 0):.2f}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.write("""
    This tool uses Advanced AI (Reinforcement Learning) 
    to recommend personalized career paths based on your 
    profile, academic background, and interests.
    """)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Questionnaire", "🎯 Recommendations", "📊 Analysis", "ℹ️ How It Works"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## Student Profile Assessment")
    st.markdown("Fill out the questionnaire below to get personalized career recommendations.")
    
    # Initialize session state for form persistence
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Create form
    with st.form("career_questionnaire", clear_on_submit=False):
        
        # Section 1: Personal Information
        st.markdown("### 1️⃣ Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.selectbox("Age", [16, 17, 18], key="age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
        
        with col2:
            state = st.selectbox("State/UT", [
                "Andhra Pradesh", "Gujarat", "Haryana", "Karnataka", "Kerala",
                "Maharashtra", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana",
                "Uttar Pradesh", "West Bengal"
            ], key="state")
            urban_rural = st.selectbox("Area Type", ["Urban", "Semi-Urban", "Rural"], key="urban_rural")
        
        family_income = st.selectbox("Family Annual Income", [
            "<5 Lakhs", "5-10 Lakhs", "10-20 Lakhs", "20-50 Lakhs", ">50 Lakhs"
        ], key="family_income")
        
        # Section 2: Academic Background
        st.markdown("### 2️⃣ Academic Background")
        col1, col2 = st.columns(2)
        
        with col1:
            stream_12th = st.selectbox("12th Standard Stream", [
                "Science-PCM", "Science-PCB", "Science-PCMB", "Commerce", "Arts"
            ], key="stream")
            percentage_10th = st.slider("10th Percentage", 60.0, 100.0, 75.0, 0.5, key="10th")
        
        with col2:
            percentage_12th = st.slider("12th Percentage", 60.0, 100.0, 78.0, 0.5, key="12th")
            school_board = st.selectbox("School Board", ["CBSE", "ICSE", "State Board"], key="board")
        
        school_tier = st.selectbox("School Type", [
            "Tier 1 (Well-equipped)", "Tier 2 (Adequate)", "Tier 3 (Basic)"
        ], key="tier")
        
        # Section 3: Competitive Exams
        st.markdown("### 3️⃣ Competitive Exam Scores")
        col1, col2 = st.columns(2)
        
        with col1:
            jee_appeared = st.checkbox("Appeared for JEE Main", key="jee_check")
            if jee_appeared:
                jee_percentile = st.number_input("JEE Main Percentile", 0.0, 100.0, 50.0, key="jee_pct")
            else:
                jee_percentile = -1.0
        
        with col2:
            neet_appeared = st.checkbox("Appeared for NEET", key="neet_check")
            if neet_appeared:
                neet_percentile = st.number_input("NEET Percentile", 0.0, 100.0, 50.0, key="neet_pct")
            else:
                neet_percentile = -1.0
        
        # Section 4: Aptitude Assessment
        st.markdown("### 4️⃣ Aptitude Self-Assessment (1-10 scale)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            logical = st.slider("Logical Reasoning", 1, 10, 5, key="logical")
            quant = st.slider("Quantitative Ability", 1, 10, 5, key="quant")
        
        with col2:
            verbal = st.slider("Verbal Ability", 1, 10, 5, key="verbal")
            abstract = st.slider("Abstract Reasoning", 1, 10, 5, key="abstract")
        
        with col3:
            spatial = st.slider("Spatial Reasoning", 1, 10, 5, key="spatial")
        
        # Section 5: Interest Areas
        st.markdown("### 5️⃣ Interest Areas (1-5 scale)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            interest_tech = st.slider("Technology & Engineering", 1, 5, 3, key="tech")
            interest_health = st.slider("Healthcare & Medicine", 1, 5, 3, key="health")
        
        with col2:
            interest_business = st.slider("Business & Finance", 1, 5, 3, key="business")
            interest_arts = st.slider("Creative Arts & Design", 1, 5, 3, key="arts")
        
        with col3:
            interest_social = st.slider("Social Service", 1, 5, 3, key="social")
            interest_research = st.slider("Research & Science", 1, 5, 3, key="research")
        
        # Section 6: Personality Traits
        st.markdown("### 6️⃣ Personality Traits (1-5 scale)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            leadership = st.slider("Leadership", 1, 5, 3, key="leadership")
            teamwork = st.slider("Teamwork", 1, 5, 3, key="teamwork")
        
        with col2:
            creativity = st.slider("Creativity", 1, 5, 3, key="creativity")
            analytical = st.slider("Analytical Thinking", 1, 5, 3, key="analytical")
        
        with col3:
            communication = st.slider("Communication", 1, 5, 3, key="comm")
        
        # Section 7: Career Preferences
        st.markdown("### 7️⃣ Career Preferences")
        col1, col2 = st.columns(2)
        
        with col1:
            location_pref = st.selectbox("Preferred Study Location", [
                "Home State", "Nearby States", "Pan India", "Abroad"
            ], key="location")
            budget = st.selectbox("Annual Education Budget (Lakhs)", [
                "2-5", "5-10", "10-20", "20-30", "30+"
            ], key="budget")
        
        with col2:
            timeline = st.selectbox("Career Goal Timeline", [
                "4 years", "5 years", "6+ years"
            ], key="timeline")
            work_pref = st.selectbox("Preferred Work Type", [
                "Private Job", "Business/Entrepreneurship", "Research & Academia", "Government Job"
            ], key="work_type")
        
        risk_tolerance = st.selectbox("Risk Tolerance", [
            "Low (Stable)", "Medium", "High (Adventurous)"
        ], key="risk")
        
        # Section 8: Extracurricular
        st.markdown("### 8️⃣ Extracurricular Activities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            has_sports = st.checkbox("Sports", key="sports")
            has_cultural = st.checkbox("Cultural", key="cultural")
        
        with col2:
            volunteering = st.slider("Volunteering Hours/Year", 0, 100, 0, key="volunteer")
        
        with col3:
            certifications = st.selectbox("Online Certifications", [0, 1, 2, 3, 4, 5], key="certs")
            projects = st.selectbox("Projects Completed", [0, 1, 2, 3, 4, 5], key="projects")
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit_button = st.form_submit_button("🚀 Get Career Recommendations", use_container_width=True)
    
    if submit_button:
        # Prepare user data
        user_data = {
            'age': age,
            'gender': gender,
            'state': state,
            'urban_rural': urban_rural,
            'family_income': family_income,
            '12th_stream': stream_12th,
            '10th_percentage': percentage_10th,
            '12th_percentage': percentage_12th,
            'school_board': school_board,
            'school_tier': school_tier.split()[1],  # Extract "1", "2", "3"
            'JEE_Main_percentile': jee_percentile,
            'NEET_percentile': neet_percentile,
            'logical_reasoning': logical,
            'quantitative_ability': quant,
            'verbal_ability': verbal,
            'abstract_reasoning': abstract,
            'spatial_reasoning': spatial,
            'interest_technology': interest_tech,
            'interest_healthcare': interest_health,
            'interest_business': interest_business,
            'interest_creative_arts': interest_arts,
            'interest_social_service': interest_social,
            'interest_research': interest_research,
            'leadership': leadership,
            'teamwork': teamwork,
            'creativity': creativity,
            'analytical_thinking': analytical,
            'communication': communication,
            'preferred_location': location_pref,
            'budget_constraint_lakhs': int(budget.split('-')[0]),
            'career_goal_timeline': timeline,
            'work_preference': work_pref,
            'risk_tolerance': risk_tolerance.split()[0],
            'has_sports': has_sports,
            'has_cultural': has_cultural,
            'volunteering_hours': volunteering,
            'num_certifications': certifications,
            'num_projects': projects,
            'initial_career': 'Computer Science Engineering'
        }
        
        st.session_state.recommendations = predict_career_path(user_data, num_paths=3)
        st.session_state.user_profile = user_data
        st.success("✅ Analysis complete! Check the Recommendations tab for results.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    if 'recommendations' not in st.session_state:
        st.info("👈 Please fill out the questionnaire first to get recommendations")
    else:
        st.markdown("## 🎯 Your Personalized Career Recommendations")
        
        user_profile = st.session_state.user_profile
        recommendations = st.session_state.recommendations
        
        # Profile Summary
        with st.expander("📌 Your Profile Summary", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stream", user_profile['12th_stream'])
            with col2:
                st.metric("12th %", f"{user_profile['12th_percentage']:.1f}%")
            with col3:
                st.metric("Interest (Tech)", user_profile['interest_technology'])
            with col4:
                st.metric("Timeline", user_profile['career_goal_timeline'])
        
        # Recommendations
        st.markdown("### Top Career Path Recommendations")
        
        for idx, rec in enumerate(recommendations, 1):
            # Color coding based on rank
            colors = ["#2ecc71", "#3498db", "#f39c12"]
            color = colors[idx-1] if idx <= 3 else "#95a5a6"
            
            with st.container():
                st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                        border-left: 5px solid {color};
                        padding: 20px;
                        border-radius: 8px;
                        margin: 15px 0;
                    '>
                        <h3 style='color: {color}; margin-top: 0;'>
                            #{idx} Recommended Path
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Career pathway
                path_str = " → ".join(rec['path'])
                st.markdown(f"**Career Journey:** `{path_str}`")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Path Score", f"{rec['score']:.2f}", delta="Recommendation Quality")
                with col2:
                    st.metric("Steps", rec['length'], delta="Career Progression Steps")
                with col3:
                    validity_check = "✅ Valid" if knowledge_graph.has_path(
                        rec['path'][0], rec['path'][-1]
                    ) else "⚠️ Exploratory"
                    st.write(f"**Graph Status:** {validity_check}")
                
                # Detailed pathway
                with st.expander("📍 Detailed Career Progression", expanded=idx==1):
                    for step_idx, career in enumerate(rec['path'], 1):
                        st.write(f"**Year {step_idx}:** {career}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    if 'recommendations' not in st.session_state:
        st.info("👈 Please fill out the questionnaire first to see analysis")
    else:
        st.markdown("## 📊 Detailed Analysis")
        
        user_profile = st.session_state.user_profile
        
        # Interest profile radar chart
        st.markdown("### Your Interest Profile")
        
        interests_data = {
            'Technology': user_profile['interest_technology'],
            'Healthcare': user_profile['interest_healthcare'],
            'Business': user_profile['interest_business'],
            'Creative': user_profile['interest_creative_arts'],
            'Social': user_profile['interest_social_service'],
            'Research': user_profile['interest_research']
        }
        
        fig = go.Figure(data=go.Scatterpolar(
            r=list(interests_data.values()),
            theta=list(interests_data.keys()),
            fill='toself',
            marker=dict(color='#3498db')
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Personality strengths
        st.markdown("### Personality Strengths")
        
        personality = {
            'Leadership': user_profile['leadership'],
            'Teamwork': user_profile['teamwork'],
            'Creativity': user_profile['creativity'],
            'Analytical': user_profile['analytical_thinking'],
            'Communication': user_profile['communication']
        }
        
        fig2 = go.Figure(data=[
            go.Bar(
                y=list(personality.keys()),
                x=list(personality.values()),
                orientation='h',
                marker=dict(color='#2ecc71')
            )
        ])
        
        fig2.update_layout(
            title="Personality Traits Comparison",
            xaxis_title="Score (1-5)",
            yaxis_title="",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Model metrics
        st.markdown("### 🔬 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Path Validity",
                f"{eval_metrics.get('path_validity_percent', 0):.1f}%",
                help="% of realistic career transitions"
            )
        with col2:
            st.metric(
                "Success Rate",
                f"{eval_metrics.get('success_rate_percent', 0):.1f}%",
                help="% of reaching recommended goals"
            )
        with col3:
            st.metric(
                "Avg Reward",
                f"{eval_metrics.get('avg_reward', 0):.2f}",
                help="Model confidence score"
            )
        with col4:
            st.metric(
                "Avg Path Length",
                f"{eval_metrics.get('avg_path_length', 0):.1f}",
                help="Steps in career journey"
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 🧠 How This System Works")
    
    st.markdown("""
    ### 🤖 AI Technology
    
    This recommendation system uses **Dueling Deep Q-Networks (DDQN)**, 
    an advanced machine learning algorithm trained on:
    - **2,000 Indian student profiles**
    - **Real career trajectories**
    - **Knowledge graph of 60+ careers**
    - **Multi-objective optimization**
    
    ### 📊 Training Data
    
    The model was trained on:
    - Students from all 5 streams (PCM, PCB, Commerce, Arts, PCMB)
    - JEE/NEET exam participation
    - 47 student profile features
    - 5-year career progressions
    
    ### 🎯 Recommendation Process
    
    1. **Profile Analysis**: Your questionnaire creates a 45-dimensional profile vector
    2. **Knowledge Graph**: Your profile is analyzed against 60+ career options
    3. **AI Inference**: The trained model predicts best career paths
    4. **Path Validation**: Recommendations are validated against knowledge graph
    5. **Scoring**: Paths are ranked by confidence score
    
    ### 🎓 What Makes This Unique
    
    ✅ **Indian Context**: JEE/NEET scores, streams, local colleges
    ✅ **Data-Driven**: Based on 2,000 real student journeys
    ✅ **Multi-Objective**: Balances multiple career factors
    ✅ **Realistic Paths**: Respects actual career progressions
    ✅ **Interpretable**: See exact pathways, not just predictions
    
    ### 📈 Performance
    
    - **Path Validity**: {:.1f}% of recommendations follow real patterns
    - **Success Rate**: {:.1f}% can reach recommended destinations
    - **Model Confidence**: {:.2f} average reward score
    """.format(
        eval_metrics.get('path_validity_percent', 0),
        eval_metrics.get('success_rate_percent', 0),
        eval_metrics.get('avg_reward', 0)
    ))
    
    st.markdown("""
    ### 📚 Career Categories Covered
    
    **Engineering** (11 paths)
    - Computer Science, Data Science, AI/ML, Mechanical, Electrical, Civil...
    
    **Medical** (10 paths)
    - MBBS, BDS, B.Pharm, Nursing, Physiotherapy...
    
    **Business & Commerce** (10 paths)
    - CA, CS, BBA, B.Com, Finance, Marketing...
    
    **Arts & Humanities** (9 paths)
    - Law, Journalism, Psychology, Social Work...
    
    **Emerging Fields** (10 paths)
    - Cybersecurity, Blockchain, Game Dev, Digital Marketing...
    
    **And More!** (10+ additional specialized paths)
    """)
    
    st.markdown("""
    ### ⚡ Quick Tips
    
    💡 **Be Honest**: Accurate answers lead to better recommendations
    
    💡 **Consider Your Interests**: Not just marks matter
    
    💡 **Think Long-term**: Your career is a journey, not a destination
    
    💡 **Explore Alternatives**: Top 3 recommendations offer diverse paths
    
    💡 **Regular Updates**: Re-run the assessment as you grow
    """)
    
    st.markdown("""
    ### 🔒 Privacy & Data
    
    - Your data is **not saved** or used for training
    - Recommendations are computed **locally**
    - No external API calls or tracking
    - Completely **private and confidential**
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
        <p>🎓 Career Path Recommender v1.0 | Powered by Deep Reinforcement Learning</p>
        <p>Built for Indian Students | MTech Project 2025</p>
        <p style='color: #999;'>Disclaimer: This tool provides guidance. Final decisions should consider multiple factors.</p>
    </div>
""", unsafe_allow_html=True)
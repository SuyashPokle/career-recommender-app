import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Career Path Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5em;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.3em;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .recommendation-box {
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        background-color: #f0f5ff;
        border-radius: 5px;
    }
    .metric-badge {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        margin-right: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING & INITIALIZATION
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load trained model and artifacts"""
    try:
        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        with open("knowledge_graph.gpickle", "rb") as f:
            import pickle as pk
            G = pk.load(f)
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
                features = self.feature_layers(state)
                value = self.value_stream(features)
                advantages = self.advantage_stream(features)
                q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
                return q_values
        
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

# ============================================================================
# LABEL ENCODERS - RECREATE FROM ACTUAL DATASET
# ============================================================================

def create_label_encoders():
    """Recreate label encoders matching the training dataset"""
    encoders = {}
    
    label_encoder_data = {
        'gender': ['Female', 'Male', 'Other'],
        'state': ['Andhra Pradesh', 'Assam', 'Bihar', 'Delhi', 'Goa', 'Gujarat', 
                  'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 
                  'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
                  'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
                  'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'],
        'urban_rural': ['Rural', 'Semi-Urban', 'Urban'],
        'family_income': ['<5 Lakhs', '5-10 Lakhs', '10-20 Lakhs', '20-50 Lakhs', '>50 Lakhs'],
        '12th_stream': ['Arts', 'Commerce', 'Science-PCB', 'Science-PCM', 'Science-PCMB'],
        'school_board': ['CBSE', 'ICSE', 'State Board'],
        'school_tier': ['Tier 1', 'Tier 2', 'Tier 3'],
        'preferred_location': ['Home State', 'Nearby States', 'Pan India', 'Abroad'],
        'career_goal_timeline': ['4 years', '5 years', '6+ years'],
        'work_preference': ['Job', 'Business', 'Research', 'Government'],
        'risk_tolerance': ['Low', 'Medium', 'High']
    }
    
    for col, classes in label_encoder_data.items():
        le = LabelEncoder()
        le.fit(classes)
        encoders[col] = le
    
    return encoders

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_user_input(user_data, encoders, scaler_params):
    """Convert user input to model state vector (38 dimensions)"""
    state = []
    
    # Encode categorical features
    for col in categorical_features:
        if col in encoders and col in user_data:
            try:
                encoded_val = encoders[col].transform([user_data[col]])
                state.append(float(encoded_val))
            except (ValueError, KeyError):
                state.append(0.0)
        else:
            state.append(0.0)
    
    # Normalize numerical features
    for i, col in enumerate(numerical_features):
        if col in user_data:
            val = float(user_data[col])
        else:
            val = 0.0
        
        # ✅ FIX: Handle both old and new scaler_params formats
        if isinstance(scaler_params, dict):
            if 'min' in scaler_params and 'scale' in scaler_params:
                # New format: {'min': [...], 'scale': [...]}
                min_val = scaler_params['min'][i] if i < len(scaler_params['min']) else 0
                scale = scaler_params['scale'][i] if i < len(scaler_params['scale']) else 1
            else:
                # Fallback: no scaling info
                min_val = 0
                scale = 1
        else:
            min_val = 0
            scale = 1
        
        normalized = (val - min_val) / scale if scale != 0 else 0.5
        state.append(normalized)
    
    return np.array(state, dtype=np.float32)

def get_valid_next_careers(current_career):
    """Get valid next career moves from knowledge graph"""
    try:
        if current_career not in knowledge_graph:
            return list(career_to_id.keys())
        successors = list(knowledge_graph.successors(current_career))
        if not successors:
            return list(career_to_id.keys())
        return successors
    except:
        return list(career_to_id.keys())

def predict_career_path(user_data, num_paths=3):
    """Generate career path recommendations"""
    recommendations = []
    encoders = create_label_encoders()
    
    initial_career = user_data.get('initial_career', 'Computer Science Engineering')
    if isinstance(initial_career, list):
        initial_career = initial_career[0]
    
    # ✅ FIX: Handle list initial_career
    if isinstance(initial_career, list):
        initial_career = initial_career
    
    if initial_career not in career_to_id:
        st.warning(f"⚠️ Career '{initial_career}' not found in knowledge graph.")
        return []
    
    for path_num in range(num_paths):
        current_career = initial_career
        path = [current_career]
        total_reward = 0
        state = encode_user_input(user_data, encoders, scaler_params)
        
        for step in range(4):  # 4 more steps after initial career
            valid_careers = get_valid_next_careers(current_career)
            valid_action_ids = [career_to_id[c] for c in valid_careers if c in career_to_id]
            
            if not valid_action_ids:
                break
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor).squeeze()
            
            q_array = q_values.cpu().numpy()
            valid_q = [(q_array[action_id], action_id) for action_id in valid_action_ids]
            
            if valid_q:
                _, best_action = max(valid_q, key=lambda x: x)
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

# ============================================================================
# MAIN APP INTERFACE
# ============================================================================

st.markdown('<h1 class="main-title">🎓 Career Path Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Career Guidance for Indian Students (Post-12th)</p>', 
            unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Questionnaire", "📊 Model Info", "ℹ️ About"])

with tab1:
    st.markdown('<div class="section-header">📝 Student Profile Questionnaire</div>', 
                unsafe_allow_html=True)
    
    encoders = create_label_encoders()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        gender = st.selectbox("Gender", options=encoders['gender'].classes_)
        state = st.selectbox("State", options=sorted(encoders['state'].classes_))
        urban_rural = st.selectbox("Area Type", options=encoders['urban_rural'].classes_)
        age = st.slider("Age", min_value=16, max_value=22, value=18)
        
        st.subheader("📚 Academic Background")
        stream = st.selectbox("12th Stream", options=encoders['12th_stream'].classes_)
        board = st.selectbox("School Board", options=encoders['school_board'].classes_)
        tier = st.selectbox("School Tier", options=encoders['school_tier'].classes_)
        
        percent_10 = st.slider("10th Percentage", min_value=50, max_value=100, value=75)
        percent_12 = st.slider("12th Percentage", min_value=50, max_value=100, value=80)
    
    with col2:
        st.subheader("💰 Family & Budget")
        income = st.selectbox("Family Income", options=encoders['family_income'].classes_)
        budget = st.slider("Education Budget (Lakhs)", min_value=2, max_value=50, value=10)
        
        st.subheader("🎯 Career Preferences")
        career_timeline = st.selectbox("Career Goal Timeline", 
                                       options=encoders['career_goal_timeline'].classes_)
        work_pref = st.selectbox("Work Preference", options=encoders['work_preference'].classes_)
        risk = st.selectbox("Risk Tolerance", options=encoders['risk_tolerance'].classes_)
        location = st.selectbox("Preferred Location", options=encoders['preferred_location'].classes_)
        
        st.subheader("✨ Aptitude Scores (1-10 scale)")
        logical = st.slider("Logical Reasoning", min_value=1, max_value=10, value=6)
        quantitative = st.slider("Quantitative Ability", min_value=1, max_value=10, value=6)
        verbal = st.slider("Verbal Ability", min_value=1, max_value=10, value=6)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🎪 Interests (1-5 scale)")
        interest_tech = st.slider("Technology Interest", min_value=1, max_value=5, value=3)
        interest_health = st.slider("Healthcare Interest", min_value=1, max_value=5, value=3)
        interest_biz = st.slider("Business Interest", min_value=1, max_value=5, value=3)
    
    with col4:
        st.subheader("🏅 Personality Traits (1-5 scale)")
        leadership = st.slider("Leadership", min_value=1, max_value=5, value=3)
        teamwork = st.slider("Teamwork", min_value=1, max_value=5, value=3)
        creativity = st.slider("Creativity", min_value=1, max_value=5, value=3)
    
    # Collect user data
    user_data = {
        'age': age,
        'gender': gender,
        'state': state,
        'urban_rural': urban_rural,
        'family_income': income,
        '12th_stream': stream,
        '10th_percentage': percent_10,
        '12th_percentage': percent_12,
        'school_board': board,
        'school_tier': tier,
        'logical_reasoning': logical,
        'quantitative_ability': quantitative,
        'verbal_ability': verbal,
        'interest_technology': interest_tech,
        'interest_healthcare': interest_health,
        'interest_business': interest_biz,
        'leadership': leadership,
        'teamwork': teamwork,
        'creativity': creativity,
        'preferred_location': location,
        'budget_constraint_lakhs': budget,
        'career_goal_timeline': career_timeline,
        'work_preference': work_pref,
        'risk_tolerance': risk,
    }
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("🚀 Get Career Recommendations", use_container_width=True):
            # Select initial career based on stream
            if stream == 'Science-PCM':
                initial_careers = ['Computer Science Engineering', 'Mechanical Engineering', 
                                'Electrical Engineering', 'AI/ML Engineering']
            elif stream == 'Science-PCB':
                initial_careers = ['MBBS', 'B.Pharm', 'BDS', 'Nursing']
            elif stream == 'Commerce':
                initial_careers = ['B.Com', 'BBA', 'CA', 'CS']
            elif stream == 'Arts':
                initial_careers = ['Law (BA LLB)', 'Mass Communication', 'Social Work']
            else:
                initial_careers = list(career_to_id.keys())[:5]
            
            # ALWAYS assign to a string, never a list
            best_career = initial_careers[0] if initial_careers else list(career_to_id.keys())[0]
            user_data['initial_career'] = best_career

            with st.spinner("🔄 Analyzing your profile and generating recommendations..."):
                recommendations = predict_career_path(user_data, num_paths=3)
            
            if recommendations:
                st.success("✅ Recommendations Generated!")
                st.markdown('<div class="section-header">🎯 Your Career Path Recommendations</div>', 
                            unsafe_allow_html=True)
                
                for idx, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f'<div class="recommendation-box">', unsafe_allow_html=True)
                        st.markdown(f"### 🔥 Recommendation {idx}")
                        st.markdown(f"**Path:** {' → '.join(rec['path'])}")
                        score_badge = f'<span class="metric-badge">Score: {rec["score"]:.3f}</span>'
                        length_badge = f'<span class="metric-badge">Years: {rec["length"]}</span>'
                        st.markdown(f"{score_badge} {length_badge}", unsafe_allow_html=True)
                        st.markdown(f'</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Could not generate recommendations. Please try again.")

with tab2:
    st.markdown('<div class="section-header">📊 Model Information</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("State Dimension", artifacts['model_config']['state_dim'])
    with col_m2:
        st.metric("Action Space", artifacts['model_config']['action_dim'])
    with col_m3:
        st.metric("Model Type", "Dueling DQN")
    
    st.markdown("**Model Architecture:**")
    st.code("""Input (38) → Linear(256) → BatchNorm → ReLU → Dropout
         → Linear(256) → BatchNorm → ReLU → Dropout
         → Linear(128) → BatchNorm → ReLU
         ↓
    ┌────────────────────────┐
    ↓ Value Stream           ↓ Advantage Stream
 Linear(64) → ReLU      Linear(64) → ReLU
 Linear(1)              Linear(72)""")
    
    st.markdown("**Evaluation Metrics:**")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Average Reward", f"{eval_metrics['avg_reward']:.3f}")
    with col_e2:
        st.metric("Path Validity", f"{eval_metrics['path_validity']:.1f}%")
    with col_e3:
        st.metric("Avg Path Length", f"{eval_metrics['avg_path_length']:.1f}")

with tab3:
    st.markdown("### 📖 About This Application")
    st.write("""
    This application uses **Deep Reinforcement Learning** (Dueling DQN) to recommend optimal career paths 
    for Indian students post-12th standard.
    
    **How it works:**
    1. You provide your academic profile, interests, and preferences
    2. The AI model analyzes your profile against a career knowledge graph
    3. It generates 3 different career pathway recommendations
    4. Each path includes career progression over 5 years
    
    **Dataset:** 2000 Indian student profiles with real career trajectories
    **Model:** Dueling Double DQN with 72 career nodes and 188 transition edges
    **Training:** 500 episodes on GPU
    """)

st.markdown("---")
st.markdown("### 💡 Tips for Best Results:")
st.info("""
✓ Be honest with aptitude scores - they significantly impact recommendations
✓ Consider multiple recommendations - each path shows different growth trajectory
✓ Your stream selection heavily influences career recommendations
✓ Visit a career counselor to validate recommendations
""")

st.markdown("---")
st.markdown("Built for MTech Project 2025 | VIT Vellore | AI/ML Specialization")


import pandas as pd
import numpy as np
import json

# Create comprehensive synthetic dataset structure for Indian students after 12th

print("="*80)
print("SYNTHETIC DATASET DESIGN FOR INDIAN CAREER PATH RECOMMENDATION")
print("After 12th Standard - Indian Context")
print("="*80)

# Define the schema for synthetic dataset
dataset_schema = {
    "student_profile": {
        "student_id": "unique_id",
        "demographics": {
            "age": "16-18 years",
            "gender": "Male/Female/Other",
            "state": "29 states + 8 UTs",
            "urban_rural": "Urban/Rural",
            "family_income": "Annual income brackets"
        },
        "academic_background": {
            "12th_stream": "Science (PCM/PCB/PCMB)/Commerce/Arts",
            "10th_percentage": "60-100%",
            "12th_percentage": "60-100%",
            "school_board": "CBSE/ICSE/State Board",
            "school_tier": "Tier 1/2/3 (based on infrastructure)",
        },
        "competitive_exams": {
            "JEE_Main_percentile": "0-100 (if appeared)",
            "JEE_Advanced_rank": "1-250000 (if qualified)",
            "NEET_percentile": "0-100 (if appeared)",
            "NEET_rank": "1-2000000 (if qualified)",
            "CUET_score": "0-800 (if appeared)",
            "other_exams": "CAT/CLAT/NDA/etc"
        },
        "skills_assessment": {
            "aptitude_scores": {
                "logical_reasoning": "1-10 scale",
                "quantitative_ability": "1-10 scale",
                "verbal_ability": "1-10 scale",
                "abstract_reasoning": "1-10 scale",
                "spatial_reasoning": "1-10 scale"
            },
            "interest_areas": {
                "technology": "1-5 scale",
                "healthcare": "1-5 scale",
                "business": "1-5 scale",
                "creative_arts": "1-5 scale",
                "social_service": "1-5 scale",
                "research": "1-5 scale"
            },
            "personality_traits": {
                "leadership": "1-5 scale",
                "teamwork": "1-5 scale",
                "creativity": "1-5 scale",
                "analytical_thinking": "1-5 scale",
                "communication": "1-5 scale"
            }
        },
        "preferences": {
            "preferred_location": "Home State/Nearby States/Pan India/Abroad",
            "budget_constraint": "Annual budget for education (INR)",
            "career_goal_timeline": "4 years/5 years/6+ years",
            "work_preference": "Job/Business/Research/Government",
            "risk_tolerance": "Low/Medium/High"
        },
        "extracurricular": {
            "sports": "Yes/No (type)",
            "cultural_activities": "Yes/No (type)",
            "volunteering": "Yes/No (hours)",
            "certifications": "List of online certifications",
            "projects": "Number of projects done"
        }
    },
    
    "career_paths": {
        "engineering_paths": [
            "Computer Science Engineering",
            "Mechanical Engineering",
            "Electrical Engineering",
            "Electronics Engineering",
            "Civil Engineering",
            "Chemical Engineering",
            "Aerospace Engineering",
            "Biotechnology",
            "Information Technology",
            "AI/ML Engineering",
            "Data Science"
        ],
        "medical_paths": [
            "MBBS (Doctor)",
            "BDS (Dentistry)",
            "BAMS (Ayurveda)",
            "BHMS (Homeopathy)",
            "B.Pharm (Pharmacy)",
            "Nursing (B.Sc Nursing)",
            "Physiotherapy",
            "Veterinary Science",
            "Medical Lab Technology",
            "Biotechnology"
        ],
        "commerce_business_paths": [
            "B.Com (Commerce)",
            "BBA (Business Administration)",
            "CA (Chartered Accountancy)",
            "CS (Company Secretary)",
            "CMA (Cost Management)",
            "B.Com + CA",
            "Economics",
            "Finance",
            "Marketing",
            "Actuarial Science"
        ],
        "arts_humanities_paths": [
            "BA Psychology",
            "BA Journalism",
            "BA Mass Communication",
            "Law (BA LLB)",
            "Design (Fashion/Interior/Graphics)",
            "Fine Arts",
            "Hotel Management",
            "Social Work",
            "Political Science",
            "Literature/Languages"
        ],
        "science_research_paths": [
            "B.Sc Physics",
            "B.Sc Chemistry",
            "B.Sc Mathematics",
            "B.Sc Biology",
            "B.Sc Statistics",
            "Environmental Science",
            "Agriculture",
            "Forestry",
            "Marine Biology",
            "Astronomy"
        ],
        "emerging_paths": [
            "Data Science",
            "Artificial Intelligence",
            "Cyber Security",
            "Blockchain Technology",
            "Game Development",
            "Animation & VFX",
            "Digital Marketing",
            "Content Creation",
            "Entrepreneurship",
            "Renewable Energy"
        ]
    },
    
    "intermediate_milestones": {
        "year_1": "Foundation courses/diploma",
        "year_2": "Core subjects/specialization choice",
        "year_3": "Advanced topics/internships",
        "year_4": "Final year/placements/further studies",
        "additional": "Masters/PhD/Professional courses"
    }
}

print("\n" + "="*80)
print("DATASET SCHEMA STRUCTURE")
print("="*80)
print(json.dumps(dataset_schema, indent=2))

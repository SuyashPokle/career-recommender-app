# ============================================================================
# Career_trajectories_improved.py
# ============================================================================
# USAGE: This file contains helper functions for career trajectory analysis
# Import this into your Jupyter notebook or use standalone
# ============================================================================

import pandas as pd
import numpy as np
from collections import defaultdict

"""
HOW TO USE THIS FILE:
=====================

Option 1: Import functions into your Jupyter notebook
--------------------------------------------------
In your notebook first cell, add:

from Career_trajectories_improved import *

# Then use functions like:
df = pd.read_csv('indian_students_career_dataset_synthetic_5000.csv')
graph_data = build_career_knowledge_graph(df)
validate_trajectories(df)
print_career_statistics(df)


Option 2: Run this file directly
--------------------------------
python Career_trajectories_improved.py
# This will generate analysis and statistics


Option 3: Use in training code
------------------------------
In your Kaggle notebook training code, copy the functions you need
and call them during training initialization
"""

# ============================================================================
# FUNCTION 1: BUILD CAREER KNOWLEDGE GRAPH
# ============================================================================

def build_career_knowledge_graph(df):
    """
    Build a knowledge graph of career transitions from dataset trajectories
    
    Input: DataFrame with career trajectory columns
    Output: Dictionary with career transitions and probabilities
    """
    
    transitions = defaultdict(lambda: defaultdict(int))
    total_transitions = 0
    
    print("Building career knowledge graph...")
    
    # Extract trajectories
    for idx, row in df.iterrows():
        # Get trajectory from career_trajectory column
        try:
            trajectory = eval(row['career_trajectory']) if isinstance(row['career_trajectory'], str) else \
                         [row['career_path_year1'], row['career_year2'], row['career_year3'], 
                          row['career_year4'], row['career_year5']]
        except:
            continue
        
        # Count transitions
        for i in range(len(trajectory) - 1):
            current = trajectory[i]
            next_career = trajectory[i + 1]
            
            # Only count actual transitions (not repetitions)
            if current != next_career:
                transitions[current][next_career] += 1
                total_transitions += 1
    
    # Convert to probabilities
    transition_probs = {}
    for source, targets in transitions.items():
        total = sum(targets.values())
        transition_probs[source] = {
            target: count / total 
            for target, count in targets.items()
        }
    
    print(f"✅ Graph built: {len(transition_probs)} career nodes, {total_transitions} transitions")
    
    return {
        'transitions': dict(transitions),
        'probabilities': transition_probs,
        'total_transitions': total_transitions
    }

# ============================================================================
# FUNCTION 2: ANALYZE CAREER PATHS
# ============================================================================

def analyze_career_paths(df):
    """
    Analyze career progression patterns in dataset
    """
    
    print("\n" + "=" * 80)
    print("CAREER PATH ANALYSIS")
    print("=" * 80)
    
    # Extract all unique careers
    all_careers = set()
    path_lengths = []
    stagnation_count = 0
    progression_count = 0
    
    for idx, row in df.iterrows():
        try:
            trajectory = eval(row['career_trajectory']) if isinstance(row['career_trajectory'], str) else \
                         [row['career_path_year1'], row['career_year2'], row['career_year3'],
                          row['career_year4'], row['career_year5']]
        except:
            continue
        
        all_careers.update(trajectory)
        
        # Count unique careers in path
        unique_in_path = len(set(trajectory))
        path_lengths.append(unique_in_path)
        
        # Count stagnation vs progression
        if unique_in_path == 1:
            stagnation_count += 1
        else:
            progression_count += 1
    
    print(f"\nTotal Unique Careers: {len(all_careers)}")
    print(f"\nPath Diversity:")
    print(f"  No progression (same career all 5 years): {stagnation_count} ({100*stagnation_count/len(df):.1f}%)")
    print(f"  With progression: {progression_count} ({100*progression_count/len(df):.1f}%)")
    print(f"  Average unique careers per path: {np.mean(path_lengths):.2f}")
    
    return {
        'unique_careers': all_careers,
        'path_lengths': path_lengths,
        'stagnation_count': stagnation_count,
        'progression_count': progression_count
    }

# ============================================================================
# FUNCTION 3: VALIDATE CAREER TRAJECTORIES
# ============================================================================

def validate_career_trajectories(df):
    """
    Validate that career trajectories make sense
    """
    
    print("\n" + "=" * 80)
    print("TRAJECTORY VALIDATION")
    print("=" * 80)
    
    valid_count = 0
    invalid_count = 0
    issues = []
    
    for idx, row in df.iterrows():
        try:
            trajectory = eval(row['career_trajectory']) if isinstance(row['career_trajectory'], str) else \
                         [row['career_path_year1'], row['career_year2'], row['career_year3'],
                          row['career_year4'], row['career_year5']]
            
            # Check length
            if len(trajectory) != 5:
                invalid_count += 1
                issues.append(f"  Row {idx}: Invalid trajectory length {len(trajectory)}")
                continue
            
            # Check for None values
            if any(x is None for x in trajectory):
                invalid_count += 1
                issues.append(f"  Row {idx}: Trajectory contains None values")
                continue
            
            valid_count += 1
            
        except Exception as e:
            invalid_count += 1
            issues.append(f"  Row {idx}: Parse error - {str(e)[:50]}")
    
    print(f"\nValid trajectories: {valid_count}/{len(df)} ({100*valid_count/len(df):.1f}%)")
    print(f"Invalid trajectories: {invalid_count}/{len(df)}")
    
    if issues:
        print("\nFirst 5 issues:")
        for issue in issues[:5]:
            print(issue)
    
    return {
        'valid': valid_count,
        'invalid': invalid_count,
        'issues': issues
    }

# ============================================================================
# FUNCTION 4: PRINT CAREER STATISTICS
# ============================================================================

def print_career_statistics(df):
    """
    Print detailed career statistics
    """
    
    print("\n" + "=" * 80)
    print("CAREER STATISTICS")
    print("=" * 80)
    
    # Year 1 careers
    print("\n📊 Year 1 Career Distribution:")
    career_dist = df['career_path_year1'].value_counts().head(15)
    for career, count in career_dist.items():
        pct = 100 * count / len(df)
        print(f"  {career}: {count} ({pct:.1f}%)")
    
    # Stream distribution
    print("\n🎓 Stream Distribution:")
    stream_dist = df['12th_stream'].value_counts()
    for stream, count in stream_dist.items():
        pct = 100 * count / len(df)
        print(f"  {stream}: {count} ({pct:.1f}%)")
    
    # Exam participation
    print("\n📝 Exam Participation:")
    jee_count = df['JEE_Main_percentile'].notna().sum()
    neet_count = df['NEET_percentile'].notna().sum()
    cuet_count = df['CUET_score'].notna().sum()
    print(f"  JEE Main: {jee_count} ({100*jee_count/len(df):.1f}%)")
    print(f"  NEET: {neet_count} ({100*neet_count/len(df):.1f}%)")
    print(f"  CUET: {cuet_count} ({100*cuet_count/len(df):.1f}%)")
    
    # Academic performance
    print("\n📈 Academic Performance:")
    print(f"  10th Percentage: {df['10th_percentage'].mean():.2f}% ± {df['10th_percentage'].std():.2f}%")
    print(f"  12th Percentage: {df['12th_percentage'].mean():.2f}% ± {df['12th_percentage'].std():.2f}%")
    
    # Aptitude averages
    print("\n🧠 Average Aptitude Scores (1-10):")
    aptitude_cols = ['logical_reasoning', 'quantitative_ability', 'verbal_ability',
                     'abstract_reasoning', 'spatial_reasoning']
    for col in aptitude_cols:
        avg = df[col].mean()
        print(f"  {col}: {avg:.2f}")
    
    # Interest averages
    print("\n💡 Average Interest Scores (1-5):")
    interest_cols = ['interest_technology', 'interest_healthcare', 'interest_business',
                     'interest_creative_arts', 'interest_social_service', 'interest_research']
    for col in interest_cols:
        avg = df[col].mean()
        print(f"  {col}: {avg:.2f}")

# ============================================================================
# FUNCTION 5: GET CAREER PROGRESSIONS
# ============================================================================

def get_career_progressions(df):
    """
    Extract common career progression sequences
    """
    
    print("\n" + "=" * 80)
    print("COMMON CAREER PROGRESSIONS")
    print("=" * 80)
    
    progressions = defaultdict(int)
    
    for idx, row in df.iterrows():
        try:
            trajectory = eval(row['career_trajectory']) if isinstance(row['career_trajectory'], str) else \
                         [row['career_path_year1'], row['career_year2'], row['career_year3'],
                          row['career_year4'], row['career_year5']]
            
            # Convert to tuple (hashable)
            prog = tuple(trajectory)
            progressions[prog] += 1
            
        except:
            continue
    
    # Sort by frequency
    sorted_progs = sorted(progressions.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Most Common Progressions:")
    for i, (prog, count) in enumerate(sorted_progs[:20], 1):
        pct = 100 * count / len(df)
        prog_str = " → ".join(prog)
        print(f"  {i}. {prog_str}")
        print(f"     Count: {count} ({pct:.1f}%)\n")
    
    return dict(sorted_progs)

# ============================================================================
# FUNCTION 6: MAIN - RUN ALL ANALYSES
# ============================================================================

def main():
    """
    Run all analyses on the dataset
    """
    
    print("=" * 80)
    print("CAREER TRAJECTORY ANALYSIS SUITE")
    print("=" * 80)
    
    # Load dataset
    try:
        df = pd.read_csv('indian_students_career_dataset_synthetic_5000.csv')
        print(f"\n✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ Dataset file not found!")
        print("Make sure 'indian_students_career_dataset_synthetic_5000.csv' is in the same directory")
        return
    
    # Run analyses
    print("\n" + "=" * 80)
    print("Running all analyses...")
    print("=" * 80)
    
    # 1. Validate trajectories
    validation = validate_career_trajectories(df)
    
    # 2. Build knowledge graph
    graph_data = build_career_knowledge_graph(df)
    
    # 3. Analyze paths
    path_analysis = analyze_career_paths(df)
    
    # 4. Print statistics
    print_career_statistics(df)
    
    # 5. Get progressions
    progressions = get_career_progressions(df)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    
    # Summary
    print(f"""
SUMMARY:
  • Total students: {len(df)}
  • Valid trajectories: {validation['valid']}
  • Unique careers: {len(path_analysis['unique_careers'])}
  • Career transitions: {graph_data['total_transitions']}
  • Career paths with progression: {path_analysis['progression_count']}
  • Average path diversity: {np.mean(path_analysis['path_lengths']):.2f}
    """)

# ============================================================================
# RUN IF EXECUTED DIRECTLY
# ============================================================================

if __name__ == "__main__":
    main()

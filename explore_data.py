"""
Data exploration and visualization script for TAME Pain Dataset
Analyze pain distribution, audio characteristics, and data quality
"""

import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_metadata(metadata_file='../meta_audio.csv'):
    """Load audio metadata"""
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    df = pd.read_csv(metadata_file)
    return df

def explore_pain_distribution(df):
    """Analyze pain level distribution"""
    print("\n" + "=" * 60)
    print("Pain Level Distribution")
    print("=" * 60)
    
    print("\nRaw Statistics:")
    print(df['REVISED PAIN'].describe())
    
    print("\nValue Counts:")
    print(df['REVISED PAIN'].value_counts().sort_index())
    
    # Create class distribution
    pain_classes = {
        'low': (1, 3),
        'medium': (4, 7),
        'high': (8, 10),
    }
    
    class_mapping = []
    for pain_level in df['REVISED PAIN']:
        pain_level = int(pain_level)
        if pain_classes['low'][0] <= pain_level <= pain_classes['low'][1]:
            class_mapping.append('Low (1-3)')
        elif pain_classes['medium'][0] <= pain_level <= pain_classes['medium'][1]:
            class_mapping.append('Medium (4-7)')
        elif pain_classes['high'][0] <= pain_level <= pain_classes['high'][1]:
            class_mapping.append('High (8-10)')
    
    class_dist = pd.Series(class_mapping).value_counts()
    print("\nClass Distribution (for training):")
    print(class_dist)
    
    return class_dist

def explore_audio_characteristics(df, dataset_path='../mic1_trim_v2'):
    """Analyze audio file characteristics"""
    print("\n" + "=" * 60)
    print("Audio Characteristics")
    print("=" * 60)
    
    print("\nDuration Statistics (seconds):")
    print(df['DURATION (SEC)'].describe())
    
    print("\nCondition Distribution:")
    print(df['COND'].value_counts())
    
    print("\nAudio Quality (ACTION LABEL) Distribution:")
    print(df['ACTION LABEL'].value_counts().sort_index())
    print("  0 = Highest quality")
    print("  4 = Lowest quality")
    
    # Sample audio analysis
    print("\nSampling audio files for spectral analysis...")
    
    durations = []
    sample_rates = []
    rms_energies = []
    
    for idx, row in df.head(100).iterrows():  # Sample first 100
        pid = row['PID']
        cond = row['COND']
        uttnum = int(row['UTTNUM'])
        uttid = int(row['UTTID'])
        
        audio_file = f"{pid}.{cond}.{uttnum}.{uttid}.wav"
        audio_path = os.path.join(dataset_path, pid, audio_file)
        
        if os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=None)
                durations.append(len(y) / sr)
                sample_rates.append(sr)
                rms_energies.append(librosa.feature.rms(y=y).mean())
            except Exception as e:
                print(f"  Error loading {audio_file}: {e}")
    
    if durations:
        print(f"\nActual Audio Duration (from samples):")
        print(f"  Mean: {np.mean(durations):.3f}s")
        print(f"  Std:  {np.std(durations):.3f}s")
        print(f"  Min:  {np.min(durations):.3f}s")
        print(f"  Max:  {np.max(durations):.3f}s")
    
    if sample_rates:
        print(f"\nSample Rates (from audio files):")
        print(f"  Unique rates: {set(sample_rates)}")
        print(f"  Mode: {max(set(sample_rates), key=sample_rates.count)} Hz")
    
    if rms_energies:
        print(f"\nRMS Energy (loudness):")
        print(f"  Mean: {np.mean(rms_energies):.4f}")
        print(f"  Std:  {np.std(rms_energies):.4f}")

def explore_participants(participant_file='../meta_participant.csv'):
    """Analyze participant demographics"""
    print("\n" + "=" * 60)
    print("Participant Demographics")
    print("=" * 60)
    
    if not os.path.exists(participant_file):
        print("Participant file not found")
        return
    
    df = pd.read_csv(participant_file)
    
    print(f"\nTotal Participants: {len(df)}")
    
    print("\nGender Distribution:")
    print(df['GENDER'].value_counts())
    
    print("\nAge Statistics:")
    print(df['AGE'].describe())
    
    print("\nRace/Ethnicity Distribution:")
    print(df['RACE/ETHNICITY'].value_counts())
    
    print("\nExperimental Condition Completion:")
    conditions = df[['LC', 'LW', 'RC', 'RW']]
    for col in conditions.columns:
        completed = (conditions[col] == 1).sum()
        print(f"  {col}: {completed}/{len(df)} completed")

def visualize_distributions(df):
    """Create visualization plots"""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pain level distribution
    axes[0, 0].hist(df['REVISED PAIN'], bins=10, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Pain Level')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Pain Level Distribution (1-10)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Class distribution
    pain_classes = {
        'low': (1, 3),
        'medium': (4, 7),
        'high': (8, 10),
    }
    
    class_counts = defaultdict(int)
    for pain_level in df['REVISED PAIN']:
        pain_level = int(pain_level)
        if pain_classes['low'][0] <= pain_level <= pain_classes['low'][1]:
            class_counts['Low (1-3)'] += 1
        elif pain_classes['medium'][0] <= pain_level <= pain_classes['medium'][1]:
            class_counts['Medium (4-7)'] += 1
        elif pain_classes['high'][0] <= pain_level <= pain_classes['high'][1]:
            class_counts['High (8-10)'] += 1
    
    colors = ['green', 'yellow', 'red']
    axes[0, 1].bar(class_counts.keys(), class_counts.values(), color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Training Class Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Duration distribution
    axes[1, 0].hist(df['DURATION (SEC)'], bins=50, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Duration (seconds)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Audio Duration Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Condition distribution
    cond_counts = df['COND'].value_counts()
    axes[1, 1].bar(cond_counts.index, cond_counts.values, color='lightsteelblue', edgecolor='black')
    axes[1, 1].set_xlabel('Condition')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Experimental Condition Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=100)
    print("✓ Saved data_distribution_analysis.png")
    
    # Pain by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = df['COND'].unique()
    pain_by_cond = [df[df['COND'] == cond]['REVISED PAIN'].values for cond in sorted(conditions)]
    
    bp = ax.boxplot(pain_by_cond, labels=sorted(conditions), patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Pain Level')
    ax.set_xlabel('Condition')
    ax.set_title('Pain Level by Experimental Condition')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('pain_by_condition.png', dpi=100)
    print("✓ Saved pain_by_condition.png")

def sample_mel_spectrograms(df, dataset_path='../mic1_trim_v2', n_samples=3):
    """Visualize sample mel spectrograms from each class"""
    print("\nGenerating sample mel spectrograms...")
    
    pain_classes = {
        'low': (1, 3),
        'medium': (4, 7),
        'high': (8, 10),
    }
    
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 10))
    
    class_names = ['Low (1-3)', 'Medium (4-7)', 'High (8-10)']
    
    for class_idx, (class_name, (low, high)) in enumerate(
        zip(class_names, pain_classes.values())
    ):
        # Find samples for this class
        class_df = df[(df['REVISED PAIN'] >= low) & (df['REVISED PAIN'] <= high)]
        samples = class_df.sample(n=min(n_samples, len(class_df)), random_state=42)
        
        for sample_idx, (_, row) in enumerate(samples.iterrows()):
            pid = row['PID']
            cond = row['COND']
            uttnum = int(row['UTTNUM'])
            uttid = int(row['UTTID'])
            pain_level = int(row['REVISED PAIN'])
            
            audio_file = f"{pid}.{cond}.{uttnum}.{uttid}.wav"
            audio_path = os.path.join(dataset_path, pid, audio_file)
            
            if os.path.exists(audio_path):
                try:
                    y, sr = librosa.load(audio_path, sr=16000)
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    ax = axes[class_idx, sample_idx]
                    im = ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
                    ax.set_title(f"{class_name}\nPain={pain_level}")
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Freq')
                except Exception as e:
                    print(f"  Error: {e}")
    
    plt.tight_layout()
    plt.savefig('sample_spectrograms.png', dpi=100)
    print("✓ Saved sample_spectrograms.png")

def main():
    print("=" * 60)
    print("TAME Pain Dataset - Exploration & Analysis")
    print("=" * 60)
    
    # Load metadata
    print("\nLoading metadata...")
    df = load_metadata()
    print(f"Loaded {len(df)} audio records")
    
    # Explore distributions
    class_dist = explore_pain_distribution(df)
    
    # Explore audio
    explore_audio_characteristics(df)
    
    # Explore participants
    explore_participants()
    
    # Visualizations
    visualize_distributions(df)
    sample_mel_spectrograms(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("Generated files:")
    print("  - data_distribution_analysis.png")
    print("  - pain_by_condition.png")
    print("  - sample_spectrograms.png")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()

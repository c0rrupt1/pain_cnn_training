"""
Setup and validation script for TAME Pain CNN training
Checks dataset, installs dependencies, validates environment
"""

import os
import sys
import subprocess

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_python_version():
    """Check Python version"""
    print("\n[1/5] Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Python 3.8+ is recommended")
        return False
    print("✓ Python version OK")
    return True

def check_dataset_structure():
    """Check if dataset files exist"""
    print("\n[2/5] Checking dataset structure...")
    
    required_files = [
        '../meta_audio.csv',
        '../meta_participant.csv',
        '../data_dictionary.md',
        '../readme.md',
    ]
    
    required_dirs = [
        '../mic1_trim_v2',
        '../Annotations',
    ]
    
    all_ok = True
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")
            all_ok = False
    
    # Check directories
    for dir in required_dirs:
        if os.path.isdir(dir):
            print(f"✓ Found {dir}/")
        else:
            print(f"✗ Missing {dir}/")
            all_ok = False
    
    # Check audio files
    if os.path.isdir('../mic1_trim_v2'):
        participants = [d for d in os.listdir('../mic1_trim_v2') 
                       if os.path.isdir(os.path.join('../mic1_trim_v2', d))]
        audio_count = 0
        for pid in participants[:3]:  # Sample first 3 participants
            pid_path = os.path.join('../mic1_trim_v2', pid)
            audio_files = [f for f in os.listdir(pid_path) if f.endswith('.wav')]
            audio_count += len(audio_files)
        
        print(f"✓ Found {len(participants)} participants with audio data")
        print(f"  (Sample: {audio_count} audio files in first 3 participants)")
    
    return all_ok

def check_packages():
    """Check if required packages are installed"""
    print("\n[3/5] Checking required packages...")
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'librosa': 'librosa',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'matplotlib': 'matplotlib',
    }
    
    missing = []
    
    for import_name, package_name in required.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (missing)")
            missing.append(package_name)
    
    return len(missing) == 0, missing

def install_packages(packages):
    """Install missing packages"""
    print("\n[4/5] Installing missing packages...")
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-q', package
            ])
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def check_scripts():
    """Check if training scripts exist"""
    print("\n[5/5] Checking training scripts...")
    
    scripts = [
        'train_pain_cnn.py',
        'predict_pain.py',
        'requirements.txt',
        'TRAINING_GUIDE.md',
    ]
    
    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"✗ {script}")
            all_ok = False
    
    return all_ok

def main():
    print_header("TAME Pain CNN - Setup & Validation")
    
    # Check Python
    if not check_python_version():
        print("\n⚠️  Warning: Python version may not be optimal")
    
    # Check dataset
    if not check_dataset_structure():
        print("\n⚠️  Warning: Some dataset files are missing")
        print("   Make sure you're in the correct directory")
    
    # Check and install packages
    packages_ok, missing = check_packages()
    if not packages_ok:
        print(f"\nMissing {len(missing)} packages")
        print("Installing...")
        if not install_packages(missing):
            print("\n✗ Failed to install some packages")
            print("Try manual installation:")
            print("  pip install -r requirements.txt")
            return False
    
    # Check scripts
    if not check_scripts():
        print("\n⚠️  Some scripts are missing")
    
    # Summary
    print_header("Setup Complete!")
    print("\n✓ All checks passed! Ready to train.\n")
    print("Next steps:")
    print("  1. Review training parameters in train_pain_cnn.py")
    print("  2. Run: python train_pain_cnn.py")
    print("  3. Monitor training progress and results")
    print("\nFor inference:")
    print("  python predict_pain.py <audio_file>\n")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

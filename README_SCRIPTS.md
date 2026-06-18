# TAME Pain CNN Training - Complete Setup

## Overview

Complete pipeline for training a CNN model to classify pain levels (Low, Medium, High) from audio mel spectrograms using the TAME Pain Dataset. The system includes training, inference, data exploration, and setup utilities.

## Files in This Folder

### Core Training Scripts

1. **`train_pain_cnn.py`** - Main training script
   - Loads audio files and metadata
   - Extracts mel spectrograms 
   - Builds and trains a CNN model
   - Evaluates on test set
   - Saves trained model and visualizations
   - **Classes**: Low (1-3), Medium (4-7), High (8-10) pain levels

2. **`predict_pain.py`** - Inference script
   - Load trained model
   - Predict pain levels from audio files
   - Visualize predictions and mel spectrograms
   - Batch prediction support

### Utility Scripts

3. **`setup.py`** - Environment setup and validation
   - Check Python version
   - Verify dataset structure
   - Install missing packages
   - Validate scripts

4. **`explore_data.py`** - Data exploration and analysis
   - Analyze pain distribution
   - Study audio characteristics
   - Generate demographic statistics
   - Create data visualization plots
   - Sample mel spectrograms

### Configuration Files

5. **`requirements.txt`** - Python dependencies
   - TensorFlow 2.10+
   - librosa (audio processing)
   - scikit-learn
   - NumPy, Pandas, Matplotlib

6. **`TRAINING_GUIDE.md`** - Comprehensive documentation
   - Installation instructions
   - Training parameters
   - Model architecture
   - Troubleshooting guide

## Quick Start

### 1. Setup Environment
```bash
python setup.py
```
This will:
- Verify your dataset structure
- Check Python version
- Install missing dependencies

### 2. Explore Dataset (Optional)
```bash
python explore_data.py
```
Generates:
- `data_distribution_analysis.png` - Pain/duration/condition distributions
- `pain_by_condition.png` - Pain levels by experimental condition
- `sample_spectrograms.png` - Example mel spectrograms

### 3. Train Model
```bash
python train_pain_cnn.py
```
The script will:
1. Load ~7000 audio files and extract mel spectrograms
2. Split data into train/val/test (70%/10%/20%)
3. Train CNN for ~50 epochs
4. Output test accuracy and metrics
5. Save trained model (`pain_cnn_model.h5`)
6. Generate plots: `training_history.png`, `confusion_matrix.png`

### 4. Make Predictions
```bash
python predict_pain.py path/to/audio.wav
```
Output:
- Predicted class (Low/Medium/High)
- Confidence score
- All class probabilities
- Visualization plot

## Model Architecture

**CNN with 3 blocks:**
```
Input (128x121 mel spectrogram)
  ↓
Conv2D(32) + BatchNorm + MaxPool + Dropout(0.3)
  ↓
Conv2D(64) + BatchNorm + MaxPool + Dropout(0.3)
  ↓
Conv2D(128) + BatchNorm + MaxPool + Dropout(0.3)
  ↓
GlobalAveragePooling2D
  ↓
Dense(256) + BatchNorm + Dropout(0.4)
  ↓
Dense(128) + BatchNorm + Dropout(0.4)
  ↓
Dense(3, softmax) → Output [Low, Medium, High]
```

## Key Parameters

Located in `CONFIG` dictionary in `train_pain_cnn.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `sample_rate` | 16000 Hz | Audio sampling rate |
| `n_mels` | 128 | Mel frequency bins |
| `duration` | 4.0 sec | Fixed audio length |
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Batch size |
| `test_split` | 0.2 | 20% test set |

## Expected Performance

Based on dataset characteristics:
- **Dataset size**: ~7,000 audio samples across 51 participants
- **Training time**: ~5-15 minutes (depending on hardware)
- **Expected accuracy**: 65-80% (3-way classification is challenging)
- **Imbalanced classes**: Medium pain (4-7) will have more samples

## Output Files

After training:
```
pain_cnn_training/
├── pain_cnn_model.h5              # Trained model weights
├── training_history.png           # Loss/accuracy curves
└── confusion_matrix.png           # Test set performance
```

## Customization

### Change Pain Classes
Edit `PAIN_CLASSES` in `train_pain_cnn.py`:
```python
PAIN_CLASSES = {
    'low': (1, 3),
    'medium': (4, 7),
    'high': (8, 10),
}
```

### Adjust Training
Edit `CONFIG`:
```python
CONFIG = {
    'epochs': 100,           # More training
    'batch_size': 16,        # Smaller batches
    'n_mels': 256,          # More features
    'duration': 5.0,         # Longer audio
}
```

### Different Audio Features
Replace mel spectrogram with:
- MFCCs: `librosa.feature.mfcc()`
- Chroma: `librosa.feature.chroma_stft()`
- Spectral contrast: `librosa.feature.spectral_contrast()`

## Troubleshooting

**Problem**: "Model not found" when predicting
- Make sure you trained the model first: `python train_pain_cnn.py`

**Problem**: Low accuracy
- Increase epochs in CONFIG
- Try different `n_mels` (256 or 512)
- Use data augmentation (add noise, time-stretch)
- Adjust class boundaries

**Problem**: Out of memory
- Reduce `batch_size` (e.g., 16)
- Reduce `n_mels` (e.g., 64)
- Use fewer samples for development

**Problem**: Very slow training
- GPU not detected: Install TensorFlow GPU version
- Reduce `n_mels` or `duration`
- Use smaller training set for testing

## Next Steps

1. Run `python setup.py` to validate everything
2. Run `python explore_data.py` to understand the data
3. Run `python train_pain_cnn.py` to train the model
4. Run `python predict_pain.py test_audio.wav` to make predictions
5. Experiment with different parameters to improve accuracy

## Dataset Information

**TAME Pain Dataset** (Toward Affective Machine Emotion - Pain):
- 51 participants, 7,039 utterances, 311 minutes of audio
- Self-reported pain levels (1-10 scale)
- 4 experimental conditions: LC, LW, RC, RW (cold/warm, left/right)
- Annotated with audio quality and disturbance markers
- Available: https://physionet.org/

## References

- **TensorFlow/Keras**: https://tensorflow.org/
- **Librosa**: https://librosa.org/doc/latest/index.html
- **Mel Spectrograms**: https://en.wikipedia.org/wiki/Mel-scale
- **CNN Image Classification**: https://cs231n.github.io/

---

Created for TAME Pain Dataset Classification

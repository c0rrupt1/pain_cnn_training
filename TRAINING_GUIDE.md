# TAME Pain CNN Classification

Train and evaluate a CNN model for 3-class pain level classification from audio mel spectrograms using the TAME Pain Dataset.

## Dataset Structure

The script expects the parent directory to contain:

```
tame/
├── pain_cnn_training/      # This folder
│   ├── train_pain_cnn.py
│   ├── predict_pain.py
│   ├── setup.py
│   ├── explore_data.py
│   ├── requirements.txt
│   └── TRAINING_GUIDE.md
├── mic1_trim_v2/           # Audio files (parent dir)
├── meta_audio.csv          # Audio metadata (parent dir)
└── meta_participant.csv    # Participant data (parent dir)
```

## Installation

1. **Create a Python environment** (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python train_pain_cnn.py
```

The script will:
1. Load the metadata and audio files
2. Extract mel spectrograms from audio
3. Create 3 pain classes:
   - **Low**: Pain levels 1-3
   - **Medium**: Pain levels 4-7
   - **High**: Pain levels 8-10
4. Build and train a CNN model
5. Evaluate on test set
6. Save the trained model (`pain_cnn_model.h5`)
7. Generate training history and confusion matrix plots

### Training Parameters

Edit `CONFIG` in `train_pain_cnn.py` to adjust:
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 32)
- `test_split`: Test set proportion (default: 0.2)
- `val_split`: Validation set proportion (default: 0.1)
- `n_mels`: Number of mel frequency bins (default: 128)
- `duration`: Max audio duration in seconds (default: 4.0)

### Model Architecture

The CNN model consists of:
- 3 convolutional blocks (32 → 64 → 128 filters)
- Batch normalization and dropout for regularization
- Global average pooling
- 2 dense layers (256, 128 units)
- Output softmax layer (3 classes)

## Inference

Use the trained model to predict pain levels:

```bash
python predict_pain.py <audio_file>
```

Example:
```bash
python predict_pain.py ../mic1_trim_v2/p10085/p10085.LC.1.161.wav
```

The script will output:
- Predicted pain class (Low/Medium/High)
- Confidence score
- All class probabilities
- Visualization (mel spectrogram + predictions)

## Data Exploration

Analyze the dataset before training:

```bash
python explore_data.py
```

Generates:
- `data_distribution_analysis.png` - Pain/duration/condition distributions
- `pain_by_condition.png` - Pain levels by experimental condition
- `sample_spectrograms.png` - Example mel spectrograms

## Output Files

After training, the script generates:
- `pain_cnn_model.h5` - Trained model weights
- `training_history.png` - Loss and accuracy plots
- `confusion_matrix.png` - Test set confusion matrix

## Customization

### Modify Pain Class Boundaries

Edit the `PAIN_CLASSES` dictionary in `train_pain_cnn.py`:
```python
PAIN_CLASSES = {
    'low': (1, 3),
    'medium': (4, 7),
    'high': (8, 10),
}
```

### Use Different Audio Features

Replace mel spectrogram extraction with other features:
- MFCCs: `librosa.feature.mfcc()`
- Chroma: `librosa.feature.chroma_stft()`
- Spectral features: `librosa.feature.spectral_centroid()`

## Troubleshooting

**Issue**: "Metadata file not found"
- Ensure you're running from the `pain_cnn_training` folder
- Check that parent directory has `meta_audio.csv`

**Issue**: "No valid audio files found"
- Check that `../mic1_trim_v2/` directory exists with audio files
- Verify file naming matches: `PID.COND.UTTNUM.UTTID.wav`

**Issue**: Out of memory
- Reduce `batch_size` in CONFIG
- Reduce number of samples loaded at once

**Issue**: Low accuracy
- Increase `epochs`
- Adjust learning rate in model compilation
- Try different `n_mels` values

## References

TAME Pain Dataset: https://physionet.org/

Librosa documentation: https://librosa.org/

TensorFlow/Keras documentation: https://tensorflow.org/

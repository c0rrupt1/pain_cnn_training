"""
Pain Classification Inference Script
Uses trained CNN model to classify pain levels from audio files
"""

import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

CONFIG = {
    'sample_rate': 16000,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'duration': 4.0,
    'model_path': 'pain_cnn_model.h5',
}

CLASS_NAMES = ['Low Pain (1-3)', 'Medium Pain (4-7)', 'High Pain (8-10)']
CLASS_COLORS = ['green', 'yellow', 'red']


def load_mel_spectrogram(audio_path, config):
    """Load audio file and extract mel spectrogram"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=config['sample_rate'])
        
        # Pad or trim to fixed length
        max_samples = int(config['duration'] * config['sample_rate'])
        if len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)), mode='constant', value=0)
        else:
            y = y[:max_samples]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=config['sample_rate'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            n_mels=config['n_mels']
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None


def predict_pain_level(audio_path, model, scaler=None, config=None):
    """Predict pain level from audio file"""
    
    if config is None:
        config = CONFIG
    
    # Load mel spectrogram
    mel_spec = load_mel_spectrogram(audio_path, config)
    if mel_spec is None:
        return None
    
    # Normalize
    X = mel_spec[np.newaxis, ...]
    if scaler is not None:
        original_shape = X.shape
        X = scaler.transform(X.reshape(X.shape[0], -1))
        X = X.reshape(original_shape)
    else:
        # Simple normalization if no scaler
        X = (X - X.mean()) / (X.std() + 1e-8)
    
    # Predict
    predictions = model.predict(X, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return {
        'audio_path': audio_path,
        'predicted_class': predicted_class,
        'predicted_class_name': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'all_predictions': {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        },
        'mel_spectrogram': mel_spec
    }


def visualize_prediction(result):
    """Visualize mel spectrogram and predictions"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot mel spectrogram
    mel_spec = result['mel_spectrogram']
    im = axes[0].imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mel Frequency Bin')
    axes[0].set_title('Mel Spectrogram')
    plt.colorbar(im, ax=axes[0])
    
    # Plot predictions
    classes = list(result['all_predictions'].keys())
    probs = list(result['all_predictions'].values())
    colors = [CLASS_COLORS[i] for i in range(len(classes))]
    
    bars = axes[1].bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Probability')
    axes[1].set_title(f"Predictions - Predicted: {result['predicted_class_name']}")
    axes[1].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


def batch_predict(audio_dir, model, config=None):
    """Predict pain levels for all audio files in a directory"""
    
    if config is None:
        config = CONFIG
    
    results = []
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    print(f"Processing {len(audio_files)} audio files...")
    
    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        result = predict_pain_level(audio_path, model, config=config)
        if result is not None:
            results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(audio_files)} files")
    
    return results


if __name__ == '__main__':
    import sys
    
    # Check if model exists
    if not os.path.exists(CONFIG['model_path']):
        print(f"Error: Model file '{CONFIG['model_path']}' not found!")
        print("Train the model first using train_pain_cnn.py")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {CONFIG['model_path']}...")
    model = keras.models.load_model(CONFIG['model_path'])
    print("Model loaded successfully!")
    
    # Example: Predict on single audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if os.path.exists(audio_file):
            print(f"\nAnalyzing: {audio_file}")
            result = predict_pain_level(audio_file, model, config=CONFIG)
            
            if result:
                print(f"Predicted class: {result['predicted_class_name']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("\nAll predictions:")
                for class_name, prob in result['all_predictions'].items():
                    print(f"  {class_name}: {prob:.4f}")
                
                # Visualize
                fig = visualize_prediction(result)
                plt.savefig(f"prediction_{os.path.basename(audio_file)}.png", dpi=100)
                print(f"\nVisualization saved to prediction_{os.path.basename(audio_file)}.png")
                plt.show()
        else:
            print(f"Error: File not found - {audio_file}")
    else:
        print("\nUsage: python predict_pain.py <audio_file>")
        print("\nExample:")
        print("  python predict_pain.py audio_sample.wav")

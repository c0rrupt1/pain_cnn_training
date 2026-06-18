"""
TAME Pain Dataset - CNN Mel Spectrogram Classification
3-Class Pain Level Classification (Low, Medium, High)
"""

import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'dataset_path': '../mic1_trim_v2',
    'metadata_path': '../meta_audio.csv',
    'sample_rate': 16000,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'duration': 4.0,  # Max duration in seconds
    'batch_size': 32,
    'epochs': 50,
    'test_split': 0.2,
    'val_split': 0.1,
    'random_state': 42,
}

# Pain level mapping to 3 classes
PAIN_CLASSES = {
    'low': (1, 3),      # Pain level 1-3
    'medium': (4, 7),   # Pain level 4-7
    'high': (8, 10),    # Pain level 8-10
}

class PainAudioDataset:
    """Handles dataset loading and preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.metadata = None
        self.audio_paths = []
        self.pain_labels = []
        self.class_labels = []
        self.load_metadata()
        
    def load_metadata(self):
        """Load audio metadata CSV"""
        metadata_file = self.config['metadata_path']
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        self.metadata = pd.read_csv(metadata_file)
        print(f"Loaded metadata: {len(self.metadata)} audio files")
        print(f"Pain level range: {self.metadata['REVISED PAIN'].min()}-{self.metadata['REVISED PAIN'].max()}")
        
    def get_pain_class(self, pain_level):
        """Map pain level (1-10) to class (0=low, 1=medium, 2=high)"""
        # Handle missing or NaN pain levels
        if pd.isna(pain_level):
            return None, None
        pain_level = int(pain_level)
        if PAIN_CLASSES['low'][0] <= pain_level <= PAIN_CLASSES['low'][1]:
            return 0, 'low'
        elif PAIN_CLASSES['medium'][0] <= pain_level <= PAIN_CLASSES['medium'][1]:
            return 1, 'medium'
        elif PAIN_CLASSES['high'][0] <= pain_level <= PAIN_CLASSES['high'][1]:
            return 2, 'high'
        else:
            return None, None
    
    def prepare_dataset(self):
        """Prepare dataset by validating file existence and mapping labels"""
        valid_data = []
        missing_files = 0
        
        for idx, row in self.metadata.iterrows():
            pid = row['PID']
            cond = row['COND']
            uttnum = row['UTTNUM']
            uttid = int(row['UTTID'])
            pain_level = row['REVISED PAIN']
            # Skip entries without a valid pain label
            if pd.isna(pain_level):
                continue
            
            # Construct audio file path
            audio_file = f"{pid}.{cond}.{int(uttnum)}.{uttid}.wav"
            audio_path = os.path.join(self.config['dataset_path'], pid, audio_file)
            
            # Check if file exists
            if not os.path.exists(audio_path):
                missing_files += 1
                continue
            
            # Get pain class
            class_idx, class_name = self.get_pain_class(pain_level)
            if class_idx is None:
                continue
            
            valid_data.append({
                'audio_path': audio_path,
                'pain_level': int(pain_level),
                'class_idx': class_idx,
                'class_name': class_name,
                'pid': pid,
                'condition': cond,
            })
        
        print(f"Valid audio files found: {len(valid_data)}")
        print(f"Missing files: {missing_files}")
        
        # Print class distribution
        class_counts = pd.DataFrame(valid_data)['class_name'].value_counts()
        print("\nClass distribution:")
        print(class_counts)
        
        return pd.DataFrame(valid_data)
    
    def load_audio(self, audio_path):
        """Load audio file and extract mel spectrogram"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            
            # Pad or trim to fixed length
            max_samples = int(self.config['duration'] * self.config['sample_rate'])
            if len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant', value=0)
            else:
                y = y[:max_samples]
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.config['sample_rate'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length'],
                n_mels=self.config['n_mels']
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None


class CNNModel:
    """CNN model for pain classification"""
    
    @staticmethod
    def build(input_shape):
        """Build CNN model"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Expand dims for channel dimension
            layers.Reshape((*input_shape, 1)),
            
            # Block 1
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 2
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 3
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output layer (3 classes)
            layers.Dense(3, activation='softmax')
        ])
        
        return model


def train_pain_classifier(config):
    """Main training pipeline"""
    
    print("=" * 60)
    print("TAME Pain Dataset - CNN Mel Spectrogram Classification")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = PainAudioDataset(config)
    df = dataset.prepare_dataset()
    
    if len(df) == 0:
        raise ValueError("No valid audio files found!")
    
    # Load and extract mel spectrograms
    print("\n[2/5] Extracting mel spectrograms...")
    X = []
    y = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        mel_spec = dataset.load_audio(row['audio_path'])
        if mel_spec is not None:
            X.append(mel_spec)
            y.append(row['class_idx'])
            valid_indices.append(idx)
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1} files...")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Successfully loaded {len(X)} mel spectrograms")
    print(f"Mel spectrogram shape: {X.shape}")
    
    # Split dataset
    print("\n[3/5] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_split'], random_state=config['random_state'],
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config['val_split'], random_state=config['random_state'],
        stratify=y_train
    )
    
    # Normalize spectrograms
    print("Normalizing spectrograms...")
    scaler = StandardScaler()
    X_train_shape = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_train = X_train.reshape(X_train_shape)
    
    X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    print("\n[4/5] Building CNN model...")
    model = CNNModel.build(input_shape=X_train.shape[1:])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train model
    print("\n[5/5] Training model...")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    class_names = ['Low (1-3)', 'Medium (4-7)', 'High (8-10)']
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model
    print("\n" + "=" * 60)
    model_path = 'pain_cnn_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    print("Saving training history plot...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100)
    print("Training history saved to training_history.png")
    
    # Plot confusion matrix
    print("Saving confusion matrix plot...")
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.xticks(range(3), class_names, rotation=45)
    plt.yticks(range(3), class_names)
    
    # Add values to cells
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    print("Confusion matrix saved to confusion_matrix.png")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return model, history


if __name__ == '__main__':
    # Check if required libraries are installed
    required_packages = ['librosa', 'tensorflow', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        print("\nInstall packages and run again.")
        exit(1)
    
    # Train model
    model, history = train_pain_classifier(CONFIG)

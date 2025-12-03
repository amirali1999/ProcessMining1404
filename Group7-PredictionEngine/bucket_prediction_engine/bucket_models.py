"""
فاز ۲: Bucket-Based Sequence Models
Phase 2: Separate LSTM/GRU model for each prefix length bucket

Key Innovation: Each bucket has its own dedicated model with fixed sequence length
- Bucket "1": Model for prefix length 1
- Bucket "2": Model for prefix length 2
- ...
- Bucket "10+": Model for prefix length >= 10
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from typing import Dict, List, Tuple


class BucketLSTMModel:
    """
    گام ۷.۲: LSTM/GRU model for a single bucket
    
    This model handles sequences of fixed length (one bucket)
    """
    
    def __init__(self, bucket_id: str, prefix_length: int, vocab_size: int,
                 embedding_dim: int = 32, units: int = 64, model_type: str = 'gru'):
        """
        Initialize bucket-specific model
        
        Args:
            bucket_id: Bucket identifier (e.g., "3" or "10+")
            prefix_length: Fixed prefix length for this bucket
            vocab_size: Size of activity vocabulary
            embedding_dim: Embedding dimension
            units: Number of LSTM/GRU units
            model_type: 'lstm' or 'gru'
        """
        self.bucket_id = bucket_id
        # Clean bucket_id for model naming (replace + with _plus)
        self.bucket_name = bucket_id.replace('+', '_plus')
        self.prefix_length = prefix_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.model_type = model_type
        
        # Models for next activity and remaining time
        self.next_activity_model = None
        self.remaining_time_model = None
        
        self.is_trained = False
        
    def build_next_activity_model(self):
        """
        گام ۷.۲: Build next activity prediction model
        
        Architecture:
        - Embedding layer
        - GRU/LSTM layer
        - Dense output (softmax for activity classification)
        """
        print(f"\n🔨 Building Next Activity model for Bucket {self.bucket_id} (length={self.prefix_length})")
        
        model = Sequential(name=f'next_activity_bucket_{self.bucket_name}')
        
        # Embedding
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.prefix_length,
            name='embedding'
        ))
        
        # Sequence layer (LSTM or GRU)
        if self.model_type == 'lstm':
            model.add(LSTM(self.units, name='lstm'))
        else:
            model.add(GRU(self.units, name='gru'))
        
        # Output
        model.add(Dense(self.vocab_size, activation='softmax', name='output'))
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.next_activity_model = model
        
        return model
    
    def build_remaining_time_model(self):
        """
        گام ۷.۳: Build remaining time prediction model
        
        Architecture:
        - Embedding layer
        - LSTM layer
        - Dense output (linear for time regression)
        """
        print(f"\n🔨 Building Remaining Time model for Bucket {self.bucket_id} (length={self.prefix_length})")
        
        model = Sequential(name=f'remaining_time_bucket_{self.bucket_name}')
        
        # Embedding
        model.add(Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.prefix_length,
            name='embedding'
        ))
        
        # LSTM layer
        model.add(LSTM(self.units, name='lstm'))
        
        # Output (regression)
        model.add(Dense(1, activation='linear', name='output'))
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='mae',
            metrics=['mae', 'mse']
        )
        
        self.remaining_time_model = model
        
        return model
    
    def train_next_activity(self, X, y, epochs=10, batch_size=64, validation_split=0.1):
        """
        Train next activity model for this bucket
        
        Args:
            X: Encoded sequences (num_samples, prefix_length)
            y: Next activity labels (num_samples,)
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        if self.next_activity_model is None:
            self.build_next_activity_model()
        
        print(f"\n🚀 Training Next Activity model for Bucket {self.bucket_id}")
        print(f"   Samples: {len(X)}, Prefix length: {self.prefix_length}")
        
        # Class weights for imbalanced data
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
        ]
        
        # Train
        history = self.next_activity_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0
        )
        
        # Final metrics
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history.get('val_accuracy', [0])[-1]
        print(f"   ✅ Final accuracy: {final_acc:.4f}, Val accuracy: {final_val_acc:.4f}")
        
        return history
    
    def train_remaining_time(self, X, y, epochs=10, batch_size=64, validation_split=0.1):
        """
        Train remaining time model for this bucket
        
        Args:
            X: Encoded sequences (num_samples, prefix_length)
            y: Remaining time labels (num_samples,)
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        if self.remaining_time_model is None:
            self.build_remaining_time_model()
        
        print(f"\n🚀 Training Remaining Time model for Bucket {self.bucket_id}")
        print(f"   Samples: {len(X)}, Prefix length: {self.prefix_length}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
        ]
        
        # Train
        history = self.remaining_time_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        # Final metrics
        final_mae = history.history['mae'][-1]
        final_val_mae = history.history.get('val_mae', [0])[-1]
        print(f"   ✅ Final MAE: {final_mae:.4f}, Val MAE: {final_val_mae:.4f}")
        
        return history
    
    def evaluate_next_activity(self, X_test, y_test):
        """Evaluate next activity model"""
        if self.next_activity_model is None:
            raise ValueError("Next activity model not trained")
        
        loss, acc, top3 = self.next_activity_model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred_proba = self.next_activity_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return {
            'loss': loss,
            'accuracy': acc,
            'top_3_accuracy': top3,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def evaluate_remaining_time(self, X_test, y_test):
        """Evaluate remaining time model"""
        if self.remaining_time_model is None:
            raise ValueError("Remaining time model not trained")
        
        loss, mae, mse = self.remaining_time_model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = self.remaining_time_model.predict(X_test, verbose=0).flatten()
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'predictions': y_pred
        }
    
    def predict_next_activity(self, X, return_proba=False):
        """Predict next activity"""
        if self.next_activity_model is None:
            raise ValueError("Next activity model not trained")
        
        proba = self.next_activity_model.predict(X, verbose=0)
        
        if return_proba:
            return proba
        else:
            return np.argmax(proba, axis=1)
    
    def predict_remaining_time(self, X):
        """Predict remaining time"""
        if self.remaining_time_model is None:
            raise ValueError("Remaining time model not trained")
        
        return self.remaining_time_model.predict(X, verbose=0).flatten()
    
    def save(self, output_dir: str):
        """Save bucket models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        if self.next_activity_model:
            self.next_activity_model.save(os.path.join(output_dir, f'next_activity_bucket_{self.bucket_id}.h5'))
        
        if self.remaining_time_model:
            self.remaining_time_model.save(os.path.join(output_dir, f'remaining_time_bucket_{self.bucket_id}.h5'))
        
        # Save metadata
        metadata = {
            'bucket_id': self.bucket_id,
            'prefix_length': self.prefix_length,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'units': self.units,
            'model_type': self.model_type
        }
        joblib.dump(metadata, os.path.join(output_dir, f'bucket_{self.bucket_id}_metadata.pkl'))
    
    def load(self, output_dir: str):
        """Load bucket models"""
        # Load metadata
        metadata = joblib.load(os.path.join(output_dir, f'bucket_{self.bucket_id}_metadata.pkl'))
        self.prefix_length = metadata['prefix_length']
        self.vocab_size = metadata['vocab_size']
        self.embedding_dim = metadata['embedding_dim']
        self.units = metadata['units']
        self.model_type = metadata['model_type']
        
        # Load models
        next_act_path = os.path.join(output_dir, f'next_activity_bucket_{self.bucket_id}.h5')
        if os.path.exists(next_act_path):
            self.next_activity_model = keras.models.load_model(next_act_path)
        
        rem_time_path = os.path.join(output_dir, f'remaining_time_bucket_{self.bucket_id}.h5')
        if os.path.exists(rem_time_path):
            self.remaining_time_model = keras.models.load_model(rem_time_path)


class BucketEnsemble:
    """
    Manager for all bucket models
    Handles routing predictions to the correct bucket
    """
    
    def __init__(self, vocab_size: int, max_bucket: int = 10):
        """
        Initialize bucket ensemble
        
        Args:
            vocab_size: Size of activity vocabulary
            max_bucket: Maximum bucket size
        """
        self.vocab_size = vocab_size
        self.max_bucket = max_bucket
        self.bucket_models: Dict[str, BucketLSTMModel] = {}
        
    def prepare_bucket_data(self, buckets: Dict[str, List[dict]], activity_encoder):
        """
        گام ۷.۲: Prepare training data for each bucket
        
        Args:
            buckets: Dictionary of bucketed prefix samples
            activity_encoder: ActivityEncoder instance
            
        Returns:
            Dictionary of prepared data for each bucket
        """
        print("\n📦 Preparing data for each bucket...")
        
        bucket_data = {}
        
        for bucket_id, samples in buckets.items():
            # Determine prefix length
            if bucket_id.endswith('+'):
                prefix_length = self.max_bucket
            else:
                prefix_length = int(bucket_id)
            
            # Encode sequences
            X_next = []
            y_next = []
            X_time = []
            y_time = []
            
            for sample in samples:
                # Encode prefix
                encoded = activity_encoder.encode_prefix(sample['prefix_activities'])
                
                # Pad to fixed length (in case of bucket "10+")
                if len(encoded) < prefix_length:
                    encoded = [0] * (prefix_length - len(encoded)) + encoded
                elif len(encoded) > prefix_length:
                    encoded = encoded[:prefix_length]
                
                # Next activity
                if sample['label_next_activity']:
                    X_next.append(encoded)
                    y_next.append(activity_encoder.encode_activity(sample['label_next_activity']))
                
                # Remaining time
                if sample['remaining_time'] is not None:
                    X_time.append(encoded)
                    y_time.append(sample['remaining_time'])
            
            # Convert to numpy arrays
            X_next = np.array(X_next, dtype='int32')
            y_next = np.array(y_next, dtype='int32')
            X_time = np.array(X_time, dtype='int32')
            y_time = np.array(y_time, dtype='float32')
            
            bucket_data[bucket_id] = {
                'prefix_length': prefix_length,
                'next_activity': (X_next, y_next),
                'remaining_time': (X_time, y_time)
            }
            
            print(f"   Bucket {bucket_id}: {len(X_next)} samples, prefix_length={prefix_length}")
        
        return bucket_data
    
    def train_all_buckets(self, bucket_data: Dict, epochs=10, batch_size=64):
        """
        Train models for all buckets
        
        Args:
            bucket_data: Prepared bucket data from prepare_bucket_data()
            epochs: Training epochs
            batch_size: Batch size
        """
        print(f"\n{'='*70}")
        print("TRAINING BUCKET MODELS")
        print(f"{'='*70}")
        
        for bucket_id, data in bucket_data.items():
            print(f"\n--- Bucket {bucket_id} ---")
            
            # Create bucket model
            model = BucketLSTMModel(
                bucket_id=bucket_id,
                prefix_length=data['prefix_length'],
                vocab_size=self.vocab_size,
                embedding_dim=32,
                units=64,
                model_type='gru'
            )
            
            # Train next activity
            X_next, y_next = data['next_activity']
            if len(X_next) > 0:
                model.train_next_activity(X_next, y_next, epochs=epochs, batch_size=batch_size)
            
            # Train remaining time
            X_time, y_time = data['remaining_time']
            if len(X_time) > 0:
                model.train_remaining_time(X_time, y_time, epochs=epochs, batch_size=batch_size)
            
            # Store model
            self.bucket_models[bucket_id] = model
        
        print(f"\n✅ Trained {len(self.bucket_models)} bucket models")
    
    def evaluate_all_buckets(self, bucket_data: Dict):
        """
        گام ۸: Evaluate all bucket models
        
        Args:
            bucket_data: Test data for each bucket
            
        Returns:
            Dictionary of evaluation results
        """
        print(f"\n{'='*70}")
        print("EVALUATING BUCKET MODELS")
        print(f"{'='*70}")
        
        results = {}
        
        for bucket_id, model in self.bucket_models.items():
            print(f"\n--- Bucket {bucket_id} ---")
            
            data = bucket_data[bucket_id]
            
            # Evaluate next activity
            X_next, y_next = data['next_activity']
            if len(X_next) > 0:
                # Split for testing
                split_idx = int(len(X_next) * 0.8)
                X_test = X_next[split_idx:]
                y_test = y_next[split_idx:]
                
                if len(X_test) > 0:
                    next_results = model.evaluate_next_activity(X_test, y_test)
                    print(f"   Next Activity - Accuracy: {next_results['accuracy']:.4f}, "
                          f"Top-3: {next_results['top_3_accuracy']:.4f}")
            
            # Evaluate remaining time
            X_time, y_time = data['remaining_time']
            if len(X_time) > 0:
                split_idx = int(len(X_time) * 0.8)
                X_test = X_time[split_idx:]
                y_test = y_time[split_idx:]
                
                if len(X_test) > 0:
                    time_results = model.evaluate_remaining_time(X_test, y_test)
                    print(f"   Remaining Time - MAE: {time_results['mae']:.4f}")
            
            results[bucket_id] = {
                'next_activity': next_results if len(X_next) > 0 else None,
                'remaining_time': time_results if len(X_time) > 0 else None
            }
        
        return results
    
    def predict(self, prefix_sequence: List[int], activity_encoder):
        """
        گام ۹.۲: Make prediction by routing to appropriate bucket
        
        Args:
            prefix_sequence: Encoded activity sequence
            activity_encoder: ActivityEncoder for decoding
            
        Returns:
            Dictionary with predictions
        """
        # Determine bucket
        prefix_length = len(prefix_sequence)
        
        if prefix_length > self.max_bucket:
            bucket_id = f"{self.max_bucket}+"
            target_length = self.max_bucket
        else:
            bucket_id = str(prefix_length)
            target_length = prefix_length
        
        # Get bucket model
        if bucket_id not in self.bucket_models:
            raise ValueError(f"No model found for bucket {bucket_id}")
        
        model = self.bucket_models[bucket_id]
        
        # Prepare input
        sequence = prefix_sequence.copy()
        if len(sequence) < target_length:
            sequence = [0] * (target_length - len(sequence)) + sequence
        elif len(sequence) > target_length:
            sequence = sequence[:target_length]
        
        X = np.array([sequence], dtype='int32')
        
        # Predict
        next_activity_proba = model.predict_next_activity(X, return_proba=True)[0]
        remaining_time = model.predict_remaining_time(X)[0]
        
        # Get top-k activities
        top_k_indices = np.argsort(next_activity_proba)[-5:][::-1]
        top_k_candidates = [
            {
                'activity': activity_encoder.decode_activity(int(idx)),
                'prob': float(next_activity_proba[idx])
            }
            for idx in top_k_indices
        ]
        
        return {
            'bucket_id': bucket_id,
            'predicted_next_activity': top_k_candidates[0]['activity'],
            'top_k_candidates': top_k_candidates,
            'predicted_remaining_time': float(remaining_time)
        }
    
    def save(self, output_dir: str):
        """Save all bucket models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for bucket_id, model in self.bucket_models.items():
            model.save(output_dir)
        
        # Save ensemble metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'max_bucket': self.max_bucket,
            'bucket_ids': list(self.bucket_models.keys())
        }
        joblib.dump(metadata, os.path.join(output_dir, 'ensemble_metadata.pkl'))
        
        print(f"\n💾 Saved {len(self.bucket_models)} bucket models to {output_dir}")
    
    def load(self, output_dir: str):
        """Load all bucket models"""
        # Load metadata
        metadata = joblib.load(os.path.join(output_dir, 'ensemble_metadata.pkl'))
        self.vocab_size = metadata['vocab_size']
        self.max_bucket = metadata['max_bucket']
        
        # Load each bucket model
        for bucket_id in metadata['bucket_ids']:
            # Determine prefix length
            if bucket_id.endswith('+'):
                prefix_length = self.max_bucket
            else:
                prefix_length = int(bucket_id)
            
            model = BucketLSTMModel(
                bucket_id=bucket_id,
                prefix_length=prefix_length,
                vocab_size=self.vocab_size
            )
            model.load(output_dir)
            self.bucket_models[bucket_id] = model
        
        print(f"✅ Loaded {len(self.bucket_models)} bucket models from {output_dir}")

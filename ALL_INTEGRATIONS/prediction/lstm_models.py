"""
Phase 2: LSTM Models for Next Activity and Remaining Time Prediction
Implements deep learning models for sequence-based predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pickle
import os


class StabilityEarlyStopping(keras.callbacks.Callback):
    """
    Stop training if metrics degrade significantly.
    Policy:
    - Stop if accuracy drops by > threshold (default 0.07)
    - Stop if loss increases by > threshold (default 0.2)
    - Patience: Stop after N violations
    """
    def __init__(self, monitor_acc='val_accuracy', monitor_loss='val_loss', 
                 acc_threshold=-0.07, loss_threshold=0.2, patience=3):
        super().__init__()
        self.monitor_acc = monitor_acc
        self.monitor_loss = monitor_loss
        self.acc_threshold = acc_threshold
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.violation_count = 0
        self.prev_acc = None
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get(self.monitor_acc)
        current_loss = logs.get(self.monitor_loss)
        
        violation = False
        
        if epoch > 0 and self.prev_acc is not None and self.prev_loss is not None:
            # Check accuracy drop (if accuracy is available)
            if current_acc is not None and self.prev_acc is not None:
                acc_change = current_acc - self.prev_acc
                if acc_change < self.acc_threshold:
                    print(f"\nWarning: Accuracy dropped by {acc_change:.4f} (Threshold: {self.acc_threshold})")
                    violation = True
            
            # Check loss increase
            if current_loss is not None and self.prev_loss is not None:
                loss_change = current_loss - self.prev_loss
                if loss_change > self.loss_threshold:
                    print(f"\nWarning: Loss increased by {loss_change:.4f} (Threshold: {self.loss_threshold})")
                    violation = True
        
        if violation:
            self.violation_count += 1
            print(f"Stability violation {self.violation_count}/{self.patience}")
            if self.violation_count >= self.patience:
                print(f"\nStopping early due to instability ({self.patience} consecutive violations)")
                self.model.stop_training = True
        else:
            self.violation_count = 0
        
        self.prev_acc = current_acc
        self.prev_loss = current_loss


# Focal Loss for better handling of imbalanced data
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss - focuses training on hard examples
    Better than cross-entropy for imbalanced datasets
    
    Args:
        gamma: Focusing parameter (default=2.0, higher=focus more on hard examples)
        alpha: Balancing parameter (default=0.25)
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Clip to prevent log(0)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # One-hot encode if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1) if len(y_true.shape) > 1 else y_true
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
            y_true = tf.cast(y_true, y_pred.dtype)
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Focal loss formula: FL = -alpha * (1-p)^gamma * log(p)
        focal_weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = focal_weight * ce
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


class NextActivityLSTM:
    """
    LSTM model for predicting the next activity in a process
    Improved with Focal Loss and stronger architecture
    """
    
    def __init__(self, vocab_size: int, max_length: int, embedding_dim: int = 128, 
                 lstm_units: int = 256):
        """
        Initialize the next activity prediction model
        
        Args:
            vocab_size: Size of activity vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of activity embeddings (increased to 128)
            lstm_units: Number of LSTM units (increased to 256)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.is_trained = False
        
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM architecture"""
        print(f"\n=== Building Next Activity LSTM Model ===")
        print(f"Vocab size: {self.vocab_size}, Max length: {self.max_length}")
        
        # Input
        input_layer = layers.Input(shape=(self.max_length,), name='activity_sequence')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='activity_embedding'
        )(input_layer)
        
        # Bidirectional LSTM layers for better context
        lstm1 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2,
                name='lstm_1'
            ),
            name='bidirectional_1'
        )(embedding)
        
        lstm2 = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units // 2,
                dropout=0.3,
                recurrent_dropout=0.2,
                name='lstm_2'
            ),
            name='bidirectional_2'
        )(lstm1)
        
        # Dense layers with stronger dropout for regularization
        dense1 = layers.Dense(128, activation='relu', name='dense_1')(lstm2)
        dropout1 = layers.Dropout(0.4)(dense1)
        dense2 = layers.Dense(64, activation='relu', name='dense_2')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense2)
        
        # Output layer (softmax for classification)
        output = layers.Dense(self.vocab_size, activation='softmax', name='output')(dropout2)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output, name='next_activity_lstm')
        
        # Compile with Focal Loss for better imbalanced data handling
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=FocalLoss(gamma=2.0, alpha=0.25),  # Using Focal Loss!
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            ]
        )
        
        print(self.model.summary())
        print("\n✨ Using Focal Loss for better minority class learning!")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=64, class_weight=None):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            class_weight: Dictionary of class weights for imbalanced data
        """
        print(f"\n=== Training Next Activity LSTM ===")
        print(f"Training samples: {len(X_train)}")
        
        # If no class weights provided, calculate them for imbalanced data
        if class_weight is None:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights_array = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            class_weight = dict(zip(classes, class_weights_array))
            print(f"Using balanced class weights (showing first 5):")
            for cls in sorted(class_weight.keys())[:5]:
                print(f"  Class {cls}: {class_weight[cls]:.4f}")
        
        # Callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,  # Increased from 15 for better convergence
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,  # Increased from 7
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath="best_next_activity_model.keras",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            StabilityEarlyStopping(
                monitor_acc='val_accuracy',
                monitor_loss='val_loss',
                acc_threshold=-0.07,
                loss_threshold=0.2,
                patience=3
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train with class weights
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weight,  # This is key for imbalanced data!
            verbose=1
        )
        
        self.is_trained = True
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\n=== Evaluating Next Activity LSTM ===")
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        loss = results[0]
        accuracy = results[1]
        top_3_accuracy = results[2] if len(results) > 2 else None
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        if top_3_accuracy:
            print(f"Test Top-3 Accuracy: {top_3_accuracy:.4f}")
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Per-class accuracy (important for imbalanced data!)
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n=== Per-Class Performance ===")
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        print(f"Confusion Matrix (showing {len(unique_classes)} classes):")
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        print(cm)
        
        # Show class-wise accuracy
        print("\nPer-class accuracy:")
        for i, cls in enumerate(unique_classes[:10]):  # Show top 10
            mask = y_test == cls
            if mask.sum() > 0:
                cls_acc = (y_pred[mask] == cls).sum() / mask.sum()
                print(f"  Class {cls}: {cls_acc:.4f} ({mask.sum()} samples)")
        
        # Top-k accuracy for different k values
        top_k_accuracies = {}
        for k in [3, 5, 10]:
            if k <= self.vocab_size:
                top_k = tf.keras.metrics.sparse_top_k_categorical_accuracy(
                    y_test, y_pred_proba, k=k
                )
                top_k_acc = np.mean(top_k)
                top_k_accuracies[f'top_{k}'] = top_k_acc
                print(f"Top-{k} Accuracy: {top_k_acc:.4f}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'top_3_accuracy': top_3_accuracy,
            'top_k_accuracies': top_k_accuracies,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    
    def predict(self, X, return_proba=False):
        """
        Predict next activity
        
        Args:
            X: Input sequences
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions (or probabilities if return_proba=True)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        proba = self.model.predict(X, verbose=0)
        
        if return_proba:
            return proba
        else:
            return np.argmax(proba, axis=1)
    
    def save(self, filepath):
        """Save the model to disk"""
        if not filepath.endswith('.keras'):
            filepath += '.keras'
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Next Activity Model saved to {filepath}")

    def load(self, filepath):
        """Load the model from disk"""
        if not filepath.endswith('.keras') and not os.path.exists(filepath):
            filepath += '.keras'
            
        self.model = load_model(filepath, custom_objects={'FocalLoss': FocalLoss})
        self.is_trained = True
        print(f"Next Activity Model loaded from {filepath}")


class RemainingTimeLSTM:
    """
    LSTM model for predicting remaining time until case completion
    """
    
    def __init__(self, vocab_size: int, max_length: int, embedding_dim: int = 64,
                 lstm_units: int = 128):
        """
        Initialize the remaining time prediction model
        
        Args:
            vocab_size: Size of activity vocabulary
            max_length: Maximum sequence length
            embedding_dim: Dimension of activity embeddings
            lstm_units: Number of LSTM units
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.is_trained = False
        
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM architecture"""
        print(f"\n=== Building Remaining Time LSTM Model ===")
        print(f"Vocab size: {self.vocab_size}, Max length: {self.max_length}")
        
        # Input
        input_layer = layers.Input(shape=(self.max_length,), name='activity_sequence')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='activity_embedding'
        )(input_layer)
        
        # LSTM layers
        lstm1 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm_1'
        )(embedding)
        
        lstm2 = layers.LSTM(
            self.lstm_units // 2,
            dropout=0.2,
            recurrent_dropout=0.2,
            name='lstm_2'
        )(lstm1)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu', name='dense_1')(lstm2)
        dropout = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(32, activation='relu', name='dense_2')(dropout)
        
        # Output layer (linear for regression)
        output = layers.Dense(1, activation='linear', name='output')(dense2)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output, name='remaining_time_lstm')
        
        # Compile
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print(self.model.summary())
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=64):
        """
        Train the model
        
        Args:
            X_train: Training sequences
            y_train: Training remaining times
            X_val: Validation sequences (optional)
            y_val: Validation remaining times (optional)
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"\n=== Training Remaining Time LSTM ===")
        print(f"Training samples: {len(X_train)}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath="best_remaining_time_model.keras",
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            StabilityEarlyStopping(
                monitor_acc=None,  # No accuracy for regression
                monitor_loss='val_loss',
                loss_threshold=0.2,
                patience=3
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights if checkpoint exists
        checkpoint_path = "best_remaining_time_model.keras"
        if os.path.exists(checkpoint_path):
            print(f"Loading best weights from {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
            # Keep the checkpoint file as requested
        
        self.is_trained = True
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test: Test sequences
            y_test: Test remaining times
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\n=== Evaluating Remaining Time LSTM ===")
        
        # Evaluate
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Additional metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        print(f"Test MSE: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAPE: {mape:.2f}%")
        
        return {
            'mse': loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': y_pred
        }
    
    def predict(self, X):
        """
        Predict remaining time
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted remaining times
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, filepath: str):
        """Save the model"""
        if not filepath.endswith('.keras'):
            filepath += '.keras'
            
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        self.model.save(filepath)
        
        # Save metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'is_trained': self.is_trained
        }
        
        with open(filepath + '_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Remaining Time LSTM saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        if not filepath.endswith('.keras') and not os.path.exists(filepath):
            filepath += '.keras'
            
        self.model = keras.models.load_model(filepath)
        
        # Load metadata if exists
        metadata_path = filepath + '_metadata.pkl'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.vocab_size = metadata['vocab_size']
            self.max_length = metadata['max_length']
            self.embedding_dim = metadata['embedding_dim']
            self.lstm_units = metadata['lstm_units']
            self.is_trained = metadata['is_trained']
        else:
            print(f"Warning: Metadata not found at {metadata_path}, assuming model is trained")
            self.is_trained = True
        
        print(f"Remaining Time LSTM loaded from {filepath}")


class CombinedLSTMPredictor:
    """
    Combined predictor for both next activity and remaining time
    """
    
    def __init__(self, vocab_size: int, max_length: int):
        """Initialize both models"""
        self.next_activity_model = NextActivityLSTM(vocab_size, max_length)
        self.remaining_time_model = RemainingTimeLSTM(vocab_size, max_length)
        self.is_trained = False
    
    def train(self, data_dict, epochs=50, batch_size=64):
        """
        Train both models
        
        Args:
            data_dict: Dictionary with next_activity and remaining_time data
            epochs: Number of epochs
            batch_size: Batch size
        """
        print("\n" + "="*70)
        print("TRAINING COMBINED LSTM MODELS")
        print("="*70)
        
        # Train next activity model with class weighting
        X_train_act, X_test_act, y_train_act, y_test_act = data_dict['next_activity']
        
        # Calculate class weights for next activity
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_act)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train_act
        )
        class_weight_dict = dict(zip(classes, class_weights_array))
        
        self.next_activity_model.train(
            X_train_act, y_train_act,
            X_val=X_test_act, y_val=y_test_act,
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weight_dict  # Pass class weights!
        )
        
        # Train remaining time model
        X_train_time, X_test_time, y_train_time, y_test_time = data_dict['remaining_time']
        self.remaining_time_model.train(
            X_train_time, y_train_time,
            X_val=X_test_time, y_val=y_test_time,
            epochs=epochs, batch_size=batch_size
        )
        
        self.is_trained = True
    
    def evaluate(self, data_dict):
        """Evaluate both models"""
        print("\n" + "="*70)
        print("EVALUATING COMBINED LSTM MODELS")
        print("="*70)
        
        # Evaluate next activity
        _, X_test_act, _, y_test_act = data_dict['next_activity']
        activity_results = self.next_activity_model.evaluate(X_test_act, y_test_act)
        
        # Evaluate remaining time
        _, X_test_time, _, y_test_time = data_dict['remaining_time']
        time_results = self.remaining_time_model.evaluate(X_test_time, y_test_time)
        
        return {
            'next_activity': activity_results,
            'remaining_time': time_results
        }
    
    def predict(self, sequence):
        """
        Make combined prediction
        
        Args:
            sequence: Activity sequence
            
        Returns:
            Dictionary with both predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        next_activity = self.next_activity_model.predict(sequence)
        remaining_time = self.remaining_time_model.predict(sequence)
        
        return {
            'next_activity': next_activity,
            'remaining_time': remaining_time
        }
    
    def save(self, directory: str):
        """Save both models"""
        os.makedirs(directory, exist_ok=True)
        
        self.next_activity_model.save(os.path.join(directory, 'best_next_activity_model.keras'))
        self.remaining_time_model.save(os.path.join(directory, 'best_remaining_time_model.keras'))
        
        print(f"Combined models saved to {directory}")
    
    def load(self, directory: str):
        """Load both models"""
        self.next_activity_model.load(os.path.join(directory, 'best_next_activity_model.keras'))
        self.remaining_time_model.load(os.path.join(directory, 'best_remaining_time_model.keras'))
        
        self.is_trained = True
        print(f"Combined models loaded from {directory}")

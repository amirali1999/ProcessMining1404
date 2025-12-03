"""
فاز ۱: Outcome Prediction using Classic ML Models
Phase 1: Predict final outcome (Cancelled/Admitted/Discharged) using Decision Tree, Logistic Regression, Random Forest
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import os


class OutcomeFeatureExtractor:
    """
    گام ۶.۲: Extract simple features for classic ML models
    
    Features:
    - Last activity ID
    - Prefix length
    - Count of specific activities (e.g., LabTest, XRay)
    - Time elapsed from case start (if available)
    """
    
    def __init__(self, activity_encoder):
        self.activity_encoder = activity_encoder
        
    def extract_features(self, sample: dict) -> np.ndarray:
        """
        Extract features from a prefix sample
        
        Args:
            sample: Dictionary with 'prefix_activities', 'prefix_length', etc.
            
        Returns:
            Feature vector as numpy array
        """
        prefix = sample['prefix_activities']
        encoded = self.activity_encoder.encode_prefix(prefix)
        
        # Feature 1: Last activity ID
        last_activity_id = encoded[-1] if encoded else 0
        
        # Feature 2: Prefix length
        prefix_length = len(prefix)
        
        # Feature 3: Count of repetitions (how many times each activity appears)
        activity_counts = {}
        for act in prefix:
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        # Use max repetition count as feature
        max_repetition = max(activity_counts.values()) if activity_counts else 0
        
        # Feature 4: Number of unique activities
        num_unique_activities = len(set(prefix))
        
        # Combine features
        features = np.array([
            last_activity_id,
            prefix_length,
            max_repetition,
            num_unique_activities
        ], dtype=float)
        
        return features


class OutcomePredictor:
    """
    فاز ۱: Outcome prediction model
    
    Uses simple features + classic ML models:
    - Decision Tree
    - Logistic Regression  
    - Random Forest
    """
    
    def __init__(self, activity_encoder):
        self.activity_encoder = activity_encoder
        self.feature_extractor = OutcomeFeatureExtractor(activity_encoder)
        self.label_encoder = LabelEncoder()
        
        # Models
        self.decision_tree = None
        self.logistic_regression = None
        self.random_forest = None
        
        # Best model
        self.best_model = None
        self.best_model_name = None
        
    def prepare_data(self, prefix_samples: List[dict], max_prefix_length: int = 5):
        """
        Prepare training data from prefix samples
        
        Args:
            prefix_samples: List of prefix samples
            max_prefix_length: Only use prefixes up to this length (e.g., 3 or 5)
            
        Returns:
            X, y arrays
        """
        print(f"\n🔧 Preparing outcome prediction data (max_prefix_length={max_prefix_length})...")
        
        # Filter prefixes by length
        filtered_samples = [s for s in prefix_samples if s['prefix_length'] <= max_prefix_length]
        
        print(f"   Using {len(filtered_samples)} samples (from {len(prefix_samples)} total)")
        
        X = []
        y = []
        
        for sample in filtered_samples:
            features = self.feature_extractor.extract_features(sample)
            label = sample['label_outcome']
            
            X.append(features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"   Feature shape: {X.shape}")
        print(f"   Outcome classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train multiple classic ML models
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
        """
        print("\n🚀 Training outcome prediction models...")
        
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
        }
        
        best_accuracy = 0
        
        for name, model in models.items():
            print(f"\n  Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    F1-Score: {f1:.4f}")
            
            # Store models
            if name == 'Decision Tree':
                self.decision_tree = model
            elif name == 'Logistic Regression':
                self.logistic_regression = model
            elif name == 'Random Forest':
                self.random_forest = model
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\n✅ Best model: {self.best_model_name} (accuracy: {best_accuracy:.4f})")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate best model on test set
        
        Args:
            X_test, y_test: Test data
        """
        print(f"\n📊 Evaluating {self.best_model_name} on test set...")
        
        y_pred = self.best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        
        # Classification report (pass labels to handle missing classes in test set)
        print("\nClassification Report:")
        all_labels = list(range(len(self.label_encoder.classes_)))
        print(classification_report(y_test, y_pred, 
                                    labels=all_labels,
                                    target_names=self.label_encoder.classes_,
                                    zero_division=0))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
    
    def predict(self, prefix_sample: dict) -> Tuple[str, float]:
        """
        Predict outcome for a single prefix
        
        Args:
            prefix_sample: Dictionary with prefix information
            
        Returns:
            (predicted_outcome, confidence)
        """
        features = self.feature_extractor.extract_features(prefix_sample)
        features = features.reshape(1, -1)
        
        # Get prediction and probability
        prediction_encoded = self.best_model.predict(features)[0]
        probabilities = self.best_model.predict_proba(features)[0]
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = probabilities[prediction_encoded]
        
        return prediction, confidence
    
    def save(self, output_dir: str):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.decision_tree, os.path.join(output_dir, 'decision_tree.pkl'))
        joblib.dump(self.logistic_regression, os.path.join(output_dir, 'logistic_regression.pkl'))
        joblib.dump(self.random_forest, os.path.join(output_dir, 'random_forest.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        joblib.dump({'name': self.best_model_name}, os.path.join(output_dir, 'best_model_info.pkl'))
        
        print(f"\n💾 Saved outcome models to {output_dir}")
    
    def load(self, output_dir: str):
        """Load trained models"""
        self.decision_tree = joblib.load(os.path.join(output_dir, 'decision_tree.pkl'))
        self.logistic_regression = joblib.load(os.path.join(output_dir, 'logistic_regression.pkl'))
        self.random_forest = joblib.load(os.path.join(output_dir, 'random_forest.pkl'))
        self.label_encoder = joblib.load(os.path.join(output_dir, 'label_encoder.pkl'))
        
        best_info = joblib.load(os.path.join(output_dir, 'best_model_info.pkl'))
        self.best_model_name = best_info['name']
        
        if self.best_model_name == 'Decision Tree':
            self.best_model = self.decision_tree
        elif self.best_model_name == 'Logistic Regression':
            self.best_model = self.logistic_regression
        elif self.best_model_name == 'Random Forest':
            self.best_model = self.random_forest
        
        print(f"✅ Loaded outcome models from {output_dir}")
        print(f"   Best model: {self.best_model_name}")

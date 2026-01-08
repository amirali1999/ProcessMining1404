"""
Phase 1: Outcome Prediction Models
Implements classification models to predict case outcomes
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
import joblib
from sklearn.impute import SimpleImputer


class OutcomePredictionModel:
    """
    Model for predicting case outcomes based on prefixes
    Supports multiple classification algorithms
    """
    
    def __init__(self, model_type: str = 'decision_tree'):
        """
        Initialize the outcome prediction model
        
        Args:
            model_type: Type of model ('decision_tree', 'logistic_regression', 
                       'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        self.imputer = None

        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classification model"""
        if self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, feature_columns=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_columns: List of feature column names
        """
        print(f"\n=== Training {self.model_type} model ===")
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        self.feature_columns = feature_columns

        # Handle missing values by imputing (fit imputer on X_train)
        try:
            X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
        except Exception:
            X_train_arr = X_train

        self.imputer = SimpleImputer(strategy='most_frequent')
        X_train_imp = self.imputer.fit_transform(X_train_arr)

        # Train the model
        self.model.fit(X_train_imp, y_train)
        self.is_trained = True
        
        # Training accuracy
        train_pred = self.model.predict(X_train_imp)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
        
    def evaluate(self, X_test, y_test, outcome_encoder=None):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            outcome_encoder: LabelEncoder for outcome names
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        print(f"\n=== Evaluating {self.model_type} model ===")
        
        # Impute test features if an imputer exists
        if self.imputer is not None:
            try:
                X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
            except Exception:
                X_test_arr = X_test
            X_test_imp = self.imputer.transform(X_test_arr)
        else:
            X_test_imp = X_test

        # Predictions
        y_pred = self.model.predict(X_test_imp)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"F1-score (weighted): {f1:.4f}")
        
        # Classification report
        labels = None
        target_names = None
        if outcome_encoder is not None:
            target_names = outcome_encoder.classes_
            labels = np.arange(len(target_names))
        
        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        if labels is not None:
            cm = confusion_matrix(y_test, y_pred, labels=labels)
        else:
            cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns if self.feature_columns else range(X_test_imp.shape[1]),
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        # Impute inputs if needed
        if self.imputer is not None:
            try:
                X_arr = X.values if hasattr(X, 'values') else X
            except Exception:
                X_arr = X
            X_imp = self.imputer.transform(X_arr)
        else:
            X_imp = X

        return self.model.predict(X_imp)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Impute inputs if needed
        if self.imputer is not None:
            try:
                X_arr = X.values if hasattr(X, 'values') else X
            except Exception:
                X_arr = X
            X_imp = self.imputer.transform(X_arr)
        else:
            X_imp = X

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_imp)
        else:
            raise ValueError(f"{self.model_type} does not support probability prediction")
    
    def save(self, filepath: str):
        """Save the model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        # Also save imputer if present
        model_data['imputer'] = self.imputer

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.imputer = model_data.get('imputer', None)
        
        print(f"Model loaded from {filepath}")


class EnsembleOutcomePredictor:
    """
    Ensemble model combining multiple classifiers
    """
    
    def __init__(self):
        """Initialize ensemble with multiple models"""
        self.models = {
            'decision_tree': OutcomePredictionModel('decision_tree'),
            'logistic_regression': OutcomePredictionModel('logistic_regression'),
            'random_forest': OutcomePredictionModel('random_forest')
        }
        self.is_trained = False
    
    def train(self, X_train, y_train, feature_columns=None):
        """Train all models in the ensemble"""
        print("\n=== Training Ensemble Models ===")
        
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            model.train(X_train, y_train, feature_columns)
        
        self.is_trained = True
    
    def evaluate(self, X_test, y_test, outcome_encoder=None):
        """Evaluate all models"""
        print("\n=== Evaluating Ensemble Models ===")
        
        results = {}
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print(f"{'='*60}")
            results[name] = model.evaluate(X_test, y_test, outcome_encoder)
        
        # Compare models
        print("\n=== Model Comparison ===")
        comparison = pd.DataFrame({
            name: {
                'Accuracy': results[name]['accuracy'],
                'F1-Score': results[name]['f1_score']
            }
            for name in results
        }).T
        print(comparison.sort_values('Accuracy', ascending=False))
        
        return results
    
    def predict(self, X, method='voting'):
        """
        Make predictions using ensemble
        
        Args:
            X: Features
            method: 'voting' for majority vote, 'best' for best single model
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        if method == 'voting':
            # Majority voting
            predictions = np.array([model.predict(X) for model in self.models.values()])
            # Get mode along axis 0
            from scipy import stats
            ensemble_pred = stats.mode(predictions, axis=0)[0].flatten()
            return ensemble_pred
        elif method == 'best':
            # Use random forest (typically best performing)
            return self.models['random_forest'].predict(X)
    
    def save(self, directory: str):
        """Save all models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_model.pkl')
            model.save(filepath)
        
        print(f"Ensemble saved to {directory}")
    
    def load(self, directory: str):
        """Load all models"""
        import os
        
        for name, model in self.models.items():
            filepath = os.path.join(directory, f'{name}_model.pkl')
            if os.path.exists(filepath):
                model.load(filepath)
        
        self.is_trained = True
        print(f"Ensemble loaded from {directory}")

"""
Data Preprocessing Module for Process Mining Prediction
Handles XES file reading, cleaning, and feature extraction
"""

import pm4py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


class XESDataPreprocessor:
    """
    Preprocessor for XES event logs
    Extracts features from case prefixes and prepares data for ML models
    """
    
    def __init__(self, xes_file_path: str):
        """
        Initialize the preprocessor with XES file path
        
        Args:
            xes_file_path: Path to the XES file
        """
        self.xes_file_path = xes_file_path
        self.log = None
        self.df = None
        self.label_encoders = {}
        self.activity_encoder = LabelEncoder()
        self.outcome_encoder = LabelEncoder()
        self.time_scaler = StandardScaler()
        
    def load_xes(self):
        """Load XES file using pm4py"""
        print(f"Loading XES file from {self.xes_file_path}...")
        self.log = pm4py.read_xes(self.xes_file_path)
        print(f"Loaded {len(self.log)} cases")
        
    def convert_to_dataframe(self):
        """Convert event log to pandas DataFrame"""
        print("Converting log to DataFrame...")
        self.df = pm4py.convert_to_dataframe(self.log)
        print(f"DataFrame shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def clean_data(self):
        """Clean the data - handle missing values, duplicates"""
        print("Cleaning data...")
        initial_shape = self.df.shape
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Sort by case and timestamp
        self.df = self.df.sort_values(['case:concept:name', 'time:timestamp'])
        
        # Handle missing values in categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['case:concept:name', 'concept:name']:
                self.df[col] = self.df[col].fillna('UNKNOWN')
        
        print(f"Data cleaned. Shape changed from {initial_shape} to {self.df.shape}")
        
    def extract_case_outcome(self):
        """
        Extract outcome for each case
        Outcome can be based on closeCode or the final activity
        """
        print("Extracting case outcomes...")
        
        # Get the last event of each case
        case_outcomes = self.df.groupby('case:concept:name').agg({
            'concept:name': 'last',  # Last activity
            'closeCode': 'last' if 'closeCode' in self.df.columns else 'concept:name',
        }).reset_index()
        
        # Determine outcome column
        if 'closeCode' in self.df.columns:
            case_outcomes['outcome'] = case_outcomes['closeCode']
        else:
            # Use last activity as outcome
            case_outcomes['outcome'] = case_outcomes['concept:name']
        
        print(f"Extracted outcomes for {len(case_outcomes)} cases")
        print(f"Unique outcomes: {case_outcomes['outcome'].nunique()}")
        print(f"Outcome distribution:\n{case_outcomes['outcome'].value_counts()}")
        
        return case_outcomes
    
    def create_prefixes(self, min_prefix_length: int = 1, max_prefix_length: int = None):
        """
        Create prefixes of different lengths for each case
        
        Args:
            min_prefix_length: Minimum prefix length
            max_prefix_length: Maximum prefix length (None for all)
            
        Returns:
            DataFrame with prefixes
        """
        print(f"Creating prefixes (min_length={min_prefix_length})...")
        
        prefixes = []
        
        for case_id, case_data in self.df.groupby('case:concept:name'):
            case_data = case_data.sort_values('time:timestamp')
            case_length = len(case_data)
            
            # Determine max prefix length for this case
            max_len = max_prefix_length if max_prefix_length else case_length
            max_len = min(max_len, case_length)
            
            # Create prefixes of different lengths
            for prefix_len in range(min_prefix_length, max_len + 1):
                prefix_data = case_data.iloc[:prefix_len]
                
                prefix_dict = {
                    'case_id': case_id,
                    'prefix_length': prefix_len,
                    'activities': list(prefix_data['concept:name']),
                }
                
                # Add timestamp information
                prefix_dict['start_time'] = prefix_data['time:timestamp'].iloc[0]
                prefix_dict['current_time'] = prefix_data['time:timestamp'].iloc[-1]
                prefix_dict['elapsed_time'] = (prefix_dict['current_time'] - 
                                               prefix_dict['start_time']).total_seconds()
                
                # Add case attributes from first event
                first_event = prefix_data.iloc[0]
                for col in self.df.columns:
                    if col.startswith('case:') and col != 'case:concept:name':
                        prefix_dict[col] = first_event[col]
                    elif col not in ['concept:name', 'time:timestamp', 'case:concept:name']:
                        # Add aggregated event attributes
                        if prefix_data[col].dtype in ['object', 'category']:
                            prefix_dict[f'{col}_last'] = prefix_data[col].iloc[-1]
                        else:
                            prefix_dict[f'{col}_mean'] = prefix_data[col].mean()
                
                prefixes.append(prefix_dict)
        
        prefixes_df = pd.DataFrame(prefixes)
        print(f"Created {len(prefixes_df)} prefixes from {self.df['case:concept:name'].nunique()} cases")
        
        return prefixes_df
    
    def extract_features_from_prefixes(self, prefixes_df: pd.DataFrame):
        """
        Extract features from prefixes for ML models
        
        Args:
            prefixes_df: DataFrame with prefixes
            
        Returns:
            Feature matrix and metadata
        """
        print("Extracting features from prefixes...")
        
        features_list = []
        
        for idx, row in prefixes_df.iterrows():
            features = {}
            
            # Basic features
            features['prefix_length'] = row['prefix_length']
            features['elapsed_time'] = row['elapsed_time']
            
            # Activity-based features
            activities = row['activities']
            
            # Activity sequence encoding (one-hot for last N activities)
            for i in range(min(5, len(activities))):
                if i < len(activities):
                    features[f'activity_{i+1}'] = activities[-(i+1)]
                else:
                    features[f'activity_{i+1}'] = 'NONE'
            
            # Activity statistics
            features['unique_activities'] = len(set(activities))
            
            # Most common activity
            from collections import Counter
            activity_counts = Counter(activities)
            features['most_common_activity'] = activity_counts.most_common(1)[0][0]
            
            # Case attributes
            for col in prefixes_df.columns:
                if col.startswith('case:') and col != 'case:concept:name':
                    features[col] = row[col]
                elif col.endswith('_last') or col.endswith('_mean'):
                    features[col] = row[col]
            
            features['case_id'] = row['case_id']
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Encode categorical features
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'case_id']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[col] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
            else:
                # Handle unseen labels
                features_df[col] = features_df[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'UNKNOWN'
                )
                features_df[col] = self.label_encoders[col].transform(features_df[col])
        
        print(f"Extracted {features_df.shape[1]} features")
        
        return features_df
    
    def prepare_outcome_prediction_data(self, test_size: float = 0.2):
        """
        Prepare data for outcome prediction (Phase 1)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_columns)
        """
        print("\n=== Preparing data for Outcome Prediction (Phase 1) ===")
        
        # Create prefixes
        prefixes_df = self.create_prefixes(min_prefix_length=1, max_prefix_length=10)
        
        # Get outcomes
        case_outcomes = self.extract_case_outcome()
        
        # Merge prefixes with outcomes
        prefixes_df = prefixes_df.merge(
            case_outcomes[['case:concept:name', 'outcome']],
            left_on='case_id',
            right_on='case:concept:name',
            how='left'
        )
        
        # Extract features
        features_df = self.extract_features_from_prefixes(prefixes_df)
        
        # Add outcome
        features_df['outcome'] = prefixes_df['outcome'].values
        
        # Remove rows with missing outcome
        features_df = features_df.dropna(subset=['outcome'])
        
        # Encode outcome
        y = self.outcome_encoder.fit_transform(features_df['outcome'])
        
        # Prepare X
        X = features_df.drop(['outcome', 'case_id'], axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Outcome classes: {self.outcome_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def prepare_sequence_data_for_lstm(self, max_case_length: int = 50):
        """
        Prepare sequence data for LSTM models (Phase 2)
        
        Returns:
            Data for next activity and remaining time prediction
        """
        print("\n=== Preparing data for LSTM (Phase 2) ===")
        
        sequences = []
        next_activities = []
        remaining_times = []
        
        for case_id, case_data in self.df.groupby('case:concept:name'):
            case_data = case_data.sort_values('time:timestamp')
            
            activities = case_data['concept:name'].tolist()
            timestamps = case_data['time:timestamp'].tolist()
            
            case_length = len(activities)
            
            # Calculate total case duration
            total_duration = (timestamps[-1] - timestamps[0]).total_seconds()
            
            # Create sequences
            for i in range(1, case_length):
                # Sequence up to current point
                seq = activities[:i]
                
                # Pad or truncate
                if len(seq) < max_case_length:
                    seq = ['START'] * (max_case_length - len(seq)) + seq
                else:
                    seq = seq[-max_case_length:]
                
                # Next activity
                next_act = activities[i]
                
                # Remaining time
                elapsed = (timestamps[i] - timestamps[0]).total_seconds()
                remaining = total_duration - elapsed
                
                sequences.append(seq)
                next_activities.append(next_act)
                remaining_times.append(remaining)
        
        print(f"Created {len(sequences)} sequences")
        
        # Encode activities
        all_activities = list(self.df['concept:name'].unique()) + ['START', 'END']
        self.activity_encoder.fit(all_activities)
        
        # Encode sequences
        sequences_encoded = []
        for seq in sequences:
            sequences_encoded.append([self.activity_encoder.transform([act])[0] for act in seq])
        
        sequences_encoded = np.array(sequences_encoded)
        next_activities_encoded = self.activity_encoder.transform(next_activities)
        remaining_times = np.array(remaining_times)
        
        # Normalize Remaining Time
        # 1. Log transform to handle skewness (time data is usually log-normal)
        # Add 1 to avoid log(0)
        y_time_log = np.log1p(remaining_times)
        
        # 2. Standard Scaling
        y_time_normalized = self.time_scaler.fit_transform(y_time_log.reshape(-1, 1)).flatten()
        
        print(f"Remaining time - Normalized stats: mean={y_time_normalized.mean():.4f}, std={y_time_normalized.std():.4f}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # For next activity prediction
        X_train_act, X_test_act, y_train_act, y_test_act = train_test_split(
            sequences_encoded, next_activities_encoded, test_size=0.2, random_state=42
        )
        
        # For remaining time prediction
        X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
            sequences_encoded, y_time_normalized, test_size=0.2, random_state=42
        )
        
        print(f"Next activity - Train: {X_train_act.shape}, Test: {X_test_act.shape}")
        print(f"Remaining time - Train: {X_train_time.shape}, Test: {X_test_time.shape}")
        print(f"Vocabulary size: {len(self.activity_encoder.classes_)}")
        
        # Calculate class distribution for next activity
        unique, counts = np.unique(next_activities_encoded, return_counts=True)
        print(f"\nActivity distribution in sequences:")
        activity_dist = dict(zip(unique, counts))
        for act_idx in sorted(activity_dist.keys())[:10]:  # Show top 10
            act_name = self.activity_encoder.inverse_transform([act_idx])[0]
            pct = (activity_dist[act_idx] / len(next_activities_encoded)) * 100
            print(f"  {act_name}: {activity_dist[act_idx]} ({pct:.2f}%)")
        
        return {
            'next_activity': (X_train_act, X_test_act, y_train_act, y_test_act),
            'remaining_time': (X_train_time, X_test_time, y_train_time, y_test_time),
            'vocab_size': len(self.activity_encoder.classes_),
            'max_length': max_case_length,
            'activity_counts': activity_dist  # For class weighting
        }
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        state = {
            'label_encoders': self.label_encoders,
            'activity_encoder': self.activity_encoder,
            'outcome_encoder': self.outcome_encoder
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
        
        # Save time scaler separately
        scaler_path = filepath.replace('.pkl', '_time_scaler.pkl')
        joblib.dump(self.time_scaler, scaler_path)
        print(f"Time scaler saved to {scaler_path}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.label_encoders = state['label_encoders']
        self.activity_encoder = state['activity_encoder']
        self.outcome_encoder = state['outcome_encoder']
        print(f"Preprocessor loaded from {filepath}")
        
        # Load time scaler
        scaler_path = filepath.replace('.pkl', '_time_scaler.pkl')
        if os.path.exists(scaler_path):
            self.time_scaler = joblib.load(scaler_path)
            print(f"Time scaler loaded from {scaler_path}")

"""
Step 0-1: Load XES event log and generate prefixes for training
گام ۰-۱: آمادهسازی Traceها و تولید Prefix
"""

import pm4py
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


class EventLogProcessor:
    """Process XES event log and generate training prefixes"""
    
    def __init__(self, xes_path: str):
        """
        Load and prepare event log
        
        Args:
            xes_path: Path to XES file
        """
        print(f"📂 Loading XES file: {xes_path}")
        self.log = pm4py.read_xes(xes_path)
        self.df = pm4py.convert_to_dataframe(self.log)
        
        # Standardize column names
        self.df.rename(columns={
            'case:concept:name': 'case_id',
            'concept:name': 'activity',
            'time:timestamp': 'timestamp'
        }, inplace=True)
        
        # Sort by case and timestamp (گام ۰)
        self.df = self.df.sort_values(['case_id', 'timestamp'])
        
        print(f"✅ Loaded {len(self.df)} events from {self.df['case_id'].nunique()} cases")
        
        # Initialize storage
        self.traces: Dict[str, List[str]] = {}
        self.case_outcome: Dict[str, str] = {}
        self.case_timestamps: Dict[str, List[pd.Timestamp]] = {}
        
    def build_traces(self):
        """
        گام ۰: Build traces (sequence of activities) for each case
        
        Creates:
            - self.traces: {case_id: [activity1, activity2, ...]}
            - self.case_outcome: {case_id: outcome_label}
            - self.case_timestamps: {case_id: [timestamp1, timestamp2, ...]}
        """
        print("\n🔨 Building traces from event log...")
        
        for row in self.df.itertuples():
            cid = row.case_id
            act = row.activity
            ts = row.timestamp
            
            # Build trace
            if cid not in self.traces:
                self.traces[cid] = []
                self.case_timestamps[cid] = []
            
            self.traces[cid].append(act)
            self.case_timestamps[cid].append(ts)
        
        # Extract outcomes (using last event's outcome or infer from data)
        if 'outcome' in self.df.columns:
            self.case_outcome = self.df.groupby('case_id')['outcome'].last().to_dict()
        else:
            # If no outcome column, use last activity as outcome
            print("⚠️  No 'outcome' column found, using last activity as outcome")
            self.case_outcome = {cid: trace[-1] for cid, trace in self.traces.items()}
        
        print(f"✅ Built {len(self.traces)} traces")
        print(f"   Unique outcomes: {len(set(self.case_outcome.values()))}")
        
    def generate_prefixes(self, min_prefix_length: int = 1, max_prefix_length: int = None):
        """
        گام ۱: Generate prefix samples for training
        
        For each case trace [A, B, C, D], generates:
        - Prefix [A] → next_activity: B
        - Prefix [A, B] → next_activity: C
        - Prefix [A, B, C] → next_activity: D
        
        Args:
            min_prefix_length: Minimum prefix length to generate
            max_prefix_length: Maximum prefix length (None = use all)
            
        Returns:
            List of prefix samples with metadata
        """
        print(f"\n🔨 Generating prefix samples...")
        
        prefix_samples = []
        
        for cid, activities in self.traces.items():
            outcome_label = self.case_outcome.get(cid)
            timestamps = self.case_timestamps.get(cid, [])
            
            # Generate prefixes (except last position)
            max_len = len(activities) if max_prefix_length is None else min(max_prefix_length + 1, len(activities))
            
            for i in range(min_prefix_length, max_len):
                prefix = activities[:i]
                next_activity = activities[i] if i < len(activities) else None
                
                # Calculate remaining time (in hours)
                remaining_time = None
                if i < len(timestamps) and timestamps:
                    start_time = timestamps[0]
                    current_time = timestamps[i-1]
                    end_time = timestamps[-1]
                    
                    remaining_seconds = (end_time - current_time).total_seconds()
                    remaining_time = remaining_seconds / 3600.0  # Convert to hours
                
                sample = {
                    'case_id': cid,
                    'prefix_activities': prefix,
                    'prefix_length': len(prefix),
                    'label_next_activity': next_activity,
                    'label_outcome': outcome_label,
                    'remaining_time': remaining_time,
                    'timestamp': timestamps[i-1] if i-1 < len(timestamps) else None
                }
                
                prefix_samples.append(sample)
        
        print(f"✅ Generated {len(prefix_samples)} prefix samples")
        
        # Show distribution
        prefix_lengths = [s['prefix_length'] for s in prefix_samples]
        print(f"   Prefix length range: {min(prefix_lengths)} - {max(prefix_lengths)}")
        print(f"   Average prefix length: {np.mean(prefix_lengths):.2f}")
        
        return prefix_samples


class PrefixBucketer:
    """
    گام ۲: Bucket prefixes by length
    """
    
    def __init__(self, max_bucket: int = 10):
        """
        Args:
            max_bucket: Maximum bucket size. Longer prefixes go to "10+" bucket
        """
        self.max_bucket = max_bucket
        self.buckets: Dict[str, List[dict]] = defaultdict(list)
        
    def bucket_prefixes(self, prefix_samples: List[dict]) -> Dict[str, List[dict]]:
        """
        گام ۲: Organize prefixes into buckets by length
        
        - Prefixes of length 1 → Bucket "1"
        - Prefixes of length 2 → Bucket "2"
        - ...
        - Prefixes of length > max_bucket → Bucket "10+"
        
        Args:
            prefix_samples: List of prefix samples from generate_prefixes()
            
        Returns:
            Dictionary {bucket_id: [samples]}
        """
        print(f"\n🗂️  Bucketing prefixes (max_bucket={self.max_bucket})...")
        
        for sample in prefix_samples:
            length = sample['prefix_length']
            
            if length > self.max_bucket:
                bucket_id = f"{self.max_bucket}+"
            else:
                bucket_id = str(length)
            
            self.buckets[bucket_id].append(sample)
        
        # Print distribution
        print(f"✅ Created {len(self.buckets)} buckets:")
        for bucket_id in sorted(self.buckets.keys(), key=lambda x: int(x.replace('+', ''))):
            count = len(self.buckets[bucket_id])
            print(f"   Bucket {bucket_id}: {count} samples")
        
        return self.buckets


class ActivityEncoder:
    """
    گام ۳: Encode activities to integers
    """
    
    def __init__(self):
        self.act2id: Dict[str, int] = {}
        self.id2act: Dict[int, str] = {}
        self.vocab_size: int = 0
        
    def fit(self, df: pd.DataFrame):
        """
        Build activity vocabulary from dataframe
        
        Args:
            df: Dataframe with 'activity' column
        """
        all_activities = sorted(df['activity'].unique())
        
        # Reserve 0 for padding
        self.act2id = {act: idx + 1 for idx, act in enumerate(all_activities)}
        self.id2act = {v: k for k, v in self.act2id.items()}
        self.id2act[0] = '<PAD>'
        
        self.vocab_size = len(all_activities) + 1
        
        print(f"\n📚 Activity vocabulary: {len(all_activities)} unique activities")
        print(f"   Vocab size (with padding): {self.vocab_size}")
        
    def encode_activity(self, activity: str) -> int:
        """Encode single activity to ID"""
        return self.act2id.get(activity, 0)
    
    def encode_prefix(self, prefix: List[str]) -> List[int]:
        """Encode prefix (list of activities) to list of IDs"""
        return [self.act2id.get(a, 0) for a in prefix]
    
    def decode_activity(self, activity_id: int) -> str:
        """Decode activity ID to name"""
        return self.id2act.get(activity_id, '<UNK>')
    
    def decode_prefix(self, prefix_ids: List[int]) -> List[str]:
        """Decode list of IDs to activity names"""
        return [self.id2act.get(aid, '<UNK>') for aid in prefix_ids]

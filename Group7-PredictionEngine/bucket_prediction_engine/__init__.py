"""
Bucket-Based Predictive Process Monitoring Engine
Group 7 - Process Mining Project

This implementation follows the prefix-length bucketing strategy:
- Phase 1: Outcome prediction using classic ML models
- Phase 2: Next activity and remaining time prediction using sequence models
- Separate models for each prefix length bucket
"""

__version__ = "2.0.0"

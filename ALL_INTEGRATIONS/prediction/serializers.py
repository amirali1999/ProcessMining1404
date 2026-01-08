"""
DRF Serializers for Prediction API
"""

from rest_framework import serializers


class PredictionInputSerializer(serializers.Serializer):
    """Input validation for prediction requests"""
    source = serializers.ChoiceField(
        choices=['default', 'raw', 'cleaned'],
        default='default',
        help_text="Which version of the event log to use"
    )
    case_id = serializers.CharField(
        required=False,
        allow_blank=False,
        help_text="Case identifier from the event log"
    )
    activities = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Manual activity sequence (alternative to case_id)"
    )
    
    def validate(self, data):
        """Ensure either case_id or activities is provided"""
        if not data.get('case_id') and not data.get('activities'):
            raise serializers.ValidationError(
                "Either 'case_id' or 'activities' must be provided"
            )
        return data

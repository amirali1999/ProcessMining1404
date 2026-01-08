from rest_framework import serializers
from uploads.models import EventLog, UploadedFile


class UploadedFileSerializer(serializers.ModelSerializer):
    """Basic serializer for UploadedFile"""
    class Meta:
        model = UploadedFile
        fields = ['id', 'original_name', 'size_mb', 'uploaded_at', 'extension']


class EventLogListSerializer(serializers.ModelSerializer):
    """Serializer for listing event logs"""
    uploaded_file = UploadedFileSerializer(read_only=True)
    has_cleaned_version = serializers.BooleanField(read_only=True)
    num_cases = serializers.IntegerField(read_only=True)
    num_events = serializers.IntegerField(read_only=True)
    num_activities = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = EventLog
        fields = [
            'id', 
            'name', 
            'uploaded_file',
            'file_type',
            'has_cleaned_version',
            'default_source_for_downstream',
            'num_cases',
            'num_events',
            'num_activities',
            'meta_info',
            'created_at',
            'updated_at',
        ]


class EventLogDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for a single event log"""
    uploaded_file = UploadedFileSerializer(read_only=True)
    has_cleaned_version = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = EventLog
        fields = '__all__'


class DefaultSourceSerializer(serializers.Serializer):
    """Serializer for updating default source"""
    default_source_for_downstream = serializers.ChoiceField(
        choices=['raw', 'cleaned']
    )

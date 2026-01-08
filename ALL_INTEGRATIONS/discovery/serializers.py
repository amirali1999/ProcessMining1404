from rest_framework import serializers
from .models import DiscoveredProcessModel
from uploads.models import EventLog


class DiscoveredProcessModelSerializer(serializers.ModelSerializer):
    """Serializer for discovered process models (without PNML content)"""
    
    event_log_name = serializers.CharField(source='event_log.name', read_only=True)
    discovered_by_username = serializers.CharField(source='discovered_by.username', read_only=True, allow_null=True)
    algorithm_display = serializers.CharField(source='get_algorithm_display', read_only=True)
    source_version_display = serializers.CharField(source='get_source_version_display', read_only=True)
    complexity_score = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = DiscoveredProcessModel
        fields = [
            'id',
            'event_log',
            'event_log_name',
            'algorithm',
            'algorithm_display',
            'source_version',
            'source_version_display',
            'num_places',
            'num_transitions',
            'num_arcs',
            'complexity_score',
            'discovered_by',
            'discovered_by_username',
            'discovered_at',
        ]
        read_only_fields = [
            'id',
            'num_places',
            'num_transitions',
            'num_arcs',
            'discovered_by',
            'discovered_at',
        ]


class DiscoveryRequestSerializer(serializers.Serializer):
    """Serializer for discovery algorithm requests"""
    
    source = serializers.ChoiceField(
        choices=['raw', 'cleaned'],
        default='raw',
        help_text="Which version of the log to use"
    )
    
    # Heuristics Miner specific parameters
    dependency_threshold = serializers.FloatField(
        default=0.5,
        min_value=0.0,
        max_value=1.0,
        required=False,
        help_text="Dependency threshold for Heuristics Miner"
    )
    
    and_threshold = serializers.FloatField(
        default=0.1,
        min_value=0.0,
        max_value=1.0,
        required=False,
        help_text="AND threshold for Heuristics Miner"
    )
    
    loop_two_threshold = serializers.FloatField(
        default=0.5,
        min_value=0.0,
        max_value=1.0,
        required=False,
        help_text="Loop-2 threshold for Heuristics Miner"
    )


class PNMLSerializer(serializers.Serializer):
    """Serializer for PNML content response"""
    
    model_id = serializers.IntegerField()
    algorithm = serializers.CharField()
    source_version = serializers.CharField()
    pnml_content = serializers.CharField()
    num_places = serializers.IntegerField()
    num_transitions = serializers.IntegerField()
    num_arcs = serializers.IntegerField()

"""
Django management command to import a PNML file as a DiscoveredProcessModel.

Usage:
    python manage.py import_pnml <pnml_file_path> <event_log_id> [--algorithm alpha] [--source raw]
    
Example:
    python manage.py import_pnml test/heuristic_SP_DF_cleaned.pnml 2 --algorithm heuristics --source cleaned
"""
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from pm4py.objects.petri_net.importer import importer as pnml_importer
from discovery.models import DiscoveredProcessModel
from uploads.models import EventLog

User = get_user_model()


class Command(BaseCommand):
    help = 'Import a PNML file as a DiscoveredProcessModel'

    def add_arguments(self, parser):
        parser.add_argument(
            'pnml_file',
            type=str,
            help='Path to the PNML file'
        )
        parser.add_argument(
            'event_log_id',
            type=int,
            help='ID of the EventLog this model belongs to'
        )
        parser.add_argument(
            '--algorithm',
            type=str,
            default='heuristics',
            choices=['alpha', 'heuristics', 'inductive'],
            help='Mining algorithm used (default: heuristics)'
        )
        parser.add_argument(
            '--source',
            type=str,
            default='cleaned',
            choices=['raw', 'cleaned'],
            help='Source version (default: cleaned)'
        )
        parser.add_argument(
            '--name',
            type=str,
            default=None,
            help='Custom model name (optional)'
        )

    def handle(self, *args, **options):
        pnml_file = options['pnml_file']
        event_log_id = options['event_log_id']
        algorithm = options['algorithm']
        source = options['source']
        custom_name = options['name']

        # Validate event log exists
        try:
            event_log = EventLog.objects.get(id=event_log_id)
        except EventLog.DoesNotExist:
            raise CommandError(f'EventLog with id={event_log_id} does not exist')

        # Load PNML file
        self.stdout.write(f'Loading PNML from: {pnml_file}')
        try:
            net, initial_marking, final_marking = pnml_importer.apply(pnml_file)
        except Exception as e:
            raise CommandError(f'Failed to load PNML file: {e}')

        # Read PNML content as string
        with open(pnml_file, 'r', encoding='utf-8') as f:
            pnml_content = f.read()

        # Count places and transitions
        num_places = len(net.places)
        num_transitions = len(net.transitions)

        self.stdout.write(self.style.SUCCESS(
            f'✅ Loaded Petri net: {num_places} places, {num_transitions} transitions'
        ))

        # Get admin user for discovered_by field
        admin_user = User.objects.filter(is_superuser=True).first()
        if not admin_user:
            admin_user = User.objects.first()

        # Generate model name
        if custom_name:
            model_name = custom_name
        else:
            algo_name = algorithm.capitalize()
            model_name = f"{algo_name} Miner (Imported)"

        # Create DiscoveredProcessModel
        model = DiscoveredProcessModel.objects.create(
            event_log=event_log,
            discovered_by=admin_user,
            algorithm=algorithm,
            source_version=source,
            model_name=model_name,
            pnml_content=pnml_content,
            num_places=num_places,
            num_transitions=num_transitions,
            num_arcs=0,  # Can be calculated if needed
        )

        self.stdout.write(self.style.SUCCESS(
            f'\n✅ Successfully created DiscoveredProcessModel (ID: {model.id})'
        ))
        self.stdout.write(f'   - Name: {model.model_name}')
        self.stdout.write(f'   - Algorithm: {algorithm}')
        self.stdout.write(f'   - Source: {source}')
        self.stdout.write(f'   - Event Log: {event_log}')
        self.stdout.write(f'   - Places: {num_places}')
        self.stdout.write(f'   - Transitions: {num_transitions}')
        self.stdout.write(f'\n📍 You can now use this model (ID: {model.id}) for conformance checking!')

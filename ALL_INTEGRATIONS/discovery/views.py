from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404

from uploads.models import EventLog
from .models import DiscoveredProcessModel, EventLogJob
from .serializers import (
    DiscoveredProcessModelSerializer,
    DiscoveryRequestSerializer,
    PNMLSerializer,
)
from . import services

# Job Processing Views for File Upload & Processing Pipeline

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from accounts.decorators import check_max_projects, check_algorithm_access
import threading
import os
import pandas as pd
import tempfile
import traceback
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import EventLogJob
import threading
import os
import pandas as pd
import tempfile
import traceback


@require_http_methods(["POST"])
@login_required
def create_job(request):
    """
    POST /jobs/create/
    
    Accepts multipart form data:
    - file: The event log file (.xes or .csv)
    - cleaning_enabled: "true" or "false"
    - mining_method: Selected mining method (e.g., "alpha", "heuristics")
    
    Returns JSON:
    {
        "job_id": 123,
        "progress_url": "/jobs/progress/123/"
    }
    """
    try:
        # Validate file upload
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        uploaded_file = request.FILES['file']
        
        # Validate file extension
        filename = uploaded_file.name
        if not (filename.endswith('.xes') or filename.endswith('.csv')):
            return JsonResponse({'error': 'Only .xes and .csv files are supported'}, status=400)
          # Get form parameters
        cleaning_enabled = request.POST.get('cleaning_enabled', 'false').lower() == 'true'
        mining_method = request.POST.get('mining_method', 'alpha')
        project_name = request.POST.get('project_name', '').strip()
        
        # Validate project name
        if not project_name:
            return JsonResponse({'error': 'Project name is required'}, status=400)
        
        # Validate mining method
        valid_methods = [choice[0] for choice in EventLogJob.MINING_METHOD_CHOICES]
        if mining_method not in valid_methods:
            return JsonResponse({'error': f'Invalid mining method. Choose from: {valid_methods}'}, status=400)
        
        # 🔒 LICENSE CHECK: Validate algorithm access
        if not request.user.can_use_algorithm(mining_method):
            allowed = ', '.join([alg.title() for alg in request.user.get_allowed_algorithms()])
            return JsonResponse({
                'error': f'⚠️ You do not have access to the {mining_method.title()} algorithm.',
                'message': f'Your plan allows: {allowed}. Please upgrade to Premium for access to all algorithms.',
                'upgrade_url': '/accounts/activate-license/'
            }, status=403)        # 🔒 LICENSE CHECK: Validate project limit
        if not request.user.is_premium:
            max_projects = request.user.max_projects
            if max_projects > 0:  # 0 means unlimited
                # Count unique project names for this user
                current_count = EventLogJob.objects.filter(
                    user=request.user
                ).exclude(
                    project_name__isnull=True
                ).exclude(
                    project_name__exact=''
                ).values('project_name').distinct().count()
                
                if current_count >= max_projects:
                    return JsonResponse({
                        'error': f'⚠️ Project limit reached ({max_projects} projects).',
                        'message': 'Please upgrade to Premium for unlimited projects or delete an existing project.',
                        'upgrade_url': '/accounts/activate-license/'
                    }, status=403)
        
        # 🔒 LICENSE CHECK: Validate log size (for CSV files)
        if filename.endswith('.csv') and not request.user.is_premium:
            max_rows = request.user.max_log_rows
            if max_rows > 0:  # 0 means unlimited
                # Quick row count check (read first to validate size)
                try:
                    uploaded_file.seek(0)
                    df_temp = pd.read_csv(uploaded_file, nrows=max_rows + 1)
                    row_count = len(df_temp)
                    uploaded_file.seek(0)  # Reset for actual processing
                    
                    if row_count > max_rows:
                        return JsonResponse({
                            'error': f'⚠️ Log size limit exceeded ({row_count:,} rows).',
                            'message': f'Your plan allows up to {max_rows:,} rows. Please upgrade to Premium for unlimited log size.',
                            'upgrade_url': '/accounts/activate-license/'
                        }, status=403)
                except Exception as e:
                    # If we can't read the file, let it proceed (will fail in processing)
                    pass
        
        # Create job record
        job = EventLogJob.objects.create(
            user=request.user,
            original_file=uploaded_file,
            original_filename=filename,
            project_name=project_name,
            cleaning_enabled=cleaning_enabled,
            mining_method=mining_method,
            status='pending',
            progress=0,
            message='Job created, waiting to start...'
        )
          # Start processing in background thread
        thread = threading.Thread(target=process_job, args=(job.id,))
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'job_id': job.id,
            'progress_url': f'/jobs/progress/{job.id}/'
        })
        
    except Exception as e:
        # Log full traceback for debugging
        import traceback
        print("="*60)
        print("❌ ERROR in create_job:")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        print("="*60)
        
        return JsonResponse({
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@require_http_methods(["GET"])
@login_required
def job_status(request, job_id):
    """
    GET /jobs/status/<job_id>/
    
    Returns JSON:
    {
        "status": "pending" | "running" | "done" | "error",
        "progress": 0-100,
        "message": "Current status message",
        "redirect_url": "/dashboard/" (only when done),
        "output_url": "/media/outputs/..." (only when done),
        "error_message": "..." (only when error)
    }
    """
    try:
        job = EventLogJob.objects.get(id=job_id, user=request.user)
        
        response_data = {
            'status': job.status,
            'progress': job.progress,
            'message': job.message,
        }
        
        if job.status == 'done':
            response_data['redirect_url'] = '/dashboard/'
            output_url = job.get_output_url()
            if output_url:
                response_data['output_url'] = output_url
        
        if job.status == 'error':
            response_data['error_message'] = job.error_message
        
        return JsonResponse(response_data)
        
    except EventLogJob.DoesNotExist:
        return JsonResponse({'error': 'Job not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
@login_required
def job_progress_page(request, job_id):
    """
    GET /jobs/progress/<job_id>/
    
    Renders the progress page with polling UI.
    """
    try:
        job = EventLogJob.objects.get(id=job_id, user=request.user)
        return render(request, 'discovery/job_progress.html', {
            'job': job,
            'job_id': job_id
        })
    except EventLogJob.DoesNotExist:
        return HttpResponse('Job not found', status=404)


@require_http_methods(["GET"])
@login_required
def get_project_api(request, project_name):
    """
    GET /api/projects/<project_name>/
    
    Returns the latest job for the specified project name.
    Used by dashboard dropdown to load process maps dynamically.
    """
    try:
        # Get the latest job for this project and user
        job = EventLogJob.objects.filter(
            user=request.user,
            project_name=project_name
        ).order_by('-created_at').first()
        
        if not job:
            return JsonResponse({
                'error': 'Project not found'
            }, status=404)
        
        # Prepare response data
        data = {
            'project_name': job.project_name,
            'original_filename': job.original_filename,
            'file_type': job.file_type,
            'mining_method': job.mining_method,
            'mining_method_display': job.get_mining_method_display(),
            'cleaning_enabled': job.cleaning_enabled,
            'status': job.status,
            'progress': job.progress,
            'message': job.message,
            'created_at': job.created_at.isoformat() if job.created_at else None,
            'updated_at': job.updated_at.isoformat() if job.updated_at else None,
            'num_cases': job.num_cases,
            'num_events': job.num_events,
            'num_activities': job.num_activities,
            'output_map_svg': job.output_map_svg.url if job.output_map_svg else None,
            'output_map_image': job.output_map_image.url if job.output_map_image else None,
        }
        
        return JsonResponse(data)
        
    except Exception as e:
        import traceback
        print(f"Error fetching project: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({
            'error': str(e)
        }, status=500)


def process_job(job_id):
    """
    Background processing function for event log jobs.
    
    Pipeline:
    1. Load file into DataFrame
    2. Optional: Run Group3 smart_clean
    3. Run Group4 process mining
    4. Save output
    5. Update job status
    """
    job = None
    try:
        job = EventLogJob.objects.get(id=job_id)
        job.status = 'running'
        job.progress = 10
        job.message = 'Loading event log file...'
        job.save()
        
        # Get file path
        file_path = job.original_file.path
        
        # Step 1: Load file into DataFrame
        df = load_event_log(file_path)
        job.progress = 30
        job.message = f'Loaded {len(df)} events from {job.original_filename}'
        job.save()
          # Step 2: Optional cleaning
        if job.cleaning_enabled:
            job.progress = 40
            job.message = 'Running data cleaning...'
            job.save()
            
            df = run_smart_clean(df)
            
            job.progress = 60
            job.message = f'Cleaning complete. {len(df)} events after cleaning.'
            job.save()
        
        # Step 3: Run process mining
        job.progress = 70
        job.message = f'Running {job.get_mining_method_display()}...'
        job.save()
        
        output_pnml_path = run_process_mining(df, job.mining_method, job_id)
          # Step 4: Save PNML output to job
        if output_pnml_path and os.path.exists(output_pnml_path):
            with open(output_pnml_path, 'rb') as f:
                filename = f'{job.mining_method}_job_{job_id}.pnml'
                # PNML should not be saved in output fields (it's the intermediate format)
            
            job.progress = 90
            job.message = 'Generating visualization...'
            job.save()
            
            # Convert PNML to SVG image for display (requires Graphviz)
            try:
                output_image_path = os.path.join(tempfile.gettempdir(), f'job_{job_id}.svg')
                convert_pnml_to_image(output_pnml_path, output_image_path)
                
                # Save SVG image to job (FIXED: SVG goes to output_map_svg)
                if os.path.exists(output_image_path):
                    with open(output_image_path, 'rb') as f:
                        image_filename = f'{job.mining_method}_job_{job_id}.svg'
                        job.output_map_svg.save(image_filename, ContentFile(f.read()), save=False)
                    
                    # Clean up temp file
                    try:
                        os.remove(output_image_path)
                    except:
                        pass
            except Exception as graphviz_error:
                # Graphviz not available - skip SVG generation
                # The PNML file is still saved
                print(f"⚠️  Warning: Could not generate SVG image (Graphviz not available)")
                print(f"   Error: {str(graphviz_error)}")
                print(f"   The process map PNML file is still available")
        
        # Step 5: Mark as done
        job.status = 'done'
        job.progress = 100
        job.message = 'Processing complete!'
        job.save()
        
    except Exception as e:
        if job:
            job.status = 'error'
            job.progress = 0
            job.message = 'Processing failed'
            job.error_message = f'{str(e)}\n\n{traceback.format_exc()}'
            job.save()


def load_event_log(file_path):
    """Load event log file into pandas DataFrame"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.xes'):
            # Use pm4py to load XES
            try:
                from pm4py.objects.log.importer.xes import importer as xes_importer
                from pm4py.objects.conversion.log import converter as log_converter
            except ImportError as e:
                raise ImportError(
                    f"pm4py is required to load XES files but is not installed. "
                    f"Install it with: pip install pm4py\n"
                    f"Original error: {str(e)}"
                )
            
            log = xes_importer.apply(file_path)
            df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
            return df
        else:
            raise ValueError(f'Unsupported file format: {file_path}')
    except Exception as e:
        print(f"❌ Error loading event log from {file_path}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run_smart_clean(df):
    """Run Group3 smart cleaning on DataFrame"""
    import sys
    import os
    
    # Add Group3 to Python path
    # __file__ is discovery/views.py, go up 1 level to pm/, then into Group3/
    group3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Group3')
    if group3_path not in sys.path:
        sys.path.insert(0, group3_path)
    
    from log_preprocess import LogPreprocessor
    
    # Create preprocessor instance with DataFrame
    preprocessor = LogPreprocessor(df=df)
    
    # Run smart_clean with sensible defaults
    cleaned_df = preprocessor.smart_clean(
        aggressive=False,
        infer_types=True,
        normalize_names=True,
        scope="selected",
        inplace=True
    )
    
    return cleaned_df


def run_process_mining(df, method, job_id):
    """
    Run Group4 process mining and return path to output PNML file.
    
    Uses ProcessDiscovery class from Group4 with alpha_miner_service() 
    or heuristic_miner_service() methods.
    """
    import sys
    import os
    
    # Add Group4 to Python path
    # __file__ is discovery/views.py, go up 1 level to pm/, then into Group4/
    group4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Group4')
    if group4_path not in sys.path:
        sys.path.insert(0, group4_path)
    
    # Import Group4 ProcessDiscovery class
    from process_discovery import ProcessDiscovery
    
    # Create output directory for this job
    output_dir = os.path.join(settings.MEDIA_ROOT, 'process_maps', str(job_id))
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ProcessDiscovery instance with DataFrame
    pd_instance = ProcessDiscovery(df=df)
    
    # Temporarily set a fake path for the instance (Group4 uses it for filename extraction)
    pd_instance.inOut_path = f"job_{job_id}.csv"
    
    # Run appropriate mining algorithm
    if method == 'alpha':
        net, im, fm = pd_instance.alpha_miner_service(output_dir=output_dir)
        pnml_file = os.path.join(output_dir, f"alpha_job_{job_id}.pnml")
    elif method == 'heuristics':
        net, im, fm = pd_instance.heuristic_miner_service(output_dir=output_dir)
        pnml_file = os.path.join(output_dir, f"heuristic_job_{job_id}.pnml")
    else:
        raise ValueError(f"Unknown mining method: {method}")
    
    # Verify the PNML file was created
    if not os.path.exists(pnml_file):
        raise FileNotFoundError(f"Process mining failed to create output file: {pnml_file}")
    
    return pnml_file


def convert_pnml_to_image(pnml_path, output_image_path):
    """
    Convert PNML file to SVG image using pm4py visualization.
    
    Args:
        pnml_path: Path to input PNML file
        output_image_path: Path to save output SVG image
        
    Returns:
        Path to generated SVG image
    """
    import pm4py
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    import os
    
    try:
        # Set Graphviz path if not in system PATH
        # Common Graphviz installation paths on macOS
        graphviz_paths = [
            '/opt/local/bin',      # MacPorts
            '/usr/local/bin',      # Homebrew
            '/opt/homebrew/bin',   # Homebrew on Apple Silicon
        ]
        
        # Add Graphviz paths to environment PATH
        current_path = os.environ.get('PATH', '')
        for gv_path in graphviz_paths:
            if os.path.exists(gv_path) and gv_path not in current_path:
                os.environ['PATH'] = f"{gv_path}:{current_path}"
                print(f"✅ Added Graphviz path to environment: {gv_path}")
                break
        
        # Read PNML file
        net, im, fm = pm4py.read_pnml(pnml_path)
        
        # Generate visualization with SVG format
        parameters = {
            pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"
        }
        gviz = pn_visualizer.apply(net, im, fm, parameters=parameters)
        
        # Save as SVG
        pn_visualizer.save(gviz, output_image_path)
        
        return output_image_path
        
    except Exception as e:
        raise Exception(f"Failed to convert PNML to SVG: {str(e)}")


class DiscoveryViewSet(viewsets.ViewSet):
    """
    API endpoints for Process Discovery (Group 4).
    
    Endpoints:
    - POST /api/event-logs/{event_log_id}/discovery/alpha/
    - POST /api/event-logs/{event_log_id}/discovery/heuristics/
    - GET /api/event-logs/{event_log_id}/discovery/
    """
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['post'], url_path='alpha')
    def run_alpha(self, request, pk=None):
        """
        Run Alpha Miner on an event log.
        
        Request body:
        {
            "source": "raw" | "cleaned"
        }
        """
        # Validate event log exists
        event_log = get_object_or_404(EventLog, pk=pk)
        
        # Validate request data
        serializer = DiscoveryRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        source = serializer.validated_data['source']
        
        try:
            # Run Alpha Miner
            model = services.run_alpha_miner(
                event_log_id=event_log.id,
                source=source,
                user_id=request.user.id
            )
            
            # Return the created model
            response_serializer = DiscoveredProcessModelSerializer(model)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Discovery failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'], url_path='heuristics')
    def run_heuristics(self, request, pk=None):
        """
        Run Heuristics Miner on an event log.
        
        Request body:
        {
            "source": "raw" | "cleaned",
            "dependency_threshold": 0.5,  // optional
            "and_threshold": 0.1,          // optional
            "loop_two_threshold": 0.5      // optional
        }
        """
        # Validate event log exists
        event_log = get_object_or_404(EventLog, pk=pk)
        
        # Validate request data
        serializer = DiscoveryRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        source = serializer.validated_data['source']
        dependency_threshold = serializer.validated_data.get('dependency_threshold', 0.5)
        and_threshold = serializer.validated_data.get('and_threshold', 0.1)
        loop_two_threshold = serializer.validated_data.get('loop_two_threshold', 0.5)
        
        try:
            # Run Heuristics Miner
            model = services.run_heuristics_miner(
                event_log_id=event_log.id,
                source=source,
                user_id=request.user.id,
                dependency_threshold=dependency_threshold,
                and_threshold=and_threshold,
                loop_two_threshold=loop_two_threshold
            )
            
            # Return the created model
            response_serializer = DiscoveredProcessModelSerializer(model)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Discovery failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='models')
    def list_models(self, request, pk=None):
        """
        List all discovered models for an event log.
        
        GET /api/event-logs/{event_log_id}/discovery/models/
        """
        # Validate event log exists
        event_log = get_object_or_404(EventLog, pk=pk)
        
        # Get all models for this log
        models = services.get_discovered_models(event_log.id)
        
        # Serialize and return
        serializer = DiscoveredProcessModelSerializer(models, many=True)
        return Response(serializer.data)


class DiscoveredModelViewSet(viewsets.ViewSet):
    """
    API endpoints for accessing discovered models.
    
    Endpoints:
    - GET /api/discovered-models/{model_id}/pnml/
    """
    permission_classes = [IsAuthenticated]
    
    @action(detail=True, methods=['get'], url_path='pnml')
    def get_pnml(self, request, pk=None):
        """
        Get PNML content for a discovered model.
        
        This is the main hand-off endpoint for Groups 5 (visualization)
        and 6 (conformance checking).
        
        GET /api/discovered-models/{model_id}/pnml/
        """
        try:
            model = get_object_or_404(DiscoveredProcessModel, pk=pk)
            
            # Return PNML content
            data = {
                'model_id': model.id,
                'algorithm': model.algorithm,
                'source_version': model.source_version,
                'pnml_content': model.pnml_content,
                'num_places': model.num_places,
                'num_transitions': model.num_transitions,
                'num_arcs': model.num_arcs,
            }
            
            serializer = PNMLSerializer(data)
            return Response(serializer.data)
            
        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['get'], url_path='petrinet-image')
    def petrinet_image(self, request, pk=None):
        """
        Get Petri net visualization as PNG image.
        
        GROUP 5 ENDPOINT - Minimal visualization implementation.
        
        GET /api/discovered-models/{model_id}/petrinet-image/
        
        Returns PNG image bytes directly.
        """
        from django.http import HttpResponse
        
        try:
            # Render the Petri net to PNG
            png_bytes = services.render_petrinet_png_from_model(model_id=pk)
              # Return as image
            return HttpResponse(png_bytes, content_type='image/png')
            
        except DiscoveredProcessModel.DoesNotExist:
            return Response(
                {'error': f'Discovered model {pk} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            import traceback
            print(f"Error rendering Petri net: {str(e)}")
            print(traceback.format_exc())
            return Response(
                {'error': f'Failed to render Petri net: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='petrinet-svg')
    def petrinet_svg(self, request, pk=None):
        """
        Get Petri net visualization as SVG image.
        
        GET /api/discovered-models/{model_id}/petrinet-svg/
        
        Returns SVG for inline display or download based on 'download' query parameter.
        """
        from django.http import HttpResponse
        try:
            # Render the Petri net to SVG
            svg_bytes = services.render_petrinet_svg_from_model(model_id=pk)
            
            # Check if download is requested
            is_download = request.GET.get('download', 'false').lower() == 'true'
            
            response = HttpResponse(svg_bytes, content_type='image/svg+xml')
            
            if is_download:
                response['Content-Disposition'] = f'attachment; filename="petri_net_model_{pk}.svg"'
            
            return response
            
        except DiscoveredProcessModel.DoesNotExist:
            return Response(
                {'error': f'Discovered model {pk} not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            import traceback
            print(f"Error rendering Petri net SVG: {str(e)}")
            print(traceback.format_exc())
            return Response(
                {'error': f'Failed to render Petri net: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# ============================================
# CSV IMPORT VIEWS
# ============================================

@require_http_methods(["POST"])
@login_required
def create_csv_import_session(request):
    """
    POST /import/csv/create/
    
    Accepts CSV file and creates a temporary import session.
    Returns JSON with session_id and redirect_url.
    """
    try:
        from .models import CSVImportSession
        
        # Validate file upload
        if 'file' not in request.FILES:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        uploaded_file = request.FILES['file']
        
        # Validate file extension
        filename = uploaded_file.name
        if not filename.lower().endswith('.csv'):
            return JsonResponse({'error': 'Only .csv files are supported'}, status=400)
          # Get project name from request
        project_name = request.POST.get('project_name', '').strip()
        if not project_name:
            # If no project name provided, use filename without extension
            project_name = filename.rsplit('.', 1)[0]
          # Check log size limit for free users
        if not request.user.is_premium:
            max_rows = request.user.max_log_rows
            if max_rows > 0:  # 0 means unlimited
                # Quick check: count lines in file
                uploaded_file.seek(0)
                # Read content as bytes and decode
                content = uploaded_file.read().decode('utf-8')
                line_count = content.count('\n')
                if line_count > 0 and not content.endswith('\n'):
                    line_count += 1
                line_count -= 1  # Subtract header row
                
                # Reset file pointer after reading
                uploaded_file.seek(0)
                
                if line_count > max_rows:
                    return JsonResponse({
                        'error': f'CSV file exceeds your limit of {max_rows:,} rows',
                        'message': f'Your file has {line_count:,} rows. Upgrade to Premium for unlimited log size.',
                        'upgrade_url': '/accounts/activate-license/'
                    }, status=403)
          # Check project limit for free users
        if not request.user.is_premium:
            max_projects = request.user.max_projects
            if max_projects > 0:  # 0 means unlimited
                from .models import EventLogJob
                # Count unique project names for this user
                current_count = EventLogJob.objects.filter(
                    user=request.user
                ).exclude(
                    project_name__isnull=True
                ).exclude(
                    project_name__exact=''
                ).values('project_name').distinct().count()
                
                if current_count >= max_projects:
                    return JsonResponse({
                        'error': f'You have reached your project limit ({max_projects} projects)',
                        'message': 'Please upgrade to Premium for unlimited projects or delete an existing project.',
                        'upgrade_url': '/accounts/activate-license/'
                    }, status=403)
        
        # Create import session
        session = CSVImportSession.objects.create(
            user=request.user,
            uploaded_file=uploaded_file,
            original_filename=filename,
            project_name=project_name
        )
        
        return JsonResponse({
            'session_id': session.id,
            'redirect_url': f'/import/csv/{session.id}/'
        })
        
    except Exception as e:
        import traceback
        print(f"CSV import session creation error: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
@login_required
def csv_import_page(request, session_id):
    """
    GET /import/csv/<session_id>/
    
    Renders the CSV import mapping UI (Disco-like).
    """
    try:
        from .models import CSVImportSession
        
        session = CSVImportSession.objects.get(id=session_id, user=request.user)
        
        # Read CSV file and get preview data (first 50 rows)
        import pandas as pd
        import os
        file_path = session.uploaded_file.path
        
        try:
            # Try to read CSV with common parameters
            print(f"📊 Reading CSV file: {file_path}")
            print(f"   File exists: {os.path.exists(file_path)}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found at {file_path}")
            
            file_size = os.path.getsize(file_path)
            print(f"   File size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("CSV file is empty (0 bytes)")
            
            # Read CSV with robust parameters
            # sep=None with engine='python' auto-detects delimiter
            df = pd.read_csv(
                file_path,
                nrows=50,
                encoding='utf-8',
                on_bad_lines='skip',
                skipinitialspace=True,
                skip_blank_lines=True
            )
            
            print(f"   DataFrame shape: {df.shape}")
            print(f"   DataFrame columns: {df.columns.tolist()}")
            
            # Prepare data for template
            columns = df.columns.tolist()
            rows = df.values.tolist()
            
            print(f"✅ CSV loaded successfully!")
            print(f"   Columns ({len(columns)}): {columns}")
            print(f"   Rows: {len(rows)}")
            if rows:
                print(f"   First row: {rows[0]}")
            
            # Detect encoding (for dropdown) - default to utf-8 if chardet not available
            detected_encoding = 'utf-8'
            try:
                import chardet
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)  # Read first 10KB
                    detected = chardet.detect(raw_data)
                    detected_encoding = detected.get('encoding', 'utf-8')
                    print(f"   Detected encoding: {detected_encoding}")
            except ImportError:
                print(f"   chardet not available, using utf-8")
            
        except Exception as e:
            print(f"❌ Error reading CSV: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            columns = []
            rows = []
            detected_encoding = 'utf-8'
        
        context = {
            'session': session,
            'columns': columns,
            'rows': rows,
            'num_columns': len(columns),
            'num_rows': len(rows),
            'detected_encoding': detected_encoding,
        }
        
        print(f"📤 Sending context to template:")
        print(f"   Columns: {len(columns)}")
        print(f"   Rows: {len(rows)}")
        print(f"   Encoding: {detected_encoding}")
        
        return render(request, 'discovery/csv_import.html', context)
        
    except CSVImportSession.DoesNotExist:
        return HttpResponse('CSV import session not found', status=404)
    except Exception as e:
        import traceback
        print(f"CSV import page error: {str(e)}")
        print(traceback.format_exc())
        return HttpResponse(f'Error loading CSV import page: {str(e)}', status=500)


@require_http_methods(["POST"])
@login_required
def process_csv_import(request, session_id):
    """
    POST /import/csv/<session_id>/process/
    
    Processes the CSV import with column mappings.
    Creates an EventLogJob to trigger discovery process.
    """
    try:
        from .models import CSVImportSession, EventLogJob
        from uploads.models import EventLog, UploadedFile
        import pandas as pd
        import json
        import os
        
        # Get the import session
        session = CSVImportSession.objects.get(id=session_id, user=request.user)
        
        # Get column mappings and discovery options from request body
        data = json.loads(request.body)
        column_mappings = data.get('columnMappings', {})
        mining_method = data.get('miningMethod', 'alpha')  # Default to alpha miner
        cleaning_enabled = data.get('cleaningEnabled', False)
        
        # Validate required mappings
        if not column_mappings.get('caseId') or not column_mappings.get('activity'):
            return JsonResponse({
                'error': 'Case ID and Activity columns are required'
            }, status=400)
        
        # Read the CSV file
        file_path = session.uploaded_file.path
        df = pd.read_csv(file_path)
        
        # Rename columns according to mappings
        rename_map = {}
        if column_mappings.get('caseId'):
            rename_map[column_mappings['caseId']] = 'case:concept:name'
        if column_mappings.get('activity'):
            rename_map[column_mappings['activity']] = 'concept:name'
        if column_mappings.get('timestamp'):
            rename_map[column_mappings['timestamp']] = 'time:timestamp'
        if column_mappings.get('resource'):
            rename_map[column_mappings['resource']] = 'org:resource'
        
        df = df.rename(columns=rename_map)
          # Convert timestamp if provided
        if 'time:timestamp' in df.columns:
            try:
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
            except:
                pass  # Keep original format if conversion fails
        
        # Get project name from session or use filename
        project_name = session.project_name if session.project_name else session.original_filename.replace('.csv', '')
        processed_filename = f'csv_mapped_{session_id}_{session.original_filename}'
        relative_path = f'uploads/event_logs/{request.user.id}/{processed_filename}'
        full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Save processed CSV
        df.to_csv(full_path, index=False)
        
        # Create EventLogJob (this will trigger the discovery process)
        job = EventLogJob.objects.create(
            user=request.user,
            original_file=relative_path,
            original_filename=session.original_filename,
            project_name=project_name,
            cleaning_enabled=cleaning_enabled,
            mining_method=mining_method,
            status='pending',
            progress=0,
            message='CSV imported, starting discovery...'
        )
        
        # Delete the import session and temp file
        session.uploaded_file.delete()
        session.delete()        # Start processing in background thread
        thread = threading.Thread(target=process_job, args=(job.id,))
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'success': True,
            'job_id': job.id,
            'project_name': project_name,
            'redirect_url': f'/jobs/progress/{job.id}/'
        })
        
    except CSVImportSession.DoesNotExist:
        return JsonResponse({'error': 'CSV import session not found'}, status=404)
    except Exception as e:
        import traceback
        print(f"CSV import processing error: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)
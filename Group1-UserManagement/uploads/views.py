from django.shortcuts import render, redirect, get_object_or_404
from django.http import FileResponse, HttpResponseForbidden
from django.contrib.auth.decorators import login_required
from .forms import UploadFileForm
from .models import UploadedFile


@login_required
def uploads_list_view(request):
    # Allow analysts OR admins
    if not (request.user.roles.filter(name='Analyst').exists() or request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('دسترسی ندارید.')
    form = UploadFileForm()

    # Filtering and sorting
    q = request.GET.get('q', '').strip()
    sort = request.GET.get('s', 'date')  # date|name|size|user
    order = request.GET.get('o', 'desc')  # asc|desc

    files = UploadedFile.objects.select_related('uploader')
    if q:
        from django.db.models import Q
        files = files.filter(
            Q(original_name__icontains=q) |
            Q(description__icontains=q) |
            Q(uploader__username__icontains=q)
        )

    sort_map = {
        'date': 'uploaded_at',
        'name': 'original_name',
        'size': 'size_bytes',
        'user': 'uploader__username',
    }
    sort_field = sort_map.get(sort, 'uploaded_at')
    if order == 'desc':
        sort_field = '-' + sort_field
    files = files.order_by(sort_field)

    return render(request, 'uploads/list.html', {
        'form': form,
        'files': files,
        'q': q,
        's': sort,
        'o': order,
    })


@login_required
def upload_file_view(request):
    if not (request.user.roles.filter(name='Analyst').exists() or request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('دسترسی ندارید.')
    if request.method != 'POST':
        return HttpResponseForbidden('روش مجاز نیست.')
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
        instance = form.save(commit=False)
        up_file = request.FILES['file']
        instance.original_name = up_file.name
        instance.uploader = request.user
        # Populate metadata
        instance.size_bytes = getattr(up_file, 'size', 0) or 0
        instance.content_type = getattr(up_file, 'content_type', '') or ''
        instance.save()
        return redirect('uploads_list')
    files = UploadedFile.objects.select_related('uploader').all()
    return render(request, 'uploads/list.html', {'form': form, 'files': files})


@login_required
def download_file_view(request, pk: int):
    file_obj = get_object_or_404(UploadedFile, pk=pk)
    # Only allow analysts or admins to download
    if not (request.user.roles.filter(name='Analyst').exists() or request.user.roles.filter(name='Admin').exists()):
        return HttpResponseForbidden('دسترسی ندارید.')
    response = FileResponse(file_obj.file.open('rb'), as_attachment=True, filename=file_obj.original_name)
    return response


@login_required
def delete_file_view(request, pk: int):
    file_obj = get_object_or_404(UploadedFile, pk=pk)
    is_owner = (file_obj.uploader_id == request.user.id)
    is_admin = request.user.roles.filter(name='Admin').exists()
    if not (is_owner or is_admin):
        return HttpResponseForbidden('فقط ادمین یا آپلودکننده می‌تواند حذف کند.')
    if request.method == 'POST':
        # Delete file from storage then DB
        try:
            storage_file = file_obj.file
            file_obj.delete()  # deletes DB row
            try:
                storage_file.delete(save=False)
            except Exception:
                pass
        except Exception:
            pass
        return redirect('uploads_list')
    return HttpResponseForbidden('روش مجاز نیست.')

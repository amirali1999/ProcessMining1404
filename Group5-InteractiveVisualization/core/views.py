from django.shortcuts import render

# Create your views here.

def base_view(request):
        return render(request, 'core/base.html') 


def dashboard_view(request):
        return render(request, 'core/dashboard.html')


def petri_net_view(request):
        return render(request, 'core/petri-net.html')


def upload_view(request):
        return render(request, 'core/upload.html')
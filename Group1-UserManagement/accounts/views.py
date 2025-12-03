from functools import wraps
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render
from .forms import RegisterForm
from .models import Role

User = get_user_model()


def role_required(role_name: str):
    def decorator(view_func):
        @login_required
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if request.user.roles.filter(name=role_name).exists():
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden('دسترسی مجاز نیست.')
        return _wrapped
    return decorator


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.email = form.cleaned_data['email'].lower()
            user.save()
            analyst_role, _ = Role.objects.get_or_create(name='Analyst')
            user.roles.add(analyst_role)
            login(request, user)
            messages.success(request, 'ثبت‌نام با موفقیت انجام شد.')
            return redirect('dashboard')
    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.get_user()
        login(request, user)
        return redirect('dashboard')
    return render(request, 'accounts/login.html', {'form': form})


@login_required
def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard_view(request):
    return render(request, 'dashboard.html')


@role_required('Admin')
def admin_only_view(request):
    return render(request, 'admin_only.html')

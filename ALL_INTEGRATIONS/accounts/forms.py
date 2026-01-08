from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from .models import LicenseCode

User = get_user_model()


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'your.email@example.com'
        })
    )
    
    activate_premium = forms.BooleanField(
        required=False,
        label='فعال‌سازی نسخه پرمیوم',
        help_text='اگر کد لایسنس پرمیوم دارید، این گزینه را فعال کنید'
    )
    
    license_code = forms.CharField(
        required=False,
        max_length=50,
        label='کد لایسنس',
        help_text='کد فعال‌سازی پرمیوم را وارد کنید',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'PREMIUM-XXXX-XXXX',
            'autocomplete': 'off'
        })
    )

    class Meta:
        model = User
        fields = ('username', 'email')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'username'
            })
        }

    def clean_email(self):
        email = self.cleaned_data['email'].lower()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('این ایمیل قبلاً ثبت شده است.')
        return email
    
    def clean(self):
        cleaned_data = super().clean()
        activate_premium = cleaned_data.get('activate_premium')
        license_code = cleaned_data.get('license_code')
        
        if activate_premium and not license_code:
            raise forms.ValidationError('لطفاً کد لایسنس را وارد کنید.')
        
        if license_code:
            # Validate license code
            try:
                code_obj = LicenseCode.objects.get(
                    code=license_code.strip().upper(),
                    is_active=True,
                    is_used=False
                )
            except LicenseCode.DoesNotExist:
                raise forms.ValidationError('کد لایسنس نامعتبر یا استفاده شده است.')
            
            # Store validated code object for use in view
            self.validated_license = code_obj
        
        return cleaned_data
    
    def save(self, commit=True):
        user = super().save(commit=False)
        
        # Check if premium should be activated
        if hasattr(self, 'validated_license'):
            # User will be premium
            user.license_type = User.LICENSE_PREMIUM
        else:
            # Default free user with limitations
            user.license_type = User.LICENSE_FREE
            user.max_log_rows = 1000
            user.max_projects = 3
            user.allowed_algorithms = ['alpha']  # Only Alpha miner
        
        if commit:
            user.save()
            
            # Activate premium license if code was provided
            if hasattr(self, 'validated_license'):
                self.validated_license.use_code(user)
        
        return user


class LicenseActivationForm(forms.Form):
    """Form for activating premium license after registration"""
    license_code = forms.CharField(
        max_length=50,
        label='کد فعال‌سازی پرمیوم',
        help_text='کد لایسنس خود را وارد کنید',
        widget=forms.TextInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'PREMIUM-XXXX-XXXX-XXXX',
            'autocomplete': 'off',
            'style': 'text-transform: uppercase; letter-spacing: 2px; font-family: monospace;'
        })
    )
    
    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
    
    def clean_license_code(self):
        code = self.cleaned_data['license_code'].strip().upper()
        
        # Check if user is already premium
        if self.user and self.user.is_premium:
            raise forms.ValidationError('شما در حال حاضر کاربر پرمیوم هستید.')
        
        # Validate license code
        try:
            code_obj = LicenseCode.objects.get(
                code=code,
                is_active=True,
                is_used=False
            )
        except LicenseCode.DoesNotExist:
            raise forms.ValidationError(
                'کد لایسنس نامعتبر است، قبلاً استفاده شده یا غیرفعال است.'
            )
        
        # Store validated code object
        self.validated_license = code_obj
        
        return code
    
    def activate(self):
        """Activate premium license for user"""
        if hasattr(self, 'validated_license') and self.user:
            self.validated_license.use_code(self.user)
            return True
        return False

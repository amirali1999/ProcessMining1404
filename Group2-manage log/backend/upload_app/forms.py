from django import forms
from .models import EventLog

class EventLogUploadForm(forms.ModelForm):
    
    class Meta:
        model = EventLog
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv,.xes',
                'id': 'fileInput'
            })
        }
        labels = {
            'file': 'انتخاب فایل Event Log'
        }
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        
        if not file:
            raise forms.ValidationError('لطفاً یک فایل انتخاب کنید.')
        
        valid_extensions = ['.csv', '.xes']
        file_ext = file.name.lower().split('.')[-1]
        
        if f'.{file_ext}' not in valid_extensions:
            raise forms.ValidationError(
                f'فقط فایل‌های CSV و XES مجاز هستند. فرمت شما: .{file_ext}'
            )
        
        max_size = 50 * 1024 * 1024  # 50MB
        if file.size > max_size:
            size_mb = file.size / (1024 * 1024)
            raise forms.ValidationError(
                f'حجم فایل ({size_mb:.2f}MB) بیشتر از حد مجاز (50MB) است.'
            )
        
        return file
    
    def save(self, commit=True, user=None):
        instance = super().save(commit=False)
        instance.original_filename = self.cleaned_data['file'].name
        instance.file_size = self.cleaned_data['file'].size
        
        if user and user.is_authenticated:
            instance.uploaded_by = user
        
        if commit:
            instance.save()
        
        return instance

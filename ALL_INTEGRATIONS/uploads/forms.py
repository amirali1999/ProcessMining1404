from django import forms
from .models import UploadedFile


class UploadFileForm(forms.ModelForm):
    file = forms.FileField(label='فایل', help_text='هر فرمتی مجاز است.')
    description = forms.CharField(label='توضیحات', required=False)

    class Meta:
        model = UploadedFile
        fields = ['file', 'description']

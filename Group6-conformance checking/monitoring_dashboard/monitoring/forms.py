
from django import forms

class ConformanceUploadForm(forms.Form):
    pnml_file = forms.FileField(label="مدل فرآیند (PNML)")
    log_csv = forms.FileField(label="لاگ رویداد (CSV)")

# Generated migration for EventLog model

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('uploads', '0002_uploadedfile_content_type_uploadedfile_size_bytes'),
    ]

    operations = [
        migrations.CreateModel(
            name='EventLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='Human-readable log name', max_length=255)),
                ('file_type', models.CharField(choices=[('csv', 'CSV'), ('xes', 'XES'), ('parquet', 'Parquet')], default='csv', max_length=10)),
                ('cleaned_file_path', models.FileField(blank=True, help_text='Path to cleaned/preprocessed log file (created by Group 3)', null=True, upload_to='cleaned_logs/%Y/%m/%d')),
                ('meta_info', models.JSONField(blank=True, default=dict, help_text='Metadata: cases, events, activities, time_range, variants, etc.')),
                ('default_source_for_downstream', models.CharField(choices=[('raw', 'Raw Data'), ('cleaned', 'Cleaned Data')], default='raw', help_text='Which version to use for discovery/conformance/prediction', max_length=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('uploaded_file', models.OneToOneField(help_text='Reference to the uploaded file (Group 2)', on_delete=django.db.models.deletion.CASCADE, related_name='event_log', to='uploads.uploadedfile')),
            ],
            options={
                'verbose_name': 'Event Log',
                'verbose_name_plural': 'Event Logs',
                'ordering': ['-created_at'],
            },
        ),
    ]

from django.core.management.base import BaseCommand
from translations.models import Translation


class Command(BaseCommand):
    help = 'Populate database with initial bilingual translations'

    def handle(self, *args, **options):
        translations_data = [
            # Navigation and General
            {'phrase': 'welcome', 'fa': 'خوش آمدید', 'en': 'Welcome'},
            {'phrase': 'guest', 'fa': 'میهمان', 'en': 'Guest'},
            {'phrase': 'roles', 'fa': 'نقش‌ها', 'en': 'Roles'},
            {'phrase': 'login', 'fa': 'ورود', 'en': 'Login'},
            {'phrase': 'logout', 'fa': 'خروج', 'en': 'Logout'},
            {'phrase': 'register', 'fa': 'ثبت‌نام', 'en': 'Register'},
            {'phrase': 'dashboard', 'fa': 'داشبورد', 'en': 'Dashboard'},
            {'phrase': 'admin_panel', 'fa': 'پنل مدیریت', 'en': 'Admin Panel'},
            {'phrase': 'file_management', 'fa': 'مدیریت فایل‌ها', 'en': 'File Management'},
            {'phrase': 'file_management_desc', 'fa': 'آپلود، مشاهده و دانلود فایل‌ها', 'en': 'Upload, view and download files'},
            {'phrase': 'admin_panel_desc', 'fa': 'مدیریت کاربران و نقش‌ها', 'en': 'Manage users and roles'},
            {'phrase': 'shortcut', 'fa': 'میانبر', 'en': 'Shortcut'},
            
            # User Management
            {'phrase': 'username', 'fa': 'نام کاربری', 'en': 'Username'},
            {'phrase': 'password', 'fa': 'رمز عبور', 'en': 'Password'},
            {'phrase': 'email', 'fa': 'ایمیل', 'en': 'Email'},
            {'phrase': 'first_name', 'fa': 'نام', 'en': 'First Name'},
            {'phrase': 'last_name', 'fa': 'نام خانوادگی', 'en': 'Last Name'},
            {'phrase': 'user_id', 'fa': 'شناسه کاربر', 'en': 'User ID'},
            
            # File Upload
            {'phrase': 'uploaded_files', 'fa': 'فایل‌های آپلود شده', 'en': 'Uploaded Files'},
            {'phrase': 'upload_new_file', 'fa': 'آپلود فایل جدید', 'en': 'Upload New File'},
            {'phrase': 'file_name', 'fa': 'نام فایل', 'en': 'File Name'},
            {'phrase': 'description', 'fa': 'توضیحات', 'en': 'Description'},
            {'phrase': 'optional', 'fa': 'اختیاری', 'en': 'Optional'},
            {'phrase': 'upload', 'fa': 'آپلود', 'en': 'Upload'},
            {'phrase': 'download', 'fa': 'دانلود', 'en': 'Download'},
            {'phrase': 'delete', 'fa': 'حذف', 'en': 'Delete'},
            {'phrase': 'select_file', 'fa': 'انتخاب فایل', 'en': 'Select File'},
            {'phrase': 'uploader', 'fa': 'کاربر', 'en': 'Uploader'},
            {'phrase': 'upload_date', 'fa': 'تاریخ', 'en': 'Date'},
            {'phrase': 'file_size', 'fa': 'حجم (MB)', 'en': 'Size (MB)'},
            {'phrase': 'file_type', 'fa': 'نوع/پسوند', 'en': 'Type/Extension'},
            
            # Search and Filter
            {'phrase': 'search', 'fa': 'جستجو', 'en': 'Search'},
            {'phrase': 'sort_by', 'fa': 'مرتب‌سازی', 'en': 'Sort By'},
            {'phrase': 'order', 'fa': 'ترتیب', 'en': 'Order'},
            {'phrase': 'apply', 'fa': 'اعمال', 'en': 'Apply'},
            {'phrase': 'search_placeholder', 'fa': 'نام فایل، توضیحات یا کاربر', 'en': 'File name, description or user'},
            {'phrase': 'sort_date', 'fa': 'تاریخ', 'en': 'Date'},
            {'phrase': 'sort_name', 'fa': 'نام', 'en': 'Name'},
            {'phrase': 'sort_size', 'fa': 'حجم', 'en': 'Size'},
            {'phrase': 'sort_user', 'fa': 'کاربر', 'en': 'User'},
            {'phrase': 'order_desc', 'fa': 'نزولی', 'en': 'Descending'},
            {'phrase': 'order_asc', 'fa': 'صعودی', 'en': 'Ascending'},
            {'phrase': 'search_results', 'fa': 'نتیجه جستجو برای', 'en': 'Search results for'},
            
            # Messages
            {'phrase': 'no_files', 'fa': 'هنوز فایلی آپلود نشده است.', 'en': 'No files uploaded yet.'},
            {'phrase': 'confirm_delete', 'fa': 'حذف این فایل؟', 'en': 'Delete this file?'},
            {'phrase': 'admin_only', 'fa': 'فقط ادمین', 'en': 'Admin Only'},
            {'phrase': 'admin_only_message', 'fa': 'این صفحه فقط برای نقش Admin در دسترس است.', 'en': 'This page is only accessible to Admin role.'},
            
            # Dashboard
            {'phrase': 'view_files', 'fa': 'مشاهده فایل‌ها', 'en': 'View Files'},
            {'phrase': 'upload_file', 'fa': 'آپلود فایل', 'en': 'Upload File'},
            
            # Language
            {'phrase': 'language', 'fa': 'زبان', 'en': 'Language'},
            {'phrase': 'persian', 'fa': 'فارسی', 'en': 'Persian'},
            {'phrase': 'english', 'fa': 'انگلیسی', 'en': 'English'},
            
            # Login page
            {'phrase': 'no_account', 'fa': 'حساب ندارید؟', 'en': "Don't have an account?"},
            {'phrase': 'django_admin', 'fa': 'پنل ادمین Django', 'en': 'Django Admin Panel'},
            {'phrase': 'have_account', 'fa': 'قبلاً ثبت‌نام کرده‌اید؟', 'en': 'Already have an account?'},
            {'phrase': 'confirm_password', 'fa': 'تکرار رمز عبور', 'en': 'Confirm Password'},
            
            # Additional phrases
            {'phrase': 'project_title', 'fa': 'پروژه فرآیندکاوی', 'en': 'Process Mining Project'},
            {'phrase': 'admin_page', 'fa': 'صفحه ادمین', 'en': 'Admin Page'},
            {'phrase': 'user_management', 'fa': 'مدیریت کاربران', 'en': 'User Management'},
            {'phrase': 'footer_text', 'fa': 'پروژه فرآیندکاوی - دانشگاه فردوسی مشهد', 'en': 'Process Mining Project - Ferdowsi University of Mashhad'},
            {'phrase': 'size', 'fa': 'حجم', 'en': 'Size'},
            {'phrase': 'type', 'fa': 'نوع', 'en': 'Type'},
        ]

        created_count = 0
        updated_count = 0

        for data in translations_data:
            translation, created = Translation.objects.update_or_create(
                phrase=data['phrase'],
                defaults={'fa': data['fa'], 'en': data['en']}
            )
            if created:
                created_count += 1
            else:
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully populated translations: {created_count} created, {updated_count} updated'
            )
        )

from django.db import models


class Translation(models.Model):
    """
    Model for storing bilingual translations
    Each phrase has Persian and English versions with text direction
    """
    phrase = models.CharField(max_length=100, unique=True, help_text="Unique key for translation (e.g., 'user_id', 'welcome_message')")
    fa = models.CharField(max_length=500, help_text="Persian translation")
    en = models.CharField(max_length=500, help_text="English translation")
    
    class Meta:
        ordering = ['phrase']
        verbose_name = "Translation"
        verbose_name_plural = "Translations"
    
    def __str__(self) -> str:
        return f"{self.phrase}: {self.fa} / {self.en}"
    
    def get_translation(self, language: str = 'fa') -> str:
        """Get translation for specified language"""
        return self.fa if language == 'fa' else self.en
    
    @staticmethod
    def get_direction(language: str = 'fa') -> str:
        """Get text direction for specified language"""
        return 'rtl' if language == 'fa' else 'ltr'

from django import template
from translations.models import Translation

register = template.Library()


@register.simple_tag(takes_context=True)
def trans(context, phrase_key: str) -> str:
    """
    Template tag to get translation based on current language
    Usage: {% trans 'phrase_key' %}
    """
    request = context.get('request')
    language = request.session.get('language', 'fa') if request else 'fa'
    
    try:
        translation = Translation.objects.get(phrase=phrase_key)
        return translation.get_translation(language)
    except Translation.DoesNotExist:
        # Return the key itself if translation not found (for debugging)
        return f"[{phrase_key}]"


@register.simple_tag(takes_context=True)
def get_direction(context) -> str:
    """
    Get text direction based on current language
    Usage: {% get_direction %}
    """
    request = context.get('request')
    language = request.session.get('language', 'fa') if request else 'fa'
    return Translation.get_direction(language)


@register.simple_tag(takes_context=True)
def get_language(context) -> str:
    """
    Get current language code
    Usage: {% get_language %}
    """
    request = context.get('request')
    return request.session.get('language', 'fa') if request else 'fa'

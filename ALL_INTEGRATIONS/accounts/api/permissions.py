from rest_framework.permissions import BasePermission


class IsAdminRole(BasePermission):
    """Allows access only to users having the Admin role property."""

    def has_permission(self, request, view):  # type: ignore[override]
        user = request.user
        return bool(user and user.is_authenticated and getattr(user, "is_admin", False))

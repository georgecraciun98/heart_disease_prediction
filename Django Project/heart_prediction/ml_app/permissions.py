from rest_framework import permissions
from django.contrib.auth.models import User

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object to edit it.
    """

    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request,
        # so we'll always allow GET, HEAD or OPTIONS requests.MySQL Database
        if request.method in permissions.SAFE_METHODS:
            return True

        # Write permissions are only allowed to the owner of the snippet.
        return obj.user == request.user





class UserDetailPermission(permissions.BasePermission):
    def has_permission(self, request, view):
        pass
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request,
        # so we'll always allow GET, HEAD or OPTIONS requests.

        user=User(pk=request.user.pk)
        if user.has_perm('ml_app | user detail model | Can add user detail model'):
            return True

        # Write permissions are only allowed to the owner of the snippet.
        return False
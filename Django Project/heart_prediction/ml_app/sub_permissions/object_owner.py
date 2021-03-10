from rest_framework import permissions
from django.contrib.auth.models import User

class IsDetailOwner(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.user.groups.filter(name='doctor'):
            return True
        return False

class IsPatient(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.user.groups.filter(name='patient'):
            return True
        return False
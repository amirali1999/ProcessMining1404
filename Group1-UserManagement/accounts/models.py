from django.contrib.auth.models import AbstractUser
from django.db import models


class Role(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self) -> str:
        return self.name


class User(AbstractUser):
    roles = models.ManyToManyField(Role, blank=True, related_name='users')

    @property
    def is_admin(self) -> bool:
        return self.roles.filter(name='Admin').exists()

    @property
    def is_analyst(self) -> bool:
        return self.roles.filter(name='Analyst').exists()

from django.urls import path
from .views import *

urlpatterns = [
    path('', get_image_data, name='get_image_data')
]

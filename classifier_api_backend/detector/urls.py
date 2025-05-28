from django.urls import path
from .views import detect_hate_speech,home

urlpatterns = [
    path('api/detect/', detect_hate_speech),
    path('', home, name='home'),

]

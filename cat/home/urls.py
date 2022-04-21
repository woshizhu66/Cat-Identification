from django.urls import path
from . import views

app_name = "home"

urlpatterns = [
    path("", views.index, name="index"),
    path("change_flag/", views.change_flag, name="change_flag"),
]

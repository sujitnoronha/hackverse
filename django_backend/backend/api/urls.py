from django.urls import path
from . import views
from django.views.decorators.csrf import csrf_exempt
from api.views import *


urlpatterns = [
    path('', dashboard, name="dashboard"),
    path('face/<int:id>/',facedetect, name="facedetector"),
    path('lastimages/', lastimage,name="imageslast"),
    path('apianalytics/', csrf_exempt(views.AnInfo.as_view()), name="analytics"),

]
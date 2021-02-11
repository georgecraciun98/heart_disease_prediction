from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .apps import MlAppConfig
from rest_framework.decorators import api_view
import os
import numpy as np
import pandas as pd
import time
import glob
import requests
from scipy import ndimage
from scipy.ndimage import zoom
from django.apps import apps
@api_view(["POST"])

def check_result(request):
    #Get video file url
    url = request.POST.get('url')
    print(url)

    emo_jso={'Url':url,'Feeling':'Good'}
    model=apps.get_app_config('ml_app').loaded_model
    print(model.summary())
    return JsonResponse(emo_jso, safe=False)
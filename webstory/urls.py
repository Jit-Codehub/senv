from django.urls import path,include
from .views import *

urlpatterns = [
    path('',homepage, name='home'),
    path('magazine/',magazine,name='magazine'),
    path('news/',news,name='news'),
    path('fashion/',fashion,name='fashion'),
    path('lifestyle/',lifestyle,name='lifestyle'),
    path('entertainment/',entertainment,name='entertainment'),
    path('<str:category>/<str:story>/',webstories,name='webstories')
]
# import webstory.jobs  # NOQA @isort:skip
# import logging
# logging.basicConfig(level="DEBUG")
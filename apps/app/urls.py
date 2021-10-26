# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.app import views

urlpatterns = [

    # The home page
    path('', views.index, name='home'),
    path('promethee', views.promethee, name='promethee'),
    path('apriori', views.apriori, name='apriori'),
    path('genetic_algorithm', views.ag, name='genetic_a'),

    # Matches any html file
    re_path(r'^.*\.*', views.pages, name='pages'),

]

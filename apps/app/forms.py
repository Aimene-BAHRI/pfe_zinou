from django import forms
from django.db.models import fields
from django.db.models.base import Model
from .models import Data

import pandas as pd
from django.contrib.auth.models import User

class DataForm(forms.ModelForm):
    user = forms.CharField(widget = forms.Textarea(attrs = {'hidden': ''}))

    class Meta:
        model = Data
        fields = ('user', 'nuissance', 'bruit', 'impacts', 'geographique', 'equipement', 'accessibilite', 'climat')


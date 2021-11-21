# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models
from django.contrib.auth.models import User
import json
# Create your models here.
ROLE_CHOICES = (
    ("initiateur", 'Initiateur'),
    ("decideur", 'Decideur'),
)
class Data(models.Model):
	user = models.OneToOneField(User, on_delete=models.CASCADE, related_name = "datas")
	mp = models.FileField(upload_to='media/Matrice_de_performance/', blank = True, null = True) #Matrice de Performance 
	role = models.CharField(max_length = 200, choices = ROLE_CHOICES, default="decideur")
	nuissance = models.FloatField(null=True)
	bruit = models.FloatField(null=True, )
	impacts = models.FloatField(null=True, )
	geographique = models.FloatField(null=True, )
	equipement = models.FloatField(null=True, )
	accessibilite = models.FloatField(null=True, )
	climat = models.FloatField(null=True, )
	
	def __str__(self):
		return "".join(str(self.user) + str(self.role))

	def get_weights(self):
		s = []
		s.extend([self.nuissance, self.bruit, self.impacts, self.geographique, self.equipement, self.accessibilite, self.climat])
		return s

from django.db.models.signals import post_save
from django.dispatch import receiver
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Data.objects.create(user=instance)


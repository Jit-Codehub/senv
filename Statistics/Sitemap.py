from django.contrib.sitemaps import Sitemap
from .urls import urlpatterns
from django.shortcuts import reverse
from datetime import datetime
from django.contrib import admin
from django.urls import path, include
from .models import *

class Median_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Median_Model.objects.all()

    def location(self, obj):
        return "/statistics/median-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class Mean_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Mean_Model.objects.all()

    def location(self, obj):
        return "/statistics/mean-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class Mode_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Mode_Model.objects.all()

    def location(self, obj):
        return "/statistics/mode-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class First_Quartile_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return First_Quartile_Model.objects.all()

    def location(self, obj):
        return "/statistics/first-quartile-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class Third_Quartile_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Third_Quartile_Model.objects.all()

    def location(self, obj):
        return "/statistics/third-quartile-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class Maximum_Number_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Maximum_Number_Model.objects.all()

    def location(self, obj):
        return "/statistics/maximum-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class Minimum_Number_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Minimum_Number_Model.objects.all()

    def location(self, obj):
        return "/statistics/minimum-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

class Five_Summary_Model_Sitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=5000
    def items(self):
        return Five_Summary_Model.objects.all()

    def location(self, obj):
        return "/statistics/five-number-summary-of-"+obj.slug+"/"

    def lastmod(self, obj): 
        return obj.generateDate

statistics={
    'median-sitemap':Median_Model_Sitemap,
    'mean-sitemap':Mean_Model_Sitemap,
    'mode-sitemap':Mode_Model_Sitemap,
    'first-quartile-sitemap':First_Quartile_Model_Sitemap,
    'third-quartile-sitemap':Third_Quartile_Model_Sitemap,
    'maximum-number-sitemap':Maximum_Number_Model_Sitemap,
    'minimum-number-sitemap':Minimum_Number_Model_Sitemap,
    'five-number-summary-sitemap':Five_Summary_Model_Sitemap,
}
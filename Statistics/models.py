from django.db import models

# Create your models here.
class Median_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class Mean_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class Mode_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class First_Quartile_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class Third_Quartile_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class Maximum_Number_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class Minimum_Number_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()

class Five_Summary_Model(models.Model):
    inputEnter = models.CharField(max_length=250)
    detailStep = models.TextField()
    finalAnswer = models.CharField(max_length=300)
    slug = models.CharField(max_length=300)
    solutionTitle = models.CharField(max_length=250)
    generateDate = models.DateField()
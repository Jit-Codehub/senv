from django.shortcuts import render
from portal.models import *;
def home(request):
    portal_obj = HomepageBlog.objects.filter(app="statistics",status="Published")
    context = {"portal_obj":portal_obj.first() if portal_obj else None}
    return render(request,'base/statistics-calculator.html',context)
def aboutus(request):
         return render(request,'base/about_us.html')
def contactus(request):
         return render(request,'base/contact_us.html')
def disclaimer(request):
         return render(request,'base/disclaimer.html')
def privacy_policy(request):
         return render(request,'base/privacy_policy.html')

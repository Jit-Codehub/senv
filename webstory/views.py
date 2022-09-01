from re import I
from django.shortcuts import render
import json
from portal.models import *
def lifestyle(request):
    file=open('media/web-stories/title-description-pinkvilla-lifestyle.json')
    data=json.load(file)
    context={}
    context['titles']=data
   
    context['category']='lifestyle'
    return render(request,'web_stories/home.html',context)

def fashion(request):
    file=open('media/web-stories/title-description-pinkvilla-fashion.json')
    data=json.load(file)
    context={}
 
    context['titles']=data
   
    context['category']='fashion'
    return render(request,'web_stories/home.html',context)

def entertainment(request):
    file=open('media/web-stories/title-description-pinkvilla-entertainment.json')
    data=json.load(file)
    context={}
    context['titles']=data
    context['category']='entertainment'
    return render(request,'web_stories/home.html',context)

def news(request):
    file=open('media/web-stories/title-description.json')
    data=json.load(file)
    context={}
    context['titles']=data
    context['category']='news'
    return render(request,'web_stories/home.html',context)
def homepage(request):
    portal_obj = HomepageBlog.objects.filter(app="web-stories",status="Published")
    context = {"portal_obj":portal_obj.first() if portal_obj else None}
    return render(request,'web_stories/home_base.html',context)

def magazine(request):
    file=open('media/web-stories/title-description-vogue.json')
    data=json.load(file)
    context={}
    context['titles']=data
    context['category']='magazine'
    return render(request,'web_stories/home.html',context)
#web stories
def webstories(request,story,category):
    context={}
    webno=story
    file=open('media/web-stories/webstory-'+str(webno)+"/webstory-"+str(webno)+".json")
    data=json.load(file)
    context['data']=data
    context['webno']=webno
    if category == 'fashion':
        file=open('media/web-stories/title-description-pinkvilla-fashion.json')
    elif category == 'lifestyle':
        file=open('media/web-stories/title-description-pinkvilla-lifestyle.json')
    elif category=='entertainment':
        file=open('media/web-stories/title-description-pinkvilla-entertainment.json')
    else:
        file=open('media/web-stories/title-description-vogue.json')
    data=json.load(file)
    for i in data:
        if i['url']==webno:
            context['titles']=i
            # return render(request,'web_stories/webstory.html',context)
            print(context)
            return render(request,'web_stories/ZZ.html',context)
    




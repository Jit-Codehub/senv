import requests
import shutil
from bs4 import BeautifulSoup
import os
import json
def download(src,path):
    r = requests.get(src, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
def run():
    file1 = open("vogue.txt","r")
    links=(file1.readlines()) 
    for i in range(len(links)):
        links[i]=links[i].replace("\n","")
    output=[]
    for i in range(len(links)):
        d={}
        Web_url = links[i]
        r = requests.get(Web_url)
        soup = BeautifulSoup(r.content, 'html5lib')
        d['content']=soup.find('meta',{'property':['og:description']}).get('content')
        d['title']=soup.find('title').text
        url=Web_url
        url=list(url.split('/'))
        d['url']=url[-2]
        print(d)
        output.append(d)
    json_dump=json.dumps(output)
    file_name="title-description.json"
    with open(file_name, "w") as outfile:
        outfile.write(json_dump)
    for i in range(len(links)):
        Web_url = links[i]
        print(Web_url)
        r = requests.get(Web_url)
        webstoryno=i+1
        os.makedirs("webstory-"+str(webstoryno))
        soup = BeautifulSoup(r.content, 'html5lib')
        stories=soup.findAll('amp-story-page')
        output=[]
        items=0
        img_count=1
        for st in stories:
            story={}
            story['url']=Web_url.split('/')[-2]
            headings=(st.findAll('h1'))
            subheadings=(st.findAll('h2'))
            paras=(st.findAll('p'))
            story['heading']=[]
            for heading in headings:
                story['heading'].append(heading.text)
            story['subheading']=[]
            for subheading in subheadings:
                story['subheading'].append(subheading.text)
            story['para']=[]
            for para in paras:
                story['para'].append(para.text)
            image=st.find('amp-img')
            poster=st.find('amp-video')
            video=st.findAll('source')
            story['image']=""
            story['poster']=""
            story['video']=""
            if image:
                items+=1
                src=image['src']
                img_src=src
                src=list(src.split('/'))
                src="webstory-"+str(webstoryno)+"-"+str(img_count)+".jpg"
                img_count+=1
                story['image']=src
                src="webstory-"+str(webstoryno)+'/'+src
                
                try:
                    download(img_src,src)
                except:
                    print(img_src,src,webstoryno)
            if poster:
                items+=1
                src=poster['poster']
                post_src=src
                src=list(src.split('/'))
                src="webstory-"+str(webstoryno)+"-"+str(img_count)+".jpg"
                img_count+=1
                story['poster']=src
                src="webstory-"+str(webstoryno)+'/'+src
                try:
                    download(post_src,src)
                except:
                    print(post_src,src,webstoryno)
            if video:
                items+=1
                src=video[-1]['src']
                video_src=src
                src=list(src.split('/'))
                src="webstory-"+str(webstoryno)+"-"+str(img_count)+".mp4"
                img_count+=1
                story['video']=src
                src="webstory-"+str(webstoryno)+'/'+src
                try:
                    download(video_src,src)
                except:
                    print(video_src,src,webstoryno)
            output.append(story)
        print("No of items:",items+1)
        json_dump=json.dumps(output)
        file_name="webstory-"+str(webstoryno)+'/'+"webstory-"+str(webstoryno)+".json"
        with open(file_name, "w") as outfile:
            outfile.write(json_dump)
run()
from django.contrib.sitemaps import Sitemap
import os,json



class EntertainmentView(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=2000
    def items(self):
        file=open('media/web-stories/title-description-pinkvilla-entertainment.json')
        data=json.load(file)
        return data
    
    def location(self, item) -> str:
        urls="/web-stories/entertainment/"+item['url']+"/"
        return urls



class FashionView(Sitemap):
    changefreq = "monthly"
    priority = 0.9
    limit=2000
    def items(self):
        file=open('media/web-stories/title-description-pinkvilla-fashion.json')
        data=json.load(file)
        return data
    
    def location(self, item) -> str:
        urls="/web-stories/news/"+item['url']+"/"
        return urls




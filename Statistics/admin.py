from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(Median_Model)
admin.site.register(Mean_Model)
admin.site.register(Mode_Model)
admin.site.register(First_Quartile_Model)
admin.site.register(Third_Quartile_Model)
admin.site.register(Maximum_Number_Model)
admin.site.register(Minimum_Number_Model)
admin.site.register(Five_Summary_Model)
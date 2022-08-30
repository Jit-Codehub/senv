from django import template
import re
register = template.Library()


def find(value):
    try:
        value=value[0]
        if value.lower().find('usa today')!=-1:
            print(value.lower(),value.lower().find('usa today'))
            return False
        if value.lower().find('10best')!=-1:
            return False
        print(value.lower())
        return True
    except:
        return False

register.filter('find', find)
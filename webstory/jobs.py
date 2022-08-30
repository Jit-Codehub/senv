from .task import *;
from apscheduler.schedulers.background import BackgroundScheduler

from django_apscheduler.jobstores import DjangoJobStore, register_events, register_job

scheduler = BackgroundScheduler({'apscheduler.job_defaults.max_instances': 4})
scheduler.add_jobstore(DjangoJobStore(), "default")

@register_job(scheduler, "interval",hours=1002, replace_existing=True)
def usa_today_scrapper_job():
    pinkvilla('pinkvilla-entertainment')
    # pinkvilla('pinkvilla-fashion')
    # pinkvilla('pinkvilla-lifestyle')
    # vogue()
    
register_events(scheduler)

scheduler.start()
print("Scheduler started!")




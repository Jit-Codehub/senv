from django.shortcuts import render,redirect
import datetime

import matplotlib
from .utilities import *
from .models import *
from django.contrib import messages
import numpy as np
from scipy.stats import binom,norm,t,lognorm
import random
import operator as op
from functools import reduce
from .main import *
import numpy as np
from .mannwhitney import *
import matplotlib.pyplot as plt
import random
import operator as op
from functools import reduce
import os
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean as me
from scipy.stats import sem
from scipy.stats import t

from math import pow

from django.shortcuts import render, redirect , HttpResponse
import math 
import numpy as np
import scipy.stats as stats
import statistics
import collections
from django.contrib import messages
import random
# Create your views here.
#exponential_growth_prediction
from django.shortcuts import render, redirect , HttpResponse
import math 
import numpy as np
import scipy.stats as stats
import statistics
import collections
from django.contrib import messages

# Create your views here

#odds
def odds(request):
    try:
        cs = str(request.POST.get('cs'))
        cl = str(request.POST.get('cl'))
    

        if request.method == "POST":
            cl = float(request.POST.get('cl'))
            cs = float(request.POST.get('cs'))
            al = cs+cl
            pw = cs / al
            pl = cl / al

            context = {
                'cl':cl,
                'cs':cs,
                'pw':pw,
                'pl':pl,
                'result':re,
                'ot':True
            }
            return render(request,'Statistics/odds.html',context)
        return render(request,'Statistics/odds.html',{'cl':4,'cs':5,'ot':False})
    except Exception as e:
        print(e)
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/odds.html',{'cl':4,'cs':5,'ot':False})

#accuracy
def accuracy(request):
    try:
        Given = request.POST.get('Given','form1')
        tp = request.POST.get('tp')
        tn = request.POST.get('tn')
        fp = request.POST.get('fp')
        fn = request.POST.get('fn')
        p = request.POST.get('p')
        ov = request.POST.get('ov')
        av = request.POST.get('av')
    

        if request.method == "POST":
            if Given == 'form1' and tp and tn and fn and fp and request.method == "POST":
                Given = request.POST.get('Given')
                tp = float(request.POST.get('tp'))
                tn = float(request.POST.get('tn'))
                fp = float(request.POST.get('fp'))
                fn = float(request.POST.get('fn'))

                re = ((tp + tn) / (tp + tn + fp + fn))*100

                context = {
                'tp':tp,
                'Given':Given,
                'tn':tn,
                'fp':fp,
                'fn':fn,
                'result':round(re,4),
                'ot':True
                }
                return render(request,'Statistics/accuracy.html',context)

            elif Given == 'form2' and p and tp and tn and fn and fp and request.method == "POST":
                Given = request.POST.get('Given')
                p = float(request.POST.get('p'))
                tp = float(request.POST.get('tp'))
                tn = float(request.POST.get('tn'))
                fp = float(request.POST.get('fp'))
                fn = float(request.POST.get('fn'))

                sp = (tn / (fp + tn))
                se = (tp / (tp + fn))

                re = (((se)* (p/100)) + ((sp)* (1 - (p/100))))*100


                context = {
                'tp':tp,
                'Given':Given,
                'tn':tn,
                'fp':fp,
                'fn':fn,
                'sp':round(sp,4),
                'se':round(se,4),
                'p':p,
                'result':round(re,4),
                'ot':True
                }
                return render(request,'Statistics/accuracy.html',context)

            elif Given == 'form3' and av and ov and request.method == "POST":
                Given = request.POST.get('Given')
                ov = float(request.POST.get('ov'))
                av = float(request.POST.get('av'))

                re = (abs(ov - av) / av) * 100

                context = {
                'ov':ov,
                'av':av,
                'Given':Given,
                'result':round(re,4),
                'ot':True
                }
                return render(request,'Statistics/accuracy.html',context)
            return render(request,'Statistics/accuracy.html',{'ov':50,'av':25,'tp':30,'tn':50,'fp':40,'fn':60,'p':40,'ot':False,'Given':Given})
        return render(request,'Statistics/accuracy.html',{'ov':50,'av':25,'tp':30,'tn':50,'fp':40,'fn':60,'p':40,'ot':False,'Given':Given})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/accuracy.html',{'ov':50,'av':25,'tp':30,'tn':50,'fp':40,'fn':60,'p':40,'ot':False,'Given':Given})
#random_number
def random_number(request):
    try:
        ul = str(request.POST.get('ul'))
        ll = str(request.POST.get('ll'))
    

        if request.method == "POST":
            ul = int(request.POST.get('ul'))
            ll = int(request.POST.get('ll'))
            re = random.randint(ll,ul)
            context = {
                'ul':ul,
                'll':ll,
                'result':re,
                'ot':True
            }
            return render(request,'Statistics/random_number.html',context)
        return render(request,'Statistics/random_number.html',{'ll':1,'ul':50,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/random_number.html',{'ll':1,'ul':50,'ot':False})

#geometric_distribution
def geometric_distribution(request):
    try:
        nf = request.POST.get('nf')
        ps = request.POST.get('ps')
                
        ot = False
        if request.method == "POST":
            nf = float(request.POST.get('nf'))
            ps = float(request.POST.get('ps'))
            
            m = (1-ps)/ps
            v = (1-ps)/ps**2
            sd = math.sqrt((1-ps)/ps**2)
            re = ((1-ps)**nf) * ps
            
            context = {
            'nf':nf,
            'm':m,
            'v':v,
            'result':round(re,5),
            'sd':sd,
            'ps':ps,
            'ot':True             
             }

            return render(request,'Statistics/geometric_distribution.html',context)
        else:
            return render(request,'Statistics/geometric_distribution.html',{'nf':1,'ps':0.5,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/geometric_distribution.html',{'nf':1,'ps':0.5,'ot':False})

#hypergeometric_distribution
def hypergeometric_distribution(request):
    try:
        N = request.POST.get('N')
        n = request.POST.get('n')
        K = request.POST.get('K')
        k = request.POST.get('k')
        def f(n):
            x=1
            for i in range(1,int(n+1)):
                x=x*i
            return x 
        ot = False
        if request.method == "POST":
            N = float(request.POST.get('N'))
            n = float(request.POST.get('n'))
            K = float(request.POST.get('K'))
            k = float(request.POST.get('k'))
            p=f(K)*f(N-K)*f(n)*f(N-n)
            p1 = f(N)*f(k)*f(K-k)*f(n-k)*f(N-K-n+k)
            re = p/p1            
            context = {
                'N':N,
                'n':n,
                'K':K,
                'k':k,
                'ot':True,
                'result':round(re,5)          
             }

            return render(request,'Statistics/hypergeometric_distribution.html',context)
        else:
            return render(request,'Statistics/hypergeometric_distribution.html',{'N':10,'K':5,'n':10,'k':5,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/hypergeometric_distribution.html',{'N':10,'K':5,'n':10,'k':5,'ot':False})

#negative_binomial_distribution
def negative_binomial_distribution(request):
    try:
        n = request.POST.get('n')
        p = request.POST.get('p')
        r = request.POST.get('r')
                
        ot = False
        if request.method == "POST":
            n = float(request.POST.get('n'))
            p = float(request.POST.get('p'))
            r = float(request.POST.get('r'))
            def f(n):
                x=1
                for i in range(1,int(n+1)):
                    x=x*i
                return x
            nCr = f(n-1)/(f(r-1)* f(n-r))
            re = nCr * (p**r) * ((1-p)**(n-r))
            print(re)
            context = {
            'n':n,
            'p':p,
            'r':r,
            'result':round(re,5),
            'nCr':nCr,
            'ot':True           
             }

            return render(request,'Statistics/negative_binomial_distribution.html',context)
        else:
            return render(request,'Statistics/negative_binomial_distribution.html',{'n':10,'r':5,'p':0.5,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/negative_binomial_distribution.html',{'n':10,'r':5,'p':0.5,'ot':False})


#poisson_distribution
def poisson_distribution(request):
    try:
        y = request.POST.get('y')
        x = request.POST.get('x')
        def f(n):
            x=1
            for i in range(1,int(n+1)):
                x=x*i
            return x       
        ot = False
        if request.method == "POST":
            y = float(request.POST.get('y'))
            x = float(request.POST.get('x'))
            re =  (2.71828**-y)*(y**x) / f(x)
            context = {
            'x':x,
            'y':y,
            'result':round(re,5),
            'ot':True           
             }

            return render(request,'Statistics/poisson_distribution.html',context)
        else:
            return render(request,'Statistics/poisson_distribution.html',{'x':5,'y':6,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/poisson_distribution.html',{'x':5,'y':6,'ot':False})

#exponential_distribution
def exponential_distribution(request):
    try:
        a = request.POST.get('a')
        X = request.POST.get('X')
              
        ot = False
        if request.method == "POST":
            a = float(request.POST.get('a'))
            X = float(request.POST.get('X'))
            p1 = math.exp(-a*X)
            p2 = 1 - math.exp(-a*X)
            m = 1/a
            me = math.log(2)/a
            v = 1/(a**2)
            sd = math.sqrt(1/(a**2))

            context = {
            'sd':round(sd,5),
            'm':round(m,5),
            'me':round(me,5),
            'v':round(v,5),
            'p1':round(p1,5),
            'p2':round(p2,5),
            'X':X,
            'a':a,
            'ot':True           
             }

            return render(request,'Statistics/exponential_distribution.html',context)
        else:
            return render(request,'Statistics/exponential_distribution.html',{'a':5,'X':1,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/exponential_distribution.html',{'X':1,'a':5,'ot':False})

def exponential_growth_prediction(request):
    try:
        r = request.POST.get('r')
        x0 = request.POST.get('x0')
        t = request.POST.get('t')
        ti_op = request.POST.get('ti_op')
        ot = False
        def conv(t,l):
            if t == "min":
                l = l / 525600
            elif t == "hrs":
                l = l / 8760
            elif t == "sec":
                l = l / 3.154e+7
            elif t == "wks":
                l = l / 52.143
            elif t == "days":
                l = l / 365
            elif t == "mon":
                l = l / 12
            return l
        if request.method == "POST":
            r = float(request.POST.get('r'))
            x0 = float(request.POST.get('x0'))
            t = float(request.POST.get('t'))
            ti_op = request.POST.get('ti_op')
            
            tv = conv(ti_op,t)
            re = x0 * (1 + r / 100)**tv
            context = {
            'x0':x0,
            'r':r,
            't':t,
            'ti_op':ti_op,
            'tv':tv,
            'result':re,
            'ot':True                }
            return render(request,'Statistics/exponential_growth_prediction.html',context)
        else:
            return render(request,'Statistics/exponential_growth_prediction.html',{'r':5,'x0':10,'t':1,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/exponential_growth_prediction.html',{'r':5,'x0':10,'t':1,'ot':False})

#binomial_distribution
def binomial_distribution(request):
    try:
        r = request.POST.get('r')
        p = request.POST.get('p')
        n = request.POST.get('n')
        
        ot = False
        if request.method == "POST":
            r = float(request.POST.get('r'))
            p = float(request.POST.get('p'))
            n = float(request.POST.get('n'))
            def facto(n):
                f=1
                for i in range(1,int(n+1)):
                    f=f*i
                return f
            
            nf = facto(n)
            rf = facto(r)
            nrf = facto(n-r)
            ncr = nf/ (rf * nrf)
            pr = p**r
            p1 = 1-p 
            nr = n-r
            re =  ncr * pr * (p1**nr)


            context = {
            'p':p,
            'r':r,
            'n':n,
            'nf':nf,
            'rf':rf,
            'nrf':nrf,
            'pr':pr,
            'p1':p1,
            'nr':nr,
            'ncr':ncr,
            'result':round(re,5),
            'ot':True               
             }

            return render(request,'Statistics/binomial_distribution.html',context)
        else:
            return render(request,'Statistics/binomial_distribution.html',{'r':7,'n':12,'p':0.5,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/binomial_distribution.html',{'r':7,'n':12,'p':0.5,'ot':False})


#chi_square
def chi_square(request):
    try:
        ov = request.POST.get('ov')
        ev = request.POST.get('ev')
        
        
        ot = False
        if request.method == "POST":
            ov = float(request.POST.get('ov'))
            ev = float(request.POST.get('ev'))
            
            re = ((ov - ev)**2) / ev            

            context = {
            'ov':ov,
            'ev':ev,
            'result':round(re,5),
            'ot':True               
             }

            return render(request,'Statistics/chi_square.html',context)
        else:
            return render(request,'Statistics/chi_square.html',{'ov':5,'ev':9,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/chi_square.html',{'ov':5,'ev':9,'ot':False})

#empirical_rule
def empirical_rule(request):
    try:
        m = request.POST.get('m')
        sd = request.POST.get('sd')
        
        
        ot = False
        if request.method == "POST":
            m = float(request.POST.get('m'))
            sd = float(request.POST.get('sd'))
            
            re68 = m - sd
            re682 = m + sd          

            re95 = m - (2*sd)
            re952 = m + (2*sd)

            re99 = m - (3*sd)
            re992 = m + (3*sd)

            context = {
            'm':m,
            'sd':sd,
            're68':re68,
            're682':re682,
            're95':re95,
            're952':re952,
            're99':re99,
            're992':re992,
            'ot':True               
             }

            return render(request,'Statistics/empirical_rule.html',context)
        else:
            return render(request,'Statistics/empirical_rule.html',{'m':5,'sd':2.5,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/empirical_rule.html',{'m':5,'sd':2.5,'ot':False})

#Coefficient_of_Determination
def coefficient_of_determination(request):
    try:
        x = str(request.POST.get('x'))
        y = str(request.POST.get('y'))
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            x2 = x.replace(',',' ')
            y2 = y.replace(',',' ')
            x1 = x2.split(' ')
            y1 = y2.split(' ')
            xo = [float(i) for i in x1]
            yo = [float(i) for i in y1]
            xmean = sum(xo)/len(xo)
            l = []
            for i in xo:
                l.append(i-xmean)
            s1 = [(i**2) for i in l]
            s1 = sum(s1)
            l2 = []
            ymean = sum(yo)/len(yo)
            for i in yo:
                l2.append(i-ymean)
            s2 = [(i**2) for i in l2]
            s2 = sum(s2)
            l3 = []
            for i in range(len(xo)):
                l3.append(l[i]*l2[i])
            asum = sum(l3)
            sq = s1*s2
            r = asum / math.sqrt(sq)
            re = r**2
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'result':re,
                'asum':asum,
                'sq':sq,
                'x':x,
                'y':y,
                'ot':True
            }
            return render(request,'Statistics/Coefficient_of_Determination.html',context)
        return render(request,'Statistics/Coefficient_of_Determination.html',{'x':"1,2,3,4,5",'y':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Coefficient_of_Determination.html')


#Confidence Interval
def confidence_interval(request):
    try:
        m =request.POST.get('m')
        n =request.POST.get('n')
        sd =request.POST.get('sd')
        Given = str(request.POST.get('Given','95%'))
        if Given == "80%":
            z = 1.282
        elif Given == "90%":
            z = 1.645
        elif Given == "95%":
            z = 1.960
        else:
            z = 2.576
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            m = float(request.POST.get('m'))
            n = float(request.POST.get('n'))
            sd = float(request.POST.get('sd'))
            l = m - z * (sd/math.sqrt(n))
            u = m + z * (sd/math.sqrt(n))
            
            context = {
                'l':round(l,5),
                'u':round(u,5),
                'n':n,
                'm':m,
                'z':z,
                'sd':sd,
                'Given':Given,
                'ot':True
            }
            return render(request,'Statistics/Confidence_Interval.html',context)
        return render(request,'Statistics/Confidence_Interval.html',{'ot':False,'n':2,'m':2.5,'sd':2})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Confidence_Interval.html')

#Correlation_Coefficient_Matthews
def correlation_coefficient_matthews(request):
    try:
        tp =request.POST.get('tp')
        fp =request.POST.get('fp')
        tn =request.POST.get('tn')
        fn =request.POST.get('fn')
        
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            tp =float(request.POST.get('tp'))
            fp =float(request.POST.get('fp'))
            tn =float(request.POST.get('tn'))
            fn =float(request.POST.get('fn'))
            re = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
            print(re)
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'result':re,
                'tn':tn,
                'fn':fn,
                'tp':tp,
                'fp':fp,
                'ot':True
            }
            return render(request,'Statistics/Correlation_Coefficient_Matthews.html',context)
        return render(request,'Statistics/Correlation_Coefficient_Matthews.html',{'ot':False,'tn':100,
                'fn':200,
                'tp':100,
                'fp':200,})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Correlation_Coefficient_Matthews.html')


#Point_Estimate
def point_estimate(request):
    try:
        x =request.POST.get('x')
        n =request.POST.get('n')
        Given = str(request.POST.get('Given','95%'))
        if Given == "80%":
            z = 1.282
        elif Given == "85%":
            z = 1.44
        elif Given == "90%":
            z = 1.645
        elif Given == "95%":
            z = 1.960
        else:
            z = 2.576
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            x = float(request.POST.get('x'))
            n = float(request.POST.get('n'))
            
            wi = (x + (z**2)/2) / (n + (z**2))
            la = (x + 1) / (n + 2)
            je = (x + 0.5) / (n + 1)
            mx = x/n
            if mx <= 0.5:
                re = (x + (z**2)/2) / (n + (z**2))
            elif mx == 1.0:
                re = (x + 1) / (n + 2)
            elif 0.5 < mx < 0.9:
                re = x/n
            elif 0.9 <= x/n < 1.0:
                re = (x + 0.5) / (n + 1)
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            redict = {'Maximum Likelihood Estimation (MLE)':mx,'Wilson': wi,'Laplace':la,'Jeffrey''s':je}
            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'result':re,
                'n':n,
                'x':x,
                'z':z,
                'Given':Given,
                'redict':redict,
                'ot':True
            }
            return render(request,'Statistics/Point_Estimate.html',context)
        return render(request,'Statistics/Point_Estimate.html',{'ot':False,'x':4,'n':10})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Point_Estimate.html')

#Relative_Error
def relative_error(request):
    try:
        av =request.POST.get('av')
        mv =request.POST.get('mv')
        
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            av = float(request.POST.get('av'))
            mv = float(request.POST.get('mv'))
            re = (av - mv) / av
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            
            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'result':re,
                'av':av,
                'mv':mv,
                'ot':True
            }
            return render(request,'Statistics/Relative_Error.html',context)
        return render(request,'Statistics/Relative_Error.html',{'ot':False,'av':10,'mv':11,})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Relative_Error.html')

#z_test
def z_test(request):
    ztype = request.POST.get('ztype','form1')
    ap = request.POST.get('ap','form1')
    return render(request,'Statistics/Z-test.html',{'ztype':ztype,'ap':ap})


#Margin_of_Error
def margin_of_error(request):
    try:
        conf =request.POST.get('conf',"95%")
        Given = request.POST.get('Given')
        ss = request.POST.get('ss')
        sp = request.POST.get('sp')
        ps = request.POST.get('ps')
        if conf == "80%":
            z = 1.282
        elif conf == "85%":
            z = 1.44
        elif conf == "90%":
            z = 1.645
        elif conf == "95%":
            z = 1.960
        else:
            z = 2.576
    
        if request.method == "POST":
            if Given == "form2" and sp and ss:
                ss = float(request.POST.get('ss'))
                sp = float(request.POST.get('sp'))
                a = sp * (1-sp)
                a1 = np.sqrt(sp * (1-sp))
                a2 = z * a1
                re = (a2 / np.sqrt(ss))*100
                              
                context = {
                    'result':round(re,4),
                    'sp':sp,
                    'conf':conf,
                    'z':z,
                    'ss':ss,
                    'Given':Given,
                    'ot':True
                }
                return render(request,'Statistics/Margin_of_Error.html',context)
            elif Given == 'form1' and sp and ss and ps:
                ss = float(request.POST.get('ss'))
                sp = float(request.POST.get('sp'))
                ps = float(request.POST.get('ps'))

                re = (z *  np.sqrt(sp * (1 - sp) ))/ np.sqrt((ps - 1) * ss / (ps - ss)) *100
                context = {
                    'result':round(re,4),
                    'sp':sp,
                    'conf':conf,
                    'ss':ss,
                    'ps':ps,
                    'z':z,
                    'Given':Given,
                    'ot':True
                }
                return render(request,'Statistics/Margin_of_Error.html',context)
            return render(request,'Statistics/Margin_of_Error.html',{'ot':False,'Given':Given})
        else:
            return render(request,'Statistics/Margin_of_Error.html',{'ot':False,'ss':10,'ps':1000,'sp':0.5,'Given':'form1'})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Margin_of_Error.html')


#shannon_entropy
def shannon_entropy(request):
    try:
        x = str(request.POST.get('x'))
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            s = [float(i) for i in x1]

            probabilities = [n_x/len(s) for x,n_x in collections.Counter(s).items()]
            e_x = [-p_x*math.log(p_x,2) for p_x in probabilities]
            re = sum(e_x)
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'result':re,
                'x':x,
                'ot':True
            }
            return render(request,'Statistics/Shannon_Entropy.html',context)
        return render(request,'Statistics/Shannon_Entropy.html',{'x':"1,2,3,4,5",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/Shannon_Entropy.html')

#exponential_regression
def exponential_regression(request):
    try:
        x = str(request.POST.get('x'))
        y = str(request.POST.get('y'))
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            x2 = x.replace(',',' ')
            y2 = y.replace(',',' ')
            x1 = x2.split(' ')
            y1 = y2.split(' ')
            xo = [float(i) for i in x1]
            yo = [float(i) for i in y1]
            fit = np.polyfit(xo, np.log(yo), 1)
            e = 2.71828
            e1 = (e**fit[1]) 
            e2 = (e**fit[0])

            context = {
                'e1':round(e1,5),
                'e2':round(e2,5),
                'f1':fit[0],
                'f2':fit[1],
                'x':x,
                'y':y,
                'ot':True
            }
            return render(request,'Statistics/exponential_regression.html',context)
        return render(request,'Statistics/exponential_regression.html',{'x':"1,2,3,4,5",'y':"5,4,3,2,1",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/exponential_regression.html')

#fishers_exact_test
def fishers_exact_test(request):
    try:
        a = request.POST.get('a')
        b = request.POST.get('b')
        c = request.POST.get('c')
        d = request.POST.get('d')
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            a = float(request.POST.get('a'))
            b = float(request.POST.get('b'))
            c = float(request.POST.get('c'))
            d = float(request.POST.get('d'))
            n = a+b+c+d
            oddsratio, pvalue = stats.fisher_exact([[a, b],[c, d]])  
            re = pvalue
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True

            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'result':re,
                'a':a,
                'b':b,
                'c':c,
                'd':d,
                'n':n,
                'ot':True
            }
            return render(request,'Statistics/fishers_exact_test.html',context)
        return render(request,'Statistics/fishers_exact_test.html',{'a':5,'b':4,'c':5,'d':4,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/fishers_exact_test.html')



#critical_value
def critical_value(request):
    try:
        dis = request.POST.get('dis','form2')
        test = request.POST.get('test','form1')
        d = request.POST.get('d')
        d1 = request.POST.get('d1')
        d2 = request.POST.get('d2')
        sl = request.POST.get('sl')
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            if dis == 'form1' and sl:
                sl = float(request.POST.get('sl'))
                dis = request.POST.get('dis')
                test = request.POST.get('test')
                if test == 'form3':
                    q = 1-sl
                elif test == 'form2':
                    q = sl
                elif test == 'form1':
                    q = 1-sl/2
                re = stats.norm.ppf(q)
                
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True

                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'result':re,
                    'test':test,
                    'dis':dis,
                    'sl':sl,
                    'ot':True
                }

                return render(request,'Statistics/critical_value.html',context)

            elif dis == 'form2' and sl and d:
                d = float(request.POST.get('d'))
                sl = float(request.POST.get('sl'))
                dis = request.POST.get('dis')
                test = request.POST.get('test')

                if test == 'form3':
                    q = 1-sl

                elif test == 'form2':
                    q = sl
                elif test == 'form1':
                    q = 1-sl/2
                re = stats.t.ppf(q,d)
                
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True

                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'result':re,
                    'test':test,
                    'dis':dis,
                    'sl':sl,
                    'd':d,
                    'ot':True
                }

                return render(request,'Statistics/critical_value.html',context)

            elif dis == 'form3' and sl and d:
                d = float(request.POST.get('d'))
                sl = float(request.POST.get('sl'))
                dis = request.POST.get('dis')
                test = request.POST.get('test')

                if test == 'form3':
                    q = 1-sl

                elif test == 'form2':
                    q = sl
                elif test == 'form1':
                    q = 1-sl/2
                re = stats.chi2.ppf(q,d)
                
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True

                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'result':re,
                    'test':test,
                    'dis':dis,
                    'sl':sl,
                    'd':d,
                    'ot':True
                }
                return render(request,'Statistics/critical_value.html',context)

            elif dis == 'form4' and sl and d1 and d2:
                d1 = float(request.POST.get('d1'))
                d2 = float(request.POST.get('d2'))
                sl = float(request.POST.get('sl'))
                dis = request.POST.get('dis')
                test = request.POST.get('test')

                if test == 'form3':
                    q = 1-sl
                elif test == 'form2':
                    q = sl
                elif test == 'form1':
                    q = 1-sl/2
                re = stats.f.ppf(q=q, dfn=d1, dfd=d2)
                
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True

                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'result':re,
                    'test':test,
                    'dis':dis,
                    'sl':sl,
                    'd1':d1,
                    'd2':d2,
                    'ot':True
                }
                return render(request,'Statistics/critical_value.html',context)

        return render(request,'Statistics/critical_value.html',{'ot':False,'dis':dis,'test':test,'sl':0.05,'d':22,'d1':6,'d2':8})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/critical_value.html')

    
#linear correlation
def linear_regression(request):
    try:
        x = str(request.POST.get('x'))
        y = str(request.POST.get('y'))
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            x1 = x.split(" ")
            y1 = y.split(" ")
            xo = [int(i) for i in x1]
            yo = [int(i) for i in y1]
            xsum = sum(xo)
            ysum = sum(yo)
            nx = len(xo)
            ny = len(yo)
            xmean = xsum/nx
            ymean = ysum/ny
            l=[]
            for i in xo:
                l.append(i-xmean)
            l2 = [(i**2) for i in l]
            ss = sum(l2)

            z = []
            for i in xo:
                z.append(i-xmean)
            z2 = []
            for i in yo:
                z2.append(i-ymean)
            z3=[]
            for i in range(len(xo)):
                z3.append(z[i]*z2[i])
            sp = sum(z3)
            b = sp/ss
            a = ymean-(b*xmean)
            re = (b*xsum)+a
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            context = {
                'output':k1,
                'start':startpoint,
                'end':endpoint,
                'nx':nx,
                'ny':ny,
                'x':x,
                'y':y,
                'xsum':xsum,
                'ysum':ysum,
                'result':re,
                'ss':ss,
                'sp':sp,
                'a':a,
                'b':b,
                'xmean':xmean,
                'ymean':ymean,
                'ot':True

            }
            return render(request,'Statistics/linear_regression.html',context)
        return render(request,'Statistics/linear_regression.html',{'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/linear_regression.html')

def average_rate_of_change_cal(request):
    try:
        if request.method=="POST":
            x1=request.POST.get("x1")
            fx1=request.POST.get("fx1")
            x2=request.POST.get("x2")
            fx2=request.POST.get("fx2")
            if request.POST.get("x1"):
                x1_calculated=float(request.POST.get("x1"))
            if request.POST.get("fx1"):
                fx1_calculated=float(request.POST.get("fx1"))
            if request.POST.get("x2"):
                x2_calculated=float(request.POST.get("x2"))
            if request.POST.get("fx2"):
                fx2_calculated=float(request.POST.get("fx2"))
            if x1 and x2 and fx1 and fx2:
                num=x1_calculated-x2_calculated
                den=fx1_calculated-fx2_calculated
                try:
                    x=num/den
                except:
                    messages.error(request,"Divide by zero error")
                    return render(request,'Statistics/average_rate_of_change_calc.html')
                context={
                    "x1":str(x1),
                    "x2":str(x2),
                    "fx1":str(fx1),
                    "fx2":str(fx2),
                    "num":str(num),
                    "den":str(den),
                    "x":str(x)
                }
                return render(request,"Statistics/average_rate_of_change_calc.html",context)
            else:
                context={
                    "x1":"",
                    "x2":"",
                    "fx1":"",
                    "fx2":"",
                }
                return render(request,"Statistics/average_rate_of_change_calc.html",context)
        else:
            return render(request,"Statistics/average_rate_of_change_calc.html")
    except:
        return render(request,"Statistics/average_rate_of_change_calc.html")


def coin_flip_probability(request):
    try:
        if request.method=="POST":
            
            Given=request.POST.get("Given")
            n=request.POST.get("n")
            if request.POST.get("n"):
                n_calculated=int(request.POST.get("n"))
            k=request.POST.get("k")
            if request.POST.get("k"):
                k_calculated=int(request.POST.get("k"))
            p=0.5
            r_values=range(0,n_calculated)
            data_binom = [binom.pmf(r, n_calculated, p) for r in r_values ]
            if Given=="at_most" and k and n:
                string1=""
                if k_calculated<=n_calculated:
                    result=sum(data_binom[0:k_calculated])/sum(data_binom)
                    #+ "+"<sup>"+str(n_calculated)+"</sup>C<sub>1 ......."
                    string1="<sup>"+str(n_calculated)+"</sup>C<sub>0</sub> p<sup>0</sup>(1-p)<sup>"+str(n_calculated)+"</sup>  +"+"<sup>"+str(n_calculated)+"</sup>C<sub>1</sub> p<sup>1</sup>(1-p)<sup>"+str(n_calculated-1)+"</sup> + ......."+"<sup>"+str(n_calculated)+"</sup>C<sub>"+str(k_calculated)+"</sub> p<sup>"+str(k_calculated)+"</sup>(1-p)<sup>"+str(n_calculated-k_calculated)+"</sup>  "                
                    p=str(round(result*100,2))+"%"
                    context={
                        "Given":Given,
                        "n":n,
                        "k":k,
                        "n_calculated":n_calculated,
                        "k_calculated":k_calculated,
                        "result":result,
                        "p":p,
                        "string1":string1
                    }
                else:
                    messages.error(request,"Value of k should be less than or equal to n")
                    return render(request,'Statistics/coin_flip_probability.html')
                return render(request,"Statistics/coin_flip_probability.html",context)
            elif Given=="exactly" and k and n:
                if k_calculated<=n_calculated:
                    result=data_binom[k_calculated]/sum(data_binom)
                    string1="<sup>"+str(n_calculated)+"</sup>C<sub>"+str(k_calculated)+"</sub> p<sup>"+str(k_calculated)+"</sup>(1-p)<sup>"+str(n_calculated-k_calculated)+"</sup>  "                
                    p=str(round(result*100,2))+"%"
                    context={
                        "Given":Given,
                        "n":n,
                        "k":k,
                        "n_calculated":n_calculated,
                        "k_calculated":k_calculated,
                        "result":result,
                        "p":p,
                        "string1":string1
                    }
                    return render(request,"Statistics/coin_flip_probability.html",context)
                else:
                    messages.error(request,"Value of k should be less than or equal to n")
                    return render(request,'Statistics/coin_flip_probability.html')
            elif Given=="at_least" and k and n:
                if k_calculated<=n_calculated:
                    result=sum(data_binom[k_calculated:])/sum(data_binom)
                    string1="<sup>"+str(n_calculated)+"</sup>C<sub>"+str(k_calculated)+"</sub> p<sup>"+str(k_calculated)+"</sup>(1-p)<sup>"+str(n_calculated-k_calculated)+"</sup> +"+"<sup>"+str(n_calculated)+"</sup>C<sub>"+str(k_calculated+1)+"</sub> p<sup>"+str(k_calculated+1)+"</sup>(1-p)<sup>"+str(n_calculated-k_calculated-1)+"</sup>  +............."+str(n_calculated)+"</sup>C<sub>"+str(n_calculated)+"</sub> p<sup>"+str(0)+"</sup>(1-p)<sup>"+str(n_calculated)+"</sup>  "                                               
                    p=str(round(result*100,2))+"%"
                    context={
                        "Given":Given,
                        "n":n,
                        "k":k,
                        "n_calculated":n_calculated,
                        "k_calculated":k_calculated,
                        "result":result,
                        "p":p,
                        "string1":string1
                    }
                    return render(request,"Statistics/coin_flip_probability.html",context)
                else:
                    messages.error(request,"Value of k should be less than or equal to n")
                    return render(request,'Statistics/coin_flip_probability.html')
            if Given=="equal_to" and k and n:
                
                context={
                    "Given":Given,
                    "n":"",
                    "k":"",
                    
                }
                return render(request,"Statistics/coin_flip_probability.html",context)
        else:
            return render(request,"Statistics/coin_flip_probability.html")
    except:
        return render(request,"Statistics/coin_flip_probability.html")
#MEDIAN VIEW
def median(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('median-detail',num1)
    else:
        return render(request, "Statistics/median.html")
def median_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Median of {}'.format(i))
        gk.append('/statistics/median-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('Mode of {}'.format(i))
        hk.append('/statistics/mode-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Median_Model.objects.filter(inputEnter=num11).exists():
            record=Median_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_median.html",context)
        else:
            ans=medians1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Median of '+str(num11)
            generateDate=datetime.datetime.now()
            record=Median_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()        
	
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_median.html",context)
    except Exception as e:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/median.html",{'error':error,'inputEnter':num11})

#MEAN VIEW
def mean(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('mean-detail',num1)
    else:
        return render(request, "Statistics/mean.html")
def mean_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Mean of {}'.format(i))
        gk.append('/statistics/mean-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('Median of {}'.format(i))
        hk.append('/statistics/median-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Mean_Model.objects.filter(inputEnter=num11).exists():
            record=Mean_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_mean.html",context)
        else:
            ans=means1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Mean of '+str(num11)
            generateDate=datetime.datetime.now()
            record=Mean_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()        
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_mean.html",context)
    except Exception as e:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/mean.html",{'error':error,'inputEnter':num11})

#MODE VIEW
def mode(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('mode-detail',num1)
    else:
        return render(request, "Statistics/mode.html")
def mode_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Mode of {}'.format(i))
        gk.append('/statistics/mode-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('Median of {}'.format(i))
        hk.append('/statistics/median-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Mode_Model.objects.filter(inputEnter=num11).exists():
            record=Mode_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_mode.html",context)
        else:
            ans=modes1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Mode of '+str(num11)
            generateDate=datetime.datetime.now()
            record=Mode_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_mode.html",context)
    except Exception as e:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/mode.html",{'error':error,'inputEnter':num11})

#FIRST QUARTILE VIEW
def first_quartile(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('first-quartile-detail',num1)
    else:
        return render(request, "Statistics/first_quartile.html")
def first_quartile_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('First Quartile of {}'.format(i))
        gk.append('/statistics/first-quartile-of-{}/'.format(i.replace(', ','-and-')))
    for i in ran1[5:]:
        hk1.append('Third Quartile of {}'.format(i))
        hk.append('/statistics/third-quartile-of-{}/'.format(i.replace(', ','-and-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if First_Quartile_Model.objects.filter(inputEnter=num11).exists():
            record=First_Quartile_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_first_quartile.html",context)
        else:
            ans=lower_quartiles1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='First Quartile of '+str(num11)
            generateDate=datetime.datetime.now()
            record=First_Quartile_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_first_quartile.html",context)
    except Exception as e:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/first_quartile.html",{'error':error,'inputEnter':num11})

#THIRD QUARTILE VIEW
def third_quartile(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('third-quartile-detail',num1)
    else:
        return render(request, "Statistics/third_quartile.html")
def third_quartile_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Third Quartile of {}'.format(i))
        gk.append('/statistics/third-quartile-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('First Quartile of {}'.format(i))
        hk.append('/statistics/first-quartile-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Third_Quartile_Model.objects.filter(inputEnter=num11).exists():
            record=Third_Quartile_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_third_quartile.html",context)
        else:
            ans=upper_quartiles1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Third Quartile of '+str(num11)
            generateDate=datetime.datetime.now()
            record=Third_Quartile_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_third_quartile.html",context)
    except Exception as e:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/third_quartile.html",{'error':error,'inputEnter':num11})

#MAXIMUM VIEW
def maximum_number(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('maximum-number-detail',num1)
    else:
        return render(request, "Statistics/maximum_number.html")
def maximum_number_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Maximum Number of {}'.format(i))
        gk.append('/statistics/maximum-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('Minimum Number of {}'.format(i))
        hk.append('/statistics/minimum-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Maximum_Number_Model.objects.filter(inputEnter=num11).exists():
            record=Maximum_Number_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_maximum_number.html",context)
        else:
            ans=find_maxs1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Maximum of '+str(num11)
            generateDate=datetime.datetime.now()
            record=Maximum_Number_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_maximum_number.html",context)
    except:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/maximum_number.html",{'error':error,'inputEnter':num11})

#MINIMUM VIEW
def minimum_number(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('minimum-number-detail',num1)
    else:
        return render(request, "Statistics/minimum_number.html")
def minimum_number_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Minimum Number of {}'.format(i))
        gk.append('/statistics/minimum-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('Maximum Number of {}'.format(i))
        hk.append('/statistics/maximum-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Minimum_Number_Model.objects.filter(inputEnter=num11).exists():
            record=Minimum_Number_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_minimum_number.html",context)
        else:
            ans=find_mins1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Minimum of '+str(num11)
            generateDate=datetime.datetime.now()
            record=Minimum_Number_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2}
            return render(request, "Statistics/detail_minimum_number.html",context)
    except:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/minimum_number.html",{'error':error,'inputEnter':num11})

#FIVE NUMBER SUMMARY VIEW
def five_summary(request):
    if request.method == 'POST': #Condition for the after submit
        num1=request.POST.get("input1")
        num1=num1.replace(',','-').replace(' ','')
        return redirect('five-summary-detail',num1)
    else:
        return render(request, "Statistics/five_summary.html")
def five_summary_detail(request,num1):
    num11=num1.replace('-',',')
    from random import randint
    k2=num11.split(',')
    ran=[]
    ran1=[]
    import random
    for i in range(10):
        for j in (k2):
            if '.' in j:
                kz=round(random.uniform(int('1'+(len(j.split('.')[0])-1)*'0'),int(len(j.split('.')[0])*'9')), int(len(j.split('.')[1])))
                if kz==0.0:
                    kz=round(20/3,j)
            else:
                kz=randint(int('1'+(len(j)-1)*'0'),int(len(j)*'9'))
                if kz==0:
                    kz=8
            ran.append(str(kz))
        ran1.append(', '.join(ran))
        ran=[]
    print(ran1,'ran1')
    gk1=[]
    gk=[]
    hk1=[]
    hk=[]
    for i in ran1[0:5]:
        gk1.append('Five Number Summary of {}'.format(i))
        gk.append('/statistics/five-number-summary-of-{}/'.format(i.replace(', ','-')))
    for i in ran1[5:]:
        hk1.append('Median of {}'.format(i))
        hk.append('/statistics/median-of-{}/'.format(i.replace(', ','-')))
    example1=zip(gk,gk1)
    example2=zip(hk,hk1)
    try:
        if Five_Summary_Model.objects.filter(inputEnter=num11).exists():
            record=Five_Summary_Model.objects.filter(inputEnter=num11)
            for i in record:
                inputEnter=i.inputEnter
                detailStep=i.detailStep
                finalAnswer=i.finalAnswer
                slug=i.slug
                solutionTitle=i.solutionTitle
                generateDate=i.generateDate
            min_value=finalAnswer.split(',')[0]
            max_value=finalAnswer.split(',')[1]
            median=finalAnswer.split(',')[2]
            first_quartile=finalAnswer.split(',')[3]
            third_quartile=finalAnswer.split(',')[4]
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2,'min_value':min_value,'max_value':max_value,'median':median,'first_quartile':first_quartile,'third_quartile':third_quartile}
            return render(request, "Statistics/detail_five_summary.html",context)
        else:
            ans=five_nums1(num11)
            inputEnter=num11
            detailStep=ans[0]
            finalAnswer=ans[1]
            slug=num1
            solutionTitle='Five Number Summary of '+str(num11)
            min_value=finalAnswer.split(',')[0]
            max_value=finalAnswer.split(',')[1]
            median=finalAnswer.split(',')[2]
            first_quartile=finalAnswer.split(',')[3]
            third_quartile=finalAnswer.split(',')[4]
            generateDate=datetime.datetime.now()
            record=Five_Summary_Model(inputEnter=inputEnter,detailStep=detailStep,finalAnswer=finalAnswer,slug=slug,solutionTitle=solutionTitle,generateDate=generateDate)
            record.save()
            context={'inputEnter':inputEnter,'detailStep':detailStep,'finalAnswer':finalAnswer,'slug':slug,'solutionTitle':solutionTitle,'generateDate':generateDate,'example1':example1,'example2':example2,'min_value':min_value,'max_value':max_value,'median':median,'first_quartile':first_quartile,'third_quartile':third_quartile}
            return render(request, "Statistics/detail_five_summary.html",context)
    except:
        error='<p>Enter a valid input</p>'
        return render(request, "Statistics/five_summary.html",{'error':error,'inputEnter':num11})
def round_func(n):
        if float(int(float(n)))==float(n):
            return round(float(n))
        else:
            return round(float(n),4)
def round_func1(n):
        if float(int(float(n)))==float(n):
            return round(float(n))
        else:
            return round(float(n),6)

#RELATIVE STANDARD DEVIATION VIEW
def relative_standard_dev(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        try:
            x1=num1.split(',')
            x2=[]
            for i in x1:
                x2.append(round_func(i))
            import statistics
            stat_data=statistics.stdev(x2)
            #stat_data=round_func1(stat_data,8)
            val=0
            for i in x2:
                val+=i
            mean_data=val/len(x2)
            print(mean_data,'mean')
            try:
                mean_data=round_func(mean_data)
            except:
                mean_data=round_func(mean_data)
            res=stat_data/mean_data
            res=round_func1(res*100,4)
            context={'input1':num1,'detailStep':res,'go_tag':'#relative_standard_deviation'}
            return render(request,'Statistics/relative_standard_dev_calc.html',context)
        except:
            messages.error(request,"Enter a valid input")
            #result=statistics.stdev(x2)
            context={'input1':num1}
            return render(request,'Statistics/relative_standard_dev_calc.html',context)
    else:
        return render(request, "Statistics/relative_standard_dev_calc.html")


#ANOVA VIEW
def anova_calc(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        num3=request.POST.get("input3")
        try:
            x1=num1.split(',')
            x3=num2.split(',')
            x5=num3.split(',')
            if len(x1)==len(x3)==len(x5):
                x2=[]
                for i in x1:
                    try:
                        x2.append(int(i))
                    except:
                        x2.append(float(i))
                x4=[]
                for i in x3:
                    try:
                        x4.append(int(i))
                    except:
                        x4.append(float(i))
                x6=[]
                for i in x5:
                    try:
                        x6.append(int(i))
                    except:
                        x6.append(float(i))
                add_1=0
                add_2=0
                add_3=0
                for i,j,k in zip(x2,x4,x6):
                    add_1+=i
                    add_2+=j
                    add_3+=k
                mean_1=round(add_1/len(x1),3)
                mean_2=round(add_2/len(x2),3)
                mean_3=round(add_3/len(x3),3)
                print(mean_1)
                print(mean_2)
                print(mean_3)
                import statistics
                sd_1=round(statistics.stdev(x2),4)
                sd_2=round(statistics.stdev(x4),4)
                sd_3=round(statistics.stdev(x6),4)
                add_11=0
                add_12=0
                add_13=0
                for i in x2:
                    add_11+=((i-mean_1)**2)
                    add_12+=((j-mean_2)**2)
                    add_13+=((k-mean_3)**2)
                    print(add_11,'add_11')
                    # print(add_12,'add_12')
                    # print(add_13,'add_13')
                # import numpy as np
                # from scipy.cluster.vq import vq
                # xx=np.array([x2,x4,x6])
                # partition,hh=vq(xx,code_book)
                # fb=[]
                # fc=xx.mean(0)
                # for i in range(partition.max()+1):
                #     fcc=xx[partition==i].mean(0)
                #     fb.append(np.bincount(partition)[i]*np.sum((fcc-fc)**2))
                # ssb=np.sum(fb)
                ssb=add_11+add_12+add_13
                ssw=(len(x2)-1)*((sd_1**2)+(sd_2**2)+(sd_3**2))
                msb=ssb/(3-1)
                msw=ssw/((len(x2)*3)-3)
                f_ratio=round(msb/msw,4)
                v=[]
                a=v.append
                a('<p><strong>Mean:</strong></p>')
                a('<p>Mean of Group 1 = {}</p>'.format(mean_1))
                a('<p>Mean of Group 2 = {}</p>'.format(mean_2))
                a('<p>Mean of Group 3 = {}</p>'.format(mean_3))
                a('<p><strong>Standard Deviation:</strong></p>')
                a('<p>SD of Group 1 = {}</p>'.format(sd_1))
                a('<p>SD of Group 2 = {}</p>'.format(sd_2))
                a('<p>SD of Group 3 = {}</p>'.format(sd_3))
                a('<p><strong>Sum of Squares:</strong></p>')
                a('<p>SSb(between groups) = {}</p>'.format(round(ssb,4)))
                a('<p>SSw(within groups) = {}</p>'.format(round(ssw,4)))
                a('<p><strong>Mean Squares</strong></p>')
                a('<p>MSb(between groups) = {}</p>'.format(round(msb,4)))
                a('<p>MSw(within groups) = {}</p>'.format(round(msw,4)))
                a('<p><strong>F-ratio :</strong>{}</p>'.format(f_ratio))
                context={'input1':num1,'input2':num2,'input3':num3,'detailStep':''.join(v),'go_tag':'#anova'}
                return render(request,'Statistics/anova_calc.html',context)
            else:
                messages.error(request,"Enter the same number of data for x and y")
                context={'input1':num1,'input2':num2,'input3':num3}
                return render(request,'Statistics/anova_calc.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1,'input2':num2,'input3':num3}
            return render(request,'Statistics/anova_calc.html',context)
    else:
        return render(request, "Statistics/anova_calc.html")

#COIN TOSS PROBABILITY VIEW
def coin_toss_prob(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        try:
            num1=int(num1)
            num2=int(num2)
            if num1<num2:
                messages.error(request,"Number of tosses value should be greater than number of heads/tails")
                return render(request,'Statistics/coin_toss_prob_calc.html',{'input1':num1,'input2':num2,'go_tag':'#coin_toss'})
            else:
                res=coin_prob_func(num1,num2)
                result=res[0]
                return render(request,'Statistics/coin_toss_prob_calc.html',{'input1':num1,'input2':num2,'detailStep':result,'go_tag':'#coin_toss'})
        except:
            messages.error(request,"Enter a valid input")
            return render(request,'Statistics/coin_toss_prob_calc.html',{'input1':num1,'input2':num2,'go_tag':'#coin_toss'})
    else:
        return render(request, "Statistics/coin_toss_prob_calc.html")

#Covariance_Calculator
def covariance_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        try:
            x1=num1.split(',')
            x2=[]
            for i in x1:
                x2.append(int(i))
            y1=num2.split(',')
            y2=[]
            for i in y1:
                y2.append(int(i))
            c1=0
            d1=0
            for i,j in zip(x2,y2):
                c1+=i
                d1+=j
            c1=c1/len(x1)
            d1=d1/len(y1)
            t1=0
            for i,j in zip(x2,y2):
                t2=(i-c1)*(j-d1)
                t1+=t2
            result1=round(t1/len(x1),5)
            result2=round(t1/(len(x1)-1),5)
            # import statistics
            # result=statistics.stdev(x2)
            # result=round(result**2,5)
            v=[]
            a=v.append
            a('<p>Sample Covariance:</p>')
            a('<p> {}</p>'.format(result1))
            a('<p>Population Covariance:</p>')
            a('<p> {}</p>'.format(result2))
            res=''.join(v)
            context={'input1':num1,'input2':num2,'detailStep':res,'go_tag':'#covariance'}
            return render(request,'Statistics/covariance-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1}
            return render(request,'Statistics/covariance-calculator.html',context)
    else:
        return render(request,'Statistics/covariance-calculator.html')

#Gamma_Function_Calculator
def gamma_function_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        try:
            try:
                n1=int(num1)
            except:
                n1=float(num1)
            if '.' in num1:
                from scipy.integrate import quad
                from numpy import exp
                f=lambda x:(x**((n1)-1))*exp(-x)
                print(f)
                res=quad(f,0,np.inf)
                res=round(res[0],8)
            else:
                import math
                res=math.factorial(int(num1)-1)
            v=[]
            a=v.append
            a('<p><strong>Gamma of {}:</strong></p>'.format(num1))
            a('<p>Γ({}) = {}</p>'.format(num1,res))
            result=''.join(v)
            context={'input1':num1,'detailStep':result,'go_tag':'#gamma_function'}
            return render(request,'Statistics/gamma-function-calculator.html',context)
        except Exception as e:
            print(e)
            messages.error(request,"Enter a valid input")
            context={'input1':num1}
            return render(request,'Statistics/gamma-function-calculator.html',context)
    else:
        return render(request,'Statistics/gamma-function-calculator.html')

#Linear_Correlation_Coefficient_Calculator
def linear_correlation_coefficient_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        try:
            x1=num1.split(',')
            x3=num2.split(',')
            if len(x1)==len(x3):
                x2=[]
                for i in x1:
                    try:
                        x2.append(int(i))
                    except:
                        x2.append(float(i))
                x4=[]
                for i in x3:
                    try:
                        x4.append(int(i))
                    except:
                        x4.append(float(i))
                mn=0
                nn=0
                square_m=0
                square_n=0
                for i in x2:
                    mn+=i
                    square_m+=i**2
                for i in x4:
                    nn+=i
                    square_n+=i**2
                add_mn=0
                for i,j in zip(x2,x4):
                    add_mn+=i*j
                import math
                numerator_res=(len(x2)*add_mn)-(mn*nn)
                denominator_ress=((len(x2)*square_m)-(mn**2))*((len(x2)*square_n)-(nn**2))
                denominator_res=math.sqrt(denominator_ress)
                res=round(numerator_res/denominator_res,5)
                v=[]
                a=v.append
                a('<p><strong>Linear correlation coefficient (r):</strong></p>')
                a('<p>{}</p>'.format(res))
                context={'input1':num1,'input2':num2,'detailStep':''.join(v),'go_tag':'#linear_correlation'}
                return render(request,'Statistics/linear-correlation-coefficient-calculator.html',context)
            else:
                messages.error(request,"Enter the same number of data for x and y")
                context={'input1':num1,'input2':num2}
                return render(request,'Statistics/linear-correlation-coefficient-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1,'input2':num2}
            return render(request,'Statistics/linear-correlation-coefficient-calculator.html',context)
    else:
        return render(request,'Statistics/linear-correlation-coefficient-calculator.html')

#Mean_Deviation_Calculator
def mean_deviation_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        try:
            x1=num1.split(',')
            x2=[]
            for i in x1:
                try:
                    x2.append(int(i))
                except:
                    x2.append(float(i))
            mn=0
            for i in x2:
                mn+=i
            mn_res=mn/len(x2)
            sum_squares=0
            for i in x2:
                b1=abs((i-mn_res))/len(x2)
                sum_squares+=b1
            v=[]
            a=v.append
            a('<p><strong>Mean:</strong></p>')
            a('<p>{}</p>'.format(mn_res))
            a('<p><strong>Mean Deviation:</strong></p>')
            a('<p>{}</p>'.format(sum_squares))
            context={'input1':num1,'detailStep':''.join(v),'go_tag':'#mean_deviation'}
            return render(request,'Statistics/mean-deviation-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1}
            return render(request,'Statistics/mean-deviation-calculator.html',context)
    else:
        return render(request,'Statistics/mean-deviation-calculator.html')

#Pearson_Correlation_Calculator
def pearson_correlation_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        try:
            x1=num1.split(',')
            x3=num2.split(',')
            if len(x1)==len(x3):
                x2=[]
                for i in x1:
                    try:
                        x2.append(int(i))
                    except:
                        x2.append(float(i))
                x4=[]
                for i in x3:
                    try:
                        x4.append(int(i))
                    except:
                        x4.append(float(i))
                mn=0
                nn=0
                square_m=0
                square_n=0
                for i in x2:
                    mn+=i
                    square_m+=i**2
                for i in x4:
                    nn+=i
                    square_n+=i**2
                add_mn=0
                for i,j in zip(x2,x4):
                    add_mn+=i*j
                import math
                numerator_res=(len(x2)*add_mn)-(mn*nn)
                denominator_ress=((len(x2)*square_m)-(mn**2))*((len(x2)*square_n)-(nn**2))
                denominator_res=math.sqrt(denominator_ress)
                res=round(numerator_res/denominator_res,5)
                v=[]
                a=v.append
                a('<p><strong>Pearson correlation coefficient (r):</strong></p>')
                a('<p>{}</p>'.format(res))
                context={'input1':num1,'input2':num2,'detailStep':''.join(v),'go_tag':'#pearson_correlation'}
                return render(request,'Statistics/pearson-correlation-calculator.html',context)
            else:
                messages.error(request,"Enter the same number of data for x and y")
                context={'input1':num1,'input2':num2}
                return render(request,'Statistics/pearson-correlation-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1,'input2':num2}
            return render(request,'Statistics/pearson-correlation-calculator.html',context)
    else:
        return render(request,'Statistics/pearson-correlation-calculator.html')

#Average_Calculator
def average_calculator(request):
	return render(request,'Statistics/average-calculator.html')

#population_mean
def population_mean(request):
    try:
        x = request.POST.get('x')
        ot = False
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            xo = [int(i) for i in x1]
            xsum = sum(xo)  
            n = len(xo)
            re = xsum/n
            context = {
            'n':n,
            'x':x,
            'xsum':xsum,
            'result':re,
            'ot':True                }
            return render(request,'Statistics/population_mean.html',context)
        else:
            return render(request,'Statistics/population_mean.html',{'x':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/population_mean.html',{'x':"10,20,30,40,50",'ot':False})

#sample_mean
def sample_mean(request):
    try:
        x = request.POST.get('x')
        ot = False
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            xo = [int(i) for i in x1]
            xsum = sum(xo)  
            n = len(xo)
            re = xsum/n
            context = {
            'n':n,
            'x':x,
            'xsum':xsum,
            'result':re,
            'ot':True                }
            return render(request,'Statistics/sample_mean.html',context)
        else:
            return render(request,'Statistics/sample_mean.html',{'x':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/sample_mean.html',{'x':"10,20,30,40,50",'ot':False})


#mid_range
def mid_range(request):
    try:
        x = request.POST.get('x')
        ot = False
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            xo = [int(i) for i in x1]
            xmin = min(xo)  
            xmax = max(xo) 
            re = (xmax+xmin)/2
            
            context = {
                    'x':x,
                    'xmin':xmin,
                    'xmax':xmax,
                    'result':re,
                    'ot':True                }
            return render(request,'Statistics/mid_range.html',context)
        else:
            return render(request,'Statistics/mid_range.html',{'x':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/mid_range.html',{'x':"10,20,30,40,50",'ot':False})

#Coefficient of Variation
def coefficient_of_variation(request):
    try:
        x = request.POST.get('x')
        ot = False
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            xo = [int(i) for i in x1]
            mean = sum(xo)/len(xo)
            l=[]
            for i in xo:
                l.append(i-mean)
            l2 = [(i**2) for i in l]            
            ss = sum(l2)
            n = len(xo)
            sd = math.sqrt(ss/(n-1))
            re = sd / mean
            k1 = False
            index_num = 0
            startpoint = 0
            endpoint = 0
            count = zerocount(re)
            if str(re) in 'e':
                r11 = str(re)
                index_num = r11.index('e')
                stratpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            elif count>5:
                r11 = str("{:.2e}".format(re))
                index_num = r11.index('e')
                startpoint = r11[:index_num]
                endpoint = r11[index_num+1:]
                k1 = True
            context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'sd':sd,
                    'result':re,
                    'ot':True,
                    'n':n,
                    'mean':mean
                }
            return render(request,'Statistics/coefficient_of_variation.html',context)
        else:
            return render(request,'Statistics/coefficient_of_variation.html',{'x':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/coefficient_of_variation.html',{'x':"10,20,30,40,50",'ot':False})

#range
def range_cal(request):
    try:
        x = request.POST.get('x')
        ot = False
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            xo = [int(i) for i in x1]
            m = max(xo)
            m1 = min(xo)
            re = m-m1
            context = {
            'x':x,
            'max':m,
            'min':m1,
            'result':re,
            'ot':True  }
            return render(request,'Statistics/range.html',context)
        else:
            return render(request,'Statistics/range.html',{'x':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/range.html',{'x':"10,20,30,40,50",'ot':False})

#percentile
def percentile(request):
    try:
        x = request.POST.get('x')
        n = request.POST.get('n')
        ot = False
        if request.method == "POST":
            n = int(request.POST.get('n'))
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            arr = [int(i) for i in x1]
            arr.sort()
            re = np.percentile(arr, n)
            context = {
            'x':x,
            'n':n,
            'arr':arr,
            'result':re,
            'ot':True  
            }
            return render(request,'Statistics/percentile.html',context)
        else:
            return render(request,'Statistics/percentile.html',{'x':"10,20,30,40,50",'n':50,'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/percentile.html',{'x':"10,20,30,40,50",'n':50,'ot':False})

#iqr
def iqr(request):
    try:
        x = request.POST.get('x')
        ot = False
        if request.method == "POST":
            x2 = x.replace(',',' ')
            x1 = x2.split(' ')
            arr = [int(i) for i in x1]
            n = len(arr)
            q3 = 3/4*(n + 1)
            q1 = 1/4*(n + 1)
            re = q3-q1
            context = {
            'x':x,
            'n':n,
            'arr':arr,
            'q1':q1,
            'q3':q3,
            'result':re,
            'ot':True  
            }
            return render(request,'Statistics/iqr.html',context)
        else:
            return render(request,'Statistics/iqr.html',{'x':"10,20,30,40,50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/iqr.html',{'x':"10,20,30,40,50",'ot':False})
#descriptive statistics calc
def descriptive_statistics_calculator(request):
    try:
        Given = request.POST.get('Given','form1')
        x = request.POST.get('x')
        def zerocount(v):
            l = str(v).count('0')
            return int(l)
        ot = False
        if request.method == "POST":
            if Given == "form1" and x:
                
                x1 = x.split(" ")
                
                xo = [int(i) for i in x1]
                s = xo
                s.sort()
                re = min(xo)
                print(re)
                context = {
                    's':s,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == "form2" and x:
                
                x1 = x.split(" ")
                
                xo = [int(i) for i in x1]
                s = xo
                s.sort()
                re = max(xo)
                context = {
                    's':s,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == "form3" and x:
                
                x1 = x.split(" ")
                
                xo = [int(i) for i in x1]
                m = max(xo)
                m1 = min(xo)
                re = m-m1
                context = {
                    'x':x,
                    'max':m,
                    'min':m1,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)
            elif Given == "form4" and x:
                
                x1 = x.split(" ")
                
                xo = [int(i) for i in x1]
                re = sum(xo)
                context = {
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == "form5" and x:
                
                x1 = x.split(" ")
                
                xo = [int(i) for i in x1]
                re = len(xo)
                context = {
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)


            elif Given == 'form6' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                xsum = sum(xo)  
                n = len(xo)
                re = xsum/n
                
                context = {
                    'n':n,
                    'x':x,
                    'xsum':xsum,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form7' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                n = len(xo)
                xo.sort()
                  
                if n % 2 == 0:
                    p = n // 2
                    median1 = xo[n//2]
                    median2 = xo[n//2 - 1]
                    median = (median1 + median2)/2
                    k = True
                else:
                    median = xo[n//2]
                    median1 = False
                    median2 = False
                    k = False
                    p = (n + 1)//2
                re = median
                
                context = {
                    'n':n,
                    'p':p,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'k':k,
                    'median1':median1,
                    'median2':median2
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)
            elif Given == 'form8' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                re = statistics.multimode(xo)
                if len(xo)==len(re):
                    re= "No Mode"

                context = {
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)
            elif Given == 'form9' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]
                ss = sum(l2)
                n = len(xo)
                re = math.sqrt(ss/(n-1))
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'ss':ss,
                    'n':n
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)
            elif Given == 'form10' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]
                ss = sum(l2)
                n = len(xo)
                re = (ss/(n-1))
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'ss':ss,
                    'n':n
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)
            elif Given == 'form11' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]
                ss = sum(l2)
                n = len(xo)
                sd = math.sqrt(ss/(n-1))
                re = (sd*100)/mean
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'ss':ss,
                    'n':n,
                    'sd':sd,
                    'mean':mean
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)
            elif Given == 'form12' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                xmin = min(xo)  
                xmax = max(xo) 
                re = (xmax+xmin)/2
                
                context = {
                    'x':x,
                    'xmin':xmin,
                    'xmax':xmax,
                    'result':re,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form13' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                q1 = np.quantile(xo, .25)
                q2 = np.quantile(xo, .50)
                q3 = np.quantile(xo, .80)
                print(q1,q2,q3)
                context = {
                    'x':x,
                    'q1':q1,
                    'q2':q2,
                    'q3':q3,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form14' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                q1 = np.quantile(xo,.25)
                q3 = np.quantile(xo,.80)
                re = q3-q1
                context = {
                    'x':x,
                    'q1':q1,
                    'result':re,
                    'q3':q3,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form15' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                q3 = np.quantile(xo, .80)
                q1 = np.quantile(xo, .25)
                iqr = q3 -q1
                
                uf = q3 + 1.5 * iqr
                lf = q1 - 1.5 * iqr
                
                
                context = {
                    'x':x,
                    'iqr':iqr,
                    'q1':q1,
                    'uf':uf,
                    'lf':lf,
                    'q3':q3,
                    'Given':Given,
                    'ot':True                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form16' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]
                re = sum(l2)
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'mean':mean
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form17' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                n = len(xo)
                for i in xo:
                    l.append(i-mean)
                ss =sum(l)
                re = ss/n
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                print(ss,l)
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'ss':ss,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'mean':mean,
                    'n':n
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)


            elif Given == 'form18' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                l= [(i**2) for i in xo]
                n = len(xo)
                ss =sum(l)
                re = math.sqrt(ss/n)
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                print(ss,l)
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'ss':ss,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'n':n
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form19' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]
                ss = sum(l2)
                n = len(xo)
                sd = math.sqrt(ss/(n-1))
                re = sd / math.sqrt(n)
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'sd':sd,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'n':n
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form20' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]            
                ss = sum(l2)
                n = len(xo)
                sd = math.sqrt(ss/(n-1))
                l3= []
                for i in xo: 
                    l3.append(((i-mean)/sd)**3)
                s3 = sum(l3)
                z1 = (n / ((n - 1)*(n - 2)))
                re = z1*(s3)
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'sd':sd,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'n':n,
                    'z1':z1,
                    's3':s3
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form21' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]            
                ss = sum(l2)
                n = len(xo)
                sd = math.sqrt(ss/(n-1))
                l3= []
                for i in xo: 
                    l3.append(((i-mean)/sd)**4)
                s3 = sum(l3)
                z1 = (n* (n +1)) / ((n - 1)* (n - 2) *(n - 3))
                re = z1*(s3)
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'sd':sd,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'n':n,
                    'z1':z1,
                    's3':s3
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form22' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]            
                ss = sum(l2)
                n = len(xo)
                sd = math.sqrt(ss/(n-1))
                l3= []
                for i in xo: 
                    l3.append(((i-mean)/sd)**4)
                s3 = sum(l3)

                z1 = (n *(n +1)) / ((n - 1) *(n - 2) *(n - 3))
                z2 =  (3*((n-1)**2)) / ((n - 2) *(n - 3))

                re = (z1*s3)-z2
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'sd':sd,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'n':n,
                    'z1':z1,
                    's3':s3,
                    'z2':z2
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)

            elif Given == 'form23' and x:
                
                x1 = x.split(" ")
                x2 = ' '.join(x1)
                xo = [int(i) for i in x1]
                mean = sum(xo)/len(xo)
                l=[]
                for i in xo:
                    l.append(i-mean)
                l2 = [(i**2) for i in l]            
                ss = sum(l2)
                n = len(xo)
                sd = math.sqrt(ss/(n-1))
                re = sd / mean
                k1 = False
                index_num = 0
                startpoint = 0
                endpoint = 0
                count = zerocount(re)
                if str(re) in 'e':
                    r11 = str(re)
                    index_num = r11.index('e')
                    stratpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                elif count>5:
                    r11 = str("{:.2e}".format(re))
                    index_num = r11.index('e')
                    startpoint = r11[:index_num]
                    endpoint = r11[index_num+1:]
                    k1 = True
                context = {
                    'output':k1,
                    'start':startpoint,
                    'end':endpoint,
                    'x':x,
                    'sd':sd,
                    'result':re,
                    'Given':Given,
                    'ot':True,
                    'n':n,
                    'mean':mean
                }
                return render(request,'Statistics/descriptive-statistics-calculator.html',context)


            return render(request,'Statistics/descriptive-statistics-calculator.html',{'Given':Given,'ot':False})
        else:
            return render(request,'Statistics/descriptive-statistics-calculator.html',{'Given':'form1','x':"10 20 30 40 50",'ot':False})
    except:
        messages.error(request,'Please Enter valid data')
        return render(request,'Statistics/descriptive-statistics-calculator.html')


#Final_Grade_Calculator
def final_grade_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        num3=request.POST.get("input3")
        try:
            try:
                n1=int(num1)
            except:
                n1=float(num1)
            try:
                n2=int(num2)
            except:
                n2=float(num2)
            try:
                n3=int(num3)
            except:
                n3=float(num3)
            n3=n3/100
            x1=(1-n3)*n2
            x2=n1-x1
            x3=x2/n3
            x3=round(x3,2)
            v=[]
            a=v.append
            a('<p>The score needed for the final exam (F) is <strong>{}</strong></p>'.format(x3))
            result=''.join(v)
            context={'input1':num1,'input2':num2,'input3':num3,'detailStep':result,'go_tag':'#final_grade'}
            return render(request,'Statistics/final-grade-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1,'input2':num2,'input3':num3}
            return render(request,'Statistics/final-grade-calculator.html',context)
    else:
	    return render(request,'Statistics/final-grade-calculator.html')

#Mean_Median_Mode_Calculator
def mean_median_mode_calculator(request):
	return render(request,'Statistics/mean-median-mode-calculator.html')

#Odds_Probability_Calculator
def odds_probability_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        num3=request.POST.get("input3")
        try:
            try:
                n1=int(num1)
            except:
                n1=float(num1)
            try:
                n2=int(num2)
            except:
                n2=float(num2)
            pa=round(n1/(n1+n2),4)
            pb=round(n2/(n1+n2),4)
            pa_p=round(pa*100,2)
            pb_p=round(pb*100,2)
            from fractions import Fraction
            if int(n1)==int(n2):
                f1='1 : 1'
            elif int(n1)>int(n2):
                g1=Fraction(int(n1),int(n2))
                g2=str(g1).split('/')
                if num3=="for winning":
                    f1='{} : {}'.format(g2[0],g2[1])
                else:
                    f1='{} : {}'.format(g2[1],g2[0])
            elif int(n1)<int(n2):
                g1=Fraction(int(n2),int(n1))
                g2=str(g1).split('/')
                if num3=="for winning":
                    f1='{} : {}'.format(g2[1],g2[0])
                else:
                    f1='{} : {}'.format(g2[0],g2[1])
            # print(g1)
            print(f1)
            v=[]
            a=v.append
            a('<p>For <strong>{} to {} odds</strong> {}:</p>'.format(num1,num2,num3))
            a('<p><strong>Probability of:</strong></p>')
            if num3=="for winning":
                a('<p>Winning = {} or {}%</p>'.format(pa,pa_p))
                a('<p>Losing = {} or {}%</p>'.format(pb,pb_p))
            else:
                a('<p>Winning = {} or {}%</p>'.format(pb,pb_p))
                a('<p>Losing = {} or {}%</p>'.format(pa,pa_p))
            a('<p>Odds {} = {}</p>'.format(num3,f1))
            result=''.join(v)
            context={'input1':num1,'input2':num2,'input3':num3,'detailStep':result,'go_tag':'#odds_probability'}
            return render(request,'Statistics/odds-probability-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1,'input2':num2,'input3':num3}
            return render(request,'Statistics/odds-probability-calculator.html',context)
    else:
	    return render(request,'Statistics/odds-probability-calculator.html')

#Standard_Deviation_Calculator
def standard_deviation_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        print(num2)
        try:
            x1=num1.split(',')
            x2=[]
            for i in x1:
                try:
                    x2.append(int(i))
                except:
                    x2.append(float(i))
            import statistics
            if num2=='Population':
                result=statistics.pstdev(x2)
                context={'input1':num1,'input21':'r','detailStep':result,'go_tag':'#standard_deviation'}
            else:
                result=statistics.stdev(x2)
                context={'input1':num1,'input22':'r','detailStep':result,'go_tag':'#standard_deviation'}
            return render(request,'Statistics/standard-deviation-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            if num2=='Population':
                result=statistics.pstdev(x2)
                context={'input1':num1,'input21':'r'}
            else:
                result=statistics.stdev(x2)
                context={'input1':num1,'input22':'r'}
            return render(request,'Statistics/standard-deviation-calculator.html',context)
    else:
	    return render(request,'Statistics/standard-deviation-calculator.html')

#Statistics_Formulas
def statistics_formulas(request):
	return render(request,'Statistics/statistics-formulas.html')

#Variance_Formulas
def variance_formulas(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        try:
            x1=num1.split(',')
            x2=[]
            for i in x1:
                try:
                    x2.append(int(i))
                except:
                    x2.append(float(i))
            mn=0
            for i in x2:
                mn+=i
            mn_res=mn/len(x2)
            sum_squares=0
            for i in x2:
                b1=(i-mn_res)**2
                sum_squares+=b1
            import statistics
            res=statistics.stdev(x2)
            result=round(res**2,5)
            v=[]
            a=v.append
            a('<p><strong>Standard Deviation:</strong></p>')
            a('<p>{}</p>'.format(res))
            a('<p><strong>Sum of Squares:</strong></p>')
            a('<p>{}</p>'.format(sum_squares))
            a('<p><strong>Variance:</strong></p>')
            a('<p>{}</p>'.format(result))
            a('<p><strong>Mean:</strong></p>')
            a('<p>{}</p>'.format(mn_res))
            a('<p><strong>Count:</strong></p>')
            a('<p>{}</p>'.format(len(x2)))
            context={'input1':num1,'detailStep':''.join(v),'go_tag':'#variance'}
            return render(request,'Statistics/variance-calculator.html',context)
        except:
            messages.error(request,"Enter a valid input")
            context={'input1':num1}
            return render(request,'Statistics/variance-calculator.html',context)
    else:
	    return render(request,'Statistics/variance-calculator.html')

#Vote_Percentage_Calculator
def vote_percentage_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        num3=request.POST.get("input3")
        try:
            num1=int(num1)
            num2=int(num2)
            num3=int(num3)
            r1=num1+num2+num3
            x1=(num1/r1)*100
            x2=(num2/r1)*100
            x3=(num3/r1)*100
            x1=round(x1,2)
            x2=round(x2,2)
            x3=round(x3,2)
            return render(request,'Statistics/vote-percentage-calculator.html',{'input1':num1,'input2':num2,'input3':num3,'res1':x1,'res2':x2,'res3':x3})
        except:
            messages.error(request,"Enter a valid input")
            return render(request,'Statistics/vote-percentage-calculator.html',{'input1':num1,'input2':num2,'input3':num3})
    else:
	    return render(request,'Statistics/vote-percentage-calculator.html')

#Z-Score_Calculator
def z_score_calculator(request):
    if request.method=='POST':
        num1=request.POST.get("input1")
        num2=request.POST.get("input2")
        num3=request.POST.get("input3")
        try:
            try:
                n1=int(num1)
            except:
                n1=float(num1)
            try:
                n2=int(num2)
            except:
                n2=float(num2)
            try:
                n3=int(num3)
            except:
                n3=float(num3)
            r1=n1-n2
            r2=r1/n3
            v=[]
            a=v.append
            a('<p><strong>Z-score:</strong></p>')
            a('<p>{}</p>'.format(r2))
            return render(request,'Statistics/z-score-calculator.html',{'input1':num1,'input2':num2,'input3':num3,'detailStep':''.join(v),'go_tag':'#z_score'})
        except:
            messages.error(request,"Enter a valid input")
            return render(request,'Statistics/z-score-calculator.html',{'input1':num1,'input2':num2,'input3':num3})
    else:
	    return render(request,'Statistics/z-score-calculator.html')

def bayes_theorem(request):
    if request.method=="POST":
        a=request.POST['a']
        b=request.POST['b']
        c=request.POST['c']
        description={}
        description['l1_bold']="Bayes Theorem : P(A/B) = (P(B/A)*P(B)) / P(A)"
        if(int(a)>100 or int(b)>100 or int(c)>100 or int(a)<1 or int(b)<1 or int(c)<1):
            messages.info(request,"Invalid Value")
        else:
            description['l2']="P(A/B) = ( "+c+" * "+b+" ) / "+a
            d=int(c)*int(b)/int(a)
            description['l3']="P(A/B) = "+str(d)
            return render(request, 'Statistics/bayes-theorem-calculator.html',context={'a':a,'b':b,"c":c,"d":d,"answer":description})
    return render(request, 'Statistics/bayes-theorem-calculator.html')
def birthday_paradox(request):
    if request.method=="POST":
        a=request.POST['a']
        if(int(a)<1):
            messages.info(request,"Invalid Value")
        else:
            d=int(a)*(int(a)-1)/2
            ans=(1-((364/365)**d))*100
            return render(request, 'Statistics/birthday-paradox-calculator.html',context={'a':a,"d":ans})
    return render(request, 'Statistics/birthday-paradox-calculator.html')

def cheb_theorem(request):
    if request.method=="POST":
        f_unit=request.POST['f_type']
        a=request.POST['a']
        b=request.POST['b']
        if(f_unit=="one"):
            d=(int(b)/int(a))*100
        else:
            d=(1/(int(a)**2))*100
        return render(request, 'Statistics/chebyshevs-theorem-calculator.html', context={'a': a,"b":b,"d":d,"f_unit":f_unit})
    return render(request, 'Statistics/chebyshevs-theorem-calculator.html')

def perm(request):
    if request.method=="POST":
        a=request.POST['a']
        b=request.POST['b']
        description={}
        if(int(a)<1 or int(b)<1):
            messages.info(request,"Invalid Value")
        else:
            description['l1_bold']="P(n,r) = n!/(n-r)!"
            description['l2']="where p is the number of permutations;"
            description['l3']="n is the total number of elements in the set; and"
            description['l4']="r is the number of elements you choose from this set"
            c=(math.factorial(int(a))/math.factorial(int(a)-int(b)))
            description['l5'] = "Permutation without repitition = "+str(c)
            d=int(a)**int(b)
            description['l6'] = "Permutation with repitition = " + str(d)
            return render(request, 'Statistics/permutation-calculator.html',context={'a':a,'b':b,"c":int(c),"d":int(d),"answer":description})
    return render(request, 'Statistics/permutation-calculator.html')

def comb(request):
    if request.method=="POST":
        a=request.POST['a']
        b=request.POST['b']
        description={}
        if(int(a)<1 or int(b)<1):
            messages.info(request,"Invalid Value")
        else:
            description['l1_bold']="C(n,r) = n!/(r!(n-r)!)"
            description['l2']="where C is the number of combinations;"
            description['l3']="n is the total number of elements in the set; and"
            description['l4']="r is the number of elements you choose from this set"
            c=(math.factorial(int(a))/(math.factorial(int(a)-int(b))*math.factorial(int(b))))
            description['l5'] = "Combination without repitition = "+str(c)
            d=(math.factorial(int(b)+int(a)-1)/(math.factorial(int(a)-1)*math.factorial(int(b))))
            description['l6'] = "Combination with repitition = " + str(d)
            return render(request, 'Statistics/combination-calculator.html',context={'a':a,'b':b,"c":int(c),"d":int(d),"answer":description})
    return render(request, 'Statistics/combination-calculator.html')

def relative_risk(request):
    if request.method=="POST":
        a=request.POST['a']
        b=request.POST['b']
        c=request.POST['c']
        d=request.POST['d']
        description={}
        if(int(a)<1 or int(b)<1 or int(c)<1 or int(d)<1):
            messages.info(request,"Invalid Value")
        else:
            description['l1_bold']="RR = [a / (a + b)] / [c / (c + d)]"
            description['l2']="where a is the number of members of the exposed group who developed the disease;"
            description['l3']="b is the number of members of the exposed group who didn't develop the disease;"
            description['l4']="c is the number of members of the control group who developed the disease;"
            description['l5']="d is the number of members of the control group who didn't develop the disease;"
            description['l6'] ="RR is the relative risk."
            e=(int(a)/((int(a)+int(b)))/(int(c)/((int(c)+int(d)))))
            description['l7']="Relative Risk = "+str(e)
            return render(request, 'Statistics/Relative_risk_calculator.html',context={'a':a,'b':b,"c":c,"d":d,"e":e,"answer":description})
    return render(request, 'Statistics/Relative_risk_calculator.html')

def population_variance(request):
    if request.method=="POST":
        try:
            a=request.POST['a']
            l=a.split(",")
            data=[]
            for i in l:
                if(i!=" "):
                    data.append(int(i))
            description={}
            description['l1_bold']="σ2 = ∑(xi - μ)2 / N"
            description['l2']="σ2 is the variance"
            description['l3']="μ is the mean; and"
            description['l4']="xᵢ represents the ith data point out of N total data points"
            mean=sum(data)/len(data)
            description['l5']="mean = "+str(mean)
            var = sum((x-mean)**2 for x in data) / len(data)
            description['l6']="Variance = "+str(var)
            std=math.sqrt(var)
            description['l7']="Standard Deviation = "+str(std)
            return render(request, 'Statistics/population-variance-calculator.html',context={"a":a,"variance":var,"answer":description})
        except:
            messages.info(request,"Wrong input format")
    return render(request, 'Statistics/population-variance-calculator.html')

def pearson_coeff(request):
    if request.method=="POST":
        try:
            a=request.POST['a']
            b=request.POST['b']
            l=a.split(",")
            data=[]
            data1=[]
            s1=0;s2=0;
            for i in l:
                if(i!=" "):
                    data.append(int(i))
                    s1+=int(i)**2
            for i in b.split(","):
                if(i!=" "):
                    data1.append(int(i))
                    s2+=int(i)**2
            if(len(data)<3 or len(data)!=len(data1)):
                messages.info(request,"Enter Atleast three values")
            else:
                description={}
                description['l1_bold']="r<sub>xy</sub>=(Σx<sub>i</sub> * y<sub>i</sub> - n * x̄ * ȳ)/√(Σx<sub>i</sub><sup>2</sup> - n * x̄<sup>2</sup>)*√(Σy<sub>i</sub><sup>2</sup> - n * ȳ<sup>2</sup>)"
                description['l2']="r<sub>xy</sub> is the Pearson-correlation-coefficient"
                description['l3']="x̄,ȳ is the mean; and"
                description['l4']="xᵢ,yᵢ represents the ith data point out of N total data points"
                mean=sum(data)/len(data)
                mean1=sum(data1)/len(data1)
                description['l5']="x̄ = "+str(mean)
                description['l6']="ȳ = "+str(mean1)
                description['l7']="Σx<sub>i</sub><sup>2</sup> = "+str(s1)
                description['l8']="Σy<sub>i</sub><sup>2</sup> = "+str(s2)
                s3=0;
                for i in range(0,len(data)):
                    s3+=data[i]*data1[i]
                description['l9']="Σx<sub>i</sub>y<sub>i</sub> = "+str(s3)
                n=len(data)
                r=(s3-len(data)*mean*mean1)/(math.sqrt(s1-n*mean**2)*math.sqrt(s2-n*mean1**2))
                description['l10']="Pearon Correlation Coefficient = "+str(r)
                return render(request, 'Statistics/Pearson-correlation-coefficient.html',context={"a":a,"b":b,"coeff":r,"answer":description})
        except:
            messages.info(request,"Wrong input format")

    return render(request, 'Statistics/Pearson-correlation-coefficient.html')

def check_decimal_values(value):
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        return value
def class_width(request):
    if request.method == "POST":
        min = check_decimal_values(request.POST.get('min'))
        max = check_decimal_values(request.POST.get('max'))
        classes = check_decimal_values(request.POST.get('classes'))
        if min >= max:
            messages.error(
                request, "The maximum value should be greater than minimum value ")

            context = {
                 'classes':classes,
            }
            return render(request, 'Statistics/classwidth.html', context)
        
        classwidth = round(((max-min)/classes),4)
        
        context = {
            'min' : min,
            'max': max,
            'classes':classes,
            'classwidth' : classwidth,
            'id':1
        }
        return render(request, "Statistics/classwidth.html", context)
    else:
        d = {'min':10,'max':50,'classes':10}
        return render(request, "Statistics/classwidth.html",d)

def standard_deviation_index(request):
    if request.method == "POST":
        Laboratorymean = check_decimal_values(request.POST.get('Laboratorymean'))
        Consensusgroupmean = check_decimal_values(request.POST.get('Consensusgroupmean'))
        Consensusgroupstandarddeviation = check_decimal_values(request.POST.get('Consensusgroupstandarddeviation'))
        SDI = round(((Laboratorymean-Consensusgroupmean)/Consensusgroupstandarddeviation),3)
        context = {
            'Laboratorymean' : Laboratorymean,
            'Consensusgroupmean': Consensusgroupmean,
            'Consensusgroupstandarddeviation':Consensusgroupstandarddeviation,
            'SDI' : SDI,
            'id':1
        }
        return render(request, "Statistics/standarddeviationindex.html", context)
    else:
        d = {'Laboratorymean':10,'Consensusgroupmean':50,'Consensusgroupstandarddeviation':10}
        return render(request, "Statistics/standarddeviationindex.html",d)

def odds_calculator(request):
    if request.method=="POST":
        a=request.POST['a']
        b=request.POST['b']
        description={}
        if(int(a)<1 or int(b)<1):
            messages.info(request,"Invalid Value")
        else:
            description['l1_bold']="Probability of Winning= Chances of Winning / (Chances of Winning + Chances Of Losing)"
            description['l2_bold']="Probability of Losing= Chances of Losing / (Chances of Winning + Chances Of Losing)"
            description['l3_bold']="Fractional Odds = Chances of Losing / Chances Of Winning"
            description['l4_bold']="Decimal Odds = Represent the total return for every $1 stake"
            description['l5']="Probabiblity of Winning = "+a+" / "+str(int(a)+int(b))
            k=int(a)+int(b)
            c=int(a)/k
            description['l6']="Probabiblity of Winning = "+str(c*100)+" %"
            description['l7']="Probabiblity of Losing = "+b+" / "+str(int(a)+int(b))
            d=int(b)/k
            description['l8']="Probabiblity of Losing = "+str(d*100)+" %"
            
            k1=math.gcd(int(b),int(a))
            e=str(int(b)//k1)+" / "+str(int(a)//k1)
            description['l9']="Fractional Odds = "+str(e)
            f=1+float(d/c)
            description['l10']="Decimal Odds = "+str(f)
            return render(request, 'Statistics/odds-calculator.html',context={'a':a,'b':b,"c":c*100,"d":d*100,"e":e,"f":f,"answer":description})
    return render(request, 'Statistics/odds-calculator.html')

def normal_distribution(request):
    if request.method=="POST":
        mean=request.POST['mean']
        stdev=request.POST['stdev']
        xcritical=request.POST['xcritical']
        zscore=((int(xcritical)-int(mean))/int(stdev))
        prob_2=norm.cdf(zscore)
        prob_1=1-prob_2
        return render(request, 'Statistics/normal_distribution.html',context={"mean":mean,"stdev":stdev,"xcritical":xcritical,"zscore":zscore,"prob_1":prob_1,"prob_2":prob_2})
    return render(request, 'Statistics/normal_distribution.html')

def probability_calculator(request):
    if request.method=="POST":
        a=int(request.POST['a'])
        b=int(request.POST['b'])
        if(a<0 or b<0):
            messages.info(request,"invalid value")
        c=(a*b)/100
        d=a+b-c
        e=d-c
        f=100-d
        g=100-a
        h=100-b 
        return render(request, 'Statistics/probability-calculator.html',context={"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"g":g,"h":h})
    return render(request, 'Statistics/probability-calculator.html')
def p_value(request):
    if request.method=="POST":
        de=""
        w_unit=request.POST['w_unit']
        h_unit=request.POST['h_unit']
        score=request.POST['score']
        if(h_unit=="xr"):
            de=request.POST['de']
            if(w_unit=="t"):
                p=2*(1 - stats.chi2.cdf(float(score), int(de)))
            elif(w_unit=="l"):
                p=stats.chi2.cdf(float(score), int(de))
            else:
                p=1 - stats.chi2.cdf(float(score), int(de))
            stats.chi2.cdf(3.84, 1)
        elif(h_unit=="ts"):
            de=request.POST['de']
            if(w_unit=="t"):
                p=2*(1 - t.cdf(abs(float(score)), int(de)))
            elif(w_unit=="l"):
                p=t.cdf(abs(float(score)), int(de))
            else:
                p=1 - t.cdf(abs(float(score)), int(de))
            
        else:
            if(w_unit=="t"):
                p=2*(1 - norm.cdf(float(score)))
            elif(w_unit=="l"):
                p=norm.cdf(float(score))
            else:
                p=1 - norm.cdf(float(score))
        return render(request, 'Statistics/p-value.html',context={"h_unit":h_unit,"w_unit":w_unit,"score":score,"de":de,"p":p})
    return render(request, 'Statistics/p-value.html')
def sample_size(request):
    #n₁ = Z² * p * (1-p) / ME²
    if request.method=="POST":
        c=request.POST['c']
        m=request.POST['m']
        pe=request.POST['pe']
        pe1=float(pe)/100
        z=2*norm.ppf(float(c)/100)
        sz=(z*(pe1)*(1-pe1))/((float(m)/100)**2)
        description={}
        description['l1_bold']="n₁ = Z² * p * (1-p) / ME²"
        description['l2']="where Z — The z-score associated with the confidence level you chose"
        description['l3']="ME — Margin of error, also known as the confidence interval"
        description['l4']="p — Your initial proportion estimate"
        description['l5']="n₁ — Required sample size."
        description['l6']="z-score = "+str(z)
        description['l7']="Sample Size = "+str(int(sz))
        return render(request, 'Statistics/sample-size-calculator.html',context={"c":c,"m":m,"pe":pe,"sz":int(sz),"answer":description})
    return render(request, 'Statistics/sample-size-calculator.html')

def numRollsToTarget( d, f, target):
    dp = [1] + [0]*target
    for i in range(d):
        for j in range(target, -1, -1):
            dp[j] = sum([dp[j-k] for k in range(1, 1+min(f, j))] or [0])
    return dp[target] 
def dicecalculator(request):

    if request.method=='POST':
        if "f1" in request.POST:
            selectdice = request.POST.get('selectdice')

            noofdice = request.POST.get('noofdice')
            gamerule = request.POST.get('gamerule')
            valueondice = request.POST.get('valueondice')

            if gamerule[0]== '1':
                k=pow(int(selectdice),int(noofdice))
                probability=1/k
                context = {
                    'k' : k,
                    'no_of_outcome':1,
                    'probability' : probability,
                    # 'max': max,
                    # 'classes':classes,
                    # 'classwidth' : classwidth,
                    'id':1
                }
                print(probability)
            if gamerule[0]== '2':
                k=pow(int(selectdice),int(noofdice))
                no_of_outcome = pow((int(selectdice)-int(valueondice)+1),int(noofdice))
                probability=no_of_outcome/k
                context = {
                    'k' : k,
                    'no_of_outcome':no_of_outcome,
                    'probability' : probability,
                    # 'max': max,
                    # 'classes':classes,
                    # 'classwidth' : classwidth,
                    'id':1
                }
                print(k,no_of_outcome)
            if gamerule[0]== '3':
                k=pow(int(selectdice),int(noofdice))
                no_of_outcome = pow((int(valueondice)),int(noofdice))
                probability=no_of_outcome/k
                context = {
                    'k' : k,
                    'no_of_outcome':no_of_outcome,
                    'probability' : probability,
                    # 'max': max,
                    # 'classes':classes,
                    # 'classwidth' : classwidth,
                    'id':1
                }
                print(k,no_of_outcome)
            if gamerule[0]== '4':
                k=pow(int(selectdice),int(noofdice))
                no_of_outcome = numRollsToTarget( int(noofdice), int(selectdice), int(valueondice))
                probability=no_of_outcome/k
                context = {
                    'k' : k,
                    'no_of_outcome':no_of_outcome,
                    'probability' : probability,
                    # 'max': max,
                    # 'classes':classes,
                    # 'classwidth' : classwidth,
                    'id':1
                }
                print(k,no_of_outcome)
            if gamerule[0]== '5':
                k=pow(int(selectdice),int(noofdice))
                no_of_outcome = 0
                max_sum = int(selectdice)*int(noofdice)
                p = int(valueondice)
                for sum in range(p,max_sum+1):
                    no_of_outcome+=numRollsToTarget( int(noofdice), int(selectdice), sum)
                probability=no_of_outcome/k
                context = {
                    'k' : k,
                    'no_of_outcome':no_of_outcome,
                    'probability' : probability,
                    # 'max': max,
                    # 'classes':classes,
                    # 'classwidth' : classwidth,
                    'id':1
                }
                print(k,no_of_outcome)
            if gamerule[0]== '6':
                k=pow(int(selectdice),int(noofdice))
                no_of_outcome = 0
                p=int(valueondice)
                for sum in range(1,p+1):
                    no_of_outcome+=numRollsToTarget( int(noofdice), int(selectdice), sum)
                probability=no_of_outcome/k
                context = {
                    'k' : k,
                    'no_of_outcome':no_of_outcome,
                    'probability' : probability,
                    # 'max': max,
                    # 'classes':classes,
                    # 'classwidth' : classwidth,
                    'id':1
                }
                print(k,no_of_outcome)
            
            return render(request,'Statistics/dicecalculator.html',context)
        else:
            gamerule = request.POST.get('gamerule')
            selectdice = request.POST.get('selectdice')
            noofdice = request.POST.get('noofdice')
            valueondice = request.POST.get('valueondice')
            id1 = 0
            if len(gamerule)!=0:
                if gamerule[0] == '7':
                    id1=7
                    context = {
                        'noofdice':str(noofdice),
                        'valueondice':str(valueondice),
                        'id1':id1
                    }
                    return render(request,'Statistics/dicecalculator.html',context)
                if gamerule[0] == '8':
                    id1=8
                    context = {
                        'noofdice':str(noofdice),
                        'valueondice':str(valueondice),
                        'id1':id1
                    }
                    return render(request,'Statistics/dicecalculator.html',context)
                else:
                    context = {
                        'noofdice':str(noofdice),
                        'valueondice':str(valueondice),
                        'selectdice':str(selectdice),
                        'gamerule':str(gamerule)
                    }
                    return render(request,'Statistics/dicecalculator.html',context)
            else:
                selectdice = request.POST.get('selectdice')
                context = {
                    'noofdice':str(noofdice),
                    'valueondice':str(valueondice),
                    'selectdice':str(selectdice)
                }
                return render(request,'Statistics/dicecalculator.html',context)



                

                
        

        
    else:
        d = {'no-of-dice':1}
        return render(request,'Statistics/dicecalculator.html',d)

def risk_calculator(request):
    
    
    if request.method == "POST":
        user_input = check_decimal_values(request.POST.get('probability'))
        user_input2 = check_decimal_values(request.POST.get('loss'))

        risk = round((user_input*user_input2),4)
        d={'risk':risk,'probability':user_input,'loss':user_input2,'id':1}
        return render(request, "Statistics/riskcalc.html",d)
        
    else:
        d={'risk':10,'probability':5,'loss':2}
        return render(request, "Statistics/riskcalc.html",d)
def fact(n):
     
    res = 1
    for i in range(2, n + 1):
        res = res * i
    return res
 
# Applying the formula
def count_heads(n, r):
     
    output = fact(n) / (fact(r) * fact(n - r))
    output = output / (pow(2, n))
    return output



def coin_flip(request):
    if request.method == "POST":
        flips = request.POST.get('flips')
        heads = request.POST.get('heads')
        probability = count_heads(int(flips),int(heads))
                
        
        context = {
            'heads' : heads,
            'flips': flips,
            #'classes':classes,
            'probability' : probability,
            'id':1
        }
        return render(request, "Statistics/coin_flip.html", context)
    else:
        d = {'heads':4,'flips':4}
        return render(request, "Statistics/coin_flip.html",d)

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer

        
def lottery_calculator(request):
    if request.method == "POST":
        m = (request.POST.get('m'))
        n = (request.POST.get('n'))
        t = (request.POST.get('t'))
        m1 = int(m)
        n1 = int(n)
        t1 = int(t)
        f = ncr(t1, m1)/(ncr(m1, n1)*ncr(t1-m1, m1-n1))
        context = {
            'f': f,
            'm1': m1,
            'n1': n1,
            't1': t1,
            'id': 1
        }
        return render(request, "Statistics/lottery.html", context)
    else:
        d = {'t': 0}
        return render(request, "Statistics/lottery.html", d)
def dice_calculator(request):
    if request.method == "POST":
        Dice = (request.POST.get('D'))
        s = (request.POST.get('ab'))
        f = s
        s1 = int(f)
        l = []
        Dice1 = Dice
        Dice = int(Dice)
        while (Dice > 0):
            N = random.randint(1, s1)
            l.append(N)
            Dice -= 1
        context = {
            'Dice': Dice1,
            'S': s,
            'N': l,
            'id': 1
        }
        return render(request, "Statistics/dice_Calculator.html", context)
    else:
        d = {'Dice': 0}
        return render(request, "Statistics/dice_Calculator.html", d)

import scipy.stats

# Create your views here.




def calculate_z_score(sample_mean,test_mean,standard_deviation,sample_size):
    z_score = (sample_mean - test_mean) / (standard_deviation / math.sqrt(sample_size))
    return z_score

def calculate_p_value(z_score):
    p_value = scipy.stats.norm.sf(abs(z_score))
    return p_value


def z_test(request):
    if request.method == "POST":
        sample_mean = request.POST.get('Sample mean x̄')
        test_mean = request.POST.get('Tested mean μ₀')
        standard_deviation = request.POST.get('Population standard deviation σ')
        sample_size= request.POST.get('Sample size n')

        z_score = calculate_z_score(float(sample_mean),float(test_mean),float(standard_deviation),float(sample_size))
        p_value = calculate_p_value(z_score)
        if p_value < 0.05:
            result = 'You can reject H₀ at the significance level 0.05, because your p-value does not exceed 0.05'
        else:
            result = 'There is not enough evidence to reject H₀ at the significance level 0.05, because your p-value is greater than 0.05.'

                
        
        context = {
            'Sample_mean' : sample_mean,
            'Tested_mean': test_mean,
            'Population_standard_deviation': standard_deviation,
            'Sample_size' : sample_size,
            'z_score' : z_score,
            'P_value' : p_value,
            'result' : result,
            'id':1
        }
        return render(request, "Statistics/z_test.html", context)
    else:
       
        return render(request, "Statistics/z_test.html")

def probability_3_events(request):
    if request.method=="POST":
        a=float(request.POST['a'])
        b=float(request.POST['b'])
        c=float(request.POST['c'])
        if(a<0 or b<0 or c<0 or a>1 or b>1 or c>1):
            messages.info(request,"invalid value")
        description={}
        description['l1_bold']="Probability of all events occuring is P(A ∩ B ∩ C) = P(A) * P(B) * P(C)"
        description['l2_bold']="Probability of atleast one event occuring is P(A ∪ B ∪ C) = P(A) + P(B) + P(C) - P(A) * P(B) - P(A) * P(C) - P(B) * P(C) + P(A) * P(B) * P(C)"
        description['l3_bold']="Probability of exactly one event occuring is P(A ∩ B' ∩ C') + P(A' ∩ B ∩ C') + P(A' ∩ B' ∩ C) = P(A) * P(B') * P(C') + P(A') * P(B) * P(C') + P(A') * P(B') * P(C')"
        description['l4_bold']="Probability of none event occuring is P(∅) = 1 - (P(A) + P(B) + P(C) - P(A) * P(B) - P(A) * P(C) - P(B) * P(C) + P(A) * P(B) * P(C))"
        d=a*b*c
        e=a+b+c-(a*b)-(a*c)-(b*c)+(a*b*c)
        f=(1-a)*b*(1-c)+(1-b)*(1-c)*a+(1-a)*(1-b)*c
        g=1-e
        description['l4']="P(A ∩ B ∩ C) = "+str(d)
        description['l5']="P(A ∪ B ∪ C) = "+str(e)
        description['l6']="P(A ∩ B' ∩ C') + P(A' ∩ B ∩ C') + P(A' ∩ B' ∩ C) = "+str(f)
        description['l7']="P(∅) = "+str(g)
        return render(request, 'Statistics/Probability-3-events-calculator.html',context={"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"g":g,"answer":description})
    return render(request, 'Statistics/Probability-3-events-calculator.html')

def central_limit(request):
    if request.method=="POST":
        a=int(request.POST['a'])
        b=int(request.POST['b'])
        c=int(request.POST['c'])
        if(a<0 or b<0 or c<0):
            messages.info(request,"invalid value")
        description={}
        description['l1_bold']="Sample Mean = Population Mean"
        description['l2_bold']="Sample Standard Deviation = Population Standard Deviation / Sample Size"
        d=a
        e=b/math.sqrt(c)
        
        description['l4']="Sample Mean = "+str(d)
        description['l5']="Sample Standard Deviation = "+str(e)
        return render(request, 'Statistics/central-limit-theorem-calculator.html',context={"a":a,"b":b,"c":c,"d":d,"e":e,"answer":description})
    return render(request, 'Statistics/central-limit-theorem-calculator.html')

def Rayleigh_distribution_calculator(request):
    if request.method=="POST":
        param2=request.POST['b']
        mode=request.POST['m_unit']
        description={}
        if(mode=="p"):
            
            b=int(param2)
            c=int(request.POST['c'])
            p_type=request.POST['p_unit']
            if(c>=0):
                temp=((-1)*c**2)/(2*(b**2))
                p=1-math.e**temp
            else:
                p=0
            if(p_type=="ge" or p_type=="gt"):
                p=1-p;
            description['l2']="Probability = "+str(p)
            return render(request, 'Statistics/rayleigh-distribution-calculator.html',context={"mode":mode,"prob":p_type,"b":b,"c":c,"j":p,"answer":description})
        elif(mode=="pd"):
            param3=request.POST['c']
           
            b=int(param2)
            c=int(param3)
            if(b<0):
                messages.info(request,"invalid value")
            description={}
             
            if(c<0):
                description['l1_bold']="Probability density function is f(x) = 0 if x<0"
            
                pd=0;
            else:
                description['l1_bold']="Probability density function is f(x) = (x * σ<sup>-2</sup>)* e <sup>-(x<sup>2</sup>/<sub>2 * σ<sup>2</sup></sup> if x>=0"
            
                temp1=((-1)*c**2)/(2*(b**2))
                pd=(c/b**2)*math.e**temp1
            
            description['l2']="Probability density function is f(x) = "+str(pd)
            return render(request, 'Statistics/rayleigh-distribution-calculator.html',context={"mode":mode,"b":b,"c":c,"j":pd,"answer":description})
        elif(mode=="q"):
            param3=request.POST['c']
            
            b=int(param2)
            c=float(param3)
            if(b<0 or c<0):
                messages.info(request,"invalid value")
            description={}
            description['l1_bold']="Quantile function is Q(p) = σ * √(-2 * ln(1-p)"
            temp3=math.sqrt((-2)*math.log(1-c))
            f2=b*temp3
            description['l2']="Quantile function is Q(p) = "+str(f2)
            return render(request, 'Statistics/rayleigh-distribution-calculator.html',context={"mode":mode,"b":b,"c":c,"j":f2,"answer":description})
        elif(mode=="c"):
            param3=request.POST['c']
            b=int(param2)
            c=float(param3)
            if(b<0):
                messages.info(request,"invalid value")
            description={}
            if(c<0):
                f1=0
                description['l1_bold']="Cumalative Distribution function is F(x) = 0 if x<0"
            else:
                description['l1_bold']="Cumalative Distribution function is F(x) = 1 - e<sup>-x<sup>2</sup>/<sub>(2 * σ)</sub><sup>2</sup></sup> if x>=0"
                temp=((-1)*c**2)/(2*(b**2))
                f1=1-math.e**temp
            description['l2']="Cumlative Distribution function is F(x) = "+str(f1)
            return render(request, 'Statistics/rayleigh-distribution-calculator.html',context={"mode":mode,"b":b,"c":c,"j":f1,"answer":description})
        elif(mode=="cm"):
            b=int(param2)
            mean=b*math.sqrt(math.pi/2)
            median=b*math.sqrt(2*math.log(2))
            mode1=b
            variance=(b**2)*((4-math.pi)/2)
            sd=math.sqrt(variance)
            skew=2*(math.pi -3)*(math.sqrt(math.pi/((4-math.pi)**3)))
            description['l1_bold']="Mean = σ * √(π / 2)"
            description['l2_bold']="Median = σ * √(2 ⋅ ln(2)"
            description['l3_bold']="Mode = σ"
            description['l4_bold']="Variance = σ² * (4 - π)/2"
            description['l6_bold']="Skewness = 2 * (π - 3) * √(π / (4 - π)³)"
            description['l7']="Mean = "+str(mean)
            description['l8']="Median = "+str(median)
            description['l9']="Mode = "+str(mode1)
            description['l10']="Variance = "+str(variance)
            description['l11']="Standard Deviation = "+str(sd)
            description['l12']="Skewness = "+str(skew)
            return render(request, 'Statistics/rayleigh-distribution-calculator.html',context={"mode":mode,"b":b,"d":mean,"e":median,"f":mode1,"g":variance,"h":sd,"i":skew,"answer":description})
    return render(request, 'Statistics/rayleigh-distribution-calculator.html')

def lognormal_distribution_calculator(request):
    if request.method=="POST":
        param1=request.POST['a']
        param2=request.POST['b']
        mode=request.POST['m_unit']
        description={}
        if(mode=="s"):
            a=int(param1)
            b=int(param2)
            c=int(request.POST['c'])
            mean=math.e**(a+((b**2)/2))
            variance=(math.e**((b**2))-1)*math.e**((2*a)+(b**2))
            sd=math.sqrt(variance)
            sample=np.random.lognormal(a, b, c)
            print(sample)
            description['l1']="Sample = "+str(sample)
            return render(request, 'Statistics/lognormal-distribution.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":sample,"answer":description})
        elif(mode=="p"):
            a=int(param1)
            b=int(param2)
            c=int(request.POST['c'])
            p_type=request.POST['p_unit']
            p=0.5+0.5*math.erf((math.log(c)-a)/(math.sqrt(2)*b))
            if(p_type=="ge" or p_type=="gt"):
                p=1-p;
            description['l2']="Probability = "+str(p)
            return render(request, 'Statistics/lognormal-distribution.html',context={"mode":mode,"prob":p_type,"a":a,"b":b,"c":c,"j":p,"answer":description})
        elif(mode=="pd"):
            param3=request.POST['c']
            a=int(param1)
            b=int(param2)
            c=int(param3)
            if(a<0 or b<0 or c<0):
                messages.info(request,"invalid value")
            description={}
            description['l1_bold']="Probability density function is f(x) = ( 1/x * σ * √2 * π )* e <sup>-(ln(x) - μ)<sup>2</sup>/<sub>2 * σ<sup>2</sup></sup>"
            temp1=1/(c*b*math.sqrt(2*math.pi))
            temp2=-(((math.log(c)-a)**2)/2*(b**2))
            pd=temp1*(math.e**temp2)
            # f1=0.5+0.5*math.erf((math.log(c)-a)/(math.sqrt(2)*b))
            # temp3=a+(b*math.sqrt(2))/(math.erf(2*c-1))
            # f2=math.e*temp3
            description['l2']="Probability density function is f(x) = "+str(pd)
            return render(request, 'Statistics/lognormal-distribution.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":pd,"answer":description})
        elif(mode=="q"):
            param3=request.POST['c']
            a=int(param1)
            b=int(param2)
            c=float(param3)
            if(a<0 or b<0 or c<0):
                messages.info(request,"invalid value")
            description={}
            description['l1_bold']="Quantile function is Q(p) = exp(μ + σ * √2 erf<sup>-1</sup>(2* p - 1)"
            # temp1=1/(c*b*math.sqrt(2*math.pi))
            # temp2=-(((math.log(c)-a)**2)/2*(b**2))
            # pd=temp1*(math.e**temp2)
            # f1=0.5+0.5*math.erf((math.log(c)-a)/(math.sqrt(2)*b))
            temp3=a+(b*math.sqrt(2))/(math.erf(2*c-1))
            f2=math.e*temp3
            description['l2']="Quantile function is Q(p) = "+str(f2)
            return render(request, 'Statistics/lognormal-distribution.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":f2,"answer":description})
        elif(mode=="c"):
            param3=request.POST['c']
            a=int(param1)
            b=int(param2)
            c=float(param3)
            if(a<0 or b<0 or c<0):
                messages.info(request,"invalid value")
            description={}
            description['l1_bold']="Cumalative Distribution function is F(x) = 1/2 + ( 1/2 * erf((ln(x) - μ)/ √2 * σ )"
            f1=0.5+0.5*math.erf((math.log(c)-a)/(math.sqrt(2)*b))
            description['l2']="Cumalative Distribution function is F(x) = "+str(f1)
            return render(request, 'Statistics/lognormal-distribution.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":f1,"answer":description})
        elif(mode=="cm"):
            a=int(param1)
            b=int(param2)
            mean=math.e**(a+((b**2)/2))
            median=math.e**a
            mode1=math.e**(a-(b**2))
            variance=(math.e**((b**2))-1)*math.e**((2*a)+(b**2))
            sd=math.sqrt(variance)
            skew=(math.e**((b**2))+2)*math.sqrt(math.e**((b**2))-1)
            description['l1_bold']="Mean = exp(μ + σ² / 2)"
            description['l2_bold']="Median = exp(μ)"
            description['l3_bold']="Mode = exp(μ - σ²)"
            description['l4_bold']="Variance = [exp(σ²) - 1] ⋅ exp(2μ + σ²)"
            description['l6_bold']="Skewness = [exp(σ²) + 2] ⋅ √[exp(σ²) - 1]"
            description['l7']="Mean = "+str(mean)
            description['l8']="Median = "+str(median)
            description['l9']="Mode = "+str(mode1)
            description['l10']="Variance = "+str(variance)
            description['l11']="Standard Deviation = "+str(sd)
            description['l12']="Skewness = "+str(skew)
            return render(request, 'Statistics/lognormal-distribution.html',context={"mode":mode,"a":a,"b":b,"d":mean,"e":median,"f":mode1,"g":variance,"h":sd,"i":skew,"answer":description})
    return render(request, 'Statistics/lognormal-distribution.html')

def weibull_distribution_calculator(request):
    if request.method=="POST":
        param1=request.POST['a']
        param2=request.POST['b']
        mode=request.POST['m_unit']
        description={}
        if(mode=="s"):
            a=int(param1)
            b=int(param2)
            c=int(request.POST['c'])
            mean=math.e**(a+((b**2)/2))
            variance=(math.e**((b**2))-1)*math.e**((2*a)+(b**2))
            sd=math.sqrt(variance)
            sample=np.random.lognormal(a, b, c)
            print(sample)
            description['l1']="Sample = "+str(sample)
            return render(request, 'Statistics/Weibull-distribution-calculator.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":sample,"answer":description})
        elif(mode=="p"):
            a=int(param1)
            b=int(param2)
            c=int(request.POST['c'])
            p_type=request.POST['p_unit']
            if(c<0):
                p=0
            else:
                temp=(-1)* ((c/a)**b)
                p=1-math.e**temp
            if(p_type=="ge" or p_type=="gt"):
                p=1-p;
            description['l2']="Probability = "+str(p)
            return render(request, 'Statistics/Weibull-distribution-calculator.html',context={"mode":mode,"prob":p_type,"a":a,"b":b,"c":c,"j":p,"answer":description})
        elif(mode=="pd"):
            param3=request.POST['c']
            a=int(param1)
            b=int(param2)
            c=int(param3)
            if(a<0 or b<0):
                messages.info(request,"invalid value")
            description={}
            if(c<0):
                description['l1_bold']="Probability density function is f(x) = 0 if x<0"
            
                pd=0;
            else:
                description['l1_bold']="Probability density function is f(x) = k * x<sup>(k-1)</sup> * λ<sup>-k</sup> * e<sup>-(x/<sub>λ</sub>)<sup>k</sup></sup>  if x>=0"
            
                temp1=(-1)*((c/a)**b)
                temp2=math.e**temp1
                temp3=1/(a**b)
                temp4=b*(c**(b-1))
                pd=temp2*temp3*temp4
            description['l2']="Probability density function is f(x) = "+str(pd)
            return render(request, 'Statistics/Weibull-distribution-calculator.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":pd,"answer":description})
        elif(mode=="q"):
            param3=request.POST['c']
            a=int(param1)
            b=int(param2)
            c=float(param3)
            if(a<0 or b<0 or c<0):
                messages.info(request,"invalid value")
            description={}
            description['l1_bold']="Quantile function is Q(p) = λ * (-ln(1-p))<sup>1<sub>k</sub></sup>"
            f2=a*((-1)*(math.log(1-c)**(1/k)))
            description['l2']="Quantile function is Q(p) = "+str(f2)
            return render(request, 'Statistics/Weibull-distribution-calculator.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":f2,"answer":description})
        elif(mode=="c"):
            param3=request.POST['c']
            a=int(param1)
            b=int(param2)
            c=float(param3)
            if(a<0 or b<0):
                messages.info(request,"invalid value")
            description={}
            if(c<0):
                f1=0
                description['l1_bold']="Cumlative Distribution function is F(x) = 0 if x<0"
            else:
                description['l1_bold']="Cumlative Distribution function is F(x) = 1 - e<sup>-x/<sub>λ</sub><sup>k</sup></sup> if x>=0"
                temp=(-1)* ((c/a)**b)
                f1=1-math.e**temp
                description['l2']="Cumlative Distribution function is F(x) = "+str(f1)
            return render(request, 'Statistics/Weibull-distribution-calculator.html',context={"mode":mode,"a":a,"b":b,"c":c,"j":f1,"answer":description})
        elif(mode=="cm"):
            a=int(param1)
            b=int(param2)
            mean=a*math.gamma(1+(1/b))
            median=a*(math.log(2)**(1/b))
            if(b>1):
                mode1=a*((b-1)/b)**(1/b)
                description['l3_bold']="Mode = λ * (k-1)<sub>k</sub><sup>1/<sub>k</sub></sup> if k>1"
            else:
                mode1=0;
                description['l3_bold']="Mode = 0 if k<1"
            variance=(a**2)*(math.gamma(1+(2/b))-(math.gamma(1+(1/b))**2))
            sd=math.sqrt(variance)
            skew=((math.gamma(1+(3/b))*(a**3))-(3*mean*(median**2))-(mean**3))/(median**3)
            description['l1_bold']="Mean = λ * Γ(1+1/k)"
            description['l2_bold']="Median = λ(ln(2))<sup>1/<sub>k</sub></sup>"
            description['l4_bold']="Variance = λ² * (Γ(1+2/k) - Γ(1+1/k)<sup>2</sup>"
          
            description['l6_bold']="Skewness = ((Γ(1+3/k) * λ<sup>3</sup> - 3 * μ *σ² - μ<sup>3</sup>)/<sub>σ<sup>3</sup></sub>"
            description['l7']="Mean = "+str(mean)
            description['l8']="Median = "+str(median)
            description['l9']="Mode = "+str(mode1)
            description['l10']="Variance = "+str(variance)
            description['l11']="Standard Deviation = "+str(sd)
            description['l12']="Skewness = "+str(skew)
            return render(request, 'Statistics/Weibull-distribution-calculator.html',context={"mode":mode,"a":a,"b":b,"d":mean,"e":median,"f":mode1,"g":variance,"h":sd,"i":skew,"answer":description})
    return render(request, 'Statistics/Weibull-distribution-calculator.html')
def mann_whitney_u_test(request):
    errors = []
    results = {}
    data = ''
    stats= []
    t_title = ''
    signif = ''
    outcomes=''
    
    


    if request.method == 'POST':
        # get url that the user has entered
        try:
        
            data1 = request.POST.get('data1')
            data2 = request.POST.get('data2')
            tailed = request.POST.get('tail')
            signif = request.POST.get('significance')

            if tailed == 'two':
                t_title = 'Two Tailed'
            elif tailed == 'less':
                t_title = 'One Tailed (smaller)'
            else:
                t_title = 'one Tailed (larger)'
            
            
            

            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )
        
        
        outcomes = mann_whitney(data1, data2, tailed, significant_level=signif)

        
            # True,'small', n1, n2,u_05, stat_a, effect, larger, data1, data2
        try:
            show_sig, sample_siz, n1, n2,u_critical, u_stat, effect_siz, huger, d1, d2 = outcomes
            
            # data1 = [a for i in data1.split(',') for a in i.split(' ') if len(a)>0]
            # data2 = [a for i in data2.split(',') for a in i.split(' ') if len(a)>0]
            data = ['Data 1 :\n\n' + (', ').join([str(d) for d in d1]), 'Data 2 :\n' + (', ').join([str(d) for d in d2])]
            stats = [('Sample Size', n1, n2),('Mean', round(np.mean(d1),3), round(np.mean(d2),3)), 
                    ('Standard Deviation', round(np.std(d1), 3), round(np.std(d2),3)), ('Median', np.median(d1), np.median(d2))]


            if show_sig:
                s = 'Yes'
            else: s = 'No'
            if sample_siz == 'Small':
                items_name = ['Significance', 
                                'Sample Size', 
                                'U Critical', 
                                'Sample Stat', 'Effect Size', 
                                'Larger Group']
            else:
                items_name = ['Significance', 
                                'Sample Size', 
                                'P Value', 
                                'Sample Stat', 'Effect Size', 
                                'Larger Group']
                u_critical = round(u_critical, 4)
            out = [s, sample_siz, u_critical, u_stat, round(effect_siz, 3), huger]
            results = zip(items_name, out)
            
    # print(stats)
        except:
            errors.append(outcomes)
   
    context = {
        'errors' : errors,
        'Datas':data,
        'Stats' : stats,
        'Tails':t_title,
        'Sig_level':signif,
        'results':results,
        'id':1,
    }


    return render(request,'Statistics/mannwhiteny.html',context )

def sensitivity_Specificity(request):
    
    
    if request.method == "POST":
        tp = check_decimal_values(request.POST.get('tp'))
        tn = check_decimal_values(request.POST.get('tn'))
        fp = check_decimal_values(request.POST.get('fp'))
        fn = check_decimal_values(request.POST.get('fn'))
        
        Sensitivity = round((tp/(tp+fn)),4)
        Specificity = round((tn/(fp+tn)),4)
        
        d={'sensitivity':Sensitivity,'specificity':Specificity,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'id':1}
        return render(request, "Statistics/Sensitivity_and_Specificity.html",d)
        
    else:
        d={'sensitivity':0.5,'Specificity':0.5,'tp':1,'tn':1,'fp':1,'fn':1}
        return render(request, "Statistics/Sensitivity_and_Specificity.html",d)


def benfords_law(request):
    
    
    if request.method == "POST":
        d1 = check_decimal_values(request.POST.get('d1'))
        d2 = check_decimal_values(request.POST.get('d2'))
        d3 = check_decimal_values(request.POST.get('d3'))
        d4 = check_decimal_values(request.POST.get('d4'))
        d5 = check_decimal_values(request.POST.get('d5'))
        d6 = check_decimal_values(request.POST.get('d6'))
        d7 = check_decimal_values(request.POST.get('d7'))
        d8 = check_decimal_values(request.POST.get('d8'))
        d9 = check_decimal_values(request.POST.get('d9'))
        
        
        """P1=math.log10(1-1/d1)
        P2=math.log10(1-1/d2)
        P3=math.log10(1-1/d3)
        P4=math.log10(1-1/d4)
        P5=math.log10(1-1/d5)
        P6=math.log10(1-1/d6)
        P7=math.log10(1-1/d7)
        P8=math.log10(1-1/d8)
        P9=math.log10(1-1/d9)
        'P1':P1,'P2':P2,'P3':P3,'P4':P4,'P5':P5,'P6':P6,'P7':P7,'P8':P8,'P9':P9,"""
        sum=d1+d2+d3+d4+d5+d6+d7+d8+d9
        if sum!=0:
            
            f1=str(round((d1/sum)*100,2))+"%"
            f2=str(round((d2/sum)*100,2))+"%"
            f3=str(round((d3/sum)*100,2))+"%"
            f4=str(round((d4/sum)*100,2))+"%"
            f5=str(round((d5/sum)*100,2))+"%"
            f6=str(round((d6/sum)*100,2))+"%"
            f7=str(round((d7/sum)*100,2))+"%"
            f8=str(round((d8/sum)*100,2))+"%"
            f9=str(round((d9/sum)*100,2))+"%"
        else:
            f1="0.00%"
            f2="0.00%"
            f3="0.00%"
            f4="0.00%"
            f5="0.00%"
            f6="0.00%"
            f7="0.00%"
            f8="0.00%"
            f9="0.00%"


        
        d={'P1':"30.10%",'P2':"17.61%",'P3':"12.49%",'P4':"9.69%",'P5':"7.92%",'P6':"6.69%",'P7':"5.80%",'P8':"5.12%",'P9':"4.58%",'d1':d1,'d2':d2,'d3':d3,'d4':d4,'d5':d5,'d6':d6,'d7':d7,'d8':d8,'d9':d9,'f1':f1,'f2':f2,'f3':f3,'f4':f4,'f5':f5,'f6':f6,'f7':f7,'f8':f8,'f9':f9,'id':1}
        return render(request, "Statistics/benford's_law.html",d)
        
    else:
        d={'P1':"30.10%",'P2':"17.61%",'P3':"12.49%",'P4':"9.69%",'P5':"7.92%",'P6':"6.69%",'P7':"5.80%",'P8':"5.12%",'P9':"4.58%",'d1':0,'d2':0,'d3':0,'d4':0,'d5':0,'d6':0,'d7':0,'d8':0,'d9':0,'f1':"0.00%",'f2':"0.00%",'f3':"0.00%",'f4':"0.00%",'f5':"0.00%",'f6':"0.00%",'f7':"0.00%",'f8':"0.00%",'f9':"0.00%"}
        return render(request, "Statistics/benford's_law.html",d)

def smpx(request):
    if request.method == "POST":
        pxmin = request.POST.get('Lower limit value of x (PXmin)')
        xmax = request.POST.get('Upper limit value of x (Xmax)')
        ml = request.POST.get('x where SMp(x) = Max (ML)')
        p1 = request.POST.get('Power (p₁)')
        p2 = request.POST.get('Power (p₂)')
        max = request.POST.get('Maximum of the model (Max)')
        x = request.POST.get('For an independent variable x') 

        if int(x) < int(pxmin):
            smpx_result = 0
            resultant_text = 'condition :  x < PXmin | SMp(x) = 0'
        elif pxmin <= x <= ml:
            smpx_result = round((((int(x) - int(pxmin))/(int(ml) - int(pxmin)))**(int(p1)) * (int(max))) ,4)
            resultant_text = 'condition : PXMin ≤ x ≤ ML | SMp(x) = [(x - PXmin)/(ML - PXmin)]^ p₁ * Max' 
        elif int(ml) <= int(x) <= int(xmax):
            smpx_result = round((((int(xmax) - int(x)) / (int(xmax) - int(ml)))**(int(p2)) * (int(max))) ,4)
            resultant_text = 'condition :  ML ≤ x ≤ XMax | SMp(x) =  [(Xmax - x) / (Xmax - ML)]^ p₂ * Max'
        elif int(x) > int(xmax):
            smpx_result = 0            
            resultant_text = 'condition :  x > Xmax | SMp(x) = 0'
        
                       
        
        context = {
            'pxmin':pxmin,
            'xmax':xmax,
            'ml':ml,
            'p1':p1,
            'p2':p2,
            'max':max,
            'x':x,
           'smpx_result': smpx_result,
           'resultant_text':resultant_text,


            'id':1
        }
        return render(request, "Statistics/smpx_dist.html", context)
    else:
       
        return render(request, "Statistics/smpx_dist.html")

def expected_value(values, probabilities):
    return sum([v * p for v, p in zip(values, probabilities)])

def expected_value_calc(request):

    
    errors = []
    
    if request.method == 'POST':
        # get url that the user has entered
        try:
        
            data1 = request.POST.get('data1')
            data2 = request.POST.get('data2')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        data1 = [b for i in data1.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        data2 = [b for i in data2.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        try:
            data1 = [float(i) for i in data1]
            data2 = [float(i) for i in data2]
        except ValueError: 
            return "ERROR: There are non-numeric elements!"

        sum = 0.0
        for i in data2:
            sum=sum+i

        print(sum)
        if sum != 1.0:
            errors.append("Sum of all probability is not equal to one")
            context = {
                'errors':errors,
                'id':1,
            }
        elif len(data1)!=len(data2):
            errors.append("Length of Both rondom variable array and Probability of random variable is not equal")
            context = {
                'data1':data1,
                'data2':data2,
                'errors':errors,
                'id':1,
            }
        else:
            k = expected_value(data1, data2)
            results=k

            context = {
                'data1':data1,
                'data2':data2,
                'results':results,
                'id':1
            }
        return render(request,'Statistics/expectedcalculator.html',context)

    else:
        return render(request,'Statistics/expectedcalculator.html')
    


def histogram(request):
    if request.method == "POST":
        #path = "static\images\Statistics\c1.png"
        matplotlib.use('Agg')
        t = (request.POST.get('t'))

        t1 = list(t.split(','))
        for i in range(0, len(t1)):
            t1[i] = int(t1[i])
        plt.hist(t1)
        st="static/images/Statistics/c.png"
        plt.savefig(st)
        plt.close()
        context = {
            
            't1': t1,
            'id': 1,
            't':t
            
        }

        return render(request, "Statistics/histogram.html", context)
    else:
        d = {'t': 0}
        return render(request, "Statistics/histogram.html", d)

 
# function for calculating the t-test for two independent sample
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = me(data1), me(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p


def dependent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = me(data1), me(data2)
	# number of paired samples
	n = len(data1)
	# sum squared difference between observations
	d1 = sum([(data1[i]-data2[i])**2 for i in range(n)])
	# sum difference between observations
	d2 = sum([data1[i]-data2[i] for i in range(n)])
	# standard deviation of the difference between means
	sd = sqrt((d1 - (d2**2 / n)) / (n - 1))
	# standard error of the difference between the means
	sed = sd / sqrt(n)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = n - 1
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p 
 
# seed the random number generator
# seed(1)
# # generate two independent samples (pretend they are dependent)
# data1 = 5 * randn(100) + 50
# data2 = 5 * randn(100) + 51
# # calculate the t test


def t_test(request):
    errors = []
    if request.method=='POST':
        try:
            version = request.POST.get('version')
            data1 = request.POST.get('data1')
            data2 = request.POST.get('data2')
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        data1 = [b for i in data1.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        data2 = [b for i in data2.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        try:
            data1 = [float(i) for i in data1]
            data2 = [float(i) for i in data2]
        except ValueError: 
            return "ERROR: There are non-numeric elements!"
        if version == "Independent Samples":
            alpha = 0.05
            t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
            result1=""
            result2=""
            if abs(t_stat) <= cv:
                result1 = "Accept null hypothesis that the means are equal."
            else:
                result1 = "Reject the null hypothesis that the means are equal."
            
            if  p > alpha:
                result2 = "Accept null hypothesis that the means are equal."
            else:
                result2 = "Reject the null hypothesis that the means are equal."
 
            context = {
                'id':1,
                't_stat' : t_stat,
                'degreeoffreedom' : df,
                'criticalvalue' : cv,
                'p_value':p,
                'result1':result1,
                'result2':result2,
                'id1':2,
            }
            return render(request,'Statistics/t_test.html',context)
        elif version == "Dependent Samples":
            alpha = 0.05
            t_stat, df, cv, p = dependent_ttest(data1, data2, alpha)
            result1=""
            result2=""
            if abs(t_stat) <= cv:
                result1 = "Accept null hypothesis that the means are equal."
            else:
                result1 = "Reject the null hypothesis that the means are equal."
            
            if  p > alpha:
                result2 = "Accept null hypothesis that the means are equal."
            else:
                result2 = "Reject the null hypothesis that the means are equal."
 
            context = {
                'id':1,
                't_stat' : t_stat,
                'degreeoffreedom' : df,
                'criticalvalue' : cv,
                'p_value':p,
                'result1':result1,
                'result2':result2,
                'id1':3,
            }
            return render(request,'Statistics/t_test.html',context)
    return render(request,'Statistics/t_test.html')

def operational_ratio(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a2 = (request.POST.get('a2'))
        a3 = (request.POST.get('a3'))
        a4 = (request.POST.get('a4'))
        a5 = (request.POST.get('a5'))
        a6 = (request.POST.get('a6'))
        t1 = (request.POST.get('t1'))
        a11 = (request.POST.get('a11'))
        a22 = (request.POST.get('a22'))
        a33 = (request.POST.get('a33'))
        a44 = (request.POST.get('a44'))
        a55 = (request.POST.get('a55'))
        a66 = (request.POST.get('a66'))
        t2 = (request.POST.get('t2'))
        m = (request.POST.get('m'))
        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)
        t1 = int(t1)
        a11 = int(a11)
        a22 = int(a22)
        a33 = int(a33)
        a44 = int(a44)
        a55 = int(a55)
        a66 = int(a66)
        t2 = int(t2)
        m = int(m)
        if a11 != 0:
            b = a1/11
            b = round(b, 4)
        else:
            b = 0
        if a22 != 0:
            b1 = a2/22
            b1 = round(b1, 4)
        else:
            b1 = 0
        if a33 != 0:
            b2 = a3/33
            b2 = round(b2, 4)
        else:
            b2 = 0
        if a44 != 0:
            b3 = a4/44
            b3 = round(b3, 4)
        else:
            b3 = 0
        if a55 != 0:
            b4 = a5/a55
            b4 = round(b4, 4)
        else:
            b4 = 0
        if a66 != 0:
            b5 = a6/a66
            b5 = round(b5, 4)
        else:
            b5 = 0
        if t2 != 0:
            b6 = t1/t2
            b6 = round(b6, 4)
        else:
            b6 = 0
        if a1 == 0 or a11 == 0:
            c1 = 'Enter value other than o'
        elif a1 > a11:
            c1 = str(b)+'% increased'
        elif a11 > a1:
            c1 = str(b)+'% decreased'
        else:
            c1 = 'No change'
        if a2 == 0 or a22 == 0:
            c2 = 'Enter value other than o'
        elif a2 > a22:
            c2 = str(b1)+'% increased'
        elif a22 > a2:
            c2 = str(b1)+'% decreased'
        else:
            c2 = 'No change'
        if a3 == 0 or a33 == 0:
            c3 = 'Enter value other than o'
        elif a3 > a33:
            c3 = str(b2)+'% increased'
        elif a33 > a3:
            c3 = str(b2)+'% decreased'
        else:
            c3 = 'No change'
        if a4 == 0 or a44 == 0:
            c4 = 'Enter value other than o'
        elif a4 > a44:
            c4 = str(b3)+'% increased'
        elif a33 > a3:
            c4 = str(b3)+'% decreased'
        else:
            c4 = 'No change'
        if a4 == 0 or a44 == 0:
            c5 = 'Enter value other than o'
        elif a4 > a44:
            c5 = str(b4)+'% increased'
        elif a55 > a5:
            c5 = str(b4)+'% decreased'
        else:
            c5 = 'No change'
        if a6 == 0 or a66 == 0:
            c6 = 'Enter value other than o'
        elif a6 > a66:
            c6 = str(b5)+'% increased'
        elif a66 > a6:
            c6 = str(b5)+'% decreased'
        else:
            c6 = 'No change'
        if t1 == 0 or t2 == 0:
            c7 = 'Enter value other than o'
        elif t1 > t2:
            c7 = str(b6)+'% increased'
        elif t2 > t1:
            c7 = str(b6)+'% decreased'
        else:
            c7 = 'No change'
        if a1 != 0 and a2 != 0:
            it = a1/a2
            it = round(it, 4)
        else:
            it = 0
        if a11 != 0 and a22 != 0:
            it1 = a11/a22
            it1 = round(it1, 4)
        else:
            it1 = 0
        if a3 != 0 and a4 != 0:
            tat = a3/a4
            tat = round(tat, 4)
        else:
            tat = 0
        if a33 != 0 and a44 != 0:
            tat1 = a33/a44
            tat1 = round(tat1, 4)
        else:
            tat1 = 0
        if a5 != 0 and (a6 != 0 and m != 0):
            acp = a5/(a6/m)
            acp = round(acp, 4)
        else:
            acp = 0
        if a55 != 0 and (a66 != 0 and m != 0):
            acp1 = a55/(a66/m)
            acp1 = round(acp1, 4)
        else:
            acp1 = 0
        if a4 != 0 and t1 != 0:
            tm = a4/t1
            tm = round(tm, 4)
        else:
            tm = 0
        if a4 != 0 and t2 != 0:
            tm1 = a4/t2
            tm1 = round(tm1, 4)
        else:
            tm1 = 0
        if it1 != 0:
            d = it/it1
            d = round(d, 4)
        else:
            d = 0
        if tat1 != 0:
            d1 = tat/tat1
            d1 = round(d1, 4)
        else:
            d1 = 0
        if acp1 != 0:
            d2 = acp/acp1
            d2 = round(d2, 4)
        else:
            d2 = 0
        if tm1 != 0:
            d3 = tm/tm1
            d3 = round(d3, 4)
        else:
            d3 = 0
        if it > it1:
            e = str(d)+'% increased '
        elif it1 > it:
            e = str(d)+"% decreased "
        else:
            e = 'no change'
        if tat > tat1:
            e1 = str(d1)+'% increased '
        elif tat1 > tat:
            e1 = str(d1)+"% decreased "
        else:
            e1 = 'no change'
        if acp > acp1:
            e2 = str(d2)+'% increased '
        elif acp1 > acp:
            e2 = str(d2)+"% decreased "
        else:
            e2 = 'no change'
        if tm > tm1:
            e3 = str(d3)+'% increased '
        elif tm1 > tm:
            e3 = str(d3)+"% decreased "
        else:
            e3 = 'no change'
        context = {
            'c1': c1, 'd': d, 'd1': d1, 'd2': d2, 'd3': d3, 'e': e, 'e1': e1, 'e2': e2, 'e3': e3, 'it': it, 'it1': it1, 'tat': tat, 'tat1': tat1, 'acp': acp,
            'c2': c2, 'acp1': acp1, 'tm': tm, 'tm1': tm1, 'm': m, 'a1': a1, 'a11': a11, 'a2': a2, 'a22': a22, 'a3': a3, 'a33': a33, 'a4': a4, 'a44': a44, 'a5': a5, 'a55': a55,
            'c3': c3, 'a6': a6, 'a66': a66, 't1': t1, 't2': t2,
            'c4': c4,
            'c5': c5,
            'c6': c6,
            'c7': c7,
            'id': 1,
        }

        return render(request, "Statistics/operational-ratio.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/operational-ratio.html", d)

def probability_ratio(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a2 = (request.POST.get('a2'))
        a3 = (request.POST.get('a3'))
        a4 = (request.POST.get('a4'))
        a5 = (request.POST.get('a5'))
        a6 = (request.POST.get('a6'))
        t1 = (request.POST.get('t1'))
        a11 = (request.POST.get('a11'))
        a22 = (request.POST.get('a22'))
        a33 = (request.POST.get('a33'))
        a44 = (request.POST.get('a44'))
        a55 = (request.POST.get('a55'))
        a66 = (request.POST.get('a66'))
        t2 = (request.POST.get('t2'))
        a7 = (request.POST.get('a7'))
        a77 = (request.POST.get('a77'))
        a1 = int(a1)
        a2 = int(a2)
        a3 = int(a3)
        a4 = int(a4)
        a5 = int(a5)
        a6 = int(a6)
        t1 = int(t1)
        a11 = int(a11)
        a22 = int(a22)
        a33 = int(a33)
        a44 = int(a44)
        a55 = int(a55)
        a66 = int(a66)
        t2 = int(t2)
        a7 = int(a7)
        a77 = int(a77)
        if a1 != 0:
            z = a11/a1
            b = (z-1)*100
            b = round(b, 4)
            b = abs(b)
        else:
            b = 0
        if a2 != 0:
            z1 = a22/a2
            b1 = (z1-1)*100
            b1 = round(b1, 4)
            b1 = abs(b1)
        else:
            b1 = 0
        if a3 != 0:
            z2 = a33/a3
            b2 = (z2-1)*100
            b2 = round(b2, 4)
            b2 = abs(b2)
        else:
            b2 = 0
        if a4 != 0:
            z3 = a44/a4
            b3 = (z3-1)*100
            b3 = round(b3, 4)
            b3 = abs(b3)
        else:
            b3 = 0
        if a5 != 0:
            z4 = a55/a5
            b4 = (z4-1)*100
            b4 = round(b4, 4)
            b4 = abs(b4)
        else:
            b4 = 0
        if a6 != 0:
            z5 = a66/a6
            b5 = (z5-1)*100
            b5 = round(b5, 4)
            b5 = abs(b5)
        else:
            b5 = 0
        if t1 != 0:
            z6 = t2/t1
            b6 = (z6-1)*100
            b6 = round(b6, 4)
            b6 = abs(b6)
        else:
            b6 = 0
        if a7 != 0:
            z7 = a77/a7
            b7 = (z7-1)*100
            b7 = round(b7, 4)
            b7 = abs(b7)
        else:
            b7 = 0
        if a1 == 0 or a11 == 0:
            c1 = 'Enter value other than o'
        elif a11 > a1:
            c1 = str(b)+'% increased'
        elif a1 > a11:
            c1 = str(b)+'% decreased'
        else:
            c1 = 'No change'
        if a2 == 0 or a22 == 0:
            c2 = 'Enter value other than o'
        elif a22 > a2:
            c2 = str(b1)+'% increased'
        elif a2 > a22:
            c2 = str(b1)+'% decreased'
        else:
            c2 = 'No change'
        if a3 == 0 or a33 == 0:
            c3 = 'Enter value other than o'
        elif a33 > a3:
            c3 = str(b2)+'% increased'
        elif a3 > a33:
            c3 = str(b2)+'% decreased'
        else:
            c3 = 'No change'
        if a4 == 0 or a44 == 0:
            c4 = 'Enter value other than o'
        elif a44 > a4:
            c4 = str(b3)+'% increased'
        elif a4 > a44:
            c4 = str(b3)+'% decreased'
        else:
            c4 = 'No change'
        if a5 == 0 or a55 == 0:
            c5 = 'Enter value other than o'
        elif a55 > a5:
            c5 = str(b4)+'% increased'
        elif a5 > a55:
            c5 = str(b4)+'% decreased'
        else:
            c5 = 'No change'
        if a6 == 0 or a66 == 0:
            c6 = 'Enter value other than o'
        elif a66 > a6:
            c6 = str(b5)+'% increased'
        elif a6 > a66:
            c6 = str(b5)+'% decreased'
        else:
            c6 = 'No change'
        if t1 == 0 or t2 == 0:
            c7 = 'Enter value other than o'
        elif t2 > t1:
            c7 = str(b6)+'% increased'
        elif t1 > t2:
            c7 = str(b6)+'% decreased'
        else:
            c7 = 'No change'
        if a7 == 0 or a77 == 0:
            c8 = 'Enter value other than o'
        elif a77 > a7:
            c8 = str(b7)+'% increased'
        elif a7 > a77:
            c8 = str(b7)+'% decreased'
        else:
            c8 = 'No change'
        if a1 != 0 and a2 != 0:
            Ra = a1/a2
            Ra = round(Ra, 4)
            Ra = abs(Ra)
        else:
            Ra = 0
        if a11 != 0 and a22 != 0:
            Ra1 = a11/a22
            Ra1 = round(Ra1, 4)
            Ra1 = abs(Ra1)
        else:
            Ra1 = 0
        if a1 != 0 and a3 != 0:
            re = a1/a3
            re = round(re, 4)
            re = abs(re)
        else:
            re = 0
        if a11 != 0 and a33 != 0:
            re1 = a11/a33
            re1 = round(re1, 4)
            re1 = abs(re1)
        else:
            re1 = 0
        if a4 != 0 and a5 != 0:
            gpm = a4/a5
            gpm = round(gpm, 4)
            gpm = abs(gpm)
        else:
            gpm = 0
        if a44 != 0 and a55 != 0:
            gpm1 = a44/a55
            gpm2 = round(gpm1, 4)
            gpm2 = abs(gpm2)
        else:
            gpm2 = 0
        if a5 != 0 and a6 != 0:
            opm = a6/a5
            opm = round(opm, 4)
            opm = abs(opm)
        else:
            opm = 0
        if a55 != 0 and a66 != 0:
            opm1 = a66/a55
            opm1 = round(opm1, 4)
            opm1 = abs(opm1)
        else:
            opm1 = 0
        if a5 != 0 and a1 != 0:
            npm = a1/a5
            npm = round(npm, 4)
            npm = abs(npm)
        else:
            npm = 0
        if a55 != 0 and a11 != 0:
            npm1 = a11/a55
            npm1 = round(npm1, 4)
            npm1 = abs(npm1)
        else:
            npm1 = 0
        if a1 != 0 and t1 != 0:
            eps = a1/t1
            eps = round(eps, 4)
            eps = abs(eps)
        else:
            eps = 0
        if a11 != 0 and t2 != 0:
            eps1 = a11/t2
            eps1 = round(eps1, 4)
            eps1 = abs(eps1)
        else:
            eps1 = 0
        if a1 != 0 and t1 != 0:
            p = a7/(a1/t1)
            p = round(p, 4)
            p = abs(p)
        else:
            p = 0
        if a11 != 0 and t2 != 0:
            p1 = a77/(a11/t2)
            p1 = round(p1, 4)
            p1 = abs(p1)
        else:
            p1 = 0
        if Ra != 0:
            y = Ra1/Ra
            d = (y-1)*100
            d = round(d, 4)
            d = abs(d)
        else:
            d = 0
        if re != 0:
            y1 = re1/re
            d1 = (y1-1)*100
            d1 = round(d1, 4)
            d1 = abs(d1)
        else:
            d1 = 0
        if gpm != 0:
            y2 = gpm1/gpm
            d2 = (y2-1)*100
            d2 = round(d2, 4)
            d2 = abs(d2)
        else:
            d2 = 0
        if opm != 0:
            y3 = opm1/opm
            d3 = (y3-1)*100
            d3 = round(d3, 4)
            d3 = abs(d3)
        else:
            d3 = 0
        if npm != 0:
            y4 = npm1/npm
            d4 = (y4-1)*100
            d4 = round(d4, 4)
            d4 = abs(d4)
        else:
            d4 = 0
        if eps != 0:
            y5 = eps1/eps
            d5 = (y5-1)*100
            d5 = round(d5, 4)
            d5 = abs(d5)
        else:
            d5 = 0
        if p != 0:
            y6 = p1/p
            d6 = (y6-1)*100
            d6 = round(d6, 4)
            d6 = abs(d6)
        else:
            d6 = 0
        if Ra1 > Ra:
            e = str(d)+'% increased '
        elif Ra > Ra1:
            e = str(d)+"% decreased "
        else:
            e = 'no change'
        if re1 > re:
            e1 = str(d1)+'% increased '
        elif re > re1:
            e1 = str(d1)+"% decreased "
        else:
            e1 = 'no change'
        if gpm2 > gpm:
            e2 = str(d2)+'% increased '
        elif gpm > gpm2:
            e2 = str(d2)+"% decreased "
        else:
            e2 = 'no change'
        if opm1 > opm:
            e3 = str(d3)+'% increased '
        elif opm > opm1:
            e3 = str(d3)+"% decreased "
        else:
            e3 = 'no change'
        if npm1 > npm:
            e4 = str(d4)+'% increased '
        elif npm > npm1:
            e4 = str(d4)+"% decreased "
        else:
            e4 = 'no change'
        if eps < eps1:
            e5 = str(d5)+'% increased '
        elif eps > eps1:
            e5 = str(d5)+"% decreased "
        else:
            e5 = 'no change'
        if p1 > p:
            e6 = str(d6)+'% increased '
        elif p > p1:
            e6 = str(d6)+"% decreased "
        else:
            e6 = 'no change'
        context = {
            'c1': c1, 'd': d, 'd1': d1, 'd2': d2, 'd3': d3, 'e': e, 'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5, 'e6': e6, 'Ra': Ra, 'Ra1': Ra1, 're': re, 're1': re1, 'npm1': npm1, 'eps': eps, 'eps1': eps1, 'p': p, 'p1': p1,
            'c2': c2, 'gpm': gpm, 'gpm2': gpm2, 'opm': opm, 'opm1': opm1, 'npm': npm, 'npm1': npm, 'a1': a1, 'a11': a11, 'a2': a2, 'a22': a22, 'a3': a3, 'a33': a33, 'a4': a4, 'a44': a44, 'a5': a5, 'a55': a55,
            'c3': c3, 'a6': a6, 'a66': a66, 't1': t1, 't2': t2, 'a7': a7, 'a77': a77, 'c4': c4,
            'c5': c5, 'c8': c8,
            'c6': c6,
            'c7': c7,
            'id': 1,
        }

        return render(request, "Statistics/Probability-Ratio-Calculator.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/Probability-Ratio-Calculator.html", d)

def pdf(request):
    errors = []
    if request.method=='POST':
        try:
            normalrandomvariable = request.POST.get('normalrandomvariable')
            mean = request.POST.get('mean')
            standarddeviation = request.POST.get('standarddeviation')
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )
        try:
            normalrandomvariable = float(normalrandomvariable)
            mean = float(mean)
            standarddeviation = float(standarddeviation) 
        except ValueError: 
            return "ERROR: There are non-numeric elements!"

        result = (1.0 / (standarddeviation * math.sqrt(2*math.pi))) * math.exp(-0.5*((normalrandomvariable - mean) / standarddeviation) ** 2)

        context = {
                'id':1,
                'normalrandomvariable':normalrandomvariable,
                'mean':mean,
                'standarddeviation':standarddeviation,
                'result':result,
                'errors':errors
            }
        

        return render(request,'Statistics/pdf.html',context)
    return render(request,'Statistics/pdf.html')

def exprocalc(request):
    errors = []
    if request.method=='POST':
        try:
            valueoflambda = request.POST.get('valueoflambda')
            valuea = request.POST.get('valuea')
            valueb = request.POST.get('valueb')
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )
        try:
            valueoflambda = float(valueoflambda)
            valuea = float(valuea)
            valueb = float(valueb) 
        except ValueError: 
            return "ERROR: There are non-numeric elements!"

        
        e = math.e
        x = valueoflambda*valuea
        y = valueoflambda*valueb

        prlesa = 1 - math.pow(e,-x)
        
        prgreb = 1 - (1-math.pow(e,-y))

        prbetwennab = (1-math.pow(e,-y)-(1-math.pow(e,-x)))

        mean = 1/valueoflambda
        variance = 1/(valueoflambda*valueoflambda)
        standarddeviation = mean

        context ={
            'id':1,
            'valueoflambda':valueoflambda,
            'valuea':valuea,
            'valueb':valueb,
            'prlesa':prlesa,
            'prgreb':prgreb,
            'prbetwennab':prbetwennab,
            'mean':mean,
            'variance':variance,
            'standarddeviation':standarddeviation
        }

        print(prlesa,prgreb,prbetwennab)
        
        return render(request,'Statistics/exprocalc.html',context)
    return render(request,'Statistics/exprocalc.html')

def prob(request):
    
    if request.method == "POST":
        di={"id":0,"sum":0,"l":[],"val":0,"nu":0}
        nu = request.POST.get('nu')       
        prob=0
        data=[]
        ans={}
        count=0
        for i in range(int(nu)):
            d=check_decimal_values(request.POST.get('r'+str(i)))
            p=check_decimal_values(request.POST.get('s'+str(i)))
            data.append(d*p)
            prob+=p
            ans["d{0}".format(i)] = d
            ans["p{0}".format(i)] = p
            di["d{0}".format(i)] = d
            di["p{0}".format(i)] = p
            count+=1
            
        di["nu"]=nu 
        di["id"]=1
        di["l"]=ans
        di["val"]=str(count)
        print(di["l"])
        if prob==1:
            for i in range(len(data)):
                di["m{0}".format(i)] = data[i]
            
            di["sum"]=sum(data)
            
        else:
            di["sum"]="Sum of Probability Distribution must be 1 "
        
        return render(request,"Statistics/mean_probability.html",dict(di))
    else:
        di={ 'id':0, 'nu':0}
        return render(request,"Statistics/mean_probability.html",dict(di))

def freq(request):
    if request.method == "POST":
        nu=request.POST.get('txt')
        
        element={}
        
        l=[]
        di={"id":1,"a":[],"s":[],"cf":[],"rf":[],"l":[],"txt":""}
        sum=0
        txt=""
        for i in nu.split(","):
            l.append(float(i))
            txt+=i+","
        di["txt"]=txt[0:len(txt)-1]
        
        se=set(l)
        for i in sorted(se):
            sum+=l.count(i)
            element[str(i)]=l.count(i)
            di["s"].append(sum)
            di["rf"].append(round(l.count(i)*(1/len(l)),3))
            di["cf"].append(round(sum*(1/len(l)),3))

    
        di["a"]=element
        di["l"]=l
        print(di["s"])
        
        
        
        
        
        return render(request,"Statistics/frequencydistribution.html",dict(di))
    else:
        
        di={"id":1,"a":{'3.0': 2, '5.0': 1, '7.0': 1},"s":[2, 3, 4],"cf":[0.5, 0.75, 1.0],"rf":[0.5, 0.25, 0.25],"l":[3.0, 5.0, 7.0, 3.0],"txt":"3,5,7,3"}
        return render(request,"Statistics/frequencydistribution.html",di)
    
def cpc(request):
    errors = []
    result = 0.0
    if request.method=='POST':
        try:
            probabilityofaandb = request.POST.get('probabilityofaandb')
            probabilityofa = request.POST.get('probabilityofa')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )
        try:
            probabilityofaandb = float(probabilityofaandb)
            probabilityofa = float(probabilityofa)
        except ValueError: 
            return "ERROR: There are non-numeric elements!"

        if(probabilityofaandb < 0.00001 or probabilityofaandb > 0.99999):
            errors.append("P(A and B) must be a number between 0.00001 and 0.99999.")
            
        if(probabilityofa < 0.00001 or probabilityofa > 0.99999):
            errors.append("P(B) must be a number between 0.00001 and 0.99999.")
            
        
        result = probabilityofaandb/probabilityofa
        context = {
            'id':1,
            'probabilityofaandb':probabilityofaandb,
            'probabilityofa':probabilityofa,
            'result':result,
            'errors':errors
        }
        return render(request,'Statistics/cpc.html',context)
    return render(request,'Statistics/cpc.html')

def empirical_probability(request):
    if request.method == "POST":
        f = request.POST.get('number_of_times_event_occurs')
        n = request.POST.get('number_of_times_experiment_performed')
        empirical_probab = round(int(f) / int(n),4)

        context = {
            'number_of_times_event_occurs':f,
            'number_of_times_experiment_performed':n,
            'empirical_probab': empirical_probab,
            'id':1
        }
        return render(request, "Statistics/empirical_prob.html", context)
    else:
        d = {'number_of_times_event_occurs':4,'number_of_times_experiment_performed':2}
        return render(request, "Statistics/empirical_prob.html",d)


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom 

cards = {'black': 26,
            'red':26,
            'heart':13,
            'diamond':13,
            'spade':13,
            'club':13,
            'jack':4,
            'queen':4,
            'king':4,
            'ace':4,
            'facecard':16 }

def prob(no,card_name):
    total_cards = cards[card_name]
    #prob1 = ncr(52,no)
    #prob2 = ncr(total_cards,no)
    probability = ncr(total_cards,no) / ncr(52,no)
    return round(probability,3)
def cal_prob(request):
    if request.method == "POST":
        cards_drawn = int(request.POST.get('cards_drawn'))
        #card_name = request.POST.get('card to be drawn')
        card_name = request.POST['dropdown'] 
        print('*********')
        print(card_name)
        print(cards_drawn)

        probability = prob(cards_drawn,card_name)
        nE = ncr(cards[card_name],cards_drawn)
        nS = ncr(52,cards_drawn)
        total_cards = cards[card_name]
        context = {
            'cards_drawn':cards_drawn,
            'card_name':card_name,
            'probability':probability,
            'nE':nE,
            'nS':nS,
            'total_cards':total_cards,
            'id':1
        }
        return render(request, "Statistics/card_deck.html", context)
    else:
        return render(request, "Statistics/card_deck.html")

def cal_distribution_prob(p,k):
    return round(p*((1-p)**(k-1)),4)
    

def geomatric_probability(request):
    if request.method == "POST":
        success_prob = request.POST.get('success_prob')
        k_trails = request.POST.get('k_trails')
        probability = cal_distribution_prob(float(success_prob),int(k_trails))
                
        
        context = {
            'success_prob' : success_prob,
            'k_trails': k_trails,
            #'classes':classes,
            'probability' : probability,
            'id':1
        }
        return render(request, "Statistics/labor_prob.html", context)
    else:
        d = {'success_prob':0.25,'k_trails':4}
        return render(request, "Statistics/labor_prob.html",d)

def probability_density(x,a,b):
    if a <= x <= b:
        return round(1 / (b-a),4)
    if x < a and b < x:
        return 0

def lower_cumulative_distribution(x,a,b):
    return round((x-a)/(b-a),4)

def upper_cumulative_distribution(x,a,b):
    return round((b-x)/(b-a),4)


def uniform_distribution(request):
    if request.method == "POST":
        percentile_x = request.POST.get('percentile_x')
        uniform_interval_a = request.POST.get('uniform_interval_a')
        uniform_interval_b = request.POST.get('uniform_interval_b')

        prob_density = probability_density(int(percentile_x),int(uniform_interval_a),int(uniform_interval_b))
        lower_cumulative = lower_cumulative_distribution(int(percentile_x),int(uniform_interval_a),int(uniform_interval_b))
        upper_cumulative = upper_cumulative_distribution(int(percentile_x),int(uniform_interval_a),int(uniform_interval_b))

        context = {
            'percentile_x':percentile_x,
            'uniform_interval_a':uniform_interval_a,
            'uniform_interval_b':uniform_interval_b,
            'prob_density':prob_density,
            'lower_cumulative':lower_cumulative,
            'upper_cumulative':upper_cumulative,
            'id':1


        }
        return render(request, "Statistics/uniform_distri.html", context)
    else:
        d = {'percentile_x':2,'uniform_interval_a':1,'uniform_interval_b':4}
        return render(request, "Statistics/uniform_distri.html",d)
def fact(n):
     
    res = 1
    for i in range(2, n + 1):
        res = res * i
    return res
 
# Applying the formula
def count_heads(n, r):
     
    output = fact(n) / (fact(r) * fact(n - r))
    output = output / (pow(2, n))
    return output



def coin_flip(request):
    if request.method == "POST":
        flips = request.POST.get('flips')
        heads = request.POST.get('heads')
        probability = count_heads(int(flips),int(heads))
                
        
        context = {
            'heads' : heads,
            'flips': flips,
            #'classes':classes,
            'probability' : probability,
            'id':1
        }
        return render(request, "Statistics/coin_flip_probability.html", context)
    else:
        d = {'heads':4,'flips':4}
        return render(request, "Statistics/coin_flip_probability.html",d)


def rpc(request):
    errors = []
    if request.method=='POST':
        try:
            type = request.POST.get('type')
            amount = float(request.POST.get('amount'))
            betting = request.POST.get('betting')
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )
        gamerule1 = ["Red" , "Black" , "Odds" , "Evens"]
        gamerule2 = ["1-12" , "13-24" ,"25-36"]
        if type == "European":
            if betting in gamerule1:
                outcomeprobability = 18/37
                successpercentage = outcomeprobability*100
                yourwin = 2*amount
            elif betting in gamerule2:
                outcomeprobability = 12/37
                successpercentage = outcomeprobability*100
                yourwin = 3*amount
            elif betting == "Straight":
                outcomeprobability = 1/37
                successpercentage = outcomeprobability*100
                yourwin = 36*amount
            elif betting == "Split":
                outcomeprobability = 2/37
                successpercentage = outcomeprobability*100
                yourwin = 18*amount
            elif betting == "Street":
                outcomeprobability = 3/37
                successpercentage = outcomeprobability*100
                yourwin = 12*amount
            elif betting == "Square":
                outcomeprobability = 4/37
                successpercentage = outcomeprobability*100
                yourwin = 9*amount
            elif betting == "Sixline":
                outcomeprobability = 6/37
                successpercentage = outcomeprobability*100
                yourwin = 6*amount
                

            
            context = {
                'id':1,
                'errors':errors,
                'amount':amount,
                'betting':betting,
                'outcomeprobability':outcomeprobability,
                'successpercentage':successpercentage,
                'yourwin':yourwin,
                
            }
            return render(request,'Statistics/rpc.html',context)
        else:
            if betting in gamerule1:
                outcomeprobability = 18/38
                successpercentage = outcomeprobability*100
                yourwin = 2*amount
            elif betting in gamerule2:
                outcomeprobability = 12/38
                successpercentage = outcomeprobability*100
                yourwin = 3*amount
            elif betting == "Straight":
                outcomeprobability = 1/38
                successpercentage = outcomeprobability*100
                yourwin = 36*amount
            elif betting == "Split":
                outcomeprobability = 2/38
                successpercentage = outcomeprobability*100
                yourwin = 18*amount
            elif betting == "Street":
                outcomeprobability = 3/38
                successpercentage = outcomeprobability*100
                yourwin = 12*amount
            elif betting == "Square":
                outcomeprobability = 4/38
                successpercentage = outcomeprobability*100
                yourwin = 9*amount
            elif betting == "Sixline":
                outcomeprobability = 6/38
                successpercentage = outcomeprobability*100
                yourwin = 6*amount
                

            
            context = {
                'id':1,
                'errors':errors,
                'amount':amount,
                'betting':betting,
                'outcomeprobability':outcomeprobability,
                'successpercentage':successpercentage,
                'yourwin':yourwin,
                
            }
            return render(request,'Statistics/rpc.html',context)



    return render(request,'Statistics/rpc.html')

def group_frequency_distribution(request):
    errors = []
    
    if request.method == 'POST':
        # get url that the user has entered
        try:
        
            data1 = request.POST.get('data1')
            data2 = request.POST.get('data2')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        data1 = [b for i in data1.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        
        try:
            data1 = [float(i) for i in data1]
            data2 = float(data2)
        except ValueError: 
            return "ERROR: There are non-numeric elements!"

        min_element=data1[0]
        for num in data1:
            min_element=min(min_element,num)
        max_element=data1[0]
        for num in data1:
            max_element=max(max_element,num)
        datarange=(max_element-min_element)//data2
        print(datarange)
        classwidth = []
        prev = min_element
        for i in range(0,int(data2)):
            onecol = []
            min_range = prev
            max_range = prev+datarange
            onecol.append(round(min_range,1))
            onecol.append(round(max_range,1))
            classwidth.append(onecol)
            prev=max_range+1.0
        class_boundries = []

        for x in classwidth:
            onecol = []
            onecol.append(round(x[0]-0.5,1))
            onecol.append(round(x[0]+0.5,1))
            class_boundries.append(onecol)
        
        frequency = {}
        for i in range(0,len(classwidth)):
            frequency[i]=0
        for num in data1:
            for x in range(0,len(classwidth)):
                if classwidth[x][0] <= num and num <= classwidth[x][1]:
                    frequency[x]+=1
                    break
        
        v = []
        for key,value in frequency.items():
            v.append(value)
        
        data = zip(classwidth,class_boundries,v)
        
        context = {
            'id':1,
            'data1':data1,
            'data2':data2,
            'errors':errors,
            'classwidth':classwidth,
            'data':data,
        }
                    
                

        # print(datarange)
        # print(min_element)
        # print(max_element)
        return render(request,'Statistics/group-frequency-distribution.html',context)

    return render(request,'Statistics/group-frequency-distribution.html')

def cumlative_frequency(request):
    errors = []
    
    if request.method == 'POST':
        # get url that the user has entered
        try:
        
            data1 = request.POST.get('data1')
            data2 = request.POST.get('data2')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        # data1 = [b for i in data1.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        data2 = [b for i in data2.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]

        
        
        try:
            data2 = [int(i) for i in data2]
        except ValueError: 
            return "ERROR: There are non-numeric elements!"


        

        v = []
        sum=0
        for num in data2:
            sum=sum+num
            v.append(sum)
        

        
        data = zip(data1,data2,v)

        data2 = str(data2)[1:-1] 
        
        context = {
            'id':1,
            'data1':data1,
            'data2':data2,
            'errors':errors,
            'data':data,
        }
                    
                

        # # print(datarange)
        # # print(min_element)
        # # print(max_element)
        return render(request,'Statistics/cumlative-frequency.html',context)
    
    data1 = "2-10 11-19 20-28"
    data2 = "1 3 9"

    context = {
            
            'data1':data1,
            'data2':data2,
        }
    return render(request,'Statistics/cumlative-frequency.html',context)

def vft(request):
    errors = []
    
    if request.method == 'POST':
        # get url that the user has entered
        try:
            upper = request.POST.get('upper')
            lower = request.POST.get('lower')
            data2 = request.POST.get('data2')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        upper = [b for i in upper.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        lower = [b for i in lower.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        data2 = [b for i in data2.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]

        
        
        try:
            upper = [float(i) for i in upper]
            lower = [float(i) for i in lower]
            data2 = [float(i) for i in data2]
        except ValueError: 
            return "ERROR: There are non-numeric elements!"


        

        if(len(upper)!=len(lower) and len(lower)!=len(data2) and len(upper)!=len(data2)):
            errors.append("Please add all the data of equal length")
        
        for i in range(0,len(upper)):
            if upper[i] <= lower[i]:
                errors.append("Upperlimit is greater than lowerlimit")
                break

        midpoint = []
        for i in range(0,len(upper)):
            midpoint.append((upper[i]+lower[i])/2)

        
        fm = []
        sf=0
        sfm=0
        for i in range(0,len(data2)):
            fm.append(data2[i]*midpoint[i])
            sf+=data2[i]
            sfm+=fm[i]
        
        fmm = []
        sfmm = 0
        for i in range(0,len(data2)):
            fmm.append(data2[i]*midpoint[i]*midpoint[i])
            sfmm+=fmm[i]


        u = sfm/sf

        variance = (sfmm - sf*u*u)/(sf-1.0)




        # print(u,variance)

        

        data = zip(lower,upper,data2,midpoint,fmm)
        upper = str(upper)[1:-1]
        lower = str(lower)[1:-1]
        data2 = str(data2)[1:-1] 

        result=0
        if len(errors)==0:
            result = 1
        
        context = {
            'id':1,
            'result':result,
            'errors':errors,
            'lower':lower,
            'upper':upper,
            'data2':data2,
            'data':data,
            'sf':sf,
            'sfmm':sfmm,
            'u':u,
            'variance':variance
            
        }
                    
                

        # # print(datarange)
        # # print(min_element)
        # # print(max_element)
        return render(request,'Statistics/vft.html',context)
    
    upper = "369 379 389 399 409 419 429 439"
    lower = "360 370 380 390 400 410 420 430"
    data2 = "2 3 5 7 5 4 4 1"

    context = {
            
            'lower':lower,
            'upper':upper,
            'data2':data2,
        }
    return render(request,'Statistics/vft.html',context)

def probability_dependent(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a22 = (request.POST.get('a22'))
        c1 = (request.POST.get('c1'))
        a1 = int(a1)
        a22 = int(a22)
        c1 = int(c1)
        f = 0
        if c1 == 0:
            c = a1+a22
            d = (a1/c)*((a1-1)/(c-1))
            e = (a22/c)*((a22-1)/(c-1))
        else:
            c = a1+a22+c1
            d = (a1/c)*((a1-1)/(c-1))*((a1-2)/(c-2))
            e = (a22/c)*((a22-1)/(c-1))*((a22-2)/(c-2))
            f = (c1/c)*((c1-1)/(c-1))*((c1-2)/(c-2))
        context = {
            'a1': a1, 'a22': a22, 'd': d, 'e': e, 'c': c, 'c1': c1, 'f': f,
            'id': 1,
        }

        return render(request, "Statistics/probability_dependent.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/probability_dependent.html", d)

def percentage_frequency(request):
    errors = []
    
    if request.method == 'POST':
        # get url that the user has entered
        try:
        
            data1 = request.POST.get('data1')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        # data1 = [b for i in data1.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        data1 = [b for i in data1.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]

        
        
        try:
            data1 = [float(i) for i in data1]
        except ValueError: 
            return "ERROR: There are non-numeric elements!"


        
        length_of_dataset = len(set(data1))

        N = 1 + 3.322 * math.log10(length_of_dataset)

        N = round(N)

        min_element=data1[0]
        for num in data1:
            min_element=min(min_element,num)
        max_element=data1[0]
        for num in data1:
            max_element=max(max_element,num)
        datarange=(max_element-min_element) // N

        classwidth = []
        prev = min_element
        for i in range(0,N):
            onecol = []
            min_range = prev
            max_range = prev+datarange
            onecol.append(round(min_range,1))
            onecol.append(round(max_range,1))
            classwidth.append(onecol)
            prev=max_range+1.0
        class_boundries = []

        for x in classwidth:
            onecol = []
            onecol.append(round(x[0]-0.5,1))
            onecol.append(round(x[0]+0.5,1))
            class_boundries.append(onecol)
        
        frequency = {}
        for i in range(0,len(classwidth)):
            frequency[i]=0
        for num in data1:
            for x in range(0,len(classwidth)):
                if classwidth[x][0] <= num and num <= classwidth[x][1]:
                    frequency[x]+=1
                    break
        
        v = []
        sumoff=0
        for key,value in frequency.items():
            sumoff+=value
            v.append(value)
        
        r = []

        fi = []

        for num in v:
            fi.append(num/sumoff)
        
        percent = []

        for num in fi:
            percent.append(num*100)

        
        data = zip(classwidth,class_boundries,v,fi,percent)


        

        

        
        



        
        # v = []
        # sum=0
        # for num in data2:
        #     sum=sum+num
        #     v.append(sum)
        

        
        # data = zip(data1,data2,v)

        data1 = str(data1)[1:-1] 
        
        context = {
            'id':1,
            'data1':data1,
            'classwidth':classwidth,
            'data':data,
            'errors':errors,
        }
                    
                

        # # print(datarange)
        # # print(min_element)
        # # print(max_element)
        return render(request,'Statistics/pfc.html',context)
    
    data1 = "9 8 7 1 0 3 8 9"
    

    context = {
            
            'data1':data1,
        }
    return render(request,'Statistics/pfc.html',context)
def Mutually_exclusive(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a22 = (request.POST.get('a22'))
        c1 = (request.POST.get('c1'))
        a1 = int(a1)
        a22 = int(a22)
        c1 = int(c1)
        f = 0
        e = 0
        a = 0
        if c1 == 0:
            c = a1+a22
            d = (a1/c)+(a22/c)
        else:
            c = a1+a22+c1
            a = (a1/c)+((a22)/(c))+((c1)/(c))
            e = (a22/c)+(c1/c)
            f = (a1/c)+((c1)/(c))
            d = (a22/c)+(a1/c)
        context = {
            'a1': a1, 'a22': a22, 'd': d, 'e': e, 'c': c, 'c1': c1, 'f': f,
            'id': 1, 'a': a,
        }

        return render(request, "Statistics/Mutually_exclusive.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/Mutually_exclusive.html", d)
def NonMutually_exclusive(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a22 = (request.POST.get('a22'))
        c1 = (request.POST.get('c1'))
        c = (request.POST.get('c'))
        a1 = int(a1)
        a22 = int(a22)
        c1 = int(c1)
        c = int(c)
        f = 0
        d = 0
        if c == 0:
            a1, a22, c1 = 0, 0, 0
        elif c1 == 0:
            d = (a1/c)+(a22/c)
        else:
            c = a1+a22+c1
            d = (a22/c)+(a1/c)+(c1/c)
        context = {
            'a1': a1, 'a22': a22, 'd': d,  'c': c, 'c1': c1, 'f': f,
            'id': 1,
        }

        return render(request, "Statistics/NonMutually_exclusive.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/NonMutually_exclusive.html", d)
def probability_independent(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a22 = (request.POST.get('a22'))
        c1 = (request.POST.get('c1'))
        a1 = int(a1)
        a22 = int(a22)
        c1 = int(c1)
        f = 0
        if c1 == 0:
            c = a1+a22
            d = (a1/c)*((a1)/(c))
            e = (a22/c)*((a22)/(c))
        else:
            c = a1+a22+c1
            d = (a1/c)*((a1)/(c))*((a1)/(c))
            e = (a22/c)*((a22)/(c))*((a22)/(c))
            f = (c1/c)*((c1)/(c))*((c1)/(c))
        context = {
            'a1': a1, 'a22': a22, 'd': d, 'e': e, 'c': c, 'c1': c1, 'f': f,
            'id': 1,
        }

        return render(request, "Statistics/probability_independent.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/probability_independent.html", d)

def poker_probability(request):
    errors = []
    if request.method=='POST':
        try:
            type = request.POST.get('type')
            betting = request.POST.get('betting')
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )
        
        if type == "5-Hand":

            if(betting == "royal_flush"):
                
                frequency = 4
                probability = "0.000154%"
                odds_against = "649739 : 1"
            elif(betting == "straight_flush"):
                
                frequency = 36
                probability = "0.0015%"
                odds_against = "72192.33 : 1"
            elif(betting == "four_of_a_kind"):
                
                frequency = 624
                probability = "0.02401%"
                odds_against = "4164 : 1"
            elif(betting == "full_house"):
                
                frequency = 3744
                probability = "0.1441%"
                odds_against = "693.1667 : 1"
            elif(betting == "flush"):
                
                frequency = 5108
                probability = "0.1965%"
                odds_against = "507.8019 : 1"
            elif(betting == "straight"):
                
                frequency = 10200
                probability = "0.3925%"
                odds_against = "253.8 : 1"
            elif(betting == "three_of_a_kind"):
                
                frequency = 54912
                probability = "2.1128%"
                odds_against = "46.32955 : 1"
            elif(betting == "two_pair"):
              
                frequency = 123552
                probability = "4.7539%"
                odds_against = "20.03535 : 1"
            elif(betting == "one_pair"):
                
                frequency = 1098240
                probability = "42.2569%"
                odds_against = "1.366477 : 1"
            elif(betting == "no_pair"):
                
                frequency = 1302540	
                probability = "50.1177%"
                odds_against = "0.9953015 : 1"
            
            
            
            
            
            
            


            
            context = {
                'id':1,
                'errors':errors,
                'betting':betting,
                'type':type,
                'frequency':frequency,
                'probability':probability,
                'odds_against':odds_against
                
            }
            return render(request,'Statistics/ppc.html',context)
        else:
            if(betting == "royal_flush"):
                frequency = 4324		
                probability = "0.0032%"
                odds_against = "30939 : 1"
            elif(betting == "straight_flush"):
                frequency = 37260
                probability = "0.0279%"
                odds_against = "3589.6 : 1"
            elif(betting == "four_of_a_kind"):
                distinct_hand = 156
                frequency =224848		
                probability = "0.168%"
                odds_against = "594 : 1"
            elif(betting == "full_house"):
                frequency = 3473184	
                probability = "2.60%"
                odds_against = "37.5 : 1"
            elif(betting == "flush"):
                frequency = 4047644	
                probability = "3.03%"
                odds_against = "32.1 : 1"
            elif(betting == "straight"):
               
                frequency = 6180020		
                probability = "4.62%"
                odds_against = "20.6 : 1"
            elif(betting == "three_of_a_kind"):
                
                frequency = 6461620	
                probability = "4.83%"
                odds_against = "19.7 : 1"
            elif(betting == "two_pair"):
                frequency = 31,433,400	
                probability = "23.5%"
                odds_against = "3.26 : 1"
            elif(betting == "one_pair"):
                
                frequency = 58627800
                probability = "43.8%"
                odds_against = "1.28 : 1"
            elif(betting == "no_pair"):
                
                frequency = 23294460	
                probability = "17.4%"
                odds_against = "4.74 : 1"        
            context = {
                'id':1,
                'errors':errors,
                
                
            }
            context = {
                'id':1,
                'errors':errors,
                'betting':betting,
                'type':type,
                'frequency':frequency,
                'probability':probability,
                'odds_against':odds_against
                
            }
            return render(request,'Statistics/ppc.html',context)
    return render(request,'Statistics/ppc.html')

def in_dependent(request):
    if request.method == "POST":
        a1 = (request.POST.get('a1'))
        a22 = (request.POST.get('a22'))
        c1 = (request.POST.get('c1'))
        c = (request.POST.get('c'))
        a1 = float(a1)
        a22 = float(a22)
        c1 = float(c1)
        c = float(c)
        f = 0
        d = 0
        e = ''
        if c == 0:
            e = 'Enter total number other than 0'
        else:
            d = (a22/c)+(a1/c)
            if d == c1:
                e = ' Independent events'
            else:
                e = ' Dependent events'
        context = {
            'a1': a1, 'a22': a22, 'd': d,  'c': c, 'c1': c1, 'f': f,
            'id': 1, 'e': e,
        }

        return render(request, "Statistics/in_dependent.html", context)
    else:
        d = {'t1': 0}
        return render(request, "Statistics/in_dependent.html", d)

def modal_frequency_table(request):
    errors = []
    
    if request.method == 'POST':
        # get url that the user has entered
        try:
            upper = request.POST.get('upper')
            lower = request.POST.get('lower')
            data2 = request.POST.get('data2')
            
        except:
            errors.append(
                "Unable to get necessary input, please try again."
            )

        upper = [b for i in upper.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        lower = [b for i in lower.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]
        data2 = [b for i in data2.split(',') for a in i.split(' ') for b in a.split('\t') if len(b)>0]

        
        
        try:
            upper = [float(i) for i in upper]
            lower = [float(i) for i in lower]
            data2 = [int(i) for i in data2]
        except ValueError: 
            return "ERROR: There are non-numeric elements!"


        

        if(len(upper)!=len(lower) and len(lower)!=len(data2) and len(upper)!=len(data2)):
            errors.append("Please add all the data of equal length")
        
        for i in range(0,len(upper)):
            if upper[i] <= lower[i]:
                errors.append("Upperlimit is greater than lowerlimit")
                break
        

        max_index=-1
        max_element=-1
        for i in range(0,len(data2)):
            if data2[i]>max_element:
                max_element=max(max_element,data2[i])
                max_index=i

        ans1 = lower[max_index]
        ans2 = upper[max_index]


        
        data = zip(lower,upper,data2)

        upper = str(upper)[1:-1]
        lower = str(lower)[1:-1]
        data2 = str(data2)[1:-1] 

        result=0
        if len(errors)==0:
            result = 1
        
        context = {
            'id':1,
            'result':result,
            'errors':errors,
            'lower':lower,
            'upper':upper,
            'data2':data2,
            'ans1':ans1,
            'ans2':ans2,
            'max_element':max_element,
            'data':data
            
        }
                    
                

        # # print(datarange)
        # # print(min_element)
        # # print(max_element)
        return render(request,'Statistics/mft.html',context)
    
    upper = "369 379 389 399 409 419 429 439"
    lower = "360 370 380 390 400 410 420 430"
    data2 = "2 3 5 7 5 4 4 1"

    context = {
            
            'lower':lower,
            'upper':upper,
            'data2':data2,
        }
    return render(request,'Statistics/mft.html',context)
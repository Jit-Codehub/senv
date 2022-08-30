import numpy as np
def insideFunc(a,title,content):
    a('<details><summary>%s</summary>%s</details>'%(title,content))
#Function to know the median of the numbers
def medians1(res1):
    r=res1.split(',')
    r1=[]
    #'r1' is the list with sorted numbers 
    for i in r:
        r1.append(int(i))
        r1.sort()
        rs=len(r1)
    #'rs' is the number of values in 'r1'(even or odd)
    #condition if number of values in 'r1' is even i.e., 4 or 6 or 8
    if rs%2==0:
        p=int(rs/2)
        q=int((rs+2)/2)
        m=r1[int(p-1)]
        n=r1[int(q-1)]
        o=str((m+n)/2)
        if '.0' in o:
            o=int(float(o))
        else:
            pass
        d1=str(m)+','+str(n)
    else:
        p=""
        q=""
        m=""
        n=int((rs+1)/2)
        o=str(int(r1[int(n-1)]))
        if '.0' in o:
            o=int(float(o))
        else:
            pass
    a=int(rs%2)
    v=[]
    a1=v.append
    a1('<p>Median:</p> ')
    a1('<p>The Median is the middle number in a sorted, ascending or descending, list of numbers and can be more descriptive of that data set than the average.</p>')
    a1('<p>Formula for Median of Numbers:</p> <p> If we have odd set of numbers,the formula will be:</p><p style="text-align: center;">Median=[ (n+1) / 2 ]<sup>th</sup> item</p> <p style="text-align: center;">where n=Set of Numbers</p><p> If we have even set of numbers, the formula will be:</p><p style="text-align: center;">Median=Mean of two middle items</p> <p style="text-align: center;">i.e., Median=Average of [ n / 2 ]<sup>th</sup> item and [ n+2 / 2 ]<sup>th</sup> item</p> <br><hr>')
    a1('<p>The given numbers are {}</p>'.format(res1))
    a1('<p>Firstly, the numbers should be set in an ascending order.</p>')
    a1('<p style="text-align: center;">{}'.format(r1[0]))
    for i in r1[1:]:
        a1(',{}'.format(i))
    a1('</p>')
    if a==0:
        a1('<p>As these are having Even set of numbers,the Mean of the both middle numbers is the Median</p><p>The Set of Numbers, i.e., n = {}</p><p>The two middle numbers items will be n / {} and (n+2) / {}</p><p style="text-align: center;">n / {} = {} / {}</p> '.format(rs,2,2,2,rs,2))
        #title is the text to show on the dropdown box of the data
        title='The Mean of the numbers {} and {} is '.format(m,n)
        #content is the means function with the detailed steps
        content=means1(d1)[0]
        a1('<p style="text-align: center;"> n / 2 = %d</p> '%(p))
        a1('<p style="text-align: center;">(n+2) / 2 = (%d+2) / 2</p> <p style="text-align: center;">n+2 / 2 = %d</p> '%(rs,q))
        a1('<p>The two middle number items %d and %d contains the numbers %d and %d</p><p>Now these numbers should be calculated to know the mean of those numbers</p>'%(p,q,m,n))
        #insideFunc is the function used to show the minimized data with the dropdown menu along with the title
        insideFunc(a1,title,content)
        a1('<p style="text-align: center;font-size: x-large;">[ n / 2 + (n+2) / 2] / 2 = (%d+%d) / 2</p><p style="text-align: center;">Average (or) Mean of two middle numbers is [%s]</p> <p>Median of the numbers is %s</p>'%(m,n,o,o))
    else:
        a1('<p>As these are having Odd set of numbers,the number in the middle of set of numbers is the Median</p><p style="text-align: center;">Median = (n+1) / 2</p>')
        a1('<p>The Set of Numbers, i.e., n = %d</p><p style="text-align: center;">(n+1) / 2 = (%d+1) / 2</p> '%(rs,rs))
        a1('<p style="text-align: center;">(n+1) / 2 = %d</p> <p>The number item %d contains the number %s</p><p>So,The Median of the numbers is %s.</p>'%(n,n,o,o))
    return ''.join(v),rs,p,q,m,n,o
#Function to know the mean of the numbers
def means1(res1):
    r=res1.split(',')
    r1=[]
    #'r1' is the list with the numbers given
    s=0
    for i in r:
        s+=int(i)
        r1.append(int(i))
        rs=len(r1) 
    re=round(int(s)/int(rs),2)
    if '.0' in str(re):
        re=int(re)
    else:
        pass
    v=[]
    a=v.append
    a('<p>Mean:</p> <p>The Mean (or Average) of a set of Data Values is the Sum of all of the Data Values divided by the Number of Data Values.</p><p>Formula for Mean of Numbers:</p> <p style="text-align: center;">Mean (or) Average of Numbers = Sum of the given data / Number of given data</p> <p style="text-align: center;">Mean = (x<sub>0</sub> +x<sub>1</sub> +x<sub>2</sub> +x<sub>3</sub> .......+x<sub>n</sub>) / n</p> <p style="text-align: center;"> X&#772; = <span>&Sigma;</span> x<sub>n</sub> / n</p> <br><hr>')
    a('<p>The Given Numbers are <b>{}</b></p><p style="text-align: center;">Number of Data given, i.e., n = {}</p><p>Now, the Sum of the Data Values will be </p>'.format(res1,rs))
    a('<p style="text-align: center;">x<sub>0</sub> + x<sub>1</sub> + x<sub>2</sub> + x<sub>3</sub> + .... + x<sub>i</sub> = {}'.format(r1[0]))
    for i in r1[1:]:
        a('+{}'.format(i))
    a('</p><br>')
    a('<p style="text-align: center;"><span>&Sigma;</span> x<sub>n</sub> = %d</p><p> Therefore, the Mean of Data is</p><p style="text-align: center;"> X&#772; = <span>&Sigma;</span> x<sub>n</sub> / n</p> <p style="text-align: center;"> X&#772; = %d / %d</p> '%(s,s,rs))
    a('<p style="text-align: center;"> X&#772; = {}</p> <p>The Mean of Data is {}</p> '.format(re,re))
    return ''.join(v),re
#Function to know the mode of the numbers
def modes1(res1):
    r=res1.split(',')
    #'a' is the list with the numbers given
    a=[]
    #'d' is the list with the count of numbers repeated
    d=[]
    for i in r:
        a.append(int(i))
        c=a.count(int(i))
        d.append(int(c))
    #'h' is the maximum number of counts
    h=max(d)
    #'e' is the index of maximum number of counts
    e=d.index(max(d))
    #'f' is the maximum counted number
    f=a[e]
    g=zip(a,d)
    v=[]
    a1=v.append
    a1('<p>Mode:</p> <p>The Mode is the number that is repeated more often than any other numbers.</p><br><p>The given numbers are {}</p><p>Firstly, the numbers should be counted accordingly.</p>'.format(res1))
    for i,j in g:
        a1('<p style="text-align: center;">The Number {} is counted as {},</p><br>'.format(i,j))
    a1('<p>After counting the numbers, the maximum number counted (or) repeated most is {} as {} times</b>.</p><p>The Mode of the given values is {}.</p>'.format(f,h,f))
    return ''.join(v),f
#Function to know the lower quartile of the numbers
def lower_quartiles1(res1):
    r=res1.split(',')
    #'r1' is the list with the given numbers sorted 
    r1=[]
    for i in r:
        r1.append(int(i))
        r1.sort()
    a=int(len(r1))
    e=int(a%2)
    b=int(a/2)
    #'g' is the middle value of the given numbers
    g=r1[b]
    #'d' is the list with the first half of the numbers
    d=[]
    for j in range(0,b):
        c=r1[j]
        d.append(c)
    d1=str(d)
    x1=d1.replace('[','')
    x1=x1.replace(']','')
    s=medians1(x1)
    v2=s[0]
    o=s[6]
    v=[]
    a1=v.append
    a1('<p>Lower (or) First Quartile:</p> <p>The Lower Quartile (Q1) is the Median of the lower half of the data set which is the first half of the given data.</p><br><p>Given numbers are {}</p><p>Firstly, the numbers should be set in an ascending order.</p>'.format(res1))
    a1('<p>{}'.format(r1[0]))
    for i in r1[1:]:
        a1(',{}'.format(i))
    a1('</p>')
    a1('<br><p>The length of the given data values are {}.</p>'.format(a))
    if e==0:
        a1('<p>The data can be divided into two halves with {} numbers each.</p>'.format(b))
    else:
        a1('<p>If the length of the numbers are odd, the middle number {} can be used to divide the first half of the numbers and the other half.</p>'.format(g))
    a1('<p>The First half of the numbers are {}'.format(d[0]))
    for j in d[1:]:
        a1(',{}'.format(j))
    a1('</p><p>Now, the Median of the numbers should be calculated.')
    a1(v2)
    a1('<p>The Lower (or) First Quartile of the Numbers %s is %s</p>'%(res1,o))
    return ''.join(v),o
#Function to know the upper quartile of the numbers
def upper_quartiles1(res1):
    r=res1.split(',')
    #'r1' is the list with the given numbers sorted
    r1=[]
    for i in r:
        r1.append(int(i))
        r1.sort()
    a=int(len(r1))
    e=int(a%2)
    b=int(a/2)
    g=r1[b]
    #'d' is the list with the second half of the numbers
    d=[]
    if e==0:
        for j in range(b,a):
            c=r1[j]
            d.append(c)
    else:
        for j in range(b+1,a):
            c=r1[j]
            d.append(c)
    d1=str(d)
    x1=d1.replace('[','')
    x1=x1.replace(']','')
    s=medians1(x1)
    v2=s[0]
    o=s[6]
    v=[]
    a1=v.append
    a1('<p>Upper (or) Third Quartile:</p> <p>The Upper Quartile (Q3) is the Median of the upper half of the data set which is the second half of the given data.</p><br><p>Given numbers are {}</p><p>Firstly, the numbers should be set in an ascending order.</p>'.format(res1))
    a1('<p>{}'.format(r1[0]))
    for i in r1[1:]:
        a1(',{}'.format(i))
    a1('</p><br><p>The length of the given data values are %d.</p>'%(a))
    if e==0:
        a1('<p>The data can be divided into two halves with %d numbers each.</p>'%(b))
    else:
        a1('<p>If the length of the numbers are odd, the middle number %d can be used to divide the first half of the numbers and the other half.</p>'%(g))
    a1('<p>The Second half of the numbers are {}'.format(d[0]))
    for j in d[1:]:
        a1(',{}'.format(j))
    a1('</p><p>Now, the Median of the numbers should be calculated.')
    a1(v2)
    a1('<p>The Upper (or) Third Quartile of the Numbers {} is {}</p>'.format(res1,o))
    return ''.join(v),o
#Function to know the minimum of the numbers
def find_mins1(res1):
    r=res1.split(',')
    r1=[]
    for i in r:
        r1.append(int(i))
        r1.sort()
    #'r2' is the minimum value of the numbers
    r2=min(r1)
    v=[]
    a=v.append
    a('<p>Given numbers are {}</p><p>Firstly, the given numbers are made in an ascending order.</p><p style="text-align: center;">{}'.format(res1,r1[0]))
    for i in r1[1:]:
        a(',{}'.format(i))
    a('</p><p>From the order of numbers,{} is the minimum number.</p><p>Therefore, the Minimum of the given numbers {} is {}.</p>'.format(r2,res1,r2))
    return ''.join(v),r2
#Function to know the maximum of the numbers
def find_maxs1(res1):
    r=res1.split(',')
    r1=[]
    for i in r:
        r1.append(int(i))
        r1.sort()
    #'r2' is the maximum value of the numbers
    r2=max(r1)
    v=[]
    a=v.append
    a('<p>Given numbers are {}</p><p>Firstly, the given numbers are made in an ascending order.</p><p style="text-align: center;">{}'.format(res1,r1[0]))
    for i in r1[1:]:
        a(',{}'.format(i))
    a('<hr><p>From the order of numbers, {} is the maximum number.</p><p>Therefore, the Maximum of the given numbers {} is {}.</p>'.format(r2,res1,r2))
    return ''.join(v),r2
#Function to know the five number summary of the numbers
def five_nums1(res1):
    #calling the functions medians,lower quartile, upper quartile, minimum and maximum of numbers with detailed steps
    s1=medians1(res1)
    s11=s1[0]
    s2=lower_quartiles1(res1)
    s3=upper_quartiles1(res1)
    s4=find_mins1(res1)
    s5=find_maxs1(res1)
    r1='Minimum (Min) - the smallest observation :'+str(s4[1])+'Maximum (Max) - the largest observation :'+str(s5[1])+'Median (M) - the middle term'+str(s1[1])+'First Quartile (Q1) - the middle term of values below the median :'+str(s2[1])+'Third Quartile (Q3) - the middle term of values above the median :'+str(s3[1])
    final=str(s4[1])+','+str(s5[1])+','+str(s1[1])+','+str(s2[1])+','+str(s3[1])
    v=[]
    a=v.append
    a('<p> Five Number Summary:</p><p>The Five-Number Summary is a descriptive statistic that provides information about a Set of Observations. It consists of the following statistics:</p><li> Minimum (Min) - the smallest observation</li><li> Maximum (Max) - the largest observation</li><li> Median (M) - the middle term</li><li> First Quartile (Q1) - the middle term of values below the median</li><li> Third Quartile (Q3) - the middle term of values above the median</li><br>')
    a('<br><br><p><li> Minimum (Min) - the smallest observation :</li></p> ')
    a(s4[0])
    a('<p><li> Maximum (Max) - the largest observation :</li></p> ')
    a(s5[0])
    a('<p><li> Median (M) - the middle term :</li></p> ')
    a(s11)
    a('<p><li> First Quartile (Q1) - the middle term of values below the median :</li></p> ')
    a(s2[0])
    a('<p><li> Third Quartile (Q3) - the middle term of values above the median :</li></p> ')
    a(s3[0])
    return ''.join(v),final

def fact_fun(n,j):
    f1 = []
    for i in range((j+1), int(n)+1):
        f1.append(str(i))
    f2 = str(j)+'!'
    f1 = f1[::-1]
    f1.append(f2)
    return 'X'.join(f1)


def fact_fun2(n,j):
    f1 = []
    for i in range((j+1), int(n)+1):
        f1.append(str(i))
    return 'X'.join(f1[::-1])


def fact_fun3(n,j):
    m1 = []
    for i in range(j+1, int(n) + 1):
        m1.append((i))
    return m1
def multiplyList(myList):
    result = 1
    for x in myList:
         result = result * x
    return result


def final_roundup(t):
    f = str(t)
    if 'x' in f:
        s = f.split('x')
        if len(s[0]):
            d1 =float(s[0])
            d= round(d1, 5)
            z = str(d)+'x'+s[1]
            return z
    else:
        return round(t, 7)
def cm_roundUp2(t):
    y = str(t)
    if 'e' in y:  # if e found in given no
        x = y.replace("e", "x 10<sup>") + "</sup>"
        return x
    else:  # if not roundup that value
        r = round(float(t), 10)
        return r

def two_combinations(res1):
    print(res1) # this no
    s = res1.split('c')

    try:
        n1 = s[0]
        k1 = s[1]
        if int(n1) > int(k1):  # n1 must be greterthen the k1
            v = [] # list
            a = v.append
            len(res1)
            a('<p><b>Steps:</b></p>')
            a('<p>Combination of two numbers</p>')
            a('<p>Combination formula <sup>n</sup>C<sub>k</sub>=n! / k!(n-k)!</p>')
            a('<p>The number of k-combinations({}) from a given set S of n ({}) elements</p>' .format(k1, n1))
            a('<p>Calculating the number of combinations</p>')
            a('<p>Substituting our values for n= {} and k= {} we get</p>'.format(n1, k1))
            a('<p><sup>{}</sup>C<sub>{}</sub> = {}! <span>&#47;</span> {}!({}-{})!</p>' .format(n1,k1,n1,k1,n1,k1))
            d1 = (int(n1)) - (int(k1))
            a('<p>Subtracting {} from {}  is {}</p>' .format(k1, n1, d1))
            a('<p>${}!\\above 1pt{}!({})!$</p>' .format(n1, k1, d1))
            nu1 = fact_fun(n1,d1)
            a('<p>Expending factorial</p>')
            k2 = fact_fun2(k1, 0)
            a('<p>${}\\above 1pt({}) X {}!$</p>' .format(nu1, k2,d1))
            nu2 = fact_fun2(n1,d1)
            a('<p>Cancelling common factors</p>')
            a('<p>${}\\above 1pt{}$</p>' .format(nu2,k2))
            a('<p>Multiply numerator values and  denominator values</p>')
            s1 = fact_fun3(n1,d1)
            ml1 = multiplyList(s1)
            s2 = fact_fun3(k1, 0)
            ml2 = multiplyList(s2)
            a('<p>$ {}\\above 1pt{} $</p>' .format(ml1,ml2))
            a('<p>Divide the {} by {}</p>' .format(ml1, ml2))
            d1 = ml1/ml2
            d = final_roundup(cm_roundUp2(d1))
            a('<p>Final Result :{}</p>' .format(d))
            a('<p>There are {} combinations to choose {} items out of a set of {}</p>' .format(d, k1, n1))
            return ''.join(v), d
        else:
            eval('s2')
    except Exception as e:
        return e
def factorial_val(n):
    factorial = 1
    if int(n) >= 1:
        for i in range (1,int(n)+1):
            factorial = factorial * i
    return factorial
def coin_prob_func(n1,h1):
    from fractions import Fraction
    n_fact=factorial_val(n1)
    h_fact=factorial_val(h1)
    n_h_fact=factorial_val(n1-h1)
    r1=0.5**h1
    r2=0.5**(n1-h1)
    r3=n_fact/(h_fact*n_h_fact)
    r4=r3*r1*r2
    r5=Fraction(r4)
    r6=round(r4*100,2)
    v=[]
    a=v.append
    a('<p>Probability is {}</p>'.format(r5))
    a('<p>Probability = {}</p>'.format(round(r4,6)))
    a('<p>Percentage of getting {} Heads/Tails is {} %</p>'.format(h1,r6))
    return ''.join(v),r4
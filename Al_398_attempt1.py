import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.optimize as optimization

a=210;b=-0.5;c=0;

def test(x):
    return a*np.exp(b*x)+c

data = pd.read_excel('C:/Users/harry/OneDrive/Documents/Uni/Y2/S2/Labs/Al graphs/AL_398_attempt1.xlsx' , 
                        names=('time', 'w'), 
                        usecols=(0,1), 
                        nrows = 1000)

#plt.scatter(data.time, data.w, label = "raw data")

i=0
newt=[];neww=[]; err = []

while(i<67):
    val=data.w[i]
    t=data.time[i]
    if(val-20<test(t) and test(t)<val+7):
        val=data.w[i]
    t=data.time[i]
    if(val-20<test(t) and test(t)<val+20):
        newt.append(t)
        neww.append(val)
        err.append(val * 0.05)
    i=i+1
#plt.plot(data.time, test(data.time), label = 'FUN')
plt.scatter(newt,neww,label="Data Points")
plt.errorbar(newt, neww, yerr = err)

#~~~~~~~~~~~~~~~~~~~Defining variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#m = mass of the disk, r = radius of the disk
#m = mass of the disk, r = radius of the disk
m=0.194 ; r=0.3; I=0.5*m*r**2
i= 0.170; Tc = 0.01 ;
sig=42.1*(10**6);R=0.048;S=1.01*(10**-3);d=1*(10**-3); B = 210 *10**-3

Ti= sig * R * R * S *B*B* d

print(Ti)

Tc =0.011847628411874535; Tp=-0.0003370959229227097
#~~~~~~~~~~~~~~~~~~Conditions for the model~~~~~~~~~~~~~~~~~~~~~~~~
delt= 0.01; 
x= 0.0; t=0.0; u=214
h=0.05

#~~~~~~~~~~~~~~~~~~Value Storage~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ωvals=[]; tvals=[];θvals=[];

#~~~~~~~~~~~~~~~~~~Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

while(u>=0.0):
    tvals.append(t);ωvals.append(u);θvals.append(x)
    
    m1=u
    k1=(-1/I)*(Ti*u+Tc+Tp*u)
    m2= u+(h/2.)*k1
    t_2=t+(h/2.)
    x_2=x+(h/2.)*m1
    u_2=m2
    k2=(-1/I)*(Ti*u_2+Tc+Tp*u_2)
    m3=u+(h/2.)*k2
    t_3=t+(h/2.)
    x_3=x+(h/2.)*m2
    u_3=m3
    k3=(-1/I)*(Ti*u_3+Tc+Tp*u_3)
    m4=u+(h)*k3
    t_4=t+(h)
    x_4=x+(h)*m3
    u_4=m4
    k4=(-1/I)*(Ti*u_4+Tc+Tp*u_4)
    t=t+h
    x=x+(h/6.)*(m1+(2.*m2)+(2.*m3)+m4)
    u=u+(h/6.)*(k1+(2.*k2)+(2.*k3)+k4)
    
#~~~~~~~~~#~~~~~~~Least Sq Fit~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def func(params,xdata,angv):
    return (angv-np.dot(xdata,params))

#~~~~~~~~~~~~~~~~~Curve Fit for Real Data~~~~~~~~~~~~~~~~~~~~~~~~~~

def str_line(T,a,b,const):
    return a*(np.exp(-b*T))+const
T=np.linspace(0.0,7,100)
popt,pcov=curve_fit(str_line,newt,neww,sigma=None)
plt.plot(T, str_line(T,*popt),"black",label="Curve Fit")

print("ωo*= {0:5.3f} +/- {1:5.3f}".format(popt[0],pcov[0,0]**0.5))
print("exp factor= {0:5.3f} +/- {1:5.3f}".format(popt[1],pcov[1,1]**0.5))
print("Constant= {0:5.3f} +/- {1:5.3f}\n".format(popt[2],pcov[2,2]**0.5))
#~~~~~~~~~~~~~~~~~Constant Finding~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
uncωostar=((pcov[0,0]**0.5)/popt[0])*100
uncb=((pcov[1,1]**0.5)/popt[1])*100
uncconst=((pcov[2,2]**0.5)/popt[2])*100

#fnd= found values of the constants, unc= Uncertainty
fndTc=-1*popt[1]*popt[2]*I
uncTc=(-1.*uncconst+uncb)/100*fndTc

fndTp=popt[1]*I-Ti
uncTp=(uncb/100.)*fndTp

fndωo=popt[0]+popt[2]
uncωo=pcov[2,2]**0.5+pcov[0,0]**0.5

fndTi=popt[1]*I-fndTp
uncTi=pcov[1,1]**0.5+uncTp

print("Tp={0:5.3f} +/- {1:5.3f}. Percentage unc= +/- {2:3.1f}%".format(fndTp,uncTp,(uncTp*100/fndTp)))
print("Tc={0:7.5f} +/- {1:7.5f}. Percentage unc= +/- {2:3.1f}%".format(fndTc,uncTc,(uncTc*100/fndTc)))
print("ωo={0:5.3f} +/- {1:5.3f}. Percentage unc= +/- {2:3.1f}%".format(fndωo,uncωo,uncωo*100/fndωo))
print("Ti={0:5.3f} +/- {1:5.3f}. Percentage unc= +/- {2:3.1f}%".format(fndTi,uncTi,uncTi*100/fndTi))

#~~~~~~~~~~~~~~~~~Plotting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.plot(tvals,ωvals,"r",label="Runge-Approximation")

#~~~~~~~~~~~~~~~~~Graph Tidying~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plt.xlabel("Time / s")
plt.ylabel("ω(t) / rad s^(-1)")
plt.rcParams.update({'font.size':13})
plt.tick_params(direction='in',length=7,bottom='on',left='on',top='on',right='on')
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:11:07 2023

@author: Tilly
"""


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve   #to perform mu-value extraction out of N-sum 
from scipy.linalg import det  # shorter version to do determinants

from itertools import cycle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


plt.rcParams['figure.dpi'] = 400


#-------------------------------------------------------
#setting experimental constants

lamba=1 #parameter wrt which all others are defined
k_b= 1 # actually 1.380649*(10)**(-23) #J/K
hbar = 1 # actually 1.055*(10)**(-34) #J⋅s
tau=hbar/lamba

#fixed parameters

T=0.1 # actually in the range of 1*10**(-9) #K
kappa=0.1
N=300 #set to 200 usually
d=1 
Nb= d*N  #from this mu wil lbe derived
N_list=np.arange(0,N)

beta= 1/(k_b*T)   

#---------------------------------------------------------
#---------------------------------------------------------

#setting time interval
t_real=np.linspace(0,100,num=500)      #setting at time interval that gives 500 points for precision
t_list=t_real/tau                     #giving time a quantum dimension
delta_t=20/200                       #timestep for derivations               

t_list_deriv=t_list[:499]
t_list_fit=np.linspace(30,150,num=500)    #used to find gamma
#---------------------------------------------------------

#setting Temperature range

Temp_range=np.linspace(0.1,1,num=5)
# usually set to (0.5,1,num=2)
#---------------------------------------------------------

def h_0_func(N,lamba):
    h = np.zeros((N,N))
    h[0,N-1]=-lamba
    h[N-1,0]=-lamba
    for i in range(N):
        if i<N-1:
            h[i,i+1]=-lamba
            h[i,i]=2*lamba #to ensure E_v > 0
        if i>0:
            h[i,i-1]=-lamba 
            h[i,i]=2*lamba #to ensure E_v > 0
    return h

h_0 = h_0_func(N,lamba)
Ev0,Evec0=np.linalg.eigh(h_0)


#--------------------------------------------------------
#defining h_1 plus eigenvalues/ eigenvectors


def h_1_func(N,lamba,kappa):
    h_interaction=[[0 for column in range(N)] for row in range(N)] 
    l=int(N/2)
    h_interaction[l][l]= kappa
    
    h =  h_0 + h_interaction
    return h

h_1 = h_1_func(N, lamba, kappa)
Ev1,Evec1=np.linalg.eigh(h_1)


#---------------------------------------------------------
#---------------------------------------------------------

def be(beta,Ev0, Nb, mu):
    num = Nb
    for n in range(Ev0.shape[0]):
        num -= 1/(np.exp(beta*(Ev0[n]-mu))-1)
    return num
mu = fsolve(lambda mu: be(beta,Ev0, Nb, mu), np.min(Ev0)-0.0001)  
   #the fsolve function works in such a way that it finds mu in rearranged
   #and shortened lambdafct per condition of set Nb 
        
   #we have to set starting value and choose it such that 
   #the chemical potential can never be higher than the lowest
   #energy level! In our case E_lowest=0

print("The chemical potential for the given number of Nb is:",mu[0])   
#m transfor a 1 element array into a number, we add [0] to say we call the only element


#defining n function for each E_val_h1
def fct_E_n(beta,Ev0,mu):
    m=np.zeros((N,N))
    for i in range(Ev0.shape[0]):
        m[i,i]= 1/((np.exp(beta*(Ev0[i]-mu))-1) )
    return m 

    #this result is a matrix
    #n=fct_E_n(beta,Ev0,mu)

#--------------------------------------------------------

#Matrix definition of finding Decoh fct
def decoh_func(t,beta,Evec0,Evec1,Ev0,Ev1,n):
    U_1=np.zeros((N,N), dtype=np.complex128)
    sc=np.zeros((N), dtype=np.complex128)
    ex=np.zeros((N), dtype=np.complex128)
    
    for i in range(N):
        sc[i]=np.exp(1j*Ev0[i]*t)*n[i,i]
        ex[i]=np.exp(-1j*Ev1[i]*t)
        for j in range(N):
            U_1[i,j]= Evec0[:,i]@Evec1[:,j]
            # <n|n'>
            # where  <n'|m> = (U_1).T

    U_2=((U_1)*ex)
    U_3=((U_2).T*sc).T
    U_4=(U_3)@(U_1).T
    U_5=(np.eye(N)+n)- U_4
    return U_5


#--------------------------------------------------------
#---------------------------------------------------------
#PLOTTING2: 2D Bloch sphere decay Plot , plotted also in the following for-loop
#---------------------------------------------------------

#---------------------------------------------------------
#Layout - linestyles


lines = [":","-"]
linecycler = cycle(lines)

lines1= [":","-"]
linecycler1 = cycle(lines1)

lines2= [":","-"]
linecycler2 = cycle(lines2)

lines3= ["-","-"]
linecycler3 = cycle(lines3)

lines4= ["-","-"]
linecycler4 = cycle(lines4)

lines5= [":","-"]
linecycler5 = cycle(lines5)

lines6= [":","-"]
linecycler6 = cycle(lines6)

lines7= [":","-"]
linecycler7 = cycle(lines7)

lines8= [":","-"]
linecycler8 = cycle(lines8)

lines9= [":","-"]
linecycler9 = cycle(lines9)

lines10= [":","-"]
linecycler10 = cycle(lines10)

lines15= [":","-"]
linecycler15 = cycle(lines15)
#unit circle details:

theta = np.linspace( 0, 2 * np.pi , 150 ) 
radius = 1
a = radius * np.cos( theta )
b = radius * np.sin( theta ) 


#legend notations:
    
cyan_patch = mpatches.Patch(color="darkcyan", label=r' $\kappa_{low}=0.02$')
orange_patch = mpatches.Patch(color="darkorange", label=r' $\kappa_{medium}=0.04$')
magenta_patch = mpatches.Patch(color="darkmagenta", label=r' $\kappa_{high}=0.06$')


line_cold = Line2D([0], [0], label=r'$T_{cold}=0.5$',ls=":", color='grey')
line_hot = Line2D([0], [0], label=r'$T_{hot}=1$',ls="-", color='grey')

#----------------------------------------------
low_coup = Line2D([0], [0], label=r" $\kappa_{low}= 0.01$",ls="-", color='darkcyan')
low_coup_PT = Line2D([0], [0], label=r" $\kappa_{low}$ PT",ls=":", color='darkcyan')


med_coup = Line2D([0], [0], label=r"  $\kappa_{medium}= 0.02$",ls="-", color='darkorange')
med_coup_PT = Line2D([0], [0], label=r"$\kappa_{medium}$ PT",ls=":", color='darkorange')


high_coup = Line2D([0], [0], label=r"  $\kappa_{high}= 0.03$",ls="-", color='darkmagenta')
high_coup_PT = Line2D([0], [0], label=r"$\kappa_{high}$ PT",ls=":", color='darkmagenta')


initial=Line2D([0], [0], marker='o', color='w', label=r"initial $|+\rangle$",markerfacecolor='b', markersize=5)
final=Line2D([0], [0], marker='o', color='w', label=r"final $|I\rangle$",markerfacecolor='r', markersize=5)
               
#1: low
kappa=0.1
d=1
Nb= d*N 
 

gammas1=[]

for m,i in enumerate(Temp_range): 

    T_1=i
    delta_T=0.0001*T_1
    T_2=T_1+delta_T
    
    beta_1= 1/(k_b*T_1)
    beta_2=1/(k_b*T_2)
    
    #the eigenvectors and eigenvalues are not Temperature-dependant
    h_0 = h_0_func(N,lamba)
    Ev0,Evec0=np.linalg.eigh(h_0)
    
    h_1 = h_1_func(N, lamba, kappa)
    Ev1,Evec1=np.linalg.eigh(h_1)
    
    BE_1=be(beta_1,Ev0, Nb, mu)
    mu_1 = fsolve(lambda mu: be(beta_1,Ev0, Nb, mu), np.min(Ev0)-0.0001)  

    BE_2=be(beta_2,Ev0, Nb, mu)
    mu_2 = fsolve(lambda mu: be(beta_2,Ev0, Nb, mu), np.min(Ev0)-0.0001)  

    n_1=fct_E_n(beta_1,Ev0,mu_1)
    n_2=fct_E_n(beta_2,Ev0,mu_2)
    
    A_1= decoh_func(0,beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
    A_2= decoh_func(0,beta_2,Evec0,Evec1,Ev0,Ev1,n_2)

    def nu_t_func(A):
        x=(1/det(A))                
        return x

    
    nu_t_1 =np.zeros([t_list.shape[0]],dtype=np.complex128)
    nu_t_2 =np.zeros([t_list.shape[0]],dtype=np.complex128)
    
    phase_t_1 =np.zeros([t_list.shape[0]])
    phase_t_2 =np.zeros([t_list.shape[0]])
    
    for x in range(t_list.shape[0]):
        A_1=decoh_func(t_list[x],beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
        A_2=decoh_func(t_list[x],beta_2,Evec0,Evec1,Ev0,Ev1,n_2)
        nu_t_1[x]=nu_t_func(A_1)
        nu_t_2[x]=nu_t_func(A_2)
        
        phase_t_1=np.angle(nu_t_1)
        phase_t_2=np.angle(nu_t_2)

    nu_t_fit =np.zeros([t_list_fit.shape[0]],dtype=np.complex128)
    for x in range(t_list_fit.shape[0]):
        A_fit=decoh_func(t_list_fit[x],beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
        nu_t_fit[x]=nu_t_func(A_fit)
        
        x=t_list_fit
        y=np.abs(nu_t_fit)
        p= np.polyfit(x, np.log(y), 1)
    gammas1.append(-p[0])
    
    #----------------------
      
    plt.figure(1)

    plt.plot(N_list,np.diag(n_1),label="T="+str(round(i,2)))
    
    #----------------------  
"""
    plt.figure(2)

    plt.semilogy(t_list,np.abs(nu_t_1),next(linecycler2),color="darkcyan",label=r"$\kappa$=0.5,T="+str(round(i,1)))

    plt.figure(3)
    plt.plot(t_list,-phase_t_1,next(linecycler3),color="darkcyan",label=r"$\kappa$ =0.5,T = "+str(round(i,1)))
    plt.plot(t_list,kappa*d*t_list,ls=":",color="darkcyan")

    #-----------------------------------------
    #finding derivatives:
    phase_deriv_Temp=(1/delta_T)*(phase_t_2-phase_t_1)
    phase_deriv_time= np.diff(phase_t_1)*(1/delta_t)

    #finding derivatives:
    nu_t_deriv_1=(1/delta_T)*(np.abs(nu_t_2)-np.abs(nu_t_1))
    nu_t_deriv=(1/delta_t)*(np.diff(nu_t_1))
    
    #-----------------------------------------
    plt.figure(4)
    plt.plot(t_list_deriv, -phase_deriv_time,next(linecycler4),color="darkcyan",label=r"$\kappa$ =0.5,T = "+str(round(i,1)))

    plt.figure(5)
    real_values=np.real(nu_t_1)
    im_values=np.imag(nu_t_1)
    
    plt.plot(real_values,im_values,next(linecycler5),color="darkcyan",label=r"T = "+str(round(i,1)))
    #----------------------------------------
    QFI=1/(1-(np.abs(nu_t_1))**2)*(nu_t_deriv_1)**2+(np.abs(nu_t_1))**2*(phase_deriv_Temp)**2
    #-----------------------------------------
    plt.figure(6)
    plt.plot(t_list,QFI,next(linecycler6),color="darkcyan")

    plt.figure(7)
    QFI_parallel=1/(1-(np.abs(nu_t_1))**2)*(nu_t_deriv_1)**2
    QFI_perp=(np.abs(nu_t_1))**2*(phase_deriv_Temp)**2
    
    ax1=plt.subplot(2, 1, 1)
    ax1.plot(t_list,(QFI_parallel),next(linecycler7),color="darkcyan",label=r"$\kappa$=0.5, T= "+str(round(i,1)))

    ax2=plt.subplot(2, 1, 2)
    ax2.plot(t_list,(QFI_perp),next(linecycler8),color="darkcyan",label=r"$\kappa$=0.5, T= "+str(round(i,1)))

    
    plt.figure(8)
    plt.plot(t_list,i*np.sqrt(QFI),next(linecycler9),color="darkcyan",label=r"$\kappa$=0.5, T= "+str(round(i,1)))

    plt.figure(9)
    plt.plot(t_list,QFI,next(linecycler10),color="darkcyan")
    plt.ylabel(r"$\mathcal{F}_Q$",size=15)
    plt.xlabel(r"$\frac{t}{\tau}$",size=15)
    plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5), frameon=False,loc="center left", borderaxespad=0)


#2: medium
kappa=0.02
d=1
Nb= d*N  


gammas2=[]

for m,i in enumerate(Temp_range): 

    T_1=i
    delta_T=0.0001*T_1
    T_2=T_1+delta_T
    
    beta_1= 1/(k_b*T_1)
    beta_2=1/(k_b*T_2)
    
    #the eigenvectors and eigenvalues are not Temperature-dependant
    h_0 = h_0_func(N,lamba)
    Ev0,Evec0=np.linalg.eigh(h_0)
    
    h_1 = h_1_func(N, lamba, kappa)
    Ev1,Evec1=np.linalg.eigh(h_1)
    
    BE_1=be(beta_1,Ev0, Nb, mu)
    mu_1 = fsolve(lambda mu: be(beta_1,Ev0, Nb, mu), np.min(Ev0)-0.0001)  

    BE_2=be(beta_2,Ev0, Nb, mu)
    mu_2 = fsolve(lambda mu: be(beta_2,Ev0, Nb, mu), np.min(Ev0)-0.0001)  

    n_1=fct_E_n(beta_1,Ev0,mu_1)
    n_2=fct_E_n(beta_2,Ev0,mu_2)
    
    A_1= decoh_func(0,beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
    A_2= decoh_func(0,beta_2,Evec0,Evec1,Ev0,Ev1,n_2)

    def nu_t_func(A):
        x=(1/det(A))                
        return x

    
    nu_t_1 =np.zeros([t_list.shape[0]],dtype=np.complex128)
    nu_t_2 =np.zeros([t_list.shape[0]],dtype=np.complex128)
    
    phase_t_1 =np.zeros([t_list.shape[0]])
    phase_t_2 =np.zeros([t_list.shape[0]])
    
    for x in range(t_list.shape[0]):
        A_1=decoh_func(t_list[x],beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
        A_2=decoh_func(t_list[x],beta_2,Evec0,Evec1,Ev0,Ev1,n_2)
        nu_t_1[x]=nu_t_func(A_1)
        nu_t_2[x]=nu_t_func(A_2)
        
        phase_t_1=np.angle(nu_t_1)
        phase_t_2=np.angle(nu_t_2)

    nu_t_fit =np.zeros([t_list_fit.shape[0]],dtype=np.complex128)
    for x in range(t_list_fit.shape[0]):
        A_fit=decoh_func(t_list_fit[x],beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
        nu_t_fit[x]=nu_t_func(A_fit)
        
        x=t_list_fit
        y=np.abs(nu_t_fit)
        p= np.polyfit(x, np.log(y), 1)
    gammas2.append(-p[0])
    
    #----------------------
    
    plt.figure(1)

    plt.plot(N_list,np.diag(n_1),next(linecycler1),color="k",label="T="+str(round(i,2)))
    
    #----------------------  

    plt.figure(2)

    plt.semilogy(t_list,np.abs(nu_t_1),next(linecycler2),color="darkorange",label=r"$\kappa$=0.5,T="+str(round(i,1)))

    plt.figure(3)
    plt.plot(t_list,-phase_t_1,next(linecycler3),color="darkorange",label=r"$\kappa$ =0.5,T = "+str(round(i,1)))
    plt.plot(t_list,kappa*d*t_list,ls=":",color="darkorange")

    #-----------------------------------------
    #finding derivatives:
    phase_deriv_Temp=(1/delta_T)*(phase_t_2-phase_t_1)
    phase_deriv_time= np.diff(phase_t_1)*(1/delta_t)

    #finding derivatives:
    nu_t_deriv_1=(1/delta_T)*(np.abs(nu_t_2)-np.abs(nu_t_1))
    nu_t_deriv=(1/delta_t)*(np.diff(nu_t_1))
    
    #-----------------------------------------
    plt.figure(4)
    plt.plot(t_list_deriv,- phase_deriv_time,next(linecycler4),color="darkorange",label=r"$\kappa$ =0.5,T = "+str(round(i,1)))

    plt.figure(5)
    real_values=np.real(nu_t_1)
    im_values=np.imag(nu_t_1)
    
    plt.plot(real_values,im_values,next(linecycler5),color="darkorange",label=r"T = "+str(round(i,1)))
    #----------------------------------------
    QFI=1/(1-(np.abs(nu_t_1))**2)*(nu_t_deriv_1)**2+(np.abs(nu_t_1))**2*(phase_deriv_Temp)**2
    #-----------------------------------------
    plt.figure(6)
    plt.plot(t_list,QFI,next(linecycler),color="darkorange")

    plt.figure(7)

    QFI_parallel=1/(1-(np.abs(nu_t_1))**2)*(nu_t_deriv_1)**2
    QFI_perp=(np.abs(nu_t_1))**2*(phase_deriv_Temp)**2
    
    ax1=plt.subplot(2, 1, 1)
    ax1.plot(t_list,(QFI_parallel),next(linecycler7),color="darkorange",label=r"$ T= "+str(round(i,1)))

    ax2=plt.subplot(2, 1, 2)
    ax2.plot(t_list,(QFI_perp),next(linecycler8),color="darkorange",label=r"$ T= "+str(round(i,1)))

    
    plt.figure(8)
    plt.plot(t_list,i*np.sqrt(QFI),next(linecycler9),color="darkorange",label=r"$\kappa$=0.5, T= "+str(round(i,1)))

    plt.figure(9)
    plt.plot(t_list,QFI,next(linecycler1),color="darkorange")
    plt.ylabel(r"$\mathcal{F}_Q$",size=15)
    plt.xlabel(r"$\frac{t}{\tau}$",size=15)
    plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5), frameon=False,loc="center left", borderaxespad=0)



#2: high
kappa=0.03
d=1
Nb= d*N  

gammas3=[]

for m,i in enumerate(Temp_range): 

    T_1=i
    delta_T=0.0001*T_1
    T_2=T_1+delta_T
    
    beta_1= 1/(k_b*T_1)
    beta_2=1/(k_b*T_2)
    
    #the eigenvectors and eigenvalues are not Temperature-dependant
    h_0 = h_0_func(N,lamba)
    Ev0,Evec0=np.linalg.eigh(h_0)
    
    h_1 = h_1_func(N, lamba, kappa)
    Ev1,Evec1=np.linalg.eigh(h_1)
    
    BE_1=be(beta_1,Ev0, Nb, mu)
    mu_1 = fsolve(lambda mu: be(beta_1,Ev0, Nb, mu), np.min(Ev0)-0.0001)  

    BE_2=be(beta_2,Ev0, Nb, mu)
    mu_2 = fsolve(lambda mu: be(beta_2,Ev0, Nb, mu), np.min(Ev0)-0.0001)  

    n_1=fct_E_n(beta_1,Ev0,mu_1)
    n_2=fct_E_n(beta_2,Ev0,mu_2)
    
    A_1= decoh_func(0,beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
    A_2= decoh_func(0,beta_2,Evec0,Evec1,Ev0,Ev1,n_2)

    def nu_t_func(A):
        x=(1/det(A))                
        return x

    
    nu_t_1 =np.zeros([t_list.shape[0]],dtype=np.complex128)
    nu_t_2 =np.zeros([t_list.shape[0]],dtype=np.complex128)
    
    phase_t_1 =np.zeros([t_list.shape[0]])
    phase_t_2 =np.zeros([t_list.shape[0]])
    
    for x in range(t_list.shape[0]):
        A_1=decoh_func(t_list[x],beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
        A_2=decoh_func(t_list[x],beta_2,Evec0,Evec1,Ev0,Ev1,n_2)
        nu_t_1[x]=nu_t_func(A_1)
        nu_t_2[x]=nu_t_func(A_2)
        
        phase_t_1=np.angle(nu_t_1)
        phase_t_2=np.angle(nu_t_2)

    nu_t_fit =np.zeros([t_list_fit.shape[0]],dtype=np.complex128)
    for x in range(t_list_fit.shape[0]):
        A_fit=decoh_func(t_list_fit[x],beta_1,Evec0,Evec1,Ev0,Ev1,n_1)
        nu_t_fit[x]=nu_t_func(A_fit)
        
        x=t_list_fit
        y=np.abs(nu_t_fit)
        p= np.polyfit(x, np.log(y), 1)
    gammas3.append(-p[0])
    
    #----------------------
    
    plt.figure(1)

    plt.plot(N_list,np.diag(n_1),next(linecycler1),color="k",label="T="+str(round(i,2)))
    

    #----------------------  

    plt.figure(2)

    plt.semilogy(t_list,np.abs(nu_t_1),next(linecycler2),color="darkmagenta",label=r"$\kappa$=0.5,T="+str(round(i,1)))


    plt.figure(3)
    plt.plot(t_list,-phase_t_1,next(linecycler3),color="darkmagenta",label=r"$\kappa$ =0.5,T = "+str(round(i,1)))
    plt.plot(t_list,kappa*d*t_list,ls=":",color="darkmagenta")
    #-----------------------------------------
    #finding derivatives:
    phase_deriv_Temp=(1/delta_T)*(phase_t_2-phase_t_1)
    phase_deriv_time= np.diff(phase_t_1)*(1/delta_t)

    #finding derivatives:
    nu_t_deriv_1=(1/delta_T)*(np.abs(nu_t_2)-np.abs(nu_t_1))
    nu_t_deriv=(1/delta_t)*(np.diff(nu_t_1))
    
    #-----------------------------------------
    plt.figure(4)
    plt.plot(t_list_deriv,- phase_deriv_time,next(linecycler4),color="darkmagenta",label=r"$\kappa$ =0.5,T = "+str(round(i,1)))
    plt.xlim(0,20)
    plt.ylim((0,0.5))
    
    plt.figure(5)
    real_values=np.real(nu_t_1)
    im_values=np.imag(nu_t_1)
    plt.plot(real_values,im_values,next(linecycler5),color="darkmagenta",label=r"T = "+str(round(i,1)))

    #----------------------------------------
    QFI=1/(1-(np.abs(nu_t_1))**2)*(nu_t_deriv_1)**2+(np.abs(nu_t_1))**2*(phase_deriv_Temp)**2
    #-----------------------------------------
    plt.figure(6)
    plt.plot(t_list,QFI,next(linecycler6),color="darkmagenta")
    plt.ylabel(r"$\mathcal{F}_Q$",size=15)
    plt.xlabel(r"$\frac{t}{\tau}$",size=15)
    plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5), fontsize="14",frameon=False,loc="center left", borderaxespad=0)

    plt.figure(7)
    QFI_parallel=1/(1-(np.abs(nu_t_1))**2)*(nu_t_deriv_1)**2
    QFI_perp=(np.abs(nu_t_1))**2*(phase_deriv_Temp)**2
    
    ax1=plt.subplot(2, 1, 1)
    ax1.plot(t_list,(QFI_parallel),next(linecycler7),color="darkmagenta",label=r"$\kappa$=0.5, T= "+str(round(i,1)))
    ax1.set_ylabel(r"$\mathcal{F}_{\parallel}$")
    ax1.set_xlabel(r"$\frac{t}{\tau}$")
    ax1.set_ylim(0,0.12)
    #ax1.xticks(color='w')

    ax2=plt.subplot(2, 1, 2)
    ax2.plot(t_list,(QFI_perp),next(linecycler8),color="darkmagenta",label=r"$\kappa$=0.5, T= "+str(round(i,1)))
    ax2.set_ylabel(r"$\mathcal{F}_{\perp}$")
    ax2.set_xlabel(r"$\frac{t}{\tau}$")
    ax2.set_ylim(0,0.35)
    
    plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 1),fontsize="14", frameon=False,loc="center left", borderaxespad=0)

    plt.figure(8)
    plt.plot(t_list,i*np.sqrt(QFI),next(linecycler9),color="darkmagenta",label=r"$\kappa$=0.5, T= "+str(round(i,1)))
    
    #plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5), frameon=False,loc="center left", borderaxespad=0)

    plt.figure(9)
    plt.plot(t_list,QFI,next(linecycler10),color="darkmagenta")
    plt.ylabel(r"$\mathcal{F}_Q$",size=15)
    plt.xlabel(r"$\frac{t}{\tau}$",size=15)
    plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5),fontsize="14", frameon=False,loc="center left", borderaxespad=0)

"""
#layout stuff:
plt.figure(1)
#plt.title(r"Plot of the Bose-Einstein-function)
plt.ylabel(r"$b(E_n)$",size=15)
plt.xlabel(r"$N$",size=15)
plt.legend(bbox_to_anchor=(1.04, 0.5),frameon=False, loc="center left", borderaxespad=0)
"""

plt.figure(2)
#plt.title(r"Plot of the decoherence function $\nu(t)$ vs time",size=20)
plt.ylabel(r"$|\nu(t)|$",size=15)
plt.xlabel(r"$\frac{t}{\tau}$",size=15)
plt.legend(bbox_to_anchor=(1.04, 0.5),frameon=False, loc="center left", borderaxespad=0)
plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5),fontsize="14", frameon=False,loc="center left", borderaxespad=0)
    
plt.ylim([0.4,1.01])
plt.xlim([0,100])

plt.figure(3)
plt.ylabel(r"$\theta(t)$",size=15)
plt.xlabel(r"$\frac{t}{\tau}$",size=15)
pi = np.pi
theta = np.arange( -pi,  pi+pi/2, step=(pi / 2))
plt.yticks(theta,['-π', '-π/2', '0', 'π/2', 'π'])
plt.ylim(-0.1,3.5)
plt.xlim(0,100)
plt.legend(handles=[low_coup,low_coup_PT,med_coup,med_coup_PT,high_coup,high_coup_PT],bbox_to_anchor=(1.04, 0.5),fontsize="14", frameon=False,loc="center left", borderaxespad=0)

plt.figure(4)
plt.ylabel(r"$\frac{d}{dt}(\theta(t))$",size=15)
plt.xlabel(r"$\frac{t}{\tau}$",size=15)
plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5),fontsize="14", frameon=False,loc="center left", borderaxespad=0)

plt.figure(5)

plt.axis('square')
#plt.title("Plot of decay as seen looking down from z>0 axis to the xy-plane ")
plt.ylabel(r"Im|$\nu$(t)|")
plt.xlabel(r"Re|$\nu$(t)|")
ticks=np.arange( -1,1.5, step=0.5)
plt.yticks(ticks, ['-1', '-1/2', '0', '1/2', '1'])
plt.xticks(ticks, ['-1', '-1/2', '0', '1/2', '1'])

plt.plot( a, b,ls=":", color="grey",linewidth=0.5 )
plt.scatter(0,0,color="r")
plt.scatter(1,0,color="blue")

plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot,initial,final],bbox_to_anchor=(1.04, 0.5), fontsize="14",frameon=False,loc="center left", borderaxespad=0)

plt.figure(8)
plt.ylabel(r"$\mathcal{Q}$",size=15)
plt.xlabel(r"$\frac{t}{\tau}$",size=15)
plt.xlim(0,301)
plt.ylim(0,0.41)

#plt.legend(bbox_to_anchor=(1.04, 0.5),frameon=False, loc="center left", borderaxespad=0)
plt.legend(handles=[cyan_patch,orange_patch,magenta_patch,line_cold,line_hot],bbox_to_anchor=(1.04, 0.5), fontsize="15",frameon=False,loc="center left", borderaxespad=0)
"""
plt.show()

#-------------------------------------
"""
print("gammas1:")
print(gammas1)

print("gammas2:")
print(gammas2)

print("gammas3:")
print(gammas3)
"""
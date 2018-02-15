#!/usr/bin/python


""" Carry out gradient descent for the data points given in homweork3_dataFinal.txt using the
    adjiont method to compute the gradient of the cost function

    Run as python grad_desc_p1.py output_filename n_iterations """

import sys

fname, n_iters = sys.argv[1:]
n_iters = int(n_iters)

import numpy as np
import scipy.integrate as spi

data = np.loadtxt("homework3_dataFinal.txt")

nb=8 #number of basis functions to expand f(x) into


def cost(a):
    x = np.polyval(a[1:],data[0])
    return sum((x-data[1])**2)

def grad_cost(x,n,a):
    """Calculate the portion of the gradient of the cost function due to the interval t[n] to t[n+1]"""
    t_int = np.linspace(data[0,n],data[0,n+1],250)
    x_list = x[250*n:250*(n+1)]
    dt = t_int[1]-t_int[0]
    f = np.poly1d(a[1:])
    fp = np.polyder(f)
    def l_derivs(l,t):
        j = np.argmin(np.abs(t_int-t))
        return [l[1],-a[0]*l[0]+fp(x_list[j])] 
    l0 = [0,-2*np.abs(x_list[-1]-data[1,n+1])] #Fix this now that there isnt a x_list
    l_list = spi.odeint(l_derivs,l0,np.flip(t_int,0))[:,0]
    grad = [np.trapz(x_list*l_list,dx=dt)]
    for k in range(nb):
        grad.append(np.trapz(l_list*x_list**k,dx=dt))
    return np.array(grad)


params = np.zeros((n_iters+1,nb+1))
params[0,1:] = [2**(i-nb)for i in range(nb)]
params[0,0]=5.
learn_rate = 1e-6
x_derivs = lambda x,t,a: [x[1],-a[0]*x[0] + np.polyval(a[1:],x[0])]
x0 = [data[1,0],-2] 
big_t = np.linspace(data[0,0],data[0,-1],(len(data[0])-1)*250) #match the 250 points per interval

for i in range(n_iters):
    if i%100==0:
	print str(i)+" Gradient descent steps taken"
    x = spi.odeint(x_derivs,x0,big_t,args=(params[i],))[:,0]
    grad = sum([grad_cost(x,j,params[i]) for j in range(len(data[0])-1)])
    params[i+1]=params[i]-learn_rate*grad

#Save the parameters at each gradient descent step 
np.savetxt("{}.txt".format(fname),params) 

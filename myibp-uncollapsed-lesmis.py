# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:42:26 2018

@author: Administrator
"""
import pandas as pd
import scipy.io
import numpy as np
import datetime
import dill
import math
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns

from scipy.special import gammaln
np.set_printoptions(threshold=np.inf)
np.random.seed(123)

import scipy.stats.kde as kde

def hpd_grid(sample, alpha=0.05, roundto=2):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode

    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results

    Returns
    ----------
    hpd: array with the lower 
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes

def plot_post1(sample, alpha=0.05, show_mode=True, kde_plot=True, bins=6, 
    ROPE=None, comp_val=None, roundto=2):
    """Plot posterior and HPD

    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    show_mode: Bool
        If True the legend will show the mode(s) value(s), if false the mean(s)
        will be displayed
    kde_plot: Bool
        If True the posterior will be displayed using a Kernel Density Estimation
        otherwise an histogram will be used
    bins: integer
        Number of bins used for the histogram, only works when kde_plot is False
    ROPE: list or numpy array
        Lower and upper values of the Region Of Practical Equivalence
    comp_val: float
        Comparison value
        

    Returns
    -------

    post_summary : dictionary
        Containing values with several summary statistics

    """       

    post_summary = {'mean':0,'median':0,'mode':0, 'alpha':0,'hpd_low':0,
                   'hpd_high':0, 'comp_val':0, 'pc_gt_comp_val':0, 'ROPE_low':0,
                   'ROPE_high':0, 'pc_in_ROPE':0}

    post_summary['mean'] = round(np.mean(sample), roundto)
    post_summary['median'] = round(np.median(sample), roundto)
    post_summary['alpha'] = alpha

    # Compute the hpd, KDE and mode for the posterior
    hpd, x, y, modes = hpd_grid(sample, alpha, roundto)
    print(min(sample))
    post_summary['hpd'] = hpd
    post_summary['mode'] = modes

    ## Plot KDE.
    if kde_plot:
            plt.plot(x, y, color='k', lw=2)
    ## Plot histogram.
    else:
        plt.hist(sample, normed=True, bins=bins, facecolor='b', 
        edgecolor='w')

    ## Display mode or mean:
    if show_mode:
        string = '{:g} ' * len(post_summary['mode'])
        plt.plot(0, label='mode =' + string.format(*post_summary['mode']), alpha=0)
    else:
        plt.plot(0, label='mean = {:g}'.format(post_summary['mean']), alpha=0)

    ## Display the hpd.
    hpd_label = ''
    for value in hpd:
        plt.plot(value, [1, 1], linewidth=10, color='b')
        hpd_label = hpd_label +  '{:g} {:g}\n'.format(round(value[0], roundto), round(value[1], roundto)) 
    plt.plot(0, 0, linewidth=4, color='b', label='hpd {:g}%\n{}'.format((1-alpha)*100, hpd_label))
    ## Display the ROPE.
    if ROPE is not None:
        pc_in_ROPE = round(np.sum((sample > ROPE[0]) & (sample < ROPE[1]))/len(sample)*100, roundto)
        plt.plot(ROPE, [0, 0], linewidth=20, color='r', alpha=0.75)
        plt.plot(0, 0, linewidth=4, color='r', label='{:g}% in ROPE'.format(pc_in_ROPE))
        post_summary['ROPE_low'] = ROPE[0] 
        post_summary['ROPE_high'] = ROPE[1] 
        post_summary['pc_in_ROPE'] = pc_in_ROPE
    ## Display the comparison value.
    if comp_val is not None:
        pc_gt_comp_val = round(100 * np.sum(sample > comp_val)/len(sample), roundto)
        pc_lt_comp_val = round(100 - pc_gt_comp_val, roundto)
        plt.axvline(comp_val, ymax=.75, color='g', linewidth=4, alpha=0.75,
            label='{:g}% < {:g} < {:g}%'.format(pc_lt_comp_val, 
                                                comp_val, pc_gt_comp_val))
        post_summary['comp_val'] = comp_val
        post_summary['pc_gt_comp_val'] = pc_gt_comp_val
    plt.title('HPD of $\lambda$')
    plt.legend(loc=0, framealpha=1)
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    return post_summary

def plot_post2(sample, alpha=0.05, show_mode=True, kde_plot=True, bins=6, 
    ROPE=None, comp_val=None, roundto=2):
    """Plot posterior and HPD

    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    show_mode: Bool
        If True the legend will show the mode(s) value(s), if false the mean(s)
        will be displayed
    kde_plot: Bool
        If True the posterior will be displayed using a Kernel Density Estimation
        otherwise an histogram will be used
    bins: integer
        Number of bins used for the histogram, only works when kde_plot is False
    ROPE: list or numpy array
        Lower and upper values of the Region Of Practical Equivalence
    comp_val: float
        Comparison value
        

    Returns
    -------

    post_summary : dictionary
        Containing values with several summary statistics

    """       

    post_summary = {'mean':0,'median':0,'mode':0, 'alpha':0,'hpd_low':0,
                   'hpd_high':0, 'comp_val':0, 'pc_gt_comp_val':0, 'ROPE_low':0,
                   'ROPE_high':0, 'pc_in_ROPE':0}

    post_summary['mean'] = round(np.mean(sample), roundto)
    post_summary['median'] = round(np.median(sample), roundto)
    post_summary['alpha'] = alpha

    # Compute the hpd, KDE and mode for the posterior
    hpd, x, y, modes = hpd_grid(sample, alpha, roundto)
    print(min(sample))
    post_summary['hpd'] = hpd
    post_summary['mode'] = modes

    ## Plot KDE.
    if kde_plot:
            plt.plot(x, y, color='k', lw=2)
    ## Plot histogram.
    else:
        plt.hist(sample, normed=True, bins=bins, facecolor='b', 
        edgecolor='w')

    ## Display mode or mean:
    if show_mode:
        string = '{:g} ' * len(post_summary['mode'])
        plt.plot(0, label='mode =' + string.format(*post_summary['mode']), alpha=0)
    else:
        plt.plot(0, label='mean = {:g}'.format(post_summary['mean']), alpha=0)

    ## Display the hpd.
    hpd_label = ''
    for value in hpd:
        plt.plot(value, [1, 1], linewidth=10, color='b')
        hpd_label = hpd_label +  '{:g} {:g}\n'.format(round(value[0], roundto), round(value[1], roundto)) 
    plt.plot(0, 0, linewidth=4, color='b', label='hpd {:g}%\n{}'.format((1-alpha)*100, hpd_label))
    ## Display the ROPE.
    if ROPE is not None:
        pc_in_ROPE = round(np.sum((sample > ROPE[0]) & (sample < ROPE[1]))/len(sample)*100, roundto)
        plt.plot(ROPE, [0, 0], linewidth=20, color='r', alpha=0.75)
        plt.plot(0, 0, linewidth=4, color='r', label='{:g}% in ROPE'.format(pc_in_ROPE))
        post_summary['ROPE_low'] = ROPE[0] 
        post_summary['ROPE_high'] = ROPE[1] 
        post_summary['pc_in_ROPE'] = pc_in_ROPE
    ## Display the comparison value.
    if comp_val is not None:
        pc_gt_comp_val = round(100 * np.sum(sample > comp_val)/len(sample), roundto)
        pc_lt_comp_val = round(100 - pc_gt_comp_val, roundto)
        plt.axvline(comp_val, ymax=.75, color='g', linewidth=4, alpha=0.75,
            label='{:g}% < {:g} < {:g}%'.format(pc_lt_comp_val, 
                                                comp_val, pc_gt_comp_val))
        post_summary['comp_val'] = comp_val
        post_summary['pc_gt_comp_val'] = pc_gt_comp_val
    plt.title('HPD of $\alpha$')
    plt.legend(loc=0, framealpha=1)
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    return post_summary

def data_flag(data):
    n=np.shape(data)[0]
    flagi=np.zeros(n).astype(int)
    flagj=np.zeros(n).astype(int)
    flag=np.zeros(n).astype(int)
    singlevi=[]
    singlevj=[]
    for i in range(n):
        nz=np.nonzero(data[i,:])
        flagi[i]=np.shape(nz[0])[0]
        if flagi[i]==1:
            wv=sum(data[i,:])
            singlevi.append((i,wv))
            if wv<3:
                flagi[i]=0
    for i in range(n):
        nz=np.nonzero(data[:,i])
        flagj[i]=np.shape(nz[0])[0]
        if flagj[i]==1:
            wv=sum(data[:,i])
            singlevj.append((i,wv))
            if wv<3:
                flagj[i]=0
        flag[i]=max(flagi[i],flagj[i])                
    return flag,singlevi,singlevj
def clear_Z(Z,flag):
    n=np.shape(Z)[0]
    for i in range(n):
        if flag[i]==0:
            Z[i,:]=0
    return Z
def cal_totalEv(X):
    nonzero=np.nonzero(X)    
    edge_num=np.shape(nonzero)[1]
    e_v=np.zeros(edge_num).astype(int)
    log_v=np.zeros(edge_num)
    for i in range(edge_num):
        V=e_v[i]=data[nonzero[0][i],nonzero[1][i]].astype(int)
        for j in range(V):
            log_v[i]+=np.log(j+1)
    return sum(e_v),sum(log_v),e_v
def cal_expo(X,Z,total_Ev,logTEV,e_v):
    nonzero=np.nonzero(X)    
    edge_num=np.shape(nonzero)[1]
    expo=np.zeros(edge_num)
    log_expo=np.zeros(edge_num)
    log_default=np.log(1E-5)    
    part1=0    
    for i in range(edge_num):
        expo[i]=sum(Z[nonzero[0][i],:]*Z[nonzero[1][i],:])
#        expoflag=0
#    expo_min=min(expo)
#    if expo_min<0:
#        print('expoflag_min')
#        expo+=np.abs(expo_min)
#        expoflag=1
#    for i in range(edge_num):
        if expo[i]!=0:
            log_expo[i]=np.log(expo[i])
        else:
            log_expo[i]=log_default   
        part1+=e_v[i]*log_expo[i]            
        
    return part1

def likelihood(X,Z,Rho,a,b,total_Ev,e_v,part2,part3,part4,part5):
    part1=cal_expo(X,Z,total_Ev,logTEV,e_v)
    Z=np.mat(Z)
    aa=np.dot(Z,Z.T)
    totalshareC=np.sum(np.reshape(aa,(aa.size,)))
    part6=(totalshareC+b)*Rho
    return part1-part2+part3+part5-part4-part6,totalshareC       

def sampleIBP(alpha, num_objects):  
    # Initializing storage for results
    result = np.zeros([num_objects, 1000]).astype(int)
    # Draw from the prior for alpha
    alpha_N=alpha/np.arange(1,num_objects+1)
    Knews = np.random.poisson(alpha_N)
    # Filling in first row of result matrix
    if Knews[0]==0:
        Knews[0]=1
    t=Knews[0]
    result[0, 0:t] = np.ones(t) #changed form np.ones([1, t])
    # Initializing K+
    K_plus = t
    for i in range(1, num_objects):
        for j in range(0, K_plus):
            mk=np.sum(result[0:i,j])
            nmk=i - mk
            logmk=1E-5
            lognmk=1E-5
            if mk!=0:
                logmk=np.log(mk)
            if nmk!=0:
                lognmk=np.log(nmk)            
            p = np.array([logmk - np.log(i+1), 
                          lognmk - np.log(i+1)])
            p = np.exp(p - max(p))
            if(np.random.uniform(0,1) < p[0]/np.sum(p)):
                result[i, j] = 1
            else:
                result[i, j] = 0
        t = Knews[i]
        x = K_plus + 1
        y = K_plus + t
        result[i, (x-1):y] = np.ones(t) #changed form np.ones([1, t])
        K_plus = K_plus+t
#        print("---ff is:",ff()-1)
    result = result[:, 0:K_plus]
#    for k in range(K_plus):
#        print(np.shape(np.nonzero(result[:,k])[0]))
    return list([result, K_plus])
def cal_Pois(alpha,num_objects,maxNew):    
    alphaN = alpha/num_objects
    pois = np.zeros(maxNew)     
    for new in range(maxNew):
        pois[new] = new*np.log(alphaN) - alphaN - np.log(math.factorial(new))
    return pois
def Gibbs_z_a(flag,Z,data,num_objects,a,b,alpha,maxNew,Rho,total_Ev,e_v,part2,part3,part4,part5,pois):

    for i in range(0, num_objects):
        if flag[i]!=0:
            P=np.zeros(2)  
    
            K_plus=np.shape(Z)[1]
            for k in range(K_plus):
                if (k>=K_plus):
                    break
                if K_plus==1:
                    break
                if np.sum(Z[:,k])-Z[i,k]==0:
    #                print('--------------------------------------merged---------%d++++'% e)
                    Z[:, k:(K_plus - 1)] = Z[:, (k+1):K_plus]
                    
                    K_plus = K_plus - 1
                    Z = Z[:, 0:K_plus]
                    continue
                Z[i,k] = 0
                [lik,_] = likelihood(data,Z,Rho,a,b,total_Ev,e_v,part2,part3,part4,part5)
                P[0] = lik + np.log(num_objects-np.sum(Z[:,k])) - np.log(num_objects)
                Z[i,k] = 1
                [lik,_] = likelihood(data,Z,Rho,a,b,total_Ev,e_v,part2,part3,part4,part5) 
                P[1] = lik + np.log(np.sum(Z[:,k])- 1) - np.log(num_objects)
                P = np.exp(P - max(P))
                U = np.random.uniform(0,1)
                if U<(P[1]/(np.sum(P))):
                    Z[i,k] = 1
                else:
                    Z[i,k] = 0   
                #Sample number of new features
            prob = np.zeros(maxNew)
            lik = np.zeros(maxNew)
            Tsc=np.zeros(maxNew)
    
            for new in range(maxNew): # max new features is 3
                ZZ = Z 
                if new>0:
                    newcol = np.zeros((num_objects, new)).astype(int)
                    newcol[i,:] = 1                                                          
                    ZZ = np.column_stack((ZZ, newcol))
                #Calculate the probability of kNew new features for object i
    
                [ll,tsc]=likelihood(data,ZZ,Rho,a,b,total_Ev,e_v,part2,part3,part4,part5)            
                lik[new] =ll
                Tsc[new] =tsc
                prob[new] = pois[new] + ll
            #normalize prob and select the most likely number of new features
            prob = np.exp(prob - max(prob))
            prob = prob/sum(prob)
            U = np.random.uniform(0,1,1)
            p = 0
            kNew=0
            for new in range(maxNew):
                p = p+prob[new]
                if U<p:
                    kNew = new
                    break
            #Add kNew new columns to Z and set the values at ith row to 1 for all of them
            if kNew>0:
    
                newcol = np.zeros((num_objects, kNew)).astype(int)
                newcol[i,:] = 1                                                      
                Z = np.column_stack((Z, newcol))
            K_plus = K_plus + kNew
            loglk= lik[kNew]
            totalshareC=Tsc[kNew]
    return Z,loglk,totalshareC

def harmi(num_objects):
    HN = 0
    for i in range(0, num_objects):
        HN = HN + 1/(i+1)
    return HN    
        
def simulateNetwork(Z,Rho):
    N=np.shape(Z)[0]
    data=np.zeros((N,N)).astype(int)
    for i in range(N):
        for j in range(N-i):
            if i!=j:
                rho=sum(Z[i,:]*Z[j,:])*Rho
#            print(rho)
                data[i,j]=np.random.poisson(rho)
#                data[j,i]=data[i,j]
#    print(data)
#    st=input("continu")
    return data
def tune_Z(Z):
    K_plus=np.shape(Z)[1]
    out=np.zeros(np.shape(Z)).astype(int)
    i=0    
    for k in range(K_plus):
        if (k>=K_plus):
            break
        if K_plus==1:
            break
        if np.sum(Z[:,k])>2:
            out[:,i]=Z[:,k]
            i+=1
    out = out[:,0:i]
    return out
def hpd(post):
#    sns.distplot(post)
    figsize(10.5, 3)
    HDP=np.percentile(post,[2.5,97.5])
    plt.plot(HDP,[0,0],label='HDP{:.2f}{:.2f}'.format(*HDP),linewidth=8,color='k')
    plt.legend(fontsize=16)
    plt.xlabel(r'$\alpha$',fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
def network_from_csv(csvfile):
    d = pd.read_csv(csvfile, usecols=['Source', 'Target', 'Weight'])
    d=np.array(d)
    n=d.shape[0]
    data=np.zeros((n,n)).astype(int)    
    for i in range(1,n):
        data[d[i,0],d[i,1]]=d[i,2]
    return data    
def summary_K(chain_K):
    k1=chain_K>1
    k2=k1*chain_K
    k2=k2.astype(int)
    k3=set(k2)
    k4=list(k3)
    if np.shape(k4)[0]>1:
        k4.pop(k4[0])
    zz=[]
    kk=[]
    a_f=[]
    kmaxl=0
    for _ in list(k4):
        temp=np.where(k2==_)[0]
        kk.append(temp)
        if np.shape(temp)[0]>kmaxl:
            kmaxl=np.shape(temp)[0]
            kmax=temp
            kkk=_
    times=0
    for i in list(kmax):
        zz.append(chain_Z[i])
        a_f.append(chain_alpha[i])
        times=times+1
    return times,kmax,zz,a_f,kkk
def plot_histo(chain_ks,chain_K,chain_alpha,chain_Rho):
    figsize(10.5, 16)
    ax = plt.subplot(411)
    ax.set_autoscaley_on(False)
    plt.title("Posterior distributions of K_plus and alpha")        
    plt.xlim([1,50])
    plt.xlabel("K_plus value")
    plt.ylabel("Density")
    plt.hist(chain_ks, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of K_plus", color="#A60628", normed=True)
    plt.legend(loc="upper right") 
    ax = plt.subplot(412)
    ax.set_autoscaley_on(False)
    plt.title("Posterior distributions of K_plus and alpha")        
    plt.xlim([1,50])
    plt.xlabel("K_plus value")
    plt.ylabel("Density")
    plt.hist(chain_K, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of K_plus", color="#A60628", normed=True)
    plt.legend(loc="upper right") 
    ax = plt.subplot(413)
    ax.set_autoscaley_on(False)
    plt.xlim([0.1,6.6])
    plt.xlabel("alpha value")
    plt.ylabel("Density")
    plt.hist(chain_alpha, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_2$", color="#467821", normed=True)
    plt.legend(loc="upper right")
    ax = plt.subplot(414)
    ax.set_autoscaley_on(False)
    plt.xlim([0.1,6.6])
    plt.xlabel("Rho value")
    plt.ylabel("Density")
    plt.hist(chain_Rho, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_2$", color="#467821", normed=True)
    plt.legend(loc="upper right")
    plt.show()
def plot_scatter(chain_K,chain_alpha):
    figsize(16.5, 12)
    xValue = list(range(0, maxI)) 
    yValue = chain_K[0:maxI] 
    plt.xlabel('x-value') 
    plt.ylabel('y-label')  
    plt.scatter(xValue, yValue)
    plt.show()
    plt.figure('Line fig') 
    xValue = list(range(0, maxI)) 
    yValue = chain_K[0:maxI] 
    ax = plt.gca() #设置x轴、y轴名称 
    ax.set_xlabel('x') 
    ax.set_ylabel('y') #画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标 #参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度 
    ax.plot(xValue, yValue, color='r', linewidth=1, alpha=0.6) 
    plt.show()
    
    figsize(16.5, 12)
    xValue = list(range(0, maxI)) 
    yValue = chain_alpha[0:maxI] 
    plt.xlabel('x-value') 
    plt.ylabel('y-label')  
    plt.scatter(xValue, yValue)
    plt.show()  
def graph_stat(X):
    nonzero=np.nonzero(X)    
    edge_num=np.shape(nonzero)[1]
    e_v=np.zeros(edge_num).astype(int)
    for i in range(edge_num):
        e_v[i]=data[nonzero[0][i],nonzero[1][i]].astype(int)    
    return edge_num,max(e_v),min(e_v),sum(e_v)    
def write_graph(step,file,net,ZZ):
    nodeFlag=0
    N=np.shape(net)[0]
    K=np.shape(ZZ)[1]
    ct=str(N)+' '+str(K)+' '+str(step)
    file.write(ct)
    file.write("     \n ")
    for j in range(N):
        nb=np.nonzero(net[j])
        nbs=np.shape(nb)[1]
        nz=np.nonzero(ZZ[j])
        nzs=np.shape(nz)[1]
        if nbs>0:
            nodeFlag=1
            nbb=' '
            for nnb in nb[0]:
                ncb=str(nnb)+' '
                nbb=nbb+ncb
            if nzs>0:
                for nnz in nz[0]:
                    ncz=str(nnz)+' '
                    nbb=nbb+ncz
        else:
            nbb=' '
        content=str(nodeFlag)+' '+str(nbs)+nbb
        file.write(content)
        file.write("  \n")                            
def write_mat(seed):
    for i in list(seed):
        ZZ=chain_Z[i]
        K=chain_K[i]
        Rho=chain_Rho[i]
        alpha=chain_alpha[i]
        net=simulateNetwork(ZZ,Rho)
        fn=str(K)+'.mat'
        scipy.io.savemat(fn,{'a':a,'b':b,'Rho':Rho,'alpha':alpha,'Z':ZZ,'net':net})
#    if chain_K[i]==6:
#        print('6----',i)
#    if chain_K[i]==7:
#        print('7----',i)
#    if chain_K[i]==8:
#        print('8----',i)
def plotl(d):
    figEx1=plt.figure(figsize=(4, 4))
    n, bins, patches = plt.hist(x=d, bins='auto', color='g',
    alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('$\lambda$ Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of $\lambda$')

    plt.grid(True)

    filename='K'
    plt.show()
    figEx1.savefig(filename+'.png')
    plt.close()
def plota(d):
    figEx2=plt.figure(figsize=(4, 4))
    n, bins, patches = plt.hist(x=d, bins='auto', color='b',
    alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('alpha Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of alpha')

    plt.grid(True)

    filename='K'
    plt.show()
    figEx2.savefig(filename+'.png')
    plt.close()
def plotk(d):
    figEx3=plt.figure(figsize=(4, 4))
    n, bins, patches = plt.hist(x=d, bins='auto', color='b',
    alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('K Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of $K_s$')

    plt.grid(True)

    filename='K'
    plt.show()
    figEx3.savefig(filename+'.png')
    plt.close()
def write_graphs(step,file,net,ZZ):
    nodeFlag=0
    N=np.shape(net)[0]
    K=np.shape(ZZ)[1]
    ct=str(N)+' '+str(K)+' '+str(step)

    file.write(ct)
    file.write("     \n ")
    net+=net.T
    for j in range(N):
        nodeFlag=0
        nb=np.nonzero(net[j])
        nbs=np.shape(nb)[1]
        nbb=' '
        if nbs>0:
            nodeFlag=1
            
            for nnb in nb[0]:
                ncb=str(nnb)
                nbb=nbb+' '+ncb
        for nnz in range(K):
#                print(ZZ[j,nnz])
            ncz=str(ZZ[j,nnz])
            nbb=nbb+' '+ncz

        if K==14:
            nbb=nbb+' '+str(0)+' '+str(0)
        if K==15:
            nbb=nbb+' '+str(0) 
        if nbs==0:
            nbb=''           
        content=str(nodeFlag)+' '+str(nbs)+nbb
        file.write(content)
        file.write("  \n")
def counter():
    x=0
    def add():
        nonlocal x
        x+=1
        return x
    return add
def write_data(data,ZZZZ):
    f2n= 'originalGraph.txt'
    file=open(f2n,'a')
    N=np.shape(data)[0]    
    
    ct=str(N)+' \n'
    file.write(ct)
    for j in range(N):
        nodeFlag=0
        nb=np.nonzero(data[j])
        nbs=np.shape(nb)[1]
        
        
        K=np.shape(ZZZZ)[1]
        if nbs>0:
            nodeFlag=1
            nbb=' '
            for nnb in nb[0]:
                ncb=str(nnb)+' '
                nbb=nbb+ncb
        for nnz in range(K):
#                print(ZZ[j,nnz])
            ncz=str(ZZZZ[j,nnz])
            nbb=nbb+' '+ncz
        nbb=nbb+' '+str(0)

        content=str(nodeFlag)+' '+str(nbs)+nbb
        file.write(content)
        file.write("  \n")
a=np.random.gamma(1,1)
b=np.random.gamma(1,1)
#mat=scipy.io.loadmat('yur.mat')
#matr=scipy.io.loadmat('yu.mat')
#a=mat['a']
#b=mat['b']
#alpha=mat['alpha']
#data = np.loadtxt("eron_net.txt", delimiter=",")
Rho = np.random.gamma(a,b)
#Rho=mat['Rho']
#Z=mat['ZZ']
#data=mat['net']
csvfile='lesmis.csv'
data=network_from_csv(csvfile)
#data=data+data.T
[flag,svi,svj]=data_flag(data)
#scipy.io.savemat('net.mat',{'net':net})
#dataI=scipy.io.loadmat('net.mat')
N=np.shape(data)[0]
#N=30
HN=harmi(N)
alpha = np.random.gamma(24,1/(1+HN))
[Z, K_plus] = sampleIBP(alpha, N)
Z=clear_Z(Z,flag)
#K_plus=np.shape(Z)[1]
ZZZ=tune_Z(Z)
print(ZZZ)
st=input("continu")
#data=simulateNetwork(ZZZ,Rho)
print(data)
st=input("continu")
num_objects=np.shape(data)[0]
#Set number of iterations
maxI = 10000
Burnin =4000
gNumber=maxI-Burnin
[total_Ev,logTEV,e_v]=cal_totalEv(data)
#Set truncation limit for max number of sampled latent features
#Set storage arrays for sampled parameters
chain_Z =[]
chain_K = np.zeros(gNumber).astype(int)
chain_alpha = np.zeros(gNumber)
chain_Rho = np.zeros(gNumber)
#Initialize parameter values
#alpha = np.random.gamma(1,1/(1+HN))
#[Z, K_plus] = sampleIBP(alpha, num_objects)
loglkMax=-np.inf
maxNew=5
pois=cal_Pois(alpha,num_objects,maxNew)
part2=logTEV
chain_graphs=[]
part4=gammaln(a)
part5=a*np.log(b) 
[ne,maxe,mine,sume]=graph_stat(data)
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename= 'save'+nowTime+'.txt'
#f2= 'stat'+nowTime+'.txt'
filename= 'YU_l44.txt'
file=open(filename,'a')
#file=open(f2,'a')
#file.write(str(gNumber)+" \n")
#f2.write('*** '+str(ne)+' '+str(maxe)+' '+str(mine)+' '+str(sume)+" \n")
#nodeFlag=0
chain_a= np.zeros(gNumber)
chain_b= np.zeros(gNumber)
chain_ks=[]
trueK=14
c1=counter()
c2=counter()
c3=counter()
for step in range(0, maxI):
    if step<Burnin:
        print("At iteration", step, ": K_plus is", K_plus, ", alpha is", alpha)
    
    part3=(total_Ev+a-1)*np.log(Rho)
    [Z,loglk,totalshareC]=Gibbs_z_a(flag,Z,data,num_objects,a,b,alpha,maxNew,Rho,total_Ev,e_v,part2,part3,part4,part5,pois)  
    K_plus=np.shape(Z)[1]
    Rho=np.random.gamma(total_Ev+a,1./(totalshareC+b))
    ZZ=tune_Z(Z)
    K=np.shape(ZZ)[1]
    chain_ks.append(K) 
    if step>=Burnin:
        
        if K>0:
            chain_alpha[step-Burnin] = alpha
            chain_Rho[step-Burnin] = Rho
            chain_Z.append(ZZ)
#            chain_ll[step-Burnin] = loglk         
            chain_K[step-Burnin]=K
            chain_a[step-Burnin] = total_Ev+a
            chain_b[step-Burnin] = totalshareC+b
            if K==trueK-1 or K==trueK or K==trueK+1:
                net=simulateNetwork(ZZ,Rho)
##            graph=adaj_network(net)
#            chain_graphs.append(net)
#            [ne,maxe,mine,sume]=graph_stat(net)
#            f2.write(str(step)+' '+str(ne)+' '+str(maxe)+' '+str(mine)+' '+str(sume)+" \n")
                write_graph(step,file,net,ZZ)
            if K==trueK-1:
                c1()
            if K==trueK:
                c2()
            if K==trueK+1:
                c3()
        print("At iteration", step, ": K is", K, ", alpha is", alpha)

    alpha = np.random.gamma(1 + K_plus, 1/(1+HN)) 
#def plot_line(data,start,end,filename):
#    figx=plt.figure(figsize=(16.5, 12))
#    plt.figure('Line fig') 
#    xValue = list(range(start,end)) 
#    yValue = data[start:end] 
#    ax = plt.gca() #设置x轴、y轴名称 
#    ax.set_xlabel('iteration') 
#    ax.set_ylabel('K') #画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标 #参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度 
#    ax.plot(xValue, yValue, color='b', linewidth=1, alpha=0.6) 
#    plt.show()
#    figx.savefig(filename+'.png')
#    plt.close() 
#file.close()         
#f2.close() 
#scipy.io.savemat('yu'+nowTime+'.mat',{'a':total_Ev+a,'b':totalshareC+b,'Rho':Rho,'alpha':alpha,'ZZ':ZZ,'net':net})
#hpd(chain_alpha)
#plot_histo(chain_ks,chain_K,chain_alpha,chain_Rho)
#plot_scatter(chain_K,chain_alpha)
#for k in range(times):
#    final_Z+=zz[k]
#nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#fn='lesmis.mat'
#scipy.io.savemat(fn,{'a':a,'b':b,'r':chain_Rho,'alpha':chain_alpha,'Z':chain_Z,'net':chain_graphs})
#final_Z=(final_Z>0).astype(int)
#final_rho=final_rho/times
#Z=chain_Z[-1]
#[times,kmax,zz,a_f,kkk]=summary_K(chain_K)
#Z_result=tune_Z(Z_result)
#print("Finally: most K_plus is", kkk,"occur times: ",times, "alpha is", a_result)
#print("Z is",Z_result) 
#print("K_plus is", np.shape(Z_result)[1])
c13=c1()-1
c23=c2()-1
c33=c3()-1      
file.write('=== '+str(c13)+' '+str(c23)+' '+str(c33)+' '+str(c13+c23+c33)+" \n")
file.close() 
plotk(chain_ks)
plotl(chain_Rho)

#dill.dump_session('yuseed.pkl')
seed=[4978,4988,4990]
#write_mat(seed)
cc=chain_alpha

plot_post2(cc, alpha=0.05, show_mode=True, kde_plot=True, bins=4, 
    ROPE=[min(cc),max(cc)], comp_val=np.median(cc), roundto=2)
#ccc=chain_Rho
#
#plot_post1(ccc, alpha=0.05, show_mode=True, kde_plot=True, bins=4, 
#    ROPE=[min(ccc),max(ccc)], comp_val=np.median(ccc), roundto=2)
#plota(chain_alpha)
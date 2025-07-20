import pandas
import math
import scipy.stats
import numpy as np

def crrn(n, res=1000):
    ''' Calculates CRRN without ties(!).
    Needs the following input:
    - n: number of rows, practically given by len(df)
    - res: resolution, or number of points on the normal distribution (ignored for n<7)
    '''
    
    [XX1,Med,XX19]=[None,None,None]
    
    if n<2:
        raise CRRNError('Smallest number of objects is 2.')
    elif n==2:
        x=[0,50]
        y=[50,50]
    elif n==3:
        x=[0,50,100]
        y=[16.67,33.33,50]
    elif n==4:
        x=[0,25,50,75,100]
        y=[4.17,12.50,29.17,37.50,16.67]
    elif n==5:
        x=[0,16.67,33.3,50,66.67,83.33,100]
        y=[0.83,3.33,10.00,20.00,29.17,20.00,16.67]
    elif n==6:
        x=[0,11.11,22.22,33.33,44.44,55.56,66.67,77.78,88.89,100]
        y=[0.14,0.69,2.50,6.39,12.92,19.03,20.56,18.89,13.89,5.00]
        
    elif n<16:
        ''' Cumulative frequencies.'''
        mt = {7:62.74,8:62.74,9:64.40,10:64.21,11:65.19,12:64.99,13:65.64,14:65.48,15:65.87}[n]
        st = {7:21.62,8:19.46,9:18.36,10:17.02,11:16.23,12:15.30,13:14.70,14:14.03,15:13.59}[n]
        
        x = np.linspace(0.0, 100.0, res)
        y = [(math.tanh( (i-mt)/st ) - math.tanh( (0-mt)/st ) ) / 2 for i in x]
        
        XX1 = next(i for (i,j) in zip(x,y) if j > 0.05)
        Med = next(i for (i,j) in zip(x,y) if j > 0.5)
        XX19 = next(i for (i,j) in zip(x,y) if j > 0.95)
        
    elif n<30:
        ''' This might correspond to the ClassTH=1 case of the with-ties distributions!'''
        mean={16:66.713,17:67.021,18:66.734,19:66.952,20:66.721,21:66.913,22:66.786,23:66.908,24:66.714,
              25:66.883,26:66.800,27:66.856,28:66.834,29:66.845}[n]
        std={16:11.049,17:10.842,18:10.392,19:10.153,20:9.871,21:9.596,22:9.349,23:9.163,24:8.854,
             25:8.722,26:8.561,27:8.409,28:8.247,29:8.074}[n]

        x = np.linspace(0.0, 100.0, res)
        y = scipy.stats.norm.pdf(x,mean,std)
        
        cumFreq = scipy.stats.norm.cdf(x,mean,std)
        
        XX1 = next(i for (i,j) in zip(x,cumFreq) if j > 0.05)
        Med = next(i for (i,j) in zip(x,cumFreq) if j > 0.5)
        XX19 = next(i for (i,j) in zip(x,cumFreq) if j > 0.95)
        
    else:
        [a,b] = [2.3796,-0.3509]

        mean = 66.667
        std = 100 / ( b + a*math.sqrt(n) )
        
        x = np.linspace(0.0, 100.0, res)
        y = scipy.stats.norm.pdf(x,mean,std)
        
        cumFreq = scipy.stats.norm.cdf(x,mean,std)
        
        XX1 = next(i for (i,j) in zip(x,cumFreq) if j > 0.05)
        Med = next(i for (i,j) in zip(x,cumFreq) if j > 0.5)
        XX19 = next(i for (i,j) in zip(x,cumFreq) if j > 0.95)
        
    return [x,y,XX1,Med,XX19]

def srd_core(df,ref,normalize=True):
    ''' Shortcut for the core SRD calculation.'''
    
    refVector=calc_ref(df,ref)
    
    dfr=df.rank()
    rVr=refVector.rank()
    diffs=dfr.subtract(rVr,axis=0)

    srd_values=diffs.abs().sum()
    
    if normalize==True:
        k=math.floor(len(df)/2)
        if len(df)%2 == 0:
            maxSRD = 2*k**2
        else:
            maxSRD = 2*k*(k+1)
            
        srd_values=srd_values/maxSRD*100
    
    return srd_values

def calc_ref(df,ref,axis=1):
    '''Select reference column or produce one with a data fusion method.'''
    
    
    if axis==1:
        if ref in df.columns:
            refVector=df[ref]
        elif ref in ['min','max','mean','median']:
            refVector={
                'min': df.min(axis=axis),
                'max': df.max(axis=axis),
                'mean': df.mean(axis=axis),
                'median': df.median(axis=axis),
                        }[ref]
        else:
            raise ReferenceError('Column not found.')
    
    #'''Supports row-wise reference selection/calculation, but this is not yet implemented in srd_core!'''
    elif axis==0:
        if ref in df.index:
            refVector=df.loc[ref]
        elif ref in ['min','max','mean','median']:
            refVector={
                'min': df.min(axis=axis),
                'max': df.max(axis=axis),
                'mean': df.mean(axis=axis),
                'median': df.median(axis=axis),
                        }[ref]
        else:
            raise ReferenceError('Row not found.')
    
    return refVector
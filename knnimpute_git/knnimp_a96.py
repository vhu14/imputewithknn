__author__ = 'victor'

import numpy as np
import scipy as sp
import random 
import sys
import math
import warnings


def imputeknn(dat, k=10,rowmax=0.5,colmax=0.8,maxp=1500,seed=362436069):

    np.random.seed(seed)
    x = dat
    p = x.shape[0]
    colnas = np.dot(np.ones(p),np.isnan(x))
    if ((colnas>colmax*p).any()):
       sys.exit('a column has more than'+(colmax*100)+'missing values!')

    ximp = knnimp(x,k, maxmiss=rowmax, maxp=maxp)
    return (ximp, seed)
	
def knnimp(x,k=10,maxmiss=0.5,maxp=1500):
	pn=x.shape
	p=pn[0]
	n=pn[1]
	imiss= np.isnan(x)
	x[imiss]=0
	irmiss=np.squeeze(np.dot(imiss,np.ones(n)))
	imax=math.floor(maxmiss*n)
	imax=irmiss>imax
	simax=sum(imax)
	if(simax>0):
		row_ind=np.array(imax)[0]
		irmiss=irmiss[:,-row_ind]
		imissomit=imiss[row_ind,:].copy()
		imiss=imiss[-row_ind,:]
		xomit=x[row_ind,:].copy()
		x=x[-row_ind,:]
		dd=imax
		p=p-simax
	if(p<=maxp):
		ximp=knnimp_internal(x,k,imiss,irmiss,p,n,maxp)
	else:
		ximp=knnimp.split(x,k,imiss,irmiss,p,n,maxp)
	imissnew=np.isnan(ximp)
	newmiss=imissnew.any()
	if((simax>0) or newmiss):
		xbar=meanmiss(x,range(0,x.shape[0]),np.isnan(x))
		if(newmiss): ximp=meanimp(ximp,imissnew,xbar)
		if(simax>0):
			xomit=meanimp(xomit,imissomit,xbar)
			xout=np.zeros(pn)
			xout[-row_ind,:]=ximp.copy()
			xout[row_ind,:]=xomit
			ximp=xout
	return (ximp)
	

def knnimp_internal(x,k,imiss,irmiss,p,n,maxp): 
	
	if(p<=maxp): 
		ximp,imiss2 = knncompute(x,p,n,imiss,irmiss,k)
		ximp[imiss2==2]=np.nan
	else: ximp = x
	return (ximp)	

def knncompute(x, p,n,imiss,irmiss,kn): 
	m = min(kn+ 1, n/2.0) 
	imiss2 = np.zeros(shape=(p,n))
	imiss2[np.where(imiss)]=1
	for i in range(0,p): 
		if(irmiss[i]>0): 
			x0 = np.squeeze(np.asarray(x[i,:]))
			imiss0 = np.squeeze(np.asarray(imiss[i,:]))
			
			workp = misdis(x0,x,p,n,imiss0,imiss)
			pos = porder(m,workp,p)
			workn, iworkn = misave(x, n, imiss0, imiss, pos, kn)
			for k in range(0,n): 
				if(iworkn[0,k]>0):
					x[i,k]=workn[0,k] ## update x values, it is ximp now
					if(iworkn[0,k]==2): 
						imiss2[i,k]=2 
	return (x, imiss2)
	
## compute the mean nn for missing coords
def misave(x, n, imiss0, imiss, pos, kn): 
	x0 = np.zeros(shape=(1,n))
	iworkn2 = np.zeros(shape=(1,n))
	iworkn2[0,np.where(imiss0)]=1
	kn=int(kn)
	
	for k in range(0,n): 
		x0[0,k]=0.0 
		if(not imiss0[k]): continue
		ktot = 0 
		for j in range(0,kn): 
			jj = pos[j]
			if(imiss[jj,k]): continue
			x0[0,k] = x0[0,k] + x[jj, k]
			ktot = ktot + 1
	
		if(ktot>0): x0[0,k]=x0[0,k]/ktot
		else: iworkn2[0,k]=2 	
	return (x0, iworkn2)

##  find the distance and number of neighbors	
def misdis(x0,x,p,n,imiss0,imiss):
	dismax = 1.0e10
	workp = np.zeros(p)
	iworkp = np.zeros(p)
	for k in range(0,n):
		if (not imiss0[k]):
			for j in range(0,p):
				if (not imiss[j,k]):
					workp[j]=workp[j]+(x0[k]-x[j,k])**2
					iworkp[j] = iworkp[j]+1
	for j in range(0,p): 
		if(iworkp[j]>0): workp[j]=workp[j]/iworkp[j]
		else: workp[j]=dismax
		
	return (workp)
	
##  find the small to large ordering of distance
def porder(m,workp,p): 
	pos = np.zeros(m)
	for j in range(0,m): 
		pos[j] = p+1
	nndist = np.zeros(m)
	for j in range(0,m): 
		nndist[j] = 1e10
	for j in range(0,p):
		flag = True
		if(j< m): 
			for k in range(0, j): 
				if (workp[j] < nndist[k]): 
					for k1 in range(j-1, k-1, -1): 
						nndist[k1+1] = nndist[k1]
						pos[k1+1] = pos[k1]
					nndist[k] = workp[j]
					pos[k] =j 
				
					flag = False
					break
			if(flag): 
				nndist[j] = workp[j]
				pos[j] = j 
		else: 
			if(workp[j]>=nndist[m-1]): continue 
			for k in range(0, m): 
				if(workp[j]<nndist[k]): 
					for k1 in range(m-2, k-1, -1): 
						nndist[k1+1]= nndist[k1]
						pos[k1+1] = pos[k1]
					nndist[k] = workp[j]
					pos[k]=j 
					break
	return (pos) 
	
def meanmiss(x, index, imiss):

    pn = x.shape
    p = pn[0]
    n = pn[1]
    x[imiss] = 0
    x0, imiss0 = misave(x, n, np.ones(n), imiss,index, len(index))
    x0[imiss0==2] = np.nan
    return (x0)
	
########################################################################
######   Apply knn_split when p is large   #############################
########################################################################

def knnimp_split(x,k,imiss,irmiss,p,n,maxp): 
	size,clus = twomeansmiss(x,np.isnan(x))
	for i in range(0,2):
		p = size[i]
		index = (clus==i)
		if(p<=k): 
			imiss_idx = np.isnan(x[index,:])
			x[index,:] = meanimp(x[index,:], imiss_idx, meanmiss(x[index,:],index, imiss_idx))
		else: 
			x[index,:] = knnimp_internal(x[index,:],k,imiss[index,:],irmiss[0,index],p,n,maxp)
	return (x)


def twomeansmiss(x,imiss,imbalance=.2,maxit=5,eps=0.001): 
	
	p = x.shape[0]
	n = x.shape[1]
	if(imiss.any()): x[imiss]=0
	starters = random.sample(range(0,p),2)
	ratio, iters, nsize, clus1 = twomis(x,p,n,imiss,starters, imbalance*5, maxit,eps)
	clus2 = np.zeros(p)
	clus2[np.unique(clus1[range(0,math.floor(nsize[1])),1]).astype(int)] = 1 
	return (clus2, ratio, iters, nsize)
	
def twomis(x,p,n,imiss,starters, balancefactor, maxit=5,eps=0.001): 

	x0 = np.zeros(shape=(n,2))
	imiss0 = np.zeros(shape=(n,2))
	clust = np.zeros(shape=(p,2))
	nsize = np.zeros(2)
	iworkn = np.asarray([True]*n)
	
	if(maxit<1): maxit = 6
	for i in range(0,2): 
		for j in range(0,n): 
			x0[j,i]=x[starters[i],j].copy()
			if(imiss[starters[i],j]): imiss0[j,i]= 1
	iters = 0 
	ratio = 0.1 
	distn = np.zeros(shape=(p,2))
	
	while (iters < maxit and ratio > eps): 
		iters = iters + 1
		for i in range(0,2):  
			## update distance each dimension to the centroid of two clusters
			distn[:,i] = misdis(np.squeeze(np.asarray(x0[:,i])), x, p,n,np.squeeze(np.asarray(imiss0[:,i])),imiss)
			nsize[i]=0 
		dnew = 0.0 
		## use distance for labeling update 
		for j in range(0,p): 
			if(distn[j,0]<balancefactor*distn[j,1]): imax =0 
			else: imax = 1 
			nsize[imax]=nsize[imax]+1 
			clust[nsize[imax]-1, imax]=j  
			dnew = dnew + distn[j, imax]
			
		if(dnew!=0.0): 
			if(iters==1): dold = dnew* 0.1 
			ratio = abs(dnew-dold) / dold 
			dold = dnew 
		for i in range(0,1): 
			## use cur distance to update x0 & imiss0	(x0 is the centroid)
			x0[:,i], iworkn2 = misave(x,n,iworkn,imiss, clust[:,i], nsize[i])
			for j in range(n): 
				if(iworkn2[0,j]==2): imiss0[j,i]=1
				else: imiss0[j,i]=0
			
	return (ratio, iters, nsize, clust)	
	
# def meanimp(x,imiss=np.isnan(x),xbar=meanmiss(x,imiss=imiss)):
def meanimp(x, imiss,xbar):
    nr = x.shape[0]
    if(nr>1):
        x[imiss] = np.outer(np.ones(nr),xbar)[imiss]
    return (x)

####  testing  #####

seed=14578396
np.random.seed(seed)
x1 = np.zeros(shape=(100, 40))
p1 = x1.shape[0]
p2 = x1.shape[1]
for i in range(0,p2): 
	x1[:,i] = np.asarray([math.floor(a) for a in np.random.sample(p1)*100])
	x1[np.where(x1[:,i]>85),i] = np.nan
	
x1_imputed, seed = imputeknn(x1)

x2 = np.zeros(shape=(2183,60))
p1 = x2.shape[0]
p2 = x2.shape[1]
for i in range(0,p2): 
	x2[:,i] = np.asarray([math.floor(a) for a in np.random.sample(p1)*100])
	x2[np.where(x2[:,i]>85),i] = np.nan
	
x2_imputed, seed = imputeknn(x2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:52:10 2018

@author: berend-christiaan

Functions for calculating point cloud features

"""

import pandas as pd

import numpy as np

import scipy

from scipy import spatial

import math

from math import ceil

import time

import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor

import pickle

import sys

import gc

import queue

import laspy

from laspy.file import File

import os

import tqdm

from tqdm import tqdm


def par_calc(start,stop,points,points_remain_prop,tree,chunk,k,clf,features,tensor_filter=False,classification=True):
   
    #%% Setting up constants and output file
    a = (4/3)*math.pi
    
    
    
    output = pd.DataFrame(np.array(np.zeros([stop-start,len(features)])),index = range(start,stop), columns=features)
    #output["return_number"] = points_remain_prop.loc[start:stop,'return_number']
    #output["number_of_returns"] = points_remain_prop.loc[start:stop,'number_of_returns']
    #output["intensity"] = points_remain_prop.loc[start:stop,'intensity']
    
    if classification == True:
        output['classification'] = np.zeros([stop-start,1])
    
    
    print(output.index)
    
    chunk_range = [x for x in range(0,chunk)]
    
    #%% Looping over the to-be-evaluated points in a semi-vectorized approach
    
    for x in range(start,stop,chunk):
        
        #Since there are more indexes entries than stop gives, chunk will surpass the stop mark (in 9/10 threads). 
        #Re-evaluation of chunksize.
        
        if x + chunk > stop:

            chunk = stop - x
            chunk_range = [x for x in range(0,chunk)]
         
            
        

        #Find nearest neighbours dinstance (dd) and index (ii)
        #print("KDtree Query")
        
        #The distances_upperbound gives in theory the control we need. Max K with max distance. However, results produce make it near
        #impossible to vectorize any of the approaches. As such a simple formula is introduced which can be tweaked. 
        #The K will be determined based upon a reference K and point density. 
        #The original methodology has been tested succesfully with a K=50 with an point density of 16 pt/m2.
        #Considering that lower point density literally means less points representing equal distances 
        #Instead of having 50 points describing a distance of X, lower point density means that there are fewer points over the same
        #length of X. As such:
        # K = 3.125 * average point density. 50 is found by 3.125*16pt/m2
        #Furthermore tests have shown when moving the upper boundary up or down, nearly all points have the same amount of NN. Usually
        #differentiating by 1 or 2 extra or less neighbours.
        knndd,knnii = tree.query(points[x:x+chunk],k)  
        #test = scipy.spatial.cKDTree.query_ball_tree(tree,points[x:x+chunk],1.5) #tree.query reports 1.37 as max distance, 1.5 is conservative.
        
        neighbour_group_xyz = points[knnii.astype(np.int64)] # Extract the xyz information
     
        #%%
        #Determine which index entries of this chunk should be calculatated or dropped based upon filter planarity & sphericity
        #These filters can be set to whichever ofcourse depending on the intended use. Current filter is based on 
        #Lucas, C 2017 "Delineating linear vegetation elements in agricultural landscapes using LiDAR point clouds"
        #Incase a different combination of features is required to filter beforehand, please contact me at berend_wijers@msn.com
        #Before any of the structure tensors can be calculated, eigenvalues/vectors need to be derived from the group of points.
        
        #Covariance

        N = neighbour_group_xyz.shape[1] #1->2

        m1 = neighbour_group_xyz - neighbour_group_xyz.sum(1,keepdims=1)/N #1->2

        #Produce a 3d matrix with shape (chunk,2,2)
        
        neighbour_group_cov = np.einsum('ikj,ikl->ijl',m1,m1) /(N - 1) #ikj, ikl -> ijl ---> ijk, ilk, ijl
        
        #Retrieve the eigenvalues and vectors of each covariance matrix. Note, each covariance matrice corresponds to a
        #point to be evaluated accompaniying its K closest neighbours 
        
        knn_eigval, knn_eigvec = np.linalg.eig(neighbour_group_cov)
        
        #retrieve the IDX of the sorted eigenvalues

        idx = np.fliplr(np.array(knn_eigval).argsort())

        #Retrieve the sorted eigenvalues

        knn_eigval_sort = knn_eigval[np.arange(len(knn_eigval))[:,None], idx]
        
        #Sum of eigenvalues

        knn_eigval_sum = np.sum(knn_eigval_sort, axis=1)
        
        #Normalize the sorted eigenvalues
        
        knn_eigval_sort  = (knn_eigval_sort / np.tile(np.array([knn_eigval_sum]).transpose(), (1, 3)))
        

        #Return the normal vectors

        knn_eigvec_norm_vec = knn_eigvec[:,:,2]
        
        #Calculate structure tensors based upon the sorted eigenvalues / vectors
        
        
        #ENABLE FOR IN SITU FILTERING
        if tensor_filter == True:
            print("Filter is on!")    
            
            chunk_res = pd.DataFrame(np.array(np.zeros([chunk,2])),columns=['planarity', 'sphericity'])
            
            chunk_res['planarity'] = (knn_eigval_sort[:,1]-knn_eigval_sort[:,2])/knn_eigval_sort[:,0]
            #output.ix[x:x+chunk-1,6] = (knn_eigval_sort[:,1]-knn_eigval_sort[:,2])/knn_eigval_sort[:,0]
                #Sphericity
            chunk_res['sphericity'] = knn_eigval_sort[:,2]/knn_eigval_sort[:,0]
        
            idx_keep = np.where(np.logical_and(chunk_res["sphericity"]>0.05, chunk_res["planarity"]<0.7))[0]
         
            idx_drop = np.where(np.isin(chunk_range,idx_keep,assume_unique=True,invert = True))[0]
        
            idx_drop = idx_drop + x
            
            output.drop(idx_drop,axis=0,inplace=True)
            
            #update list from chunk idx to output idx
            idx_keep = [i+x for i in idx_keep]
            
            knn_eigval_sum = knn_eigval_sum[idx_keep]
            knn_eigval_sort = knn_eigval_sort[idx_keep]
            knn_eigvec_norm_vec = knn_eigvec_norm_vec[idx_keep]
            neighbour_group_xyz = neighbour_group_xyz[idx_keep]
            knndd = knndd[idx_keep]
         
            
            #Delta_Z
            neighbour_group_z = neighbour_group_xyz[:,:,2]
            if "delta_z" in features:
                output.loc[idx_keep,"delta_z"] = np.amax(neighbour_group_z,axis=1) - np.amin(neighbour_group_z,axis=1)
            
            #StdZ
            if "std_z" in features:
                output.loc[idx_keep,"std_z"] = np.std(neighbour_group_z,axis=1)
            
            #Local Radius
            Rl = np.amax(knndd,axis=1)
            if "radius" in features:
                output.loc[idx_keep,"radius"] = Rl

            #Local Point Density
            if "density" in features:
                output.loc[idx_keep,"density"] = k / (a * (Rl**3))

            #Norm Z
            if "norm_z" in features:
                output.loc[idx_keep,"norm_z"] = abs(knn_eigvec_norm_vec[:, 2])
            #Linearity
            if "linearity" in features:
                output.loc[idx_keep,"linearity"] = (knn_eigval_sort[:,0]-knn_eigval_sort[:,1])/knn_eigval_sort[:,0]
            #planarity
            
            output.loc[idx_keep,"planarity"] = chunk_res.ix[idx_keep,'planarity']
            #Sphericity
            
            output.loc[idx_keep,"sphericity"] = chunk_res.ix[idx_keep,'sphericity']
            #Omnivariance
            if "omnivariance" in features:
                output.loc[idx_keep,"omnivariance"] = scipy.cbrt(knn_eigval_sort[:,0]*knn_eigval_sort[:,1]*knn_eigval_sort[:,2])
            #anisotropy
            if "anisotropy" in features:
                output.loc[idx_keep,9] = (knn_eigval_sort[:,0] - knn_eigval_sort[:,2]) / knn_eigval_sort[:,0]
            #eigenentropy
            if "eigenentropy" in features:
                output.loc[idx_keep,"eigenentropy"] =  (-1 *((knn_eigval_sort[:,0] * np.log(knn_eigval_sort[:,0])) +

                      (knn_eigval_sort[:,1] * np.log(knn_eigval_sort[:,1])) +

                      (knn_eigval_sort[:,2] * np.log(knn_eigval_sort[:,2]))))     
            #Sum eigenvalues
            if "sum_eigenvalues" in features:
                output.loc[idx_keep,"sum_eigenvalues"]=knn_eigval_sum
            #Local Surface Variation
            if "curvature" in features:
                output.loc[idx_keep,"curvature"] = knn_eigval_sort[:,2] / knn_eigval_sum
            
            #Normalized returns
            #output.loc[idx_keep,"return_number"] = points.ix[idx_keep,4]
            #output.loc[idx_keep,"number_of_returns"] = points.ix[idx_keep,5]
            
            if "norm_returns" in features:
                output.loc[idx_keep,"norm_returns"] = output.loc[idx_keep,"return_number"] / output.loc[idx_keep,"number_of_returns"]
            #Classification
            if classification == True:
                if len(idx_keep) >= 1:
                    print("Classifying!")
                    #print(output.ix[idx_keep])
                    output.loc[idx_keep,"classification"] = clf.predict(output.loc[idx_keep,features])
            
           
        
        
        
        else:
            #print("Filter is off!")
                        
            #Delta Z
            neighbour_group_z = neighbour_group_xyz[:,:,2]
            if "delta_z" in features:
                output.loc[x:x+chunk-1,'delta_z'] = np.amax(neighbour_group_z,axis=1) - np.amin(neighbour_group_z,axis=1)
            #Std Z
            if "std_z" in features:
                output.loc[x:x+chunk-1,"std_z"] = np.std(neighbour_group_z,axis=1)  
            #Local Radius
            Rl = np.amax(knndd,axis=1)
            if "radius" in features:
                output.loc[x:x+chunk-1,"radius"] = Rl
            #Local Point Density
            if "density" in features:
                output.loc[x:x+chunk-1,"density"] = k / (a * (Rl**3))
            #Norm Z
            if "norm_z" in features:
                output.loc[x:x+chunk-1,"norm_z"] = abs(knn_eigvec_norm_vec[:, 2])
            #Linearity
            if "linearity" in features:
                output.loc[x:x+chunk-1,"linearity"] = (knn_eigval_sort[:,0]-knn_eigval_sort[:,1])/knn_eigval_sort[:,0]
            #Planarity
            #if "planarity" in features:
            output.loc[x:x+chunk-1,"planarity"] = (knn_eigval_sort[:,1]-knn_eigval_sort[:,2])/knn_eigval_sort[:,0]
            #Sphericity
            #if "sphericity" in features:
            output.loc[x:x+chunk-1,"sphericity"] = knn_eigval_sort[:,2]/knn_eigval_sort[:,0]
            #Omnivariance
            if "omnivariance" in features:
                output.loc[x:x+chunk-1,"omnivariance"] = scipy.cbrt(knn_eigval_sort[:,0]*knn_eigval_sort[:,1]*knn_eigval_sort[:,2])
            #Anisotropy
            if "anisotropy" in features:
                output.loc[x:x+chunk-1,"anisotropy"] = (knn_eigval_sort[:,0] - knn_eigval_sort[:,2]) / knn_eigval_sort[:,0]   
            #Eigenentropy
            if "eigenentropy" in features:
                output.loc[x:x+chunk-1,"eigenentropy"] =  (-1 *((knn_eigval_sort[:,0] * np.log(knn_eigval_sort[:,0])) +

                      (knn_eigval_sort[:,1] * np.log(knn_eigval_sort[:,1])) +

                      (knn_eigval_sort[:,2] * np.log(knn_eigval_sort[:,2]))))    
            #Sum eigenvalues
            if "sum_eigenvalues" in features:
                output.loc[x:x+chunk-1,"sum_eigenvalues"] = knn_eigval_sum
            #Local Surface Variation
            if "curvature" in features:
                output.loc[x:x+chunk-1,"curvature"] = knn_eigval_sort[:,2] / knn_eigval_sum
            #Norm returns
            if "norm_returns" in features:
                output.loc[x:x+chunk-1,"norm_returns"] = points_remain_prop.loc[x:x+chunk-1,"return_number"] / points_remain_prop.loc[x:x+chunk-1,"number_of_returns"]
            
            if "number_of_returns" in features:
                output.loc[x:x+chunk-1,"number_of_returns"] = points_remain_prop.loc[x:x+chunk-1,"number_of_returns"]
            
            if "return_number" in features:
                output.loc[x:x+chunk-1,"return_number"] = points_remain_prop.loc[x:x+chunk-1,"return_number"]
            
            if "intensity" in features:
                output.loc[x:x+chunk-1,"intensity"] = points_remain_prop.loc[x:x+chunk-1,"intensity"]
            
            if classification == True:
                 print("Classifying!")
                 output.loc[x:x+chunk-1,"classification"] = clf.predict(output.loc[x:x+chunk-1,features])
          

        #Collect garbage incase anyone forgot to pick up their thrash.
            gc.collect()
            
        
    print("Classified {} as vegetation".format(len(output.classification)-sum(output.classification)))
    print((len(output.classification)-sum(output.classification)) / len(output.classification),"%")
    return output

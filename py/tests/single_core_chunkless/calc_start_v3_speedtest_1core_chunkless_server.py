#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:25:55 2018

@author: berend-christiaan
"""
import pandas as pd

import numpy as np

import scipy

from scipy import spatial

import math

from math import ceil

import time

import multiprocessing as mp

import multiprocessing

from multiprocessing import Manager

from concurrent.futures import ThreadPoolExecutor

import pickle

import sys

import gc

import queue

import laspy

from laspy.file import File

import os

#os.chdir('/home/berend-christiaan/uni/server')

#import pcfeatures_v3_speedtest_chunkless

#from pcfeatures_v3_speedtest_chunkless import par_calc_chunkless

import glob

import psutil
from psutil import virtual_memory

import threading

import tqdm
from tqdm import tqdm




import lidar_funcs_v2_speedtest
from lidar_funcs_v2_speedtest import hasNumbers, pt_density_est, auto_k, mp_preprocessing, auto_chunk, resource_usage_psutil

if __name__ == "__main__":

        
    hccc = 1
    if hccc == 1:
        init_runs = 1  
    else:
        init_runs = 0
        
    rerun = 0
    rerun+= init_runs
    
    input_file_path = []
    for i in range(0,rerun):
        input_file_path.append('/datastore/data/input/las/nl/beesdgeldermalsen/C_39CN1.las')
        #input_file_path.append('/media/berend-christiaan/bwijers_college/MscThesis/Code/Countries/TheNetherlands/BeesdGeldermalsen/Data/AHN3/C_39CN1_RA_LAS/C_39CN1_RA.las')
    print("Number of test runs:",rerun)
    
    #Number of points to use
    input_rows = 10000000 #10 million
    testrun = 0
    
#Filtering / classification settings
    ckdtree_setting = "unbalanced" # or 'unbalanced'
    classification = False
    tensor_filter = False
#Logger settings
    manager = Manager()
    d_log = manager.dict()
    q_log = queue.Queue()
    d_log["stop"] = False
    mem_cpu_percent = []

    for i in range(0,len(input_file_path)):
        
        #%% Initialize variable for storing speed results
        print("Running test #",testrun)
        columns_speed=['start_time','LAS_header_load','points_loading','pt_density_est','load_classifier','auto_k','bcKDTree_build','auto_chunk','mp_preprocessing',"par_calc","output_retrieve","output_organize","dimension_define","setting_attributes","close_las","end_time","tot_time"]
        speed_result = pd.DataFrame([np.zeros(len(columns_speed))],columns=columns_speed)
        
        speed_result.start_time = time.time()
        
        #Start a seperate thread to monitor the cpu and memory usage.
        
        #%% Reading las header
        
        print("Memory mappping", os.path.basename(input_file_path[i]), "...")
        ###########################
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        in_LAS = File(input_file_path[i], mode='r')
        speed_result.LAS_header_load = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False #Memory and cpu logger OFF
        ###########################
        #input_rows = len(in_LAS.x)
        print(input_rows)
        print("Retrieving points to be processed from memory")
      
        ###########################
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        points = np.vstack([in_LAS.x[0:input_rows],in_LAS.y[0:input_rows],in_LAS.z[0:input_rows]]).transpose()        
        points_remain_prop = pd.DataFrame(np.vstack([in_LAS.intensity[0:input_rows],in_LAS.return_num[0:input_rows],in_LAS.num_returns[0:input_rows]]).transpose(),columns = ["intensity","return_number","number_of_returns"])
        speed_result.points_loading = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF

        ###########################
        
    
        #Determine point density  
        
        print("Determining average point density of the input file")
        ###########################
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON

        mod_time = time.time()
        est_p_density = pt_density_est(in_LAS)
        speed_result.pt_density_est = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF        
        ########################### 
        #print("Estimated point density of this file is {} points/m2".format(est_p_density))
        
        
        #%% Loading classifier
        ###########################
        print('Loading classifier')
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        with open ('classifiers/old/clf_beesd_geld_10_03_ahn3_no_edit.pkl',"rb")as f:#('/home/berend-christiaan/uni/server/Input/classifiers/clf_beesd_geld_10_03_ahn3_no_edit.pkl',"rb") as f:#
            p_load = pickle.load(f)
        
       
        clf = p_load[0]
        speed_result.load_classifier = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF 
        ###########################
    
        #Prepare data
    
        features = ['delta_z', 'std_z', 'radius', 'density', 'norm_z',
    
                'linearity', 'planarity', 'sphericity', 'omnivariance',
    
                'anisotropy', 'eigenentropy', 'sum_eigenvalues',
    
                'curvature','norm_return']
    
        #features = p_load[1]
        
        #print(features)
        
        #Trim features of the K used in the training of the classifier
        #features = [f[:-3] if hasNumbers(f) is True else f for f in features]
        
        point_cloud_prop = pd.DataFrame([], columns=features)
        
                
        #print("Calculating the following features:")
        
        #print(features)
      
    
        # %%
    
        #Wijers method
    
        #print("Defining constants..")
    
        x = None
        
        ##########################
        print('Auto_k')
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        k = auto_k(est_p_density)
        speed_result.auto_k = time.time() - mod_time ; del mod_time    
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        #print("Using K={}".format(k))
        ###########################
        
       
    
        # %%   
        
        
    
       #%%
        #print("Constructing KDTree")
    
        ##############################
        print('KD_tree construction')
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        #evaluate options for quicker kdtree building
        if ckdtree_setting == "unbalanced":
            print("Unbalanced CKDTree")
            tree=scipy.spatial.cKDTree(points,balanced_tree=0,compact_nodes=0)
        else:    
            print("Balanced CKDTree")
            tree = scipy.spatial.cKDTree(points)
        speed_result.bcKDTree_build = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        
        
        #%%
        #print("Multiprocessing")
    
       
        
        ##################################
        print('Auto_chunk')
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        chunk = auto_chunk(k,classification)
        speed_result.auto_chunk = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        #print('Using chunk size:', chunk) 
        
        #Split the work in cpu_cores, let the function run with those parameters.
       
        print("Splitting work load across n_cpu-1")
    
        #tensor_filter = True
        #classification = True
        #tensor_filter = False
        #classification = False
    
        q = queue.Queue()
        
        ##################
        print('MP preprocessing')
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        index_order = mp_preprocessing(len(points),1)
        speed_result.mp_preprocessing = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        #################
        #cpu_count = mp.cpu_count()
        cpu_count = 1
        chunk = 1
       
        #%%
        
        ################################
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        print("Parameter calculation")
        start = 0
        stop = 10000000
        #output = pd.DataFrame(np.zeros([len(points),12]), columns=['Dz','StdZ','Rl','Di','L','P','Sp','O','A','E','Su','C'])
        output = pd.DataFrame(np.zeros([len(points),14]), columns = features)
        a = (4/3)*math.pi
        #output = par_calc_chunkless(start, stop, points,points_remain_prop,tree,chunk,k,clf,features,tensor_filter,classification)
        
        for x in tqdm(range(0,len(points))):

    
    
    
            ##Height difference
            
            #Find nearest neighbours dinstance (dd) and index (ii) 
            knndd,knnii = tree.query(points[x,:],k) # knn
            #knniis = tree.query.ball_point(na1[x,:],1) #Spherical (x,y,z)
            #knniic = tree2d.query.ball_point(na1[x,:2],1) #Cylindrical (x,y)
            
            #Retrieve Z information 
            neighbour_group_z = points[knnii.astype(np.int64),2]
            
            #Delta Z
            output.ix[x,'delta_z'] = max(neighbour_group_z) - min(neighbour_group_z)
            
                
            
            ##Height standard deviation 
            output.ix[x,'std_z'] = np.std(neighbour_group_z)
            
               
            ##Local radius
                   
            #Radius_local
            #Dat kan toch ook als max(np.abs(knndd[x,:2])) ?
            Rl = max(knndd)
            
            output.ix[x,'radius'] = Rl
               
            
            ##Local point density
            
            #Di = k / ( a * math.pow(Rl,3) )
            Di = k / (a * (Rl**3))
               
            output.ix[x,'density'] = Di
            
            
            ##Eigenvalues // covariance matrices
            #x,y,z of the neighbourhood
            #for y in range(0,len(knnii),1):
            
            neighbour_group_xyz = points[knnii.astype(np.int64),:]
            #neighbour_group_xyz = points[knnii[y].astype(np.int64),:]
            #Covariance of Knn = 3 (Pi,Q1,Q2) and corresponding (X,Y,Z)
            neighbour_group_xyz_cov = np.cov(neighbour_group_xyz,rowvar= 0)
            #Eigenvectors and Eigenvalues of covariance matrix
            knn_eigval, knn_eigvec = np.linalg.eig(neighbour_group_xyz_cov)
            #knn_eigval Diagonal eigenvalues. 
            idx = np.array(knn_eigval).argsort()[::-1]
            
            knn_eigval_sort = knn_eigval[idx]
            knn_eigvec_sort = knn_eigvec[idx,:]
            
            ##Structure tensors
            
            #Linearity
            L = (knn_eigval_sort[0]-knn_eigval_sort[1])/knn_eigval_sort[0]
            
            #Planarity
            P = (knn_eigval_sort[1]-knn_eigval_sort[2])/knn_eigval_sort[0]
            
            #Sphericity
            Sp = knn_eigval_sort[2]/knn_eigval_sort[0]
            
            #Omnivariance
            O = scipy.cbrt(knn_eigval_sort[0]*knn_eigval_sort[1]*knn_eigval_sort[2])
            
            #Anisotropy
            A = (knn_eigval_sort[0] - knn_eigval_sort[2]) / knn_eigval_sort[0]
            
            #Eigenentropy
            E = (-knn_eigval_sort[0] * np.log(knn_eigval_sort[0]) ) - (knn_eigval_sort[1] * np.log(knn_eigval_sort[1])) - (knn_eigval_sort[2] * np.log(knn_eigval_sort[2]))
            
            #sum of eigenvalues
            Su = sum(knn_eigval_sort)
            
            #Local surface variation
            C = knn_eigval_sort[2] / (sum(knn_eigval_sort))
            #Save properties
            #output.ix[x+y,['L','P','Sp','O','A','E','Su','C']] = [L,P,Sp,O,A,E,Su,C]
            output.ix[x,['linearity','planarity','sphericity','omnivariance','anisotropy','eigenentropy','sum_eigenvalues','curvature']] = [L,P,Sp,O,A,E,Su,C]
            
            #print(x, "/", len(points),end="\r")















# =============================================================================
#         with ThreadPoolExecutor(max_workers=cpu_count) as executor:
#     
#             futures = []
#     
#             for j in range(0,cpu_count):
#     
#                 start = index_order[j,0]
#     
#                 stop = index_order[j,1]
#                 
#                 print("Submitting thread", j)
#                 futures.append(executor.submit(par_calc, start, stop, points,points_remain_prop,tree,chunk,k,clf,features,tensor_filter,classification))
#     
# =============================================================================
        speed_result.par_calc = time.time() - mod_time ; del mod_time  
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        #############################################################
        
        
       #%%
        ############
        print("Retrieving results")
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        
        point_cloud_prop = pd.concat([point_cloud_prop,output],axis=0)
# =============================================================================
#         output = []
#      
#         for future in futures:
#     
#             output = future.result()
#             #print(output)
#             point_cloud_prop = pd.concat([point_cloud_prop,output],axis=0)
#             #print(point_cloud_prop)
#         
#         
#         del output
# =============================================================================
        speed_result.output_retrieve = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        #####################################################################
            
            #%% Reshaping output
        
        #print("Bundeling everything together..")
        del tree,clf
        
        ############################################
        print("Output reshaping")
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        points = pd.DataFrame(points,columns = ["x","y","z"])
        points_remain_prop = pd.DataFrame(points_remain_prop,columns = ["intensity","return_number","number_of_returns"])
        
        #print(points_remain_prop.columns)
        
        points = points.ix[point_cloud_prop.index]
        points_remain_prop = points_remain_prop.ix[point_cloud_prop.index]
        #print(points_remain_prop.columns)
                
        
        #point_cloud = points
        #del points
        #point_cloud = pd.concat([point_cloud,points_remain_prop],axis=1,copy=False)
        #del points_remain_prop
        #point_cloud = pd.concat([point_cloud,point_cloud_prop],axis=1,copy=False)
        #del point_cloud_prop
        
        #Test for concat merging large dataframes
        #point_cloud = points
        #del points
        
        
        #Different way of conectating
      
 
       
# =============================================================================
#         for f in range(0,len(points_remain_prop.columns)):
#             print("Merging the column {0}".format(points_remain_prop.columns[0]))
#             point_cloud = pd.concat([point_cloud,points_remain_prop.iloc[:,0]],axis=1,copy=False)
#             print("Dropping the column {0}".format(points_remain_prop.columns[0]))
#             points_remain_prop.drop(columns=[points_remain_prop.columns[0]],inplace=True)
#             gc.collect()
#         del points_remain_prop
#         gc.collect()
#         
#         for f in range(0,len(point_cloud_prop.columns)):
#             print("Merging the column {0}".format(point_cloud_prop.columns[0]))
#             point_cloud = pd.concat([point_cloud,point_cloud_prop.iloc[:,0]],axis=1,copy=False)
#             print("Dropping the column {0}".format(point_cloud_prop.columns[0]))
#             point_cloud_prop.drop(columns=[point_cloud_prop.columns[0]],inplace=True)
#             gc.collect()
#         del point_cloud_prop
#         gc.collect()
# =============================================================================
        #print(point_cloud.columns)
        #print("removing duplicate columns")
        #point_cloud = point_cloud.loc[:,~point_cloud.columns.duplicated()]
        speed_result.output_organize = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        ####################################################################
         #%% Test run with outputting to a .LAS file.
        #print("Outputting to las file")
        #in_LAS
        #Copy over LAS header from in_LAS
        #out_LAS = File('/home/berend-christiaan/uni/server/Output/C_39CN1_RA_params.las', mode = "w", header = in_LAS.header)
        #print(i)
        las_path_root = os.path.splitext(input_file_path[i][:-4])
        las_path_base = os.path.basename(input_file_path[i][:-4])
        #print(las_path_root)
        
        out_filename = '%s_params.las' % (las_path_root[0])
        #out_filename = '%s_veg_classification_testparams.las' % ("//datastore//data//output//speed_assessment//features//"+las_path_base)
        #print(out_filename)
        out_LAS = File(out_filename, mode = "w", header = in_LAS.header)
        
        #'%s_params.las' % (las_path_root)               
                       
                       
                       
        #Define new dimension
        
        #####################
        print("Creating extra dimensions for LAS file")
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        
        
        out_LAS.define_new_dimension(name="delta_z"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="std_z"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="radius"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="density"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="norm_z"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="linearity"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="planarity"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="sphericity"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="omnivariance"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="anisotropy"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="eigenentropy"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="sum_eigenvalues"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="curvature"+"_"+str(k), data_type=9, description= "Spatial feature")
        out_LAS.define_new_dimension(name="classification"+"_"+str(k),data_type=9, description="reference")
        
        speed_result.dimension_define = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
        #####################################################################
        
        #print(point_cloud.columns)
        #Find out which points have new information
        #test = in_LAS.x == point_cloud['x'] AND in_LAS.y == point_cloud['y'] AND in_LAS.z == point_cloud['z']
        #Put data in new dimension
        
        ############################
        print("Moving output to las file, setting attributes")
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON
        mod_time = time.time()
        
        print("x")
        out_LAS.x = points['x']
        #points.drop('x',axis=1,inplace=True)#point_cloud.drop('x')
        print("y")
        out_LAS.y = points['y']
        #points.drop('y',axis=1,inplace=True)
        print("z")
        out_LAS.z = points['z']
        #points.drop('z',axis=1,inplace=True)
        print("intensity")
        out_LAS.intensity = points_remain_prop['intensity'] 
        #points_remain_prop.drop('intensity',axis=1,inplace=True)
        print("return_num")
        out_LAS.return_num = points_remain_prop['return_number']
        #points_remain_prop.drop('return_number',axis=1,inplace=True)
        #print(point_cloud["number_of_returns"])
        #point_cloud["number_of_returns"] = point_cloud["number_of_returns"]
        #print(point_cloud["number_of_returns"])
        print("number of returns")
        out_LAS.num_returns = points_remain_prop['number_of_returns']
        #points_remain_prop.drop('number_of_returns',axis=1,inplace=True)
        
    
 
        #Setting attributes Maybe do this with "try" ?
        print("delta_z")
        setattr(out_LAS,'delta_z'+"_"+str(k),point_cloud_prop['delta_z'])
        #point_cloud_prop.drop('delta_z',axis=1,inplace=True)
        print("std_z")
        setattr(out_LAS,'std_z'+"_"+str(k),point_cloud_prop['std_z'])
        #point_cloud_prop.drop('std_z',axis=1,inplace=True)
        print("radius")
        setattr(out_LAS,'radius'+"_"+str(k),point_cloud_prop['radius'])
        #point_cloud_prop.drop('radius',axis=1,inplace=True)
        print("density")
        setattr(out_LAS,'density'+"_"+str(k),point_cloud_prop['density'])
        #point_cloud_prop.drop('density',axis=1,inplace=True)
        print("norm_z")
        setattr(out_LAS,'norm_z'+"_"+str(k),point_cloud_prop['norm_z'])
        #point_cloud_prop.drop('norm_z',axis=1,inplace=True)
        print("linearity")
        setattr(out_LAS,'linearity'+"_"+str(k),point_cloud_prop['linearity'])
        #point_cloud_prop.drop('linearity',axis=1,inplace=True)
        print("planarity")
        setattr(out_LAS,'planarity'+"_"+str(k),point_cloud_prop['planarity'])
        #point_cloud_prop.drop('planarity',axis=1,inplace=True)
        print("sphericity")
        setattr(out_LAS,'sphericity'+"_"+str(k),point_cloud_prop['sphericity'])
        #point_cloud_prop.drop('sphericity',axis=1,inplace=True)
        print("omnivariance")
        setattr(out_LAS,'omnivariance'+"_"+str(k),point_cloud_prop['omnivariance'])
        #point_cloud_prop.drop('omnivariance',axis=1,inplace=True)
        if "anisotropy" in features:
            print("anisotropy")
            setattr(out_LAS,'anisotropy'+"_"+str(k),point_cloud_prop['anisotropy'])
            #point_cloud_prop.drop('anisotropy',axis=1,inplace=True)
        if "eigenentropy" in features:
            print("eigenentropy")
            setattr(out_LAS,'eigenentropy'+"_"+str(k),point_cloud_prop['eigenentropy'])
            #point_cloud_prop.drop('eigenentropy',axis=1,inplace=True)
        if "sum_eigenvalues" in features:
            print("sum_eigenvalues")
            setattr(out_LAS,'sum_eigenvalues'+"_"+str(k),point_cloud_prop['sum_eigenvalues'])
            #point_cloud_prop.drop('sum_eigenvalues',axis=1,inplace=True)
        if "curvature" in features:
            print("curvature")
            setattr(out_LAS,'curvature'+"_"+str(k),point_cloud_prop['curvature'])
            #point_cloud_prop.drop('curvature',axis=1,inplace=True)
        if "classification" in features:
            print("classification")
            setattr(out_LAS,'classification'+"_"+str(k),point_cloud_prop['classification'])
            #point_cloud_prop.drop('classification',axis=1,inplace=True)
        
        speed_result.setting_attributes = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF
# =============================================================================
#         print("Moving output to las file, setting attributes")
#         mod_time = time.time()
#         
#         out_LAS.x = point_cloud['x']
#         point_cloud.drop('x',inplace=True)#point_cloud.drop('x')
#         
#         out_LAS.y = point_cloud['y']
#         point_cloud.drop('y',inplace=True)
#         out_LAS.z = point_cloud['z']
#         point_cloud.drop('z',inplace=True)
#         out_LAS.intensity = point_cloud['intensity'] 
#         point_cloud.drop('intensity',inplace=True)
#         out_LAS.return_num = point_cloud['return_number']
#         point_cloud.drop('return_number',inplace=True)
#         #print(point_cloud["number_of_returns"])
#         #point_cloud["number_of_returns"] = point_cloud["number_of_returns"]
#         #print(point_cloud["number_of_returns"])
#         out_LAS.num_returns = point_cloud['number_of_returns']
#         point_cloud.drop('number_of_returns',inplace=True)
#         
#     
#  
#         #Setting attributes Maybe do this with "try" ?
#         setattr(out_LAS,'delta_z'+"_"+str(k),point_cloud['delta_z'])
#         point_cloud.drop('delta_z',inplace=True)
#         setattr(out_LAS,'std_z'+"_"+str(k),point_cloud['std_z'])
#         point_cloud.drop('std_z',inplace=True)
#         setattr(out_LAS,'radius'+"_"+str(k),point_cloud['radius'])
#         point_cloud.drop('radius',inplace=True)
#         setattr(out_LAS,'density'+"_"+str(k),point_cloud['density'])
#         point_cloud.drop('density',inplace=True)
#         setattr(out_LAS,'norm_z'+"_"+str(k),point_cloud['norm_z'])
#         point_cloud.drop('norm_z',inplace=True)
#         setattr(out_LAS,'linearity'+"_"+str(k),point_cloud['linearity'])
#         point_cloud.drop('linearity',inplace=True)
#         setattr(out_LAS,'planarity'+"_"+str(k),point_cloud['planarity'])
#         point_cloud.drop('planarity',inplace=True)
#         setattr(out_LAS,'sphericity'+"_"+str(k),point_cloud['sphericity'])
#         point_cloud.drop('sphericity',inplace=True)
#         setattr(out_LAS,'omnivariance'+"_"+str(k),point_cloud['omnivariance'])
#         point_cloud.drop('omnivariance',inplace=True)
#         if "anisotropy" in features:
#             setattr(out_LAS,'anisotropy'+"_"+str(k),point_cloud['anisotropy'])
#             point_cloud.drop('anisotropy',inplace=True)
#         if "eigenentropy" in features:
#             setattr(out_LAS,'eigenentropy'+"_"+str(k),point_cloud['eigenentropy'])
#             point_cloud.drop('eigenentropy',inplace=True)
#         if "curvature" in features:
#             setattr(out_LAS,'sum_eigenvalues'+"_"+str(k),point_cloud['sum_eigenvalues'])
#             point_cloud.drop('sum_eigenvalues',inplace=True)
#         if "curvature" in features:
#             setattr(out_LAS,'curvature'+"_"+str(k),point_cloud['curvature'])
#             point_cloud.drop('curvature',inplace=True)
#         if "classification" in features:
#             setattr(out_LAS,'classification'+"_"+str(k),point_cloud['classification'])
#             point_cloud.drop('classification',inplace=True)
#         
#         speed_result.setting_attributes = time.time() - mod_time ; del mod_time
# =============================================================================
        ########################################################################
        
        
        
        ##############################
        print("Closing las file")
        th = threading.Thread(target=resource_usage_psutil,args=(d_log,q_log));th.start()    #Memory and cpu logger ON        
        mod_time = time.time()
        out_LAS.close()
        speed_result.close_las = time.time() - mod_time ; del mod_time
        d_log["stop"] = True ; mem_cpu_percent.append(q_log.get()) ; th.join() ; print(mem_cpu_percent) ; d_log["stop"] = False  #Memory and cpu logger OFF

        ######################################################################
        speed_result.end_time = time.time()
        speed_result.tot_time = time.time() - speed_result.start_time
        
        #Saving speed results
        print("Saving speed results")
        #output location
        #out_filename_speed = '%s_speed_result.csv' % ("//datastore//data//output//speed_assessment//speed//C_39CN1_testrun_"+str(testrun)+"_classification_"+str(classification)+"_tensor_filter_"+str(tensor_filter)+"_input_rows_"+str(input_rows)+"_hccc_",str(hccc)+"_ckdtree_setting_"+ckdtree_setting)
        out_filename_speed = ("//datastore//data//output//speed_assessment//speed//memory_cpu//cores_1_chunks_off//C_39CN1_classification_{1}_tensorfilter_{2}_inputrows_{3}_hccc_{4}_ckdtreesetting_{5}//C_39CN1_testrun_{0}_classification_{1}_tensorfilter_{2}_inputrows_{3}_hccc_{4}_ckdtreesetting_{5}.csv".format(testrun,classification,tensor_filter,input_rows,hccc,ckdtree_setting))
        
        #Pull values from container and store them into a dataframe
        out_filename_mem_cpu = out_filename_speed[:-3] + '_mem_cpu.csv'
        df = pd.DataFrame(mem_cpu_percent)
        
        #for i in range(0,len(mem_cpu_percent)):
        #    df[i] = mem_cpu_percent[i]
        
        df.to_csv(out_filename_mem_cpu)
        speed_result.to_csv(out_filename_speed)
        testrun+= 1
        

 

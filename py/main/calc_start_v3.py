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

from concurrent.futures import ThreadPoolExecutor

import pickle

import sys

import gc

import queue

import laspy

from laspy.file import File

import os

import pcfeatures_v3

from pcfeatures_v3 import par_calc

import glob

import psutil
from psutil import virtual_memory

import lidar_funcs_v2
from lidar_funcs_v2 import hasNumbers

if __name__ == "__main__":

    ###########LAS
    las_path = []
    
    for file in glob.glob("../../**/*.las", recursive = True):
        las_path.append(file)
        
    ##Display found las files
    for i in range (0,len(las_path)):
        print(i, las_path[i])
        
        
    print("Select the files to process, seperated by a comma. Example: 0,1,2 | 1,3,2 etc. Or provide a range")  
    
    user_las_selection = input().split(',')
    
    user_las_selection = [int(x) for x in user_las_selection]
    
    
    
    print("You have selected the following files for processing:")
    ##Display found las files
    for i in range (0,len(user_las_selection)):
        print(i, las_path[user_las_selection[i]])
    
    
    #las_input_path = las_path[user_las_selection]
    
    
    
    #print("Input .LAS file path")
    
    #input_file_path = str(input())
    #input_file_path = "/home/berend-christiaan/uni/server/Input/C_39CN1_RA.las"
    #print(input_file_path)
    
    #input_file_path = las_input_path 
    
    input_file_path = [las_path[x] for x in user_las_selection]
    #input_file_path = 'Input/C_39C1N_RA_LAS_downsampled/C_39CN1_RA_2_ptm2.las'
    #print(input_file_path)
    
    for i in range(0,len(input_file_path)):
        
        
        
        print("Memory mappping", os.path.basename(input_file_path[i]), "...")
        
        in_LAS = File(input_file_path[i], mode='r')
        #in_LAS = File(input_file_path,mode='r')
        
        #Read out the size and therefore point density by len points divided by minx,y max xy distances.
        #input_rows = 'max'
        #input_rows = 100000
        input_rows = 10000000
    
        #print("Input number of rows or type max to use the full file")    
        
        #input_rows = input()
        #input_rows = 1000000
      
        
        
    
        if input_rows == "max" :
            #input_rows = None
            input_rows = int(len(in_LAS))
        else:
            input_rows = int(input_rows)
    
        #Determine point density
        #Draw a small rectangular box around a set of points, count the points divide by surface area of box.
        
        print("Determining average point density of the input file")
        #Top left x coordinate
        left_x = np.min(in_LAS.x)
        top_y = np.min(in_LAS.y[np.where(in_LAS.x == left_x)])
        
        top_left_xy = np.array((left_x,top_y)) #used
        bot_left_xy = np.array((left_x,top_y+10))# used
        top_right_xy = np.array((left_x+10,top_y)) #used
        bot_right_xy = np.array((left_x+10,top_y+10)) #unused
        
        #mini_bounding_box = [top_left_xy,bottom_left_xy,top_right_xy,bottom_right_xy]

        #Calculate distances between each set of coordinates
        
        #Top left to top right
        
        #distance_top_LR = np.sqrt(np.einsum('ij,ij->i',top_left_xy-top_right_xy, top_left_xy-top_right_xy))
        #distance_bot_LR
        #distance_L_top_bot
        #distance_R_top_bot
        
        #A - B
        A = np.array((top_left_xy,bot_left_xy,top_left_xy,top_right_xy))
        B = np.array((top_right_xy,bot_right_xy,bot_left_xy,bot_right_xy))
        
        dd = np.sqrt(np.einsum('ij,ij->i',A-B,A-B))
        
        surf_area = dd[0] * dd[2]
        
        x_range = np.where(np.logical_and(in_LAS.x > top_left_xy[0], in_LAS.x < top_right_xy[0]))
        #y_range = np.where(np.logical_and(in_LAS.y > top_left_xy[1], in_LAS.y < bot_left_xy[1]))
        
        x_range_points = np.vstack(np.array([in_LAS.x[x_range],in_LAS.y[x_range]])).transpose()
        
        
        num_points = len(np.where(np.logical_and(x_range_points[:,1] > top_left_xy[1], x_range_points[:,1] < bot_left_xy[1]))[0])
        
        
        est_p_density = num_points/surf_area
        
        print("Estimated point density of this file is {} points/m2".format(est_p_density))
        #y_range_points = np.vstack(np.array([in_LAS.x[y_range],in_LAS.y[y_range]])).transpose()
        
        
        
        #in_LAS.points[a][b][c] : a = point record number, b is tuple/ list, c = x,y,z,intens..etc
        
        
        
        
        
        
        #Loading classifier
        
        #with open('classifiers/clf_beesd_geld_ahn3_16_ptm2.pkl',"rb")as f:
            #p_load = pickle.load(f)
        
        with open ('classifiers/old/clf_beesd_geld_10_03_ahn3_no_edit.pkl',"rb")as f:
            p_load = pickle.load(f)
        
        #with open('/home/berend-christiaan/uni/server/Input/classifiers/clf_beesd_geld_10_03_ahn3_no_edit.pkl',"rb") as f:
         #   p_load = pickle.load(f)
        #print(p_load)
        clf = p_load[0]
       
    
        #Prepare data
    
        #features = ['delta_z', 'std_z', 'radius', 'density', 'norm_z',
    
         #       'linearity', 'planarity', 'sphericity', 'omnivariance',
    
          #      'anisotropy', 'eigenentropy', 'sum_eigenvalues',
    
           #     'curvature','classification']
    
        features = p_load[1]
        
        #Trim features of the K used in the training of the classifier
        
        
        

        
        features = [f[:-3] if hasNumbers(f) is True else f for f in features]
        
        
        
                
        print("Calculating the following features:")
        
        print(features)
      
    
        # %%
    
        #Wijers method
    
        print("Defining constants..")
    
        #Define constant
    
        #global chunk, k, a, q, points, tree
    
        
    
        x = None
    
        #k = 50 #16 pt/m2
        
        
        #k = 5 0.5pt/m2
        #Callibration: K = 50 for 16 pt/m2
        #50 / 16 = 3.125 K's per point
        
        k = int(3.125 * est_p_density)
        
        #introduce a minimum K check (We need atleast 3 to do anything sensible!)
        
        if k < 3:
            k = 3
            
        #Testing purposes
        k = 50
            
        print("Using K={}".format(k))

        
        #Dynamic chunk size based upon CPU/MEMORY ratio
        #500.000 seems to be good for 40 cores with 256 gb of ram.
        #256 GB / 40 = 6.4 GB/Core. 500.000 chunk size for K = 50 seems to be appropriate.
        #500.000 * 50 = 25.000.000 points per 6.4 GB/Core
        #25.000.000 / 6.4 = 3.906.205 points / GB/core
        
        #
        
        
        #500000 * 40 = 20.000.000
        #20.000.000 / 
        #chunk = 5000
    
      
    
       
    
        # %%
    
     
    
        
    
        
        print("Retrieving points to be processed from memory")
      
        points = np.vstack([in_LAS.x[0:input_rows],in_LAS.y[0:input_rows],in_LAS.z[0:input_rows]]).transpose()
    
        
        points_remain_prop = pd.DataFrame(np.vstack([in_LAS.intensity[0:input_rows],in_LAS.return_num[0:input_rows],in_LAS.num_returns[0:input_rows]]).transpose(),columns = ["intensity","return_number","number_of_returns"])
        #points_remain_prop = np.vstack([in_LAS.intensity[0:input_rows]]).transpose()
    
         
       
        
        
    
        
        print("Constructing KDTree")
    
      
        #evaluate options for quicker kdtree building
        tree=scipy.spatial.cKDTree(points,balanced_tree=0,compact_nodes=0)
        
        
        point_cloud_prop = pd.DataFrame([], columns=features)
        
        
        
        print("Multiprocessing")
    
       
        ##Dynamic chunk size based upon CPU/MEMORY ratio
        #500.000 seems to be good for 40 cores with 256 gb of ram.
        #256 GB / 40 = 6.4 GB/Core. 500.000 chunk size for K = 50 seems to be appropriate.
        #500.000 * 50 = 25.000.000 points per 6.4 GB/Core
        #25.000.000 / 6.4 = 3.906.205 points / GB/core
        #Values have been experimentally found.
        cpu_count = mp.cpu_count()
        mem=virtual_memory()
        print(mem)
        #mem_tot = mem.total/1000000000 #gb
        mem_tot = mem.free/1000000000
        print(mem_tot)
        cpu_mem_ratio = mem_tot/cpu_count
        print(cpu_mem_ratio)
        #K = 50
        #Apparently classifying points copy huge datasets to the memory (duplicates) as such a new variable H is introduced which is a integer between 0 - 1 to be able to scale.
        H = 0.5
        tot_points_cpu = (3906205*H) * cpu_mem_ratio
        print(tot_points_cpu)
        chunk = int(tot_points_cpu / k)
        
        
        print('Using chunk size:', chunk) 
        
        
        
        
        
    
       
    
        print("Total processors (hyper threading) available: ",mp.cpu_count(),". Using: ",cpu_count, "threads")
    
      
    
        #Maximum amount of points to go through
    
        num_points = len(points)
    
      
    
        #Number of points per CPU
    
        points_p_cpu = int(num_points/cpu_count)
    
      
    
        index_order = np.zeros((cpu_count,2),dtype=int)
    
        #Create an index work order per core
    
        print("Create start stop indexes for n_cpu_cores-1")
    
        for l in range(0,cpu_count):
    
            index_start = int(l * (points_p_cpu))
    
            index_order[l] = [int(index_start),int(index_start + (points_p_cpu))]
    
            if l == cpu_count-1:
    
                index_order[-1,1] = index_order[-1,1] + num_points%cpu_count              #Add remaining index to last cpu core
    
      
    
        print(index_order) 
    
        # %%  
    
           
    
           
    
        #Split the work in cpu_cores, let the function run with those parameters.
       
        print("Splitting work load across n_cpu-1")
    
       
    
        
        tensor_filter = False
        classification = True
        #classification = False
    
        q = queue.Queue()
    
        
        mod_start = time.time()
        
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
    
            futures = []
    
            for j in range(0,cpu_count):
    
                start = index_order[j,0]
    
                stop = index_order[j,1]
                
                
    
               
                print("Submitting thread", j)
                futures.append(executor.submit(par_calc, start, stop, points,points_remain_prop,tree,chunk,k,clf,features,tensor_filter,classification))
    
           
        print(time.time()-mod_start)
        
        print("Done!")
        
        #time.sleep(25)
       
        print("Retrieving results")
        mod_start = time.time()
          
    
        output = []
    
        #output = np.array([])
        
        for future in futures:
    
           
            output = future.result()
            #print(output)
            point_cloud_prop = pd.concat([point_cloud_prop,output],axis=0)
            #print(point_cloud_prop)
        
        print(time.time()-mod_start)
        print("Done!")
        
        #print(point_cloud_prop)
       
        #with open("output_threadpool.pkl","wb")as f:
         #      pickle.dump(times_parallel_chunked,f)
            
            
            #%% Reshaping output
        
            
             #print(point_cloud_prop)
    
        print("Calculations are done, just finishing up here..")
    
        print("Keeping track of parameters used...")
    
        #Rename to indicate which K has been used - build it next time that it runs through automatically.
    
        #point_cloud_prop.columns = ['delta_z'+"_"+str(k), 'std_z'+"_"+str(k), 'radius'+"_"+str(k), 'density'+"_"+str(k), 'norm_z'+"_"+str(k),
    
         #           'linearity'+"_"+str(k), 'planarity'+"_"+str(k), 'sphericity'+"_"+str(k), 'omnivariance'+"_"+str(k),
    
          #          'anisotropy'+"_"+str(k), 'eigenentropy'+"_"+str(k), 'sum_eigenvalues'+"_"+str(k),
    
           #         'curvature'+"_"+str(k), 'classification'+"_"+str(k)]
    
        
    
        print("Bundeling everything together..")
        
        #points = pd.DataFrame(points[point_cloud_prop.index],columns = ["x","y","z"])
        
        #points_remain_prop = pd.DataFrame(points_remain_prop[point_cloud_prop.index],columns = ["intensity","return_number","number_of_returns"])
        
        #Filter stats
        #print("Enabling the filter has saved the calculation of", len(points) - len(point_cloud_prop), "points.(",np.round(((len(points)-len(point_cloud_prop))/ len(points))*100),"%)") 
        #print("Classified {} points as vegetation".format(np.sum(point_cloud_prop['classification'+"_"+str(k)])))
        #print("Classified {} points as non-vegetation".format(len(point_cloud_prop)-np.sum(point_cloud_prop['classification'+"_"+str(k)])))
        
        points = pd.DataFrame(points,columns = ["x","y","z"])
        points_remain_prop = pd.DataFrame(points_remain_prop,columns = ["intensity","return_number","number_of_returns"])
        
        print(points_remain_prop.columns)
        #output.drop(idx_drop,axis=0,inplace=True)
        #points = points.drop(point_cloud_prop.index,axis=0,inplace=False)
        
        points = points.ix[point_cloud_prop.index]
        points_remain_prop = points_remain_prop.ix[point_cloud_prop.index]
        print(points_remain_prop.columns)
                
        
            
        
        #print(points.index)
        #print(points_remain_prop.index)
        #print(point_cloud_prop.index)
        #point_cloud = pd.concat([points,points_remain_prop,point_cloud_prop], axis=1)
        
        #if "intensity" in point_cloud_prop.columns:
            #print("Throwing out intensity out of points_remain_prop",points_remain_prop.columns)
         #   points_remain_prop = points_remain_prop.drop(columns="intensity")
          #  print(points_remain_prop.columns)
            
        #if "return_number" in point_cloud_prop.columns:
            #print("Throwing out intensity out of points_remain_prop",points_remain_prop.columns)
         #   points_remain_prop = points_remain_prop.drop(columns="return_number")
          #  print(points_remain_prop.columns)
            
        #if "number_of_returns" in point_cloud_prop.columns:
            #print("Throwing out intensity out of points_remain_prop",points_remain_prop.columns)
         #   point_cloud_prop = points_remain_prop.drop(columns="number_of_returns")
          #  print(point_cloud_prop.columns)
        
       
        
        point_cloud = points
        del points
        point_cloud = pd.concat([point_cloud,points_remain_prop],axis=1)
        del points_remain_prop
        point_cloud = pd.concat([point_cloud,point_cloud_prop],axis=1)
        del point_cloud_prop
        
        print(point_cloud.columns)
        print("removing duplicate columns")
        point_cloud = point_cloud.loc[:,~point_cloud.columns.duplicated()]
        #Test possibility of doing this step by step. Merge points,points_remain_prop. Remove points, points_remains
       
        #print(point_cloud)
    # =============================================================================
         #%% Test run with outputting to a .LAS file.
        print("Outputting to las file")
        #in_LAS
        #Copy over LAS header from in_LAS
        #out_LAS = File('/home/berend-christiaan/uni/server/Output/C_39CN1_RA_params.las', mode = "w", header = in_LAS.header)
        #print(i)
        #las_path_root = os.path.splitext(input_file_path[i][:-4])
        las_path_base = os.path.basename(input_file_path[i][:-4])
        #print(las_path_root)
        
        #out_filename = '%s_params.las' % (las_path_root[0])
        out_filename = '%s_veg_classification_testparams.las' % ("//datastore//data//output//"+las_path_base)
        print(out_filename)
        out_LAS = File(out_filename, mode = "w", header = in_LAS.header)
        
        #'%s_params.las' % (las_path_root)               
                       
                       
                       
        #Define new dimension
        
        
        
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
        
        print(point_cloud.columns)
        #Find out which points have new information
        #test = in_LAS.x == point_cloud['x'] AND in_LAS.y == point_cloud['y'] AND in_LAS.z == point_cloud['z']
        #Put data in new dimension
        out_LAS.x = point_cloud['x']
        #point_cloud.drop('x')
        
        out_LAS.y = point_cloud['y']
        out_LAS.z = point_cloud['z']
        out_LAS.intensity = point_cloud['intensity'] 
        out_LAS.return_num = point_cloud['return_number']
        #print(point_cloud["number_of_returns"])
        point_cloud["number_of_returns"] = point_cloud["number_of_returns"]
        #print(point_cloud["number_of_returns"])
        out_LAS.num_returns = point_cloud['number_of_returns']
# =============================================================================
#         out_LAS.delta_z+"_"+str(k) = point_cloud['delta_z'+"_"+str(k)]
#         out_LAS.std_z+"_"+str(k) = point_cloud['std_z'+"_"+str(k)]
#         out_LAS.radius+"_"+str(k) = point_cloud['radius'+"_"+str(k)]
#         out_LAS.density+"_"+str(k) = point_cloud['density'+"_"+str(k)]
#         out_LAS.norm_z+"_"+str(k) = point_cloud['norm_z'+"_"+str(k)]
#         out_LAS.linearity+"_"+str(k) = point_cloud['linearity'+"_"+str(k)]
#         out_LAS.planarity+"_"+str(k) = point_cloud['planarity'+"_"+str(k)]
#         out_LAS.sphericity+"_"+str(k) = point_cloud['sphericity'+"_"+str(k)]
#         out_LAS.omnivariance+"_"+str(k) = point_cloud['omnivariance'+"_"+str(k)]
#         out_LAS.anisotropy+"_"+str(k) = point_cloud['anisotropy'+"_"+str(k)]
#         out_LAS.eigenentropy+"_"+str(k) = point_cloud['eigenentropy'+"_"+str(k)]
#         out_LAS.sum_eigenvalues+"_"+str(k) = point_cloud['sum_eigenvalues'+"_"+str(k)]
#         out_LAS.curvature+"_"+str(k) = point_cloud['curvature'+"_"+str(k)]
#         out_LAS.classification_ML+"_"+str(k) = point_cloud['classification_ML'+"_"+str(k)]
# =============================================================================
        
        #Setting attributes Maybe do this with "try" ?
        setattr(out_LAS,'delta_z'+"_"+str(k),point_cloud['delta_z'])
        #point_cloud.drop('delta_z')
        setattr(out_LAS,'std_z'+"_"+str(k),point_cloud['std_z'])
        #point_cloud.drop('std_z')
        setattr(out_LAS,'radius'+"_"+str(k),point_cloud['radius'])
        #point_cloud.drop('radius')
        setattr(out_LAS,'density'+"_"+str(k),point_cloud['density'])
        #point_cloud.drop('density')
        setattr(out_LAS,'norm_z'+"_"+str(k),point_cloud['norm_z'])
        #point_cloud.drop('norm_z')
        setattr(out_LAS,'linearity'+"_"+str(k),point_cloud['linearity'])
        #point_cloud.drop('linearity')
        setattr(out_LAS,'planarity'+"_"+str(k),point_cloud['planarity'])
        #point_cloud.drop('planarity')
        setattr(out_LAS,'sphericity'+"_"+str(k),point_cloud['sphericity'])
        #point_cloud.drop('sphericity')
        setattr(out_LAS,'omnivariance'+"_"+str(k),point_cloud['omnivariance'])
        #point_cloud.drop('omnivariance')
        if "anisotropy" in features:
            setattr(out_LAS,'anisotropy'+"_"+str(k),point_cloud['anisotropy'])
        #point_cloud.drop('anisotropy')
        if "eigenentropy" in features:
            setattr(out_LAS,'eigenentropy'+"_"+str(k),point_cloud['eigenentropy'])
        #point_cloud.drop('eigenentropy')
        if "curvature" in features:
            setattr(out_LAS,'sum_eigenvalues'+"_"+str(k),point_cloud['sum_eigenvalues'])
        #point_cloud.drop('sum_eigenvalues')
        if "curvature" in features:
            setattr(out_LAS,'curvature'+"_"+str(k),point_cloud['curvature'])
        #point_cloud.drop('curvature')
        if "classification" in features:
            setattr(out_LAS,'classification'+"_"+str(k),point_cloud['classification'])
        #point_cloud.drop('classification')
        
        
        #setattr(out_LAS,'delta_z'+"_"+str(k),point_cloud['delta_z'+"_"+str(k)])
        #point_cloud.drop('delta_z'+"_"+str(k))
        #setattr(out_LAS,'std_z'+"_"+str(k),point_cloud['std_z'+"_"+str(k)])
        #point_cloud.drop('std_z'+"_"+str(k))
        #setattr(out_LAS,'radius'+"_"+str(k),point_cloud['radius'+"_"+str(k)])
        #point_cloud.drop('radius'+"_"+str(k))
        #setattr(out_LAS,'density'+"_"+str(k),point_cloud['density'+"_"+str(k)])
        #point_cloud.drop('density'+"_"+str(k))
        #setattr(out_LAS,'norm_z'+"_"+str(k),point_cloud['norm_z'+"_"+str(k)])
        #point_cloud.drop('norm_z'+"_"+str(k))
        #setattr(out_LAS,'linearity'+"_"+str(k),point_cloud['linearity'+"_"+str(k)])
        #point_cloud.drop('linearity'+"_"+str(k))
        #setattr(out_LAS,'planarity'+"_"+str(k),point_cloud['planarity'+"_"+str(k)])
        #point_cloud.drop('planarity'+"_"+str(k))
        #setattr(out_LAS,'sphericity'+"_"+str(k),point_cloud['sphericity'+"_"+str(k)])
        #point_cloud.drop('sphericity'+"_"+str(k))
        #setattr(out_LAS,'omnivariance'+"_"+str(k),point_cloud['omnivariance'+"_"+str(k)])
        #point_cloud.drop('omnivariance'+"_"+str(k))
        #setattr(out_LAS,'anisotropy'+"_"+str(k),point_cloud['anisotropy'+"_"+str(k)])
        #point_cloud.drop('anisotropy'+"_"+str(k))
        #setattr(out_LAS,'eigenentropy'+"_"+str(k),point_cloud['eigenentropy'+"_"+str(k)])
        #point_cloud.drop('eigenentropy'+"_"+str(k))
        #setattr(out_LAS,'sum_eigenvalues'+"_"+str(k),point_cloud['sum_eigenvalues'+"_"+str(k)])
        #point_cloud.drop('sum_eigenvalues'+"_"+str(k))
        #setattr(out_LAS,'curvature'+"_"+str(k),point_cloud['curvature'+"_"+str(k)])
        #point_cloud.drop('curvature'+"_"+str(k))
        #setattr(out_LAS,'classification'+"_"+str(k),point_cloud['classification'+"_"+str(k)])
        #point_cloud.drop('classification'+"_"+str(k))
        #Extra info: for dimension in in_LAS.point_format: dat=in_LAS.reader.get_dimension.name) out_LAS.writer.get_dimension(dimension.name,dat)
        #test = np.logical_and(in_LAS.x,point_cloud['x'] =in_LAS.y,point_cloud['y'],in_LAS.z,point_cloud['z'])
        #test = np.equal([in_LAS.x,in_LAS.y,in_LAS.z],[point_cloud['x'],point_cloud['y'],point_cloud['z']])
        
        out_LAS.close()
    # =============================================================================
       
       
        #print(point_cloud_prop["delta_z_50"])
    
        # %% Trim the data by deleting all non scatter points from the point cloud
    
        #print ("Trimming data..")
    
        #point_cloud.query('sphericity_50 > 0.05 & planarity_50 < 0.7', inplace=True)
    
        #point_cloud.reset_index(drop=True, inplace=True)
        
        #print(time.time()-mod_time)
        #mod_time = time.time()
        #print ("Done!")
    
       
    
        # %% Compute normalized return number
    
        #print("Computing normalized return numbers..")
        
        #point_cloud['norm_returns'] = (point_cloud['return_number'] /
    
         #                              point_cloud['number_of_returns'])
    
     
    
     
        #print(time.time()-mod_time)
        #mod_time = time.time()
        #print("Done")
    
       
    
        #out_filename = "/home/berend-christiaan/uni/server/Output/C_39CN1_RA_parallel_spawn_params.csv"
    
       # print("Saving data..")
    
       # point_cloud.to_csv(out_filename, index=False)
    
       # print("Done! Your file is located in: ", out_filename)
       
    
        #print("Runtime..", time.time() - start_time)
        
        #print(point_cloud.head)
       

        

 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:53:21 2018

@author: berend-christiaan
"""

# %% Library loading


# %% Point density estimation
def pt_density_est(pc):
    import numpy as np
    
    in_LAS = pc
    
    """
    Input:
        pc = point cloud which needs to be assessed
    Return:
        Float for point density / m2.
    """
    print("Determining average point density of the input file")
    #Top left x coordinate
    left_x = np.min(in_LAS.x)# left_x = np.min(in_LAS.x)
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
    return est_p_density

# %% File scanning
def laz_input(input_loc):
    
        # %% Loading environment
    import glob
    import os
        
    # %% Set variables
    laz_path = []
    
    # %% Scan for .LAS files
    for file in glob.glob((input_loc + '/**/*.laz'), recursive = True):
        laz_path.append(file)
        
    # %%Display found las files
    for i in range (0,len(laz_path)):
        print(i, laz_path[i])
        
    if laz_path == []:
        print("No files found, please specify a different input folder")
        return
        
        
    print("Select the files to process, seperated by a comma. Example: 0,1,2 | 1,3,2 etc. Or provide a range")  
    
    # %%Receive user input for selection fo files
    user_laz_selection = input().split(',')
    
    user_laz_selection = [int(x) for x in user_laz_selection]
    
    laz_input_file_path = [laz_path[x] for x in user_laz_selection]
    
    print("You have selected the following files for processing:")
    
    for i in range(0,len(laz_input_file_path)):
        print(i, laz_input_file_path[i])
        
    
    return laz_input_file_path    
    


def las_input(input_loc):
    """
    Lidar input scans for LAS files, lets the user select which one(s) should be processed as well as determine amount of points to
    be processed.
    """
    
    # %% Loading environment
    import glob
    import os
        
    # %% Set variables
    las_path = []
    
    # %% Scan for .LAS files
    for file in glob.glob((input_loc+"/**/*.las"), recursive = True):
        las_path.append(file)
        
    # %%Display found las files
    for i in range (0,len(las_path)):
        print(i, las_path[i])
        
        
    print("Select the files to process, seperated by a comma. Example: 0,1,2 | 1,3,2 etc. Or provide a range")  
    
    # %%Receive user input for selection fo files
    user_las_selection = input().split(',')
    
    user_las_selection = [int(x) for x in user_las_selection]
    
    las_input_file_path = [las_path[x] for x in user_las_selection]
    
    print("You have selected the following files for processing:")
    
    for i in range(0,len(las_input_file_path)):
        print(i, las_input_file_path[i])
        
    
    return las_input_file_path

def laz2las(laz_input_file_path):
    """
    Converts LAZ -> LAS files based upon LAZ files found to be processed
    program_files\current_build\LAStools\bin\laszip
    """
    import subprocess
    
    for i in laz_input_file_path:
       return_code = subprocess.call(['LAStools/bin/./laszip','-i',i])
       
       if return_code == 0:
           print("Conversion succesfully completed!")
           
    return return_code

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def auto_k(est_p_density):
    k = int(3.125 * est_p_density)
    if k < 4:
        k = 4
    return k

def auto_chunk(k,classification):
    import multiprocessing as mp
    import psutil
    from psutil import virtual_memory
    
    cpu_count = mp.cpu_count()
    mem=virtual_memory()
    mem_tot = mem.free/1000000000
    cpu_mem_ratio = mem_tot/cpu_count
    if classification == True:
        H = 0.5
    else:
        H = 1
    tot_points_cpu = (3906205*H) * cpu_mem_ratio
    chunk = int(tot_points_cpu / k)
    return chunk

def mp_preprocessing(n_points):
    import multiprocessing as mp
    import numpy as np    
    
    cpu_count = mp.cpu_count()
    print("Processing ",n_points, " points")
    print("Total processors (hyper threading) available: ",mp.cpu_count(),". Using: ",cpu_count, "threads")
    
         
    #Maximum amount of points to go through
    
    num_points = n_points
    
    cpu_count = mp.cpu_count()
          
    #Number of points per CPU
    
    points_p_cpu = int(num_points/cpu_count)
    print("points per cpu ", points_p_cpu)
      
    
    index_order = np.zeros((cpu_count,2),dtype=int)
    
    #Create an index work order per core
    
    print("Create start stop indexes for n_cpu_cores-1")
    
    for l in range(0,cpu_count):
    
        index_start = int(l * (points_p_cpu))
    
        index_order[l] = [int(index_start),int(index_start + (points_p_cpu))]
    
        if l == cpu_count-1:
    
            index_order[-1,1] = index_order[-1,1] + num_points%cpu_count              #Add remaining index to last cpu core
    
    print(index_order) 
        
    return index_order


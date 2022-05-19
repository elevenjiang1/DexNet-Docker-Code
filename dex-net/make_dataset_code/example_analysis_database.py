# -*- coding: utf-8 -*
"""
Extract object shape and grasp difficulty from database
Object shape use voxel to represent
"""
from tqdm import tqdm
import copy
import IPython
import logging
import numpy as np
import os
import sys
import time
import open3d as o3d
from unittest import TestCase, TestSuite, TextTestRunner

from autolab_core import RigidTransform, YamlConfig

from perception import CameraIntrinsics, RenderMode

from meshpy.obj_file import ObjFile
from meshpy.mesh_renderer import ViewsphereDiscretizer, VirtualCamera

from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS,WRITE_ACCESS
from dexnet.database import Hdf5Database, MeshProcessor, RescalingType
from dexnet.grasping import ParallelJawPtGrasp3D,GraspableObject3D, AntipodalGraspSampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper

from dexnet.visualization import DexNetVisualizer3D as vis

import shutil

Abs_Path=os.path.dirname(os.path.abspath(__file__))

    
def extract_obj_grasp_info(database_path,generate_path):
    """
    Extract object and object's q75 ability
    :param database_path: .hdf5 path
    :param generate_path: save voxels (as .ply) and record.txt file

    Note: record.txt save:

    """
    #1: Get dataset and check generate_path
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database[database.datasets[0].name]

    if os.path.exists(generate_path):
        check_data=raw_input("!!generate_path exist,input y for remove origin dataset,others for return!!")
        if check_data=='y':
            os.removedirs(generate_path)
            os.mkdir(generate_path)
            os.mkdir(os.path.join(generate_path,"voxels"))
    else:
        os.mkdir(generate_path)
        os.mkdir(os.path.join(generate_path,"voxels"))

    #2: extract each obj,their shape and grasp_rate
    print("Extracting {} objects...".format(len(dataset.object_keys)))
    for i,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        #2.1 load data(obj,grasps,grasp_metrics)
        obj=dataset[key]
        grasps=dataset.grasps(key)
        grasp_metrics=dataset.grasp_metrics(key,grasps)

        #2.2 load mesh file,change to pointcloud and normalized,Finally change to voxel shape
        #get o3d_mesh
        obj_mesh=obj.mesh
        o3d_mesh=o3d.geometry.TriangleMesh()
        o3d_mesh.vertices=o3d.utility.Vector3dVector(obj_mesh.vertices)
        o3d_mesh.triangles=o3d.utility.Vector3iVector(obj_mesh.triangles)

        #change to voxel
        o3d_voxel=o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh,voxel_size=np.max(obj_mesh.vertices)*2/63.0)
        o3d_voxel.origin=np.array([0,0,0])

        #save voxels
        save_ply_path=os.path.join(generate_path,"voxels","{}.ply".format(i))
        o3d.io.write_voxel_grid(save_ply_path,o3d_voxel,write_ascii=True)

        #2.3 sort grasp_metrics and get grasp rate
        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])
        if len(qualities)==0:#index 1049,1266,1289 has no qualities in origin data,new database has remove.
            print("!!!!!!!!!!!! Index:{} its qualities is: {}".format(i,len(qualities)))
            continue
        qualities.sort()
        q75=qualities[int(len(qualities)*0.75)]

        #2.4 index txt for "index:grasp_rate:key_name"
        save_txt=open(os.path.join(generate_path,"record.txt"),'a')
        save_txt.write("{}:{}:{}\n".format(i,q75,key))
        save_txt.close()




#####################################example code###############################
def example_getinfo_database():
    """
    example code for see data in database
    """
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    database_path=os.path.join(Abs_Path,'data/example_data/example_database.hdf5')
    generate_path=os.path.join(Abs_Path,'data/example_data/example_database_info')
    extract_obj_grasp_info(database_path, generate_path)

def extract_info_database(database_path,generate_path):
    """
    extract shape info and grasp quality from grasp
    """
    extract_obj_grasp_info(database_path, generate_path)



if __name__ == '__main__':
    ### Usage: ###
    # cd /path/to/dex-net
    # python object_synthesis_code/analysis_database.py
    ### Usage: ###

    print("********begin function***********")
    example_getinfo_database()

    # dataset_path="/home/Project/Code/data/dexnet_2_database.hdf5"
    # generate_path="/home/Project/Code/code/DexNet/dex-net/learn_dexnet/data/shape_grasp_analysis"
    # extract_info_database(database_path, generate_path)
    print("********end function***********")
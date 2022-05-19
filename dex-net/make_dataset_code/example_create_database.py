# -*- coding: utf-8 -*
"""
Specify a path which contains .obj files
Generate h5py file,including:
1. create graspable objects with MeshProcesser
3. calculate grasps
4. calculate grasp metrics
!!!Note!!!
The code should run in dex-net floder
python object_synthesis_code/create_database.py
"""
import os
import sys
import time
from tqdm import tqdm
import copy
import IPython
import logging
import numpy as np

from unittest import TestCase, TestSuite, TextTestRunner
from autolab_core import RigidTransform, YamlConfig
from perception import CameraIntrinsics, RenderMode

from meshpy.obj_file import ObjFile
from meshpy.mesh_renderer import ViewsphereDiscretizer, VirtualCamera

from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS,WRITE_ACCESS
from dexnet.database import Hdf5Database, MeshProcessor, RescalingType
from dexnet.grasping import ParallelJawPtGrasp3D,GraspableObject3D, AntipodalGraspSampler, GraspQualityConfigFactory, GraspQualityFunctionFactory, RobotGripper

from dexnet.visualization import DexNetVisualizer3D as vis

Abs_Path=os.path.dirname(os.path.abspath(__file__))

def create_database(config_path,objs_path,generate_path,dataset_name=None):
    """
    create a database(hdf5 file),which contain obj,obj.stable_poses,obj_grasp_metric
    :param config_path: example file can reference /data/cfg/generate_database.yaml
    :param objs_path: path of Folder which want to add to database file
    :param generate_path: a new .hdf5 file path
    :param dataset_name: dataset name in database,if is none,then will be mini_test
    """
    #1: load config file
    config_data=YamlConfig(config_path)
    config_data['obj_rescaling_type'] = RescalingType.RELATIVE

    #2: create database and dataset
    if not os.path.exists(generate_path):
        print("generate_path not exist,generate a new database")
        database=Hdf5Database(generate_path,access_level=WRITE_ACCESS)
    else:
        print("generate_path exist,read this database")
        database=Hdf5Database(generate_path,access_level=READ_WRITE_ACCESS)

    if dataset_name is None:
        dataset_name="mini_test"
    dataset=database.create_dataset(dataset_name)
    dataset=database.dataset(dataset_name)

    make_data_begin_time=time.time()

    #3: load graspable objects
    print("****************Adding all object from objs_path****************")
    objs_name_list=os.listdir(objs_path)
    objs_name_list.sort(key=lambda x:int(x[:-4]))#if need sort files
    for count_generate,obj_file in enumerate(objs_name_list):
        print("*****Create {} graspable objects | Finish ({}/{}) *****".format(count_generate,count_generate,len(objs_name_list)))
        if obj_file[-3:]!="obj":
            continue
            
        obj_path=os.path.join(objs_path,obj_file)
        mesh_processor=MeshProcessor(obj_path,config_data['cache_dir'])
        mesh_processor.generate_graspable(config_data)
        dataset.create_graspable(mesh_processor.key,mesh_processor.mesh,mesh_processor.sdf,mesh_processor.stable_poses,mass=1.0)
    
    print("!!!!!!!!!!!!!!!!!!!!!Add all graspable shape use time:{} min!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format((time.time()-make_data_begin_time)/60))
    make_data_begin_time=time.time()

    #4: add dataset.metric
    print("****************Adding all metrics****************")
    for metric_name in config_data['metrics'].keys():
        metric_config=config_data['metrics'][metric_name]
        dataset.create_metric(metric_name,metric_config)
    
    #5: add object's grasp and grasp metirc
    print("****************Calculating all objects' grasp quality****************")
    #5.1 init antipodal grasp sampler and grasp quality config
    GRIPPER_NAME = 'yumi_metal_spline'
    gripper=RobotGripper.load(GRIPPER_NAME)
    ags = AntipodalGraspSampler(gripper, config_data)
    # fc_quality_config = GraspQualityConfigFactory.create_config(config_data['metrics']['force_closure'])
    rfc_quality_config = GraspQualityConfigFactory.create_config(config_data['metrics']['robust_ferrari_canny'])
    # ferrari_canny_quality_config=GraspQualityConfigFactory.create_config(config_data['metrics']['ferrari_canny'])

    #add gripper parameter
    setattr(rfc_quality_config, 'force_limits', gripper.force_limit)
    setattr(rfc_quality_config, 'finger_radius', gripper.finger_radius)
    
    NUM_TEST_CASES=100
    num_all_objects=len(dataset.object_keys)
    
    for count_generate,key in enumerate(dataset.object_keys):
        print("*****Generating {} grasps and grasp metrics | Finish ({}/{}) | Use time: {} min*****".format(key,count_generate,num_all_objects,(time.time()-make_data_begin_time)/60))

        #####For some situtation when generate need run again####
        # temp_loaded_grasp=dataset.grasps(key)
        # if len(temp_loaded_grasp)!=0:
        #     print("index:{} has generate grasps! contine to generate grasps".format(count_generate))
        #     continue
        #####For some situtation when generate need run again####
        
        graspable_object_3d=dataset.graspable(key)
        #5.2 init 'force_closure grasp quality' and 'robust_ferrari_canny quality' function
        # fc_quality_fn = GraspQualityFunctionFactory.create_quality_function(graspable_object_3d, fc_quality_config)
        rfc_quality_fn = GraspQualityFunctionFactory.create_quality_function(graspable_object_3d, rfc_quality_config)
        # ferrari_canny_quality_fn=GraspQualityFunctionFactory.create_quality_function(graspable_object_3d, ferrari_canny_quality_config)

        #5.3 generate and save grasps
        grasps=ags.generate_grasps(graspable_object_3d,target_num_grasps=NUM_TEST_CASES,max_iter=5)
        dataset.store_grasps(key,grasps,gripper=GRIPPER_NAME)
        database.flush()

        #5.4 calculate and save grasp_metrics
        loaded_grasps=dataset.grasps(key)#must use loaded_grasps,otherwise grasp will not have grasp.id
        grasp_metrics = {}
        for grasp in tqdm(loaded_grasps,total=len(loaded_grasps)):
            #get metric
            # fc_metric=fc_quality_fn(grasp).quality
            rfc_metric=rfc_quality_fn(grasp).quality
            # ferrari_canny_metric=ferrari_canny_quality_fn(grasp).quality

            #save metirc
            grasp_metrics[grasp.id]={}
            # grasp_metrics[grasp.id]['force_closure']=fc_metric
            # grasp_metrics[grasp.id]['ferrari_canny']=ferrari_canny_metric
            grasp_metrics[grasp.id]['robust_ferrari_canny']=rfc_metric


        dataset.store_grasp_metrics(key,grasp_metrics,gripper=GRIPPER_NAME)
        database.flush()

    #6: close database
    print("!!!!!!Generate Finish,Save {} objects,close database!!!!!!".format(len(dataset.object_keys)))
    database.close()

def check_database(database_path,vis=False):
    """
    Use for show database_path file info
    """
    #1: load datasetbase and see datasets name
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    print("load database from:\n {} \nit has {} dataset,their names are:".format(database_path,len(database.datasets)))
    for dataset in database.datasets:
        print(dataset.name)

    #select first dataset
    dataset=database.dataset(database.datasets[0].name)

    #load gripper
    gripper_path="yumi_metal_spline"
    gripper=RobotGripper.load(gripper_path)

    #2: see graspable in dataset
    print("dataset has {} objects,their names are:".format(len(dataset.objects)))
    for index,key in enumerate(dataset.object_keys): 
        print("**** object index method ****")
        print("object key is: {}".format(key))
        print("dataset[index] is: {}".format(dataset[index]))#GraspableObject3D object

        #3: see stable poses
        print("**** object stable pose index ****")
        stable_poses=dataset.stable_poses(key)
        print("{} has {} stable poses,first is:".format(key,len(stable_poses)))
        print(stable_poses[0].T_obj_table)

        #4: see grasps 
        print("**** object grasp index ****")
        grasps=dataset.grasps(key)
        print("{} has {} grasps ,the first is:".format(key,len(grasps)))
        first_grasp=grasps[0]
        print("grasp.center:{}".format(first_grasp.center))
        print("grasp.axis:{}".format(first_grasp.axis))
        print("grasp.max_grasp_width:{}".format(first_grasp.open_width))
        print("Other detail can see class ParallelJawPtGrasp3D")
        

        #5: see grasps metrics
        print("**** object grasp_metrics index ****")
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        print("first grasp_metrics[0]['robust_ferrari_canny']:{}".format(grasp_metrics[0]['robust_ferrari_canny']))
        print("first grasp_metrics[0]['force_closure']:{}".format(grasp_metrics[0]['force_closure']))

        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])
        if len(qualities)==0:#index 1049,1266,1289 has no qualities in origin data,new database has remove.
            print("!!!!!!!!!!!! Index:{} its qualities is: {}".format(i,len(qualities)))
            continue
        qualities.sort()
        q75=qualities[int(len(qualities)*0.75)]
        print("load {} grasp_metrics,the q75 score is:{}".format(len(grasp_metrics),q75))

        #5: visualize the grasp info
        if vis:
            for index,stable_pose in enumerate(dataset.stable_poses(key)):
                grasps=dataset.grasps(key)
                aligned_grasps=[grasp.perpendicular_table(stable_pose) for grasp in grasps]

                obj=dataset[key]

                grasp_metrics=dataset.grasp_metrics(key,aligned_grasps)
                
                for i,aligned_grasp in enumerate(aligned_grasps):
                    print("{} data,grasp_metrics is: {}".format(i,grasp_metrics[i]))
                    vis.figure()
                    vis.gripper_on_object(gripper,aligned_grasp,obj,stable_pose.T_obj_world)
                    vis.show()
                    if i>3:
                        break


        



#####################################example code###############################
def example_create_database():
    """
    example for generate database in example_meshes
    """
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/example_data/example_database.hdf5")
    objs_path=os.path.join(Abs_Path,"data/example_data/example_meshes")
    # create_database(config_path,objs_path,generate_path)
    check_database(generate_path)

def create_custom_database(objs_path,generate_path):
    """
    create custom database from objs_path, generate database named generate_path
    :param objs_path: path of Folder which want to add to database file
    :param generate_path: a new .hdf5 file path
    """
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    create_database(config_path,objs_path,generate_path)
    check_database(generate_path)




if __name__ == '__main__':
    ### Usage: ###
    # cd /path/to/dex-net
    # python object_synthesis_code/create_database.py
    ### Usage: ###

    print("********begin function***********")
    example_create_database()

    # Abs_Path=os.path.dirname(os.path.abspath(__file__))
    # objs_path=os.path.join(Abs_Path,"data/example_data/example_meshes")
    # generate_path=os.path.join(Abs_Path,"data/example_data/example_database.hdf5")
    # create_custom_database(objs_path, generate_path)
    print("********end function***********")
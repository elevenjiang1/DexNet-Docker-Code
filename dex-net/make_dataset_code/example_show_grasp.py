import os
import sys
import numpy as np
from tqdm import tqdm
import mayavi.mlab as mv
import mayavi.mlab as mlab

from autolab_core import YamlConfig
from dexnet.database import Hdf5Database
from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS,WRITE_ACCESS
from autolab_core import YamlConfig, RigidTransform
from dexnet.grasping import RobotGripper

import matplotlib.pyplot as plt
from dexnet.visualization.visualizer3d import DexNetVisualizer3D as vis

def show_grasp():
    #1: load database
    
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_test_database.hdf5"
    
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/split_database/dexnet_2_database.hdf5"
    
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database.datasets[0]


    #2: load grasp:
    gripper_name="yumi_metal_spline"
    object_name="0"
    metric_name="robust_ferrari_canny"


    GRIPPER_NAME = 'yumi_metal_spline'
    gripper=RobotGripper.load(GRIPPER_NAME)

    show_index_list=[]

    for index,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        object = dataset[key]

        grasps=dataset.grasps(key)
        if len(grasps)==0:
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])

        qualities_array=np.array(qualities)
        temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
        each_object_positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
        if each_object_positive_rate<0.05:
            show_index_list.append(index)

        if index>20:
            break

    print("Find {} data".format(len(show_index_list)))


    for index,key in enumerate(dataset.object_keys):
        # if index not in show_index_list:
        #     continue
        object = dataset[key]

        grasps=dataset.grasps(key)
        if len(grasps)==0:
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)

        
        

        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])

        qualities_array=np.array(qualities)

        low = np.min(qualities_array)
        high = np.max(qualities_array)
        q_to_c = lambda quality: 0.3 * (quality - low) / (high - low)


        temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
        each_object_positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
        print("mean of quality array is:{}".format(np.sum(qualities_array)/qualities_array.shape[0]))
        print("object key is:{}".format(key))
        print("object grasp postive rate is:{}".format(each_object_positive_rate))

        i = 0
        stable_pose = dataset.stable_pose(object.key, 'pose_1')
        for grasp in grasps:
            quality=grasp_metrics[grasp.id]['robust_ferrari_canny']
            # if quality<0.001:
            #     continue
            # grasp = grasp.perpendicular_table(stable_pose)
        
            if quality>0.002:
                color=(0,1,0)
            else:
                color=(1,0,0)

            # print 'Grasp %d %s=%.5f' %(grasp.id, metric_name, metric)
            # T_obj_world = RigidTransform(from_frame='obj',
                                            # to_frame='world')
            # color = plt.get_cmap('hsv')(q_to_c(quality))[:-1]
            # T_obj_gripper = grasp.gripper_pose(gripper)
            vis.grasp(grasp, grasp_axis_color=color,
                        endpoint_color=color)
            i += 1
            # if i>30:
            #     break
        
        open3d_data=object.mesh.trimesh
        mlab.triangular_mesh(open3d_data.vertices[:,0],open3d_data.vertices[:,1],open3d_data.vertices[:,2],open3d_data.faces,color=(1,1,1))
        mlab.show()
    
    # vis.show(animate=False)


def better_show_grasp():
    #show a book, a cub a bottle
    show_index_list=[]

    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/split_database/dexnet_2_database.hdf5"
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database.datasets[0]

    for index,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        if index<30:
            continue
        # if index not in show_index_list:
        #     continue
        #1: get object property
        object=dataset[key]
        grasps=dataset.grasps(key)
        if len(grasps)==0:
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])
        qualities_array=np.array(qualities)
        low = np.min(qualities_array)
        high = np.max(qualities_array)
        q_to_c = lambda quality: 0.3 * (quality - low) / (high - low)

        temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
        each_object_positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
        print("mean of quality array is:{}".format(np.sum(qualities_array)/qualities_array.shape[0]))
        print("object key is:{}".format(key))
        print("object grasp postive rate is:{}".format(each_object_positive_rate))

        #2: get show list
        num_show_grasp=40
        num_positive_count=num_show_grasp*each_object_positive_rate
        num_negative_count=num_show_grasp*(1-each_object_positive_rate)

        all_show_grasp_list=[]
        count_positve=0
        count_negative=0
        for g in grasps:
            grasp_quality=grasp_metrics[g.id]['robust_ferrari_canny']
            if grasp_quality<0.001:
                continue
            if grasp_quality>0.002 and count_positve<num_positive_count:
                all_show_grasp_list.append(g)
                count_positve=count_positve+1

            elif grasp_quality<0.002 and count_negative<num_negative_count:
                all_show_grasp_list.append(g)
                count_negative=count_negative+1

            if len(all_show_grasp_list)>num_show_grasp:
                break

        #3: Final show grasp
        print("show grasp data is: {}".format(len(all_show_grasp_list)))
        print("count positive:{} and count negative:{}".format(count_positve,count_negative))
        for show_grasp in all_show_grasp_list:
            quality=grasp_metrics[show_grasp.id]['robust_ferrari_canny']
            if quality>0.002:
                color=(0,1,0)
            else:
                color=(1,0,0)
            # color = plt.get_cmap('hsv')(q_to_c(quality))[:-1]
            vis.grasp(show_grasp, grasp_axis_color=color,endpoint_color=color)

        open3d_data=object.mesh.trimesh
        mlab.triangular_mesh(open3d_data.vertices[:,0],open3d_data.vertices[:,1],open3d_data.vertices[:,2],open3d_data.faces,color=(1,1,1))
        mlab.show()




print("*******************Begin function***********************")
# show_grasp()
better_show_grasp()
print("*******************End function***********************")
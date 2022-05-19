# -*- coding: utf-8 -*
"""
Split a database to two databse
"""
import os
import sys
import time
import copy
import random
import pandas
import IPython
import logging

import numpy as np
import open3d as o3d
from tqdm import tqdm

from autolab_core import YamlConfig
from dexnet.database import Hdf5Database
from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS,WRITE_ACCESS

Abs_Path=os.path.dirname(os.path.abspath(__file__))

def generate_record_csv():
    """
    split train and test objects.In all 3dnet(1363 objects),random shuffle, then select 1000 as train and 363 as test
    """
    save_dict={}
    all_index=list(range(1363))
    save_dict['index']=copy.deepcopy(all_index)
    random.shuffle(all_index)
    train_index=all_index[:1000]
    train_index=sorted(train_index)
    test_index=all_index[1000:]
    test_index=sorted(test_index)

    print(test_index)
    all_list=[]
    for index in range(1363):
        if index in test_index:
            all_list.append('test')
        else:
            all_list.append('train')

    save_dict['split']=all_list
    data_frame=pandas.DataFrame(save_dict)
    data_frame.to_csv(os.path.join(Abs_Path,"data/split_database/split.csv"),index=False)


def example_split_database():
    #1: load origin databse and create train,test database
    # database_path=os.path.join(Abs_Path,"data/example_data/example_database.hdf5")
    database_path=os.path.join(Abs_Path,"data/split_database/dexnet_2_database.hdf5")
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database.datasets[0]

    train_database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    train_database=Hdf5Database(train_database_path,access_level=WRITE_ACCESS)
    train_dataset=train_database.create_dataset("train_dataset")

    test_database_path=os.path.join(Abs_Path,"data/split_database/test_database.hdf5")
    test_database=Hdf5Database(test_database_path,access_level=WRITE_ACCESS)
    test_dataset=test_database.create_dataset("test_dataset")

    #init new dataset
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    config_data=YamlConfig(config_path)
    for metric_name in config_data['metrics'].keys():
        metric_config=config_data['metrics'][metric_name]
        test_dataset.create_metric(metric_name,metric_config)
        train_dataset.create_metric(metric_name,metric_config)

    #2: load split dataset
    dataframe=pandas.read_csv(os.path.join(Abs_Path,"data/split_database/split.csv"))
    split_list=list(dataframe['split'])

    #3: split database
    save_key_list=[]
    for index,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        #load graspable,stable_poses,grasps,grasp_metrics
        obj=dataset[key]
        save_key_list.append(key)
        stable_poses=dataset.stable_poses(key)

        grasps=dataset.grasps(key)
        grasp_metrics=dataset.grasp_metrics(key,grasps)

        if split_list[index]=='train':
            train_dataset.create_graspable(obj.key,obj.mesh,obj.sdf,stable_poses,obj.mass)
            train_dataset.store_grasps(key,grasps,gripper='yumi_metal_spline')
            train_dataset.store_grasp_metrics(key,grasp_metrics,gripper='yumi_metal_spline')
        else:
            test_dataset.create_graspable(obj.key,obj.mesh,obj.sdf,stable_poses,obj.mass)
            test_dataset.store_grasps(key,grasps,gripper='yumi_metal_spline')
            test_dataset.store_grasp_metrics(key,grasp_metrics,gripper='yumi_metal_spline')

    dataframe['key']=save_key_list
    dataframe.to_csv(os.path.join(Abs_Path,"data/split_database/all_info.csv"),index=False)

    #4: check new database
    print("test dataset have:")
    print(len(test_dataset.object_keys))
    for key in test_dataset.object_keys:
        print(key)

if __name__ == '__main__':
    print("********begin function***********")
    # generate_record_csv()
    # example_split_database()
    print("********end function***********")

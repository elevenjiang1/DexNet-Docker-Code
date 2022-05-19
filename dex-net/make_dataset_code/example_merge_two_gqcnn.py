import argparse
import copy
import IPython
import logging
import numpy as np
import os
import sys
from tqdm import tqdm

import autolab_core.utils as utils
from autolab_core import Point, YamlConfig
from perception import BinaryImage, ColorImage, DepthImage, GdImage, GrayscaleImage, RgbdImage, RenderMode

from gqcnn import Grasp2D
from gqcnn import Visualizer as vis2d

from dexnet.learning import TensorDataset
from dexnet.constants import READ_ONLY_ACCESS,WRITE_ACCESS,READ_WRITE_ACCESS

Abs_Path=os.path.dirname(os.path.abspath(__file__))


def add_tensor_dataset(dataset_path1,dataset_path2,generate_dataset_path,config_path=None,all_count_num=None):
    """
    :param dataset_path1: the gqcnn path of NO.1 dataset
    :param dataset_path2: the gqcnn path of NO.2 dataset
    :param config_path: yaml file for merage
    """
    #1: load both dataset and config path
    dataset_1=TensorDataset.open(dataset_path1)
    dataset_2=TensorDataset.open(dataset_path2)

    #2: generate a new dataset for saveing two dataset
    #2.1 get tensor config from config file
    if config_path is None:
        config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    config=YamlConfig(config_path)
    tensor_config=config['tensors']

    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']

    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    tensor_config['fields']['obj_masks']['height'] = im_final_height
    tensor_config['fields']['obj_masks']['width'] = im_final_width

    tensor_config['fields']['robust_ferrari_canny'] = {}
    tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
    # tensor_config['fields']['force_closure'] = {}
    # tensor_config['fields']['force_closure']['dtype'] = 'float32'

    #2.2 create new_gqcnn_dataset
    new_gqcnn_dataset=TensorDataset(generate_dataset_path,tensor_config)
    print("New dataset will generate in : {}".format(generate_dataset_path))

    #2.3 begin to merge too dataset
    # all_count_num=10000
    generated_count=0
    print("Ready to save dataset1 to new dataset")
    print("dataset1 include data: {}".format(len(dataset_1.datapoint_indices)))
    for i,data in tqdm(enumerate(dataset_1),total=len(dataset_1.datapoint_indices)):
        new_gqcnn_dataset.add(data)
        if all_count_num is not None:
            generated_count=generated_count+1
            if generated_count>all_count_num:
                break

    print("Ready to save dataset2 to new dataset")
    print("dataset2 include data: {}".format(len(dataset_2.datapoint_indices)))
    for i,data in tqdm(enumerate(dataset_2),total=len(dataset_2.datapoint_indices)):
        new_gqcnn_dataset.add(data)
        if all_count_num is not None:
            generated_count=generated_count+1
            if generated_count>all_count_num:
                break

    ###For debug###
    # for i,data in tqdm(enumerate(range(10))):
    #     new_gqcnn_dataset.add(dataset_1[i])

    # for i,data in tqdm(enumerate(range(10))):
    #     new_gqcnn_dataset.add(dataset_2[i])
    ###For debug###


    #2.4 generate all tensor
    new_gqcnn_dataset.flush()
    print("new generate dataset consist data : {}".format(len(new_gqcnn_dataset.datapoint_indices)))

def add_specify_label_tensor_dataset(dataset_path1,dataset_path2,generate_dataset_path,specify_label_list=None,config_path=None,all_count_num=None):
    """
    :param dataset_path1: the gqcnn path of NO.1 dataset
    :param dataset_path2: the gqcnn path of NO.2 dataset
    :param config_path: yaml file for merage
    """
    #1: load both dataset and config path
    dataset_1=TensorDataset.open(dataset_path1)
    dataset_2=TensorDataset.open(dataset_path2)

    #2: generate a new dataset for saveing two dataset
    #2.1 get tensor config from config file
    if config_path is None:
        config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    config=YamlConfig(config_path)
    tensor_config=config['tensors']

    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']

    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    tensor_config['fields']['obj_masks']['height'] = im_final_height
    tensor_config['fields']['obj_masks']['width'] = im_final_width

    tensor_config['fields']['robust_ferrari_canny'] = {}
    tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
    # tensor_config['fields']['force_closure'] = {}
    # tensor_config['fields']['force_closure']['dtype'] = 'float32'

    #2.2 create new_gqcnn_dataset
    new_gqcnn_dataset=TensorDataset(generate_dataset_path,tensor_config)
    print("New dataset will generate in : {}".format(generate_dataset_path))

    #2.3 begin to merge too dataset
    # all_count_num=100
    generated_count=0
    print("Ready to save dataset1 to new dataset")
    print("dataset1 include data: {}".format(len(dataset_1.datapoint_indices)))
    for i,data in tqdm(enumerate(dataset_1),total=len(dataset_1.datapoint_indices)):
        if specify_label_list is not None:
            if data['obj_labels'] in specify_label_list:
                new_gqcnn_dataset.add(data)

        if all_count_num is not None:
            generated_count=generated_count+1
            if generated_count>all_count_num:
                break


    print("Ready to save dataset2 to new dataset")
    print("dataset2 include data: {}".format(len(dataset_2.datapoint_indices)))
    for i,data in tqdm(enumerate(dataset_2),total=len(dataset_2.datapoint_indices)):
        new_gqcnn_dataset.add(data)
        if all_count_num is not None:
            generated_count=generated_count+1
            if generated_count>all_count_num:
                break

    ###For debug###
    # for i,data in tqdm(enumerate(range(10))):
    #     new_gqcnn_dataset.add(dataset_1[i])

    # for i,data in tqdm(enumerate(range(10))):
    #     new_gqcnn_dataset.add(dataset_2[i])
    ###For debug###


    #2.4 generate all tensor
    new_gqcnn_dataset.flush()
    print("new generate dataset consist data : {}".format(len(new_gqcnn_dataset.datapoint_indices)))


def check_new_dataset(dataset_path,visualization_config_path=None):
    #1: open new dataset
    dataset=TensorDataset.open(dataset_path)
    if visualization_config_path is None:
        visualization_config_path=os.path.join(Abs_Path,'data/cfg/visualize_gqcnn_dataset.yaml')
    config=YamlConfig(visualization_config_path)
    print("dataset consist data:")
    print(len(dataset.datapoint_indices))


    print("Visualization dataset")
    indices = dataset.datapoint_indices
    np.random.shuffle(indices)
    

    # read config
    field_name = config['field_name']
    field_type = config['field_type']
    font_size = config['font_size']
    print_fields = config['print_fields']
    gripper_width_px = config['gripper_width_px']

    # see each data
    for i,index in enumerate(indices):
        datapoint=dataset[index]
        data = datapoint[field_name]
        if field_type == RenderMode.SEGMASK:
            image = BinaryImage(data)
        elif field_type == RenderMode.DEPTH:
            image = DepthImage(data)
        else:
            raise ValueError('Field type %s not supported!' %(field_type))


        print("****DATAPOINT {}****".format(index))
        for f in print_fields:
            data = datapoint[f]
            print("filed name : {} | its data is".format(f))
            print(data)
        print("success at index: {}".format(index))

        grasp_2d = Grasp2D(Point(image.center), 0, datapoint['hand_poses'][2])

        vis2d.figure()
        if field_type == RenderMode.RGBD:
            vis2d.subplot(1,2,1)
            vis2d.imshow(image.color)
            vis2d.grasp(grasp_2d, arrow_head_width=gripper_width_px)
            vis2d.subplot(1,2,2)
            vis2d.imshow(image.depth)
            vis2d.grasp(grasp_2d, arrow_head_width=gripper_width_px)
        elif field_type == RenderMode.GD:
            vis2d.subplot(1,2,1)
            vis2d.imshow(image.gray)
            vis2d.grasp(grasp_2d, arrow_head_width=gripper_width_px)
            vis2d.subplot(1,2,2)
            vis2d.imshow(image.depth)
            vis2d.grasp(grasp_2d, arrow_head_width=gripper_width_px)
        else:
            vis2d.imshow(image)
            vis2d.grasp(grasp_2d, arrow_head_width=gripper_width_px)
        vis2d.title('Datapoint %d: %s' %(index, field_type))
        vis2d.show()

def custom_add_tensor_dataset(dataset_path1,dataset_path2,generate_dataset_path,config_path=None,all_count_num=None):
    """
    :param dataset_path1: the gqcnn path of NO.1 dataset
    :param dataset_path2: the gqcnn path of NO.2 dataset
    :param config_path: yaml file for merage
    """
    #1: load both dataset and config path
    dataset_1=TensorDataset.open(dataset_path1)
    dataset_2=TensorDataset.open(dataset_path2)

    #2: generate a new dataset for saveing two dataset
    #2.1 get tensor config from config file
    if config_path is None:
        config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    config=YamlConfig(config_path)
    tensor_config=config['tensors']

    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']

    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    tensor_config['fields']['obj_masks']['height'] = im_final_height
    tensor_config['fields']['obj_masks']['width'] = im_final_width

    tensor_config['fields']['robust_ferrari_canny'] = {}
    tensor_config['fields']['robust_ferrari_canny']['dtype'] = 'float32'
    # tensor_config['fields']['force_closure'] = {}
    # tensor_config['fields']['force_closure']['dtype'] = 'float32'

    #2.2 create new_gqcnn_dataset
    new_gqcnn_dataset=TensorDataset(generate_dataset_path,tensor_config)
    print("New dataset will generate in : {}".format(generate_dataset_path))

    #2.3 begin to merge too dataset
    print("Ready to save dataset1 to new dataset")
    print("dataset1 include data: {}".format(len(dataset_1.datapoint_indices)))
    generate_count=0
    for i,data in tqdm(enumerate(dataset_1),total=len(dataset_1.datapoint_indices)):
        if i%1000==0:
            print("load {} data, generate count is:{}".format(i,generate_count))

        if 0.0015<data['robust_ferrari_canny']<0.004:
            new_gqcnn_dataset.add(data)
            generate_count=generate_count+1
        else:
            # print("not generate")
            # print(data['robust_ferrari_canny'])
            continue

        

    print("Ready to save dataset2 to new dataset")
    print("dataset2 include data: {}".format(len(dataset_2.datapoint_indices)))
    for i,data in tqdm(enumerate(dataset_2),total=len(dataset_2.datapoint_indices)):
        new_gqcnn_dataset.add(data)
        

    ###For debug###
    # for i,data in tqdm(enumerate(range(10))):
    #     new_gqcnn_dataset.add(dataset_1[i])

    # for i,data in tqdm(enumerate(range(10))):
    #     new_gqcnn_dataset.add(dataset_2[i])
    ###For debug###


    #2.4 generate all tensor
    new_gqcnn_dataset.flush()
    print("new generate dataset consist data : {}".format(len(new_gqcnn_dataset.datapoint_indices)))

#####################################example code###############################
def example_merge_twogqcnn_data():
    """
    example for merge two generated gqcnn dataset
    """
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/example_data/example_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/example_data/example_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/example_data/example_merge_gqcnn_data')
    add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

    check_new_dataset(generate_dataset_path)

def example_merge_specify_label_twogqcnn_data():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/example_data/example_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/example_data/example_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/example_data/example_merge_gqcnn_data')

    add_specify_label_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path,specify_label_list=[0,2])



def merge_custom_data(dataset_path1,dataset_path2,generate_dataset_path):
    add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)
    check_new_dataset(generate_dataset_path)

def see_data(dataset_path):
    dataset=TensorDataset.open(dataset_path)
    indices = dataset.datapoint_indices

    visualization_config_path=os.path.join(Abs_Path,'data/cfg/visualize_gqcnn_dataset.yaml')
    config=YamlConfig(visualization_config_path)
    field_name = config['field_name']
    field_type = config['field_type']
    font_size = config['font_size']
    print_fields = config['print_fields']
    gripper_width_px = config['gripper_width_px']

    for i,index in enumerate(indices):
        datapoint=dataset[index]
        data = datapoint[field_name]
        metric=datapoint['robust_ferrari_canny']
        print("data is:")
        print(data.shape)
        print("metric is:")
        print(metric)


        return




if __name__ == '__main__':
    ### Usage: ###
    # cd /path/to/dex-net
    # python object_synthesis_code/merge_two_gqcnn.py
    ### Usage: ###
    # parse args
    # logging.getLogger().setLevel(logging.INFO)
    
    print("************begin to run code***********************")
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    # example_merge_twogqcnn_data()
    example_merge_specify_label_twogqcnn_data()

    # dataset_path1="/home/Project/Code/code/DexNet/dex-net/learn_dexnet/data/newmesh_gqcnn_data"
    # dataset_path2="/home/Project/Code/code/DexNet/dex-net/make_gqcnn_data/data/train_gqcnn_data"
    # generate_dataset_path=os.path.join(Abs_Path,"data/compare_dataset/single_10000_train_gqcnn_data")
    # merge_custom_data(dataset_path1, dataset_path2, generate_dataset_path)
    print("************end the code***********************")
    # see_data(os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data"))
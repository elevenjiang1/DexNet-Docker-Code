# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Generates datasets of synthetic point clouds, grasps, and grasp robustness metrics from a Dex-Net HDF5 database for GQ-CNN training.

Author
------
Jeff Mahler

YAML Configuration File Parameters
----------------------------------
database_name : str
    full path to a Dex-Net HDF5 database
target_object_keys : :obj:`OrderedDict`
    dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
env_rv_params : :obj:`OrderedDict`
    parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
gripper_name : str
    name of the gripper to use
"""
import argparse
import collections
import cPickle as pkl
import gc
import IPython
import json
import logging
import numpy as np
import os
import random
import shutil
import sys
import time

from autolab_core import Point, RigidTransform, YamlConfig
import autolab_core.utils as utils
from gqcnn import Grasp2D
from gqcnn import Visualizer as vis2d
from meshpy import ObjFile, RenderMode, SceneObject, UniformPlanarWorksurfaceImageRandomVariable
from perception import CameraIntrinsics, BinaryImage, DepthImage

from dexnet.constants import READ_ONLY_ACCESS
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.learning import TensorDataset

from dexnet.visualization import DexNetVisualizer3D as vis


Abs_Path=os.path.dirname(os.path.abspath(__file__))

logging.root.name = 'dex-net'

# seed for deterministic behavior when debugging
SEED = 197561

# name of the grasp cache file
CACHE_FILENAME = 'grasp_cache.pkl'

class GraspInfo(object):
    """ Struct to hold precomputed grasp attributes.
    For speeding up dataset generation.
    """
    def __init__(self, grasp, collision_free, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.phi = phi

def generate_gqcnn_dataset(generate_path,database_path,config_path,select_index_list=None):
    """
    Generates a GQ-CNN TensorDataset for training models with new grippers, quality metrics, objects, and cameras.

    :param generate_path: strpath to save the dataset to
    :param database_path: Dex-Net database path containing the 3D meshes, grasps, and grasp metrics
    :param config_path: config path
    :param select_index_list: contain need to generate object's index.If is None,generate all objects
    """
    #region
    #1: Init all gen params
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    config=YamlConfig(config_path)
        
    #1.1 setup render related param
    image_samples_per_stable_pose = config['images_per_stable_pose']
    stable_pose_min_p = config['stable_pose_min_p']
    env_rv_params=config['env_rv_params']#parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)

    #1.2 setup grasp related param
    #gripper
    gripper_name=config['gripper']
    gripper=RobotGripper.load(gripper_name)

    #get phi_offsets
    table_alignment_params = config['table_alignment']
    min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
    num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

    phi_offsets = []
    if max_grasp_approach_offset == min_grasp_approach_offset:
        phi_inc = 1
    elif num_grasp_approach_samples == 1:
        phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
    else:
        phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)
                                                            
    phi = min_grasp_approach_offset
    while phi <= max_grasp_approach_offset:
        phi_offsets.append(phi)
        phi += phi_inc

    #setup collision checking
    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    if not os.path.isabs(table_mesh_filename):
        table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', table_mesh_filename)
    table_mesh = ObjFile(table_mesh_filename).read()

    #1.3 setup save tensor related param
    #load gqcnn data param
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2

    #set tensor dataset config
    tensor_config = config['tensors']
    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    tensor_config['fields']['obj_masks']['height'] = im_final_height
    tensor_config['fields']['obj_masks']['width'] = im_final_width

    #add available metrics (assuming same are computed for all objects)
    metric_names=[]
    datasets=database.datasets
    dataset = datasets[0]
    if len(dataset.object_keys) == 0:
        raise ValueError('No valid objects in dataset %s' %(dataset.name))
    grasps = dataset.grasps(dataset.object_keys[0])
    grasp_metrics = dataset.grasp_metrics(dataset.object_keys[0], grasps)
    metric_names = grasp_metrics[grasp_metrics.keys()[0]].keys()
    for metric_name in metric_names:
        tensor_config['fields'][metric_name] = {}
        tensor_config['fields'][metric_name]['dtype'] = 'float32'
    #endregion

    #2. Init presave dataset
    # init tensor dataset
    tensor_dataset = TensorDataset(generate_path, tensor_config)
    tensor_datapoint = tensor_dataset.datapoint_template

    # setup log file
    experiment_log_filename = os.path.join(generate_path, 'dataset_generation.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    hdlr = logging.FileHandler(experiment_log_filename)
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr)
    root_logger = logging.getLogger()

    # copy config file
    out_config_filename = os.path.join(generate_path, 'dataset_generation.json')
    ordered_dict_config = collections.OrderedDict()
    for key in config.keys():
        ordered_dict_config[key] = config[key]
    with open(out_config_filename, 'w') as outfile:
        json.dump(ordered_dict_config, outfile)

    #3: Begin to generate
    #3.1 Precompute the set of valid grasps for each stable pose:
    #  i) Perpendicular to the table
    # ii) Collision-free along the approach direction

    # load grasps if they already exist
    grasp_cache_filename = os.path.join(generate_path, CACHE_FILENAME)
    if os.path.exists(grasp_cache_filename):
        logging.info('Loading grasp candidates from file')
        candidate_grasps_dict = pkl.load(open(grasp_cache_filename, 'rb'))
    # otherwise re-compute by reading from the database and enforcing constraints
    else:
        print("*****************************")
        print("*****Generate Cache File*****")
        print("*****************************")
        # create grasps dict
        candidate_grasps_dict = {}
        
        # loop through dataset.objects
        dataset=datasets[0]
        logging.info('Reading dataset %s' %(dataset.name))
        for all_index,obj in enumerate(dataset):
            ####select data in select_index_list###
            # if select_index_list is not None:
            #     if all_index not in select_index_list:
            #         print("object key {} not in select_index_list".format(obj.key))
            #         continue
            if select_index_list is not None:
                if obj.key not in select_index_list:
                    print("object key {} not in select_index_list".format(obj.key))
                    continue
            # if select_index_list is not None:
            #     if int(obj.key) not in select_index_list:
            #         print("object key {} not in select_index_list".format(obj.key))
            #         continue
            #     else:
            #         print("generate object key is:{}".format(obj.key))
            ####select key in select_index_list###

            print("**************Now is Procesing {} index data | ({}/{}) left******************".format(all_index,all_index,len(dataset.object_keys)))
            # init candidate grasp storage
            candidate_grasps_dict[obj.key] = {}

            # setup collision checker
            collision_checker = GraspCollisionChecker(gripper)
            collision_checker.set_graspable_object(obj)

            # read in the stable poses of the mesh
            try:
                stable_poses = dataset.stable_poses(obj.key)
            except:
                print("*************key {} error!!!! continue***************".format(obj.key))
                continue
            for i, stable_pose in enumerate(stable_poses):
                # render images if stable pose is valid
                if stable_pose.p > stable_pose_min_p:
                    candidate_grasps_dict[obj.key][stable_pose.id] = []

                    # setup table in collision checker
                    T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                    T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
                    T_table_obj = T_obj_table.inverse()
                    collision_checker.set_table(table_mesh_filename, T_table_obj)

                    # read grasp and metrics
                    grasps = dataset.grasps(obj.key, gripper=gripper.name)
                    logging.info('Aligning %d grasps for object %s in stable %s' %(len(grasps), obj.key, stable_pose.id))

                    # align grasps with the table
                    aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]

                    # check grasp validity
                    logging.info('Checking collisions for %d grasps for object %s in stable %s' %(len(grasps), obj.key, stable_pose.id))
                    for aligned_grasp in aligned_grasps:
                        # check angle with table plane and skip unaligned grasps
                        _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                        perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                        if not perpendicular_table: 
                            continue

                        # check whether any valid approach directions are collision free
                        collision_free = False
                        for phi_offset in phi_offsets:
                            rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                            collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
                            if not collides:
                                collision_free = True
                                break
                
                        # store if aligned to table
                        candidate_grasps_dict[obj.key][stable_pose.id].append(GraspInfo(aligned_grasp, collision_free))

                        # visualize if specified
                        if collision_free and config['vis']['candidate_grasps']:
                            logging.info('Grasp %d' %(aligned_grasp.id))
                            vis.figure()
                            vis.gripper_on_object(gripper, aligned_grasp, obj, stable_pose.T_obj_world)
                            vis.show()
                                
        # save to file
        logging.info('Saving to file')
        pkl.dump(candidate_grasps_dict, open(grasp_cache_filename, 'wb'))

    #3.2 Render a dataset of images and associate the gripper pose with image coordinates for each grasp in the Dex-Net database

    # setup variables
    obj_category_map = {}
    pose_category_map = {}

    cur_pose_label = 0
    cur_obj_label = 0
    cur_image_label = 0
                
    # render images for each stable pose of each object in the dataset
    render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]

    logging.info('Generating data for dataset %s' %(dataset.name))
    print("*****************************")
    print("*****Render dataset File*****")
    print("*****************************")

    # iterate through all object keys
    object_keys = dataset.object_keys
    for all_index,obj_key in enumerate(object_keys):
        obj = dataset[obj_key]
        ####select data in select_index_list###
        # if select_index_list is not None:
        #     if all_index not in select_index_list:
        #         print("object key {} not in select_index_list".format(obj.key))
        #         continue
        if select_index_list is not None:
            if obj.key not in select_index_list:
                print("object key {} not in select_index_list".format(obj.key))
                continue
        # if select_index_list is not None:
        #     if int(obj.key) not in select_index_list:
        #         print("object key {} not in select_index_list".format(obj.key))
        #         continue
        #     else:
        #         print("generate object key is:{}".format(obj.key))
        ####select key in select_index_list###

        print("**************Now is Procesing {} index data******************".format(all_index))
        # read in the stable poses of the mesh
        try:
            stable_poses = dataset.stable_poses(obj.key)
        except:
            print("*************key {} error!!!! continue***************".format(obj.key))
            continue
        for i, stable_pose in enumerate(stable_poses):

            # render images if stable pose is valid
            if stable_pose.p > stable_pose_min_p:
                # log progress
                logging.info('Rendering images for object %s in %s' %(obj.key, stable_pose.id))

                # add to category maps
                if obj.key not in obj_category_map.keys():
                    obj_category_map[obj.key] = cur_obj_label
                pose_category_map['%s_%s' %(obj.key, stable_pose.id)] = cur_pose_label

                # read in candidate grasps and metrics
                candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                candidate_grasps = [g.grasp for g in candidate_grasp_info]
                grasp_metrics = dataset.grasp_metrics(obj.key, candidate_grasps, gripper=gripper.name)

                # compute object pose relative to the table
                T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                T_obj_stp = obj.mesh.get_T_surface_obj(T_obj_stp)

                # sample images from random variable
                T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
                scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}
                urv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                                    render_modes,
                                                                    'camera',
                                                                    env_rv_params,
                                                                    stable_pose=stable_pose,
                                                                    scene_objs=scene_objs)
                
                render_start = time.time()
                render_samples = urv.rvs(size=image_samples_per_stable_pose)
                render_stop = time.time()
                logging.info('Rendering images took %.3f sec' %(render_stop - render_start))

                # visualize
                if config['vis']['rendered_images']:
                    d = int(np.ceil(np.sqrt(image_samples_per_stable_pose)))

                    # binary
                    vis2d.figure()
                    for j, render_sample in enumerate(render_samples):
                        vis2d.subplot(d,d,j+1)
                        vis2d.imshow(render_sample.renders[RenderMode.SEGMASK].image)

                    # depth table
                    vis2d.figure()
                    for j, render_sample in enumerate(render_samples):
                        vis2d.subplot(d,d,j+1)
                        vis2d.imshow(render_sample.renders[RenderMode.DEPTH_SCENE].image)
                    vis2d.show()

                # tally total amount of data
                num_grasps = len(candidate_grasps)
                num_images = image_samples_per_stable_pose 
                num_save = num_images * num_grasps
                logging.info('Saving %d datapoints' %(num_save))

                # for each candidate grasp on the object compute the projection
                # of the grasp into image space
                for render_sample in render_samples:
                    # read images
                    binary_im = render_sample.renders[RenderMode.SEGMASK].image
                    depth_im_table = render_sample.renders[RenderMode.DEPTH_SCENE].image
                    # read camera params
                    T_stp_camera = render_sample.camera.object_to_camera_pose
                    shifted_camera_intr = render_sample.camera.camera_intr

                    # read pixel offsets
                    cx = depth_im_table.center[1]
                    cy = depth_im_table.center[0]

                    # compute intrinsics for virtual camera of the final
                    # cropped and rescaled images
                    camera_intr_scale = float(im_final_height) / float(im_crop_height)
                    cropped_camera_intr = shifted_camera_intr.crop(im_crop_height, im_crop_width, cy, cx)
                    final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)

                    # create a thumbnail for each grasp
                    for grasp_info in candidate_grasp_info:
                        # read info
                        grasp = grasp_info.grasp
                        collision_free = grasp_info.collision_free
                        
                        # get the gripper pose
                        T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)
                        grasp_2d = grasp.project_camera(T_obj_camera, shifted_camera_intr)

                        # center images on the grasp, rotate to image x axis
                        dx = cx - grasp_2d.center.x
                        dy = cy - grasp_2d.center.y
                        translation = np.array([dy, dx])

                        binary_im_tf = binary_im.transform(translation, grasp_2d.angle)
                        depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)

                        # crop to image size
                        binary_im_tf = binary_im_tf.crop(im_crop_height, im_crop_width)
                        depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)

                        # resize to image size
                        binary_im_tf = binary_im_tf.resize((im_final_height, im_final_width), interp='nearest')
                        depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))
                        
                        # visualize the transformed images
                        if config['vis']['grasp_images']:
                            grasp_center = Point(depth_im_tf_table.center,
                                                    frame=final_camera_intr.frame)
                            tf_grasp_2d = Grasp2D(grasp_center, 0,
                                                    grasp_2d.depth,
                                                    width=gripper.max_width,
                                                    camera_intr=final_camera_intr)

                            # plot 2D grasp image
                            vis2d.figure()
                            vis2d.subplot(2,2,1)
                            vis2d.imshow(binary_im)
                            vis2d.grasp(grasp_2d)
                            vis2d.subplot(2,2,2)
                            vis2d.imshow(depth_im_table)
                            vis2d.grasp(grasp_2d)
                            vis2d.subplot(2,2,3)
                            vis2d.imshow(binary_im_tf)
                            vis2d.grasp(tf_grasp_2d)
                            vis2d.subplot(2,2,4)
                            vis2d.imshow(depth_im_tf_table)
                            vis2d.grasp(tf_grasp_2d)
                            vis2d.title('Coll Free? %d'%(grasp_info.collision_free))
                            vis2d.show()

                            # plot 3D visualization
                            #TODO
                            ##############Need to solve this bug#################
                            vis.figure()
                            vis.gripper_on_object(gripper, grasp,obj,stable_pose.T_obj_world)
                            vis.show()
                            ##############Need to solve this bug#################

                        # form hand pose array
                        hand_pose = np.r_[grasp_2d.center.y,
                                            grasp_2d.center.x,
                                            grasp_2d.depth,
                                            grasp_2d.angle,
                                            grasp_2d.center.y - shifted_camera_intr.cy,
                                            grasp_2d.center.x - shifted_camera_intr.cx,
                                            grasp_2d.width_px]
        

                        # store to data buffers
                        tensor_datapoint['depth_ims_tf_table'] = depth_im_tf_table.raw_data
                        tensor_datapoint['obj_masks'] = binary_im_tf.raw_data
                        tensor_datapoint['hand_poses'] = hand_pose
                        tensor_datapoint['collision_free'] = collision_free
                        tensor_datapoint['obj_labels'] = cur_obj_label
                        tensor_datapoint['pose_labels'] = cur_pose_label
                        tensor_datapoint['image_labels'] = cur_image_label

                        for metric_name, metric_val in grasp_metrics[grasp.id].iteritems():
                            coll_free_metric = (1 * collision_free) * metric_val
                            tensor_datapoint[metric_name] = coll_free_metric
                        tensor_dataset.add(tensor_datapoint)

                    # update image label
                    cur_image_label += 1

                # update pose label
                cur_pose_label += 1

                # force clean up
                gc.collect()

        # update object label
        cur_obj_label += 1

        # force clean up
        gc.collect()

    # save last file
    tensor_dataset.flush()

    # save category mappings
    obj_cat_filename = os.path.join(generate_path, 'object_category_map.json')
    json.dump(obj_category_map, open(obj_cat_filename, 'w'))
    pose_cat_filename = os.path.join(generate_path, 'pose_category_map.json')
    json.dump(pose_category_map, open(pose_cat_filename, 'w'))


#####################################example code###############################
def example_create_gqcnn():
    generate_path=os.path.join(Abs_Path,"data/example_data/example_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/example_data/example_database.hdf5")
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_gqcnn_dataset(generate_path, database_path, config_path)


def create_custum_gqcnn_data(generate_path,database_path,config_path,index_txt_path=None):
    # select train/test data from index.txt
    select_index_list=None
    if index_txt_path is not None:
        Abs_Path=os.path.dirname(os.path.abspath(__file__))
        text_index=open(index_txt_path,'r').read().splitlines()
        select_index_list=[]
        for data in text_index:
            temp=data.split('|')[0]
            index,grasp_class=temp.split(':')
            index,grasp_class=int(index),int(grasp_class)
            select_index_list.append(index)

    generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list)





if __name__ == '__main__':
    ### Usage: ###
    # cd /path/to/dex-net
    # python object_synthesis_code/create_gqcnn_data.py
    ### Usage: ###

    print("********begin function***********")
    logging.getLogger().setLevel(logging.INFO)
    example_create_gqcnn()


    print("********end function***********")
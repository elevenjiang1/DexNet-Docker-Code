"""
Generate all custom data here
# python object_synthesis_code/custom_generate_dataset.py
"""
import os
import sys
import shutil
import random
import pandas
import numpy as np
import example_analysis_database,example_create_database,example_create_gqcnn_data,example_merge_two_gqcnn




Abs_Path=os.path.dirname(os.path.abspath(__file__))

###############################First version make dataset####################################
def create_train_test_gqcnn_data(split='test'):
    """
    Generate train and test gqcnn dataset

    Usage:
    create_train_test_gqcnn_data(split='train')
    create_train_test_gqcnn_data(split='test')
    """
    #generate test_gqcnn_data
    if split=='test':
        generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/test_gqcnn_data")
        database_path=os.path.join(Abs_Path,"data/split_database/test_database.hdf5")
    else:
        generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/train_gqcnn_data")
        database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path)

def add_500_train_data_index():
    """
    Add select_500_data to all_info file.This is set 
    """
    #process csv file
    dataframe=pandas.read_csv(os.path.join(Abs_Path,'data/all_info.csv'))
    select_500_data=[]
    all_index=list(dataframe['index'])
    all_split=list(dataframe['split'])

    all_train_index=list(range(1000))
    for i in range(1000):
        if i<500:
            all_train_index[i]=1
        else:
            all_train_index[i]=0

    random.shuffle(all_train_index)
    print(all_train_index)
    count=0
    for i in range(len(all_index)):
        split=all_split[i]
        if split=='train':
            select_500_data.append(all_train_index[count])
            count=count+1
        else:
            select_500_data.append(0)

    dataframe['select_500_data']=select_500_data
    print("!!!Please check whether to add new data,If true, uncomment next line!!!")
    # dataframe.to_csv(os.path.join(Abs_Path,"data/all_info.csv"),index=False)


def create_select_500_gqcnn_data():
    """
    Need to change code in example_create_gqcnn_data.py
    ```
    #From:
    if select_index_list is not None:
        if all_index in select_index_list:
            continue
    #To:
    if select_index_list is not None:
        if obj.key not in select_index_list:
            continue
    ```
    """
    #1: load select_index_list
    select_index_list=[]
    dataframe=pandas.read_csv(os.path.join(Abs_Path,'data/all_info.csv'))
    all_index=list(dataframe['index'])
    all_split=list(dataframe['split'])
    all_key=list(dataframe['key'])
    all_select_index=list(dataframe['select_500_data'])

    for index in range(len(all_index)):
        if all_split[index]=='train' and all_select_index[index]==1:
            select_index_list.append(all_key[index])

    #2: generate gqcnn data
    print(select_index_list)

    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/select_500_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list=select_index_list)

def create_generate_meshes_database():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/split_database/generate_meshes_database.hdf5")
    objs_path="/home/Project/Code/code/data_augmentation/data/new_meshes/voxels"
    example_create_database.create_database(config_path,objs_path,generate_path)
    example_create_database.check_database(generate_path)

def check_generate_meshes_database():
    from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS,WRITE_ACCESS
    from dexnet.database import Hdf5Database
    database_path=os.path.join(Abs_Path,"data/split_database/generate_meshes_database.hdf5")
    record_txt=open(os.path.join(Abs_Path,'generate_record.txt'),'r')
    generate_key=record_txt.read().splitlines()

    #1: load database
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    print("load database from:\n {} \nit has {} dataset,their names are:".format(database_path,len(database.datasets)))
    for dataset in database.datasets:
        print(dataset.name)

    #select first dataset
    dataset=database.dataset(database.datasets[0].name)


    #2: see all grasp metircs

    count_compute=0
    for index,key in enumerate(dataset.object_keys):
        if key not in generate_key:
            continue
        print("*****************************************")
        grasps=dataset.grasps(key)
        if len(grasps)==0:
            print("!!!None generate grasps!!!!!!!!!!!")
        else:
            grasp_metrics=dataset.grasp_metrics(key,grasps)
            print("object key {} its grasp metrics are:{}".format(key,grasp_metrics[0]))
            count_compute=count_compute+1
            # record_txt.write("{}\n".format(key))
    print("Count amount is {}".format(count_compute))

def create_generate_meshes_gqcnn_data():
    """
    Need to change code in example_create_gqcnn_data.py
    ```
    #From:
    if select_index_list is not None:
        if all_index in select_index_list:
            continue
    #To:
    if select_index_list is not None:
        if obj.key not in select_index_list:
            continue
    ```
    """
    #1: generate gqcnn data
    select_key_list=open(os.path.join(Abs_Path,"generate_record.txt"),'r').read().splitlines()

    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/generate_meshes_gqcnn_data copy")
    database_path=os.path.join(Abs_Path,"data/split_database/generate_meshes_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list=select_key_list)

def merge_two_gqcnn_data():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/select_500_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/generate_meshes_gqcnn_data copy')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/select_500_generate_610')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def create_ycb_meshes_database():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/split_database/ycb_database.hdf5")
    objs_path="/home/Project/Code/code/data_augmentation/ycb_data/origin_data/all_meshes"
    example_create_database.create_database(config_path,objs_path,generate_path)
    example_create_database.check_database(generate_path)

def create_ycb_augmentation_database():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/split_database/ycb_augmentation_database.hdf5")
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_meshes/meshes"
    example_create_database.create_database(config_path,objs_path,generate_path)
    example_create_database.check_database(generate_path)

def create_ycb_gqcnn_data():
    #1: generate gqcnn data
    data_frame=pandas.read_csv("/home/Project/Code/code/data_augmentation/data/all_info.csv")
    select_78_data=list(data_frame['select_78_data'])
    all_index=np.array(list(data_frame['index'])) 
    select_index_list=[]
    for index,data in enumerate(select_78_data):
        if data==1:
            select_index_list.append(index)

    print("select_index_list len is:{}".format(len(select_index_list)))
    select_key_list=all_index[select_index_list]
    print("len select_key_list is:{}".format(len(select_key_list)))
    print("select key list is:")
    print(select_key_list)
    

    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/ycb_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/ycb_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list=select_key_list)

def check_ycb_gqcnn_data():
    data_frame=pandas.read_csv("/home/Project/Code/code/data_augmentation/data/all_info.csv")
    select_78_data=list(data_frame['select_78_data'])
    all_index=np.array(list(data_frame['index'])) 
    select_index_list=[]
    for index,data in enumerate(select_78_data):
        if data==1:
            select_index_list.append(index)

    print("select_index_list len is:{}".format(len(select_index_list)))
    select_key_list=all_index[select_index_list]
    print("len select_key_list is:{}".format(len(select_key_list)))
    print("select key list is:")
    print(select_key_list)
    

    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/ycb_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/ycb_database.hdf5")
    from dexnet.database import Hdf5Database
    from dexnet.constants import READ_ONLY_ACCESS

    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database.datasets[0]
    for all_index,obj in enumerate(dataset):
        print("*****Now is processing key {}*****".format(obj.key))
        if int(obj.key) not in select_index_list:
            # print("key {} not in select index list,continue~~~".format(obj.key))
            continue
        # try:
        stable_poses = dataset.stable_poses(obj.key)
        grasps=dataset.grasps(obj.key)
        print("Load {} stable poses grasps are:{}".format(len(stable_poses),len(grasps)))
        # except:
            # print("!!!!{} error!!!******************************".format(obj.key))

    # example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list=select_key_list)
    print("Finish generate ycb gqcnn data")

def create_ycb_augmentation_gqcnn_data():
    """
    Need to change code in example_create_gqcnn_data.py
    ```
    #From:
    if select_index_list is not None:
        if all_index in select_index_list:
            continue
    #To:
    if select_index_list is not None:
        if obj.key not in select_index_list:
            continue
    ```
    """
    #1: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/ycb_augmentation_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/ycb_augmentation_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path)

def merge_ycb_augmentation_gqcnn_data():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/ycb_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/ycb_augmentation_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/merge_ycb_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_positive_gqcnn_data():
    """
    """
    #1: select positivate index
    from tqdm import tqdm
    from dexnet.database import Hdf5Database
    from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS
    database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database[database.datasets[0].name]

    #save robust_ferrari_data
    all_robust_ferrari_list=[]
    positive_rate_list=[]
    select_index_list=[]

    for i,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        obj=dataset[key]
        grasps=dataset.grasps(key)
        if len(grasps)==0:
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        if len(grasp_metrics)==0:
            continue
        try:
            qualities=[]
            for g in grasps:
                qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])
                all_robust_ferrari_list.append(grasp_metrics[g.id]['robust_ferrari_canny'])

            qualities_array=np.array(qualities)
            temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
            positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
            positive_rate_list.append(positive_rate)
        except:
            print("obj key{} error!!".format(obj.key))
            break

        if positive_rate>0.2:
            select_index_list.append(i)

    print("select_index_list len is:{}".format(len(select_index_list)))

    #2: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/positive_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    # example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list)
    print("Finish generate positive gqcnn data")

def generate_500_positive_gqcnn_data():
    """
    """
    #1: select positivate index
    from tqdm import tqdm
    from dexnet.database import Hdf5Database
    from dexnet.constants import READ_WRITE_ACCESS,READ_ONLY_ACCESS
    database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    database=Hdf5Database(database_path,access_level=READ_ONLY_ACCESS)
    dataset=database[database.datasets[0].name]

    #save robust_ferrari_data
    all_robust_ferrari_list=[]
    positive_rate_list=[]
    select_index_list=[]

    for i,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        obj=dataset[key]
        grasps=dataset.grasps(key)
        if len(grasps)==0:
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        if len(grasp_metrics)==0:
            continue
        try:
            qualities=[]
            for g in grasps:
                qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])
                all_robust_ferrari_list.append(grasp_metrics[g.id]['robust_ferrari_canny'])

            qualities_array=np.array(qualities)
            temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
            positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
            positive_rate_list.append(positive_rate)
        except:
            print("obj key{} error!!".format(obj.key))
            break

        if positive_rate>0.18:
            select_index_list.append(i)

    print("select_index_list len is:{}".format(len(select_index_list)))

    #2: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/positive_500_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list)

def generate_500_lof_gqcnn_data():
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"3dnet_all_info.csv"))
    lof_all=np.array(list(data_frame['lof_all']))
    split_index=list(data_frame['split'])
    train_select_index_list=[]

    for index,lof_data in enumerate(lof_all):
        if split_index[index]=='test':
            continue
        else:
            train_select_index_list.append(index)

    train_lof_data=lof_all[train_select_index_list]
    sort_train_lof_data=sorted(train_lof_data,reverse=True)
    lof_threshold=sort_train_lof_data[500]
    print(lof_threshold)

    high_lof_index_list=[]
    for train_index in train_select_index_list:
        lof=lof_all[train_index]
        if lof>lof_threshold:
            high_lof_index_list.append(train_index)

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/highlof_500_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/dexnet_2_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,high_lof_index_list)

def merge_ycb_positive_gqcnn_data():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/positive_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/ycb_augmentation_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/merge_ycb_positive_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def create_select_100_gqcnn_data():
    """
    Need to change code in example_create_gqcnn_data.py
    ```
    #From:
    if select_index_list is not None:
        if all_index in select_index_list:
            continue
    #To:
    if select_index_list is not None:
        if obj.key not in select_index_list:
            continue
    ```
    """
    #1: load select_index_list
    select_key_list=[]
    dataframe=pandas.read_csv(os.path.join(Abs_Path,'data/all_info.csv'))
    all_index=list(dataframe['index'])
    all_split=list(dataframe['split'])
    all_key=list(dataframe['key'])
    all_select_index=list(dataframe['select_500_data'])

    for index in range(len(all_index)):
        if all_split[index]=='train' and all_select_index[index]==1:
            select_key_list.append(all_key[index])

    #2: generate gqcnn data
    print(len(select_key_list))
    import random
    random.shuffle(select_key_list)
    select_key_list=select_key_list[:100]
    print("new select_100_index:")
    print(len(select_key_list))

    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/select_100_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/train_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_index_list=select_key_list)

def merge_100_augmentation_gqcnn_data():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/select_100_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/generate_meshes_gqcnn_data copy')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/merge_100_augmentation_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def see_100_data():
    import cPickle as pkl

    #1: get select_100_data key
    select_100_key_list=[]
    grasp_cache_filename = os.path.join(Abs_Path,"data/train_test_gqcnn_data/select_100_gqcnn_data/grasp_cache.pkl")
    candidate_grasps_dict = pkl.load(open(grasp_cache_filename, 'rb'))
    for data in candidate_grasps_dict:
        select_100_key_list.append(data)
    print("select_100_key len is:{}".format(len(select_100_key_list)))


    #2: add select_100_data to csv file
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"3dnet_all_info.csv"))
    all_key_list=list(data_frame['key'])
    select_100_data=[]
    for key in all_key_list:
        if key in select_100_key_list:
            select_100_data.append(1)
        else:
            select_100_data.append(0)

    data_frame['select_100_data']=select_100_data
    data_frame.to_csv(os.path.join(Abs_Path,"3dnet_all_info.csv"),index=False)
    
def create_test_smooth_meshes_database():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/split_database/test_origin_smooth_meshes.hdf5")
    objs_path="/home/Project/Code/code/data_augmentation/data/better_generation/origin_meshes"
    example_create_database.create_database(config_path,objs_path,generate_path)

def create_augmentation_select_100_database():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/split_database/select_100_smooth_database.hdf5")
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_100_data_smooth_backup/meshes"
    example_create_database.create_database(config_path,objs_path,generate_path)
    example_create_database.check_database(generate_path)

def create_augmentation_select_100_smooth_less_database():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    generate_path=os.path.join(Abs_Path,"data/split_database/select_100_smoothless_database.hdf5")
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_100_data/meshes"
    example_create_database.create_database(config_path,objs_path,generate_path)
    example_create_database.check_database(generate_path)

def create_augmentation_select_100_gqcnn_data():
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/augmentation_select_100_smoothless_gqcnn_data")
    database_path=os.path.join(Abs_Path,"data/split_database/select_100_smoothless_database.hdf5")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path)

def merge_augmentation_select_100_gqcnn_data():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/augmentation_select_100_smoothless_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/select_100_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/merge_aug_100_smoothlessselect_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


###############################Generate in big function####################################
def generate_gqcnn_data_from_meshes(mesh_path,database_path,gqcnn_path):
    #1: generate database
    print("Generating database...")
    config_path=os.path.join(Abs_Path,"data/cfg/generate_database.yaml")
    database_path=database_path
    objs_path=mesh_path
    example_create_database.create_database(config_path,objs_path,database_path)
    
    #2: generate gqcnn data
    print("Generating gqcnn data...")
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    gqcnn_path=gqcnn_path

    ###For generate_select_data###
    # select_index_list=[]
    # for i in range(183):
    #     select_index_list.append(i)
    # example_create_gqcnn_data.generate_gqcnn_dataset(gqcnn_path, database_path, config_path,select_index_list=select_index_list)
    ###For generate_select_data###

    example_create_gqcnn_data.generate_gqcnn_dataset(gqcnn_path, database_path, config_path)

def example_generate_one_object():
    generate_type="origin_mesh"
    mesh_path=os.path.join(Abs_Path,"data/compare_data/mesh/{}".format(generate_type))
    database_path=os.path.join(Abs_Path,"data/compare_data/databases/{}.hdf5".format(generate_type))
    gqcnn_path=os.path.join(Abs_Path,"data/compare_data/gqcnn_data/{}".format(generate_type))
    generate_gqcnn_data_from_meshes(mesh_path,database_path,gqcnn_path)

def create_select_100_unsmooth_samesize_gqcnn():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_100_data_unsmooth_samesize_backup/meshes"
    database_path=os.path.join(Abs_Path,"data/split_database/select_100_unsmooth_samesize.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/select_100_unsmooth_samesize")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

def merge_select_100_unsmooth_samessize():
    Abs_Path=os.path.dirname(os.path.abspath(__file__))
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/select_100_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/select_100_unsmooth_samesize')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/merge_100_augmentation_unsmooth_samesize_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_again_origin_shapes():
    # #1: load all test shapes and select_100_shapes
    # origin_meshes_path="/home/elevenjiang/Documents/Project/ObjectSynthesis/datasets/DexNet-Dataset/model_meshes/3dnet"
    # test_meshes_path="/home/elevenjiang/Documents/Project/ObjectSynthesis/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/test_meshes"
    # select_100_meshes_path="/home/elevenjiang/Documents/Project/ObjectSynthesis/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/select_100_meshes"
    # train_meshes_path="/home/elevenjiang/Documents/Project/ObjectSynthesis/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/train_meshes"

    # data_frame=pandas.read_csv(os.path.join(Abs_Path,"3dnet_all_info.csv"))
    # select_100_list=list(data_frame['select_100_data'])
    # split_list=list(data_frame['split'])
    # key_list=list(data_frame['key'])
    # for index,key in enumerate(key_list):
    #     obj_path=os.path.join(origin_meshes_path,key+".obj")

    #     if select_100_list[index]==1:
    #         shutil.copy(obj_path,os.path.join(select_100_meshes_path,key+".obj"))
    #     if split_list[index]=='train':
    #         shutil.copy(obj_path,os.path.join(train_meshes_path,key+".obj"))
    pass
    
def generate_test_gqcnn_data_from_mesh():
    objs_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/test_meshes"
    database_path=os.path.join(Abs_Path,"data/myself_database/myself_test_database.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/myself_test_gqcnn_data_all")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

def generate_select_100_gqcnn_data_from_mesh():
    objs_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/select_100_meshes"
    database_path=os.path.join(Abs_Path,"data/myself_database/myself_select_100_database.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/myself_select_100_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    #merge dataset
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/myself_select_100_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/select_100_unsmooth_samesize')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/myself_aug_select_100_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_train_gqcnn_data_from_mesh():
    objs_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/train_meshes"
    database_path=os.path.join(Abs_Path,"data/myself_database/myself_train_database.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/myself_train_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")
    
def generate_select_100_aug_moredata_from_mesh():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_different_rate_aug100/meshes"
    database_path=os.path.join(Abs_Path,"data/split_database/aug_100_diff_rate.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/high_weiht_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    #merge dataset
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/myself_select_100_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/high_weiht_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/myself_aug_100_height_weight')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_select_100_aug_moreaugdata_lowrate():
    """
    Use 0.2 weights to generate new data,and check whether will improve
    """
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_different_rate_aug100/meshes"
    database_path=os.path.join(Abs_Path,"data/split_database/aug_100_diff_rate.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/train_test_gqcnn_data/high_weiht_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    #merge dataset
    dataset_path1=os.path.join(Abs_Path,'data/train_test_gqcnn_data/myself_select_100_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/train_test_gqcnn_data/high_weiht_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_dataset/myself_aug_100_height_weight')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


###############################New Generate in code_2####################################
def generate_high_lof_gqcnn_data():
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    lof_all=np.array(list(data_frame['lof_all']))
    split_index=list(data_frame['split'])
    key_list=list(data_frame['key'])

    #get train index data
    train_select_index_list=[]
    for index,lof_data in enumerate(lof_all):
        if split_index[index]=='test':
            continue
        else:
            train_select_index_list.append(index)

    #get train index data threshold
    train_lof_data=lof_all[train_select_index_list]
    sort_train_lof_data=sorted(train_lof_data,reverse=True)
    lof_threshold=sort_train_lof_data[100]
    print("lof threshold is:")
    print(lof_threshold)

    #get high 100 lof index list
    high_lof_index_list=[]
    high_lof_key_list=[]
    for train_index in train_select_index_list:
        lof=lof_all[train_index]
        if lof>lof_threshold:
            high_lof_index_list.append(train_index)
            high_lof_key_list.append(key_list[train_index])

    print("high lof key list len is:")
    print(len(high_lof_key_list))

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/highlof_100_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,high_lof_key_list)

def generate_easygrasp_gqcnn_data():
    """
    Same as high lof,just change index data_frame as 'positive_rate'
    """
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    grasp_positive_rate=np.array(list(data_frame['positive_rate']))
    split_index=list(data_frame['split'])
    key_list=list(data_frame['key'])

    #get train index data
    train_select_index_list=[]
    for index,_ in enumerate(grasp_positive_rate):
        if split_index[index]=='test':
            continue
        else:
            train_select_index_list.append(index)

    #get train index data threshold
    train_lof_data=grasp_positive_rate[train_select_index_list]
    sort_train_lof_data=sorted(train_lof_data,reverse=True)
    lof_threshold=sort_train_lof_data[100]
    print("lof threshold is:")
    print(lof_threshold)

    #get high 100 lof index list
    high_lof_index_list=[]
    high_lof_key_list=[]
    for train_index in train_select_index_list:
        lof=grasp_positive_rate[train_index]
        if lof>lof_threshold:
            high_lof_index_list.append(train_index)
            high_lof_key_list.append(key_list[train_index])

    print("high lof key list len is:")
    print(len(high_lof_key_list))

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/easygrasp_100_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,high_lof_key_list)

def generate_average_metric_data():
    """
    Later will try to generate average metric data and see the improvement
    """
    pass

def generate_high_critic_data():
    record_txt=open(os.path.join(Abs_Path,"better_critic.txt"),'r').read().splitlines()

    #1: load index and critic data
    index_list=[]
    critic_list=[]
    for data in record_txt:
        index,critic=data.split(':')
        index,critic=int(index),float(critic)
        index_list.append(index)
        critic_list.append(critic)

    #2: sort critic_list
    critic_list=np.array(critic_list)
    critic_list=np.abs(critic_list)
    sorted_critic_list=sorted(critic_list)
    print(sorted_critic_list[:10])
    critic_threshold=sorted_critic_list[85]


    #3: final select index
    final_select_index_list=[]
    for index,critic in enumerate(critic_list):
        if critic<critic_threshold:
            final_select_index_list.append(index)

    print("final select index {} are:".format(len(final_select_index_list)))
    print(final_select_index_list)



    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/highcritic_85_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/split_database/aug_100_diff_rate.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,final_select_index_list)
    
    dataset_path1="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/train_test_gqcnn_data/myself_select_100_gqcnn_data"
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/highcritic_85_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/aug_highcritic_select_100_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_183_lowweight_aug_data():
    #generate gqcnn data
    print("Generating gqcnn data...")
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/split_database/aug_100_diff_rate.hdf5"
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_lowweights_183_gqcnn_data")

    ###For generate_select_data###
    select_index_list=[]
    for i in range(183):
        select_index_list.append(i+183)
    # example_create_gqcnn_data.generate_gqcnn_dataset(gqcnn_path, database_path, config_path,select_index_list=select_index_list)
    ###For generate_select_data###

    # example_create_gqcnn_data.generate_gqcnn_dataset(gqcnn_path, database_path, config_path)


    #merge dataset
    dataset_path1="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/train_test_gqcnn_data/myself_select_100_gqcnn_data"
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_lowweights_183_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/myself_aug_100_low_weight_183data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_lowweight_aug100_87data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_different_rate_low_rate/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_100_lowweight.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_100_lowweight_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    #merge dataset
    dataset_path1="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/train_test_gqcnn_data/myself_select_100_gqcnn_data"
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_100_lowweight_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/myself_aug_100_low_weight')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_select_50_gqcnn_data():
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    select_50_data=np.array(list(data_frame['select_50_data']))    
    key_list=list(data_frame['key'])

    #get train index data
    select_50_index_list=[]
    for index,data in enumerate(select_50_data):
        if data==1:
            select_50_index_list.append(index)

    select_50_key_list=[]
    for index in select_50_index_list:
        select_50_key_list.append(key_list[index])
        

    print("select_50_key_list len is:")
    print(len(select_50_key_list))

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_50_key_list)

def generate_select_50_augmentation_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_50_data/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_50_data.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_50_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_50_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/aug_50_rate_1_2_gqcnndata')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_high_lof_500_gqcnn_data():
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    lof_all=np.array(list(data_frame['lof_all']))
    split_index=list(data_frame['split'])
    key_list=list(data_frame['key'])

    #get train index data
    train_select_index_list=[]
    for index,lof_data in enumerate(lof_all):
        if split_index[index]=='test':
            continue
        else:
            train_select_index_list.append(index)

    #get train index data threshold
    train_lof_data=lof_all[train_select_index_list]
    sort_train_lof_data=sorted(train_lof_data,reverse=True)
    lof_threshold=sort_train_lof_data[500]
    print("lof threshold is:")
    print(lof_threshold)

    #get high 100 lof index list
    high_lof_index_list=[]
    high_lof_key_list=[]
    for train_index in train_select_index_list:
        lof=lof_all[train_index]
        if lof>lof_threshold:
            high_lof_index_list.append(train_index)
            high_lof_key_list.append(key_list[train_index])

    print("high lof key list len is:")
    print(len(high_lof_key_list))

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/highlof_500_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,high_lof_key_list)

def generate_easygrasp_500_gqcnn_data():
    """
    Same as high lof,just change index data_frame as 'positive_rate'
    """
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    grasp_positive_rate=np.array(list(data_frame['positive_rate']))
    split_index=list(data_frame['split'])
    key_list=list(data_frame['key'])

    #get train index data
    train_select_index_list=[]
    for index,_ in enumerate(grasp_positive_rate):
        if split_index[index]=='test':
            continue
        else:
            train_select_index_list.append(index)

    #get train index data threshold
    train_lof_data=grasp_positive_rate[train_select_index_list]
    sort_train_lof_data=sorted(train_lof_data,reverse=True)
    lof_threshold=sort_train_lof_data[500]
    print("lof threshold is:")
    print(lof_threshold)

    #get high 100 lof index list
    high_lof_index_list=[]
    high_lof_key_list=[]
    for train_index in train_select_index_list:
        lof=grasp_positive_rate[train_index]
        if lof>lof_threshold:
            high_lof_index_list.append(train_index)
            high_lof_key_list.append(key_list[train_index])

    print("high lof key list len is:")
    print(len(high_lof_key_list))

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/easygrasp_500_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,high_lof_key_list)


###################################Generate in different ratio###########################
def generate_aug_50_data():
    #1: select different index
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    generate_path="/home/elevenjiang/Documents/Project/ObjectSynthesis/Code/code/data_augmentation/data/generate_select_50_data"

    #recalculate lof data from 50 data
    all_feature_vectors=np.load(os.path.join(generate_path,"all_feature_vectors.npz"))['feature_vector']
    pre_augmentation_flag=np.array(list(data_frame['select_50_data']))
    augmentation_index_list=[]
    change_back_dict={}#change back from augmentation_index_list to all_index_list
    for index,data in enumerate(pre_augmentation_flag):
        if data==1:
            augmentation_index_list.append(index)
            change_back_dict[len(augmentation_index_list)-1]=index

    augmentation_feature_vectors=all_feature_vectors[augmentation_index_list]
    from PyNomaly import loop
    m=loop.LocalOutlierProbability(augmentation_feature_vectors,n_neighbors=5).fit()
    lof_list = m.local_outlier_probabilities
    
    #save sepcify lof_dict 
    lof_dict={}
    for all_index,lof_data in enumerate(lof_list):
        lof_dict[augmentation_index_list[all_index]]=lof_data
    
    #get positive_rate
    positive_rate_list=list(data_frame['positive_rate'])


    #2: load generate txt and extract info
    generate_txt_info=open(os.path.join(generate_path,'generate_record.txt'),'r').read().splitlines()
    new_record_txt=open(os.path.join(Abs_Path,"temp.txt"),'a')
    for i,data in enumerate(generate_txt_info):
        index,pair=data.split(":")
        index=int(index)

        center_index,target_index=pair.split(",")
        center_index,target_index=int(center_index),int(target_index)

        #load their lof data and positive data
        center_lof=lof_dict[center_index]
        target_lof=lof_dict[target_index]
        object_lof_data=center_lof+target_lof

        center_positive_rate=positive_rate_list[center_index]
        target_positive_rate=positive_rate_list[target_index]
        object_positive_rate_data=center_positive_rate+target_positive_rate

        print("index:{} its lof is:{} object_positive_rate is:{}".format(index,object_lof_data,object_positive_rate_data))

        if i<43:
            new_record_txt.write("l,"+data+":{},{}\n".format(object_lof_data,object_positive_rate_data))
        else:
            new_record_txt.write("g,"+data+":{},{}\n".format(object_lof_data,object_positive_rate_data))

def generate_different_ratio_aug_50(rate='1'):
    """
    rate can only be '05' '1' '15'
    """
    #load from origin data
    recrod_txt=open(os.path.join(Abs_Path,"data/50_ratio_pair/1_{}.txt".format(rate)),'r').read().splitlines()
    record_pair_list=[]
    for data in recrod_txt:
        center_index,target_index=data.split(',')
        center_index,target_index=int(center_index),int(target_index)
        record_pair_list.append((center_index,target_index))

    #load all pair and index
    all_pair_txt=open(os.path.join(Abs_Path,"data/50_ratio_pair/all_pair.txt"),'r').read().splitlines()
    generate_index_list=[]
    for data in all_pair_txt:
        #load index
        index,pair,rate=data.split(':')
        index=int(index.split(',')[1])

        #check target in origin
        center_index,target_index=pair.split(',')
        center_index,target_index=int(center_index),int(target_index)
        if (center_index,target_index) in record_pair_list:
            generate_index_list.append(index)

    print(len(generate_index_list))
    print(generate_index_list)

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_50_1_{}".format(rate))
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_50_data.hdf5")
    print("Begin to generate {} data to path:".format(len(generate_index_list)))
    print(generate_path)
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,generate_index_list)

    #3: mer dataset
    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_50_1_{}".format(rate))
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/aug_50_rate_1_{}_gqcnndata'.format(rate))
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_aug_200_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_200_data/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_200_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_200_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/aug_200_rate_1_2_gqcnndata')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)
    
def generate_select_200_gqcnn_data():
    #1: find high lof data
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info.csv"))
    select_50_data=np.array(list(data_frame['select_200_data']))    
    key_list=list(data_frame['key'])

    #get train index data
    select_50_index_list=[]
    for index,data in enumerate(select_50_data):
        if data==1:
            select_50_index_list.append(index)

    select_50_key_list=[]
    for index in select_50_index_list:
        select_50_key_list.append(key_list[index])
        

    print("select_50_key_list len is:")
    print(len(select_50_key_list))

    #2: generate shape
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_50_key_list)

def generate_select_50_aug_25_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_50_data_aug25/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/select_50_data_aug25.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_50_aug_25_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_50_aug_25_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_select_50_aug_25_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_random_select_50_aug_25_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_random_select_50_data/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/select_50_data_random_aug25.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_50_random_aug_25_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_50_random_aug_25_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_select_50_random_aug_25_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_random_select_50_aug_25_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_random_center_select_50_data/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/select_50_data_random_center_aug25.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_50_random_center_aug_25_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_50_random_center_aug_25_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_select_50_random_center_aug_25_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


def generate_random_select_200_aug_200_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_random_center_select_200_data/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/select_200_data_random_center_aug200.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_200_random_center_aug_200_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_200_random_center_aug_200_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_select_200_random_center_aug_200_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_select_200_aug_low_rarity_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/generate_select_200_aug_low_rarity/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/select_200_aug_low_rarity.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_200_aug_low_rarity_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_200_aug_low_rarity_gqcnn_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_select_200_aug_low_rarity_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

###################################Generate Adversarial Shapes gqcnn data###########################
def store_positive_rate():
    from tqdm import tqdm
    from dexnet.constants import READ_ONLY_ACCESS
    from dexnet.database import Hdf5Database
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info_adversarial.csv"))


    train_database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    test_database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_test_database.hdf5"

    #load test database
    database=Hdf5Database(test_database_path,access_level=READ_ONLY_ACCESS)
    dataset=database[database.datasets[0].name]
    save_positive_rate_dict={}
    for i,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        grasps=dataset.grasps(key)
        if len(grasps)==0:
            print("object key {} has no grasps".format(key))
            save_positive_rate_dict[key]=0
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        if len(grasp_metrics)==0:
            print("object key {} has no grasps metric".format(key))
            save_positive_rate_dict[key]=0
            continue

        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])

        qualities_array=np.array(qualities)
        temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
        each_object_positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
        save_positive_rate_dict[key]=each_object_positive_rate
    print("save dict len is:")
    print(len(save_positive_rate_dict))

    #load test databse
    database=Hdf5Database(train_database_path,access_level=READ_ONLY_ACCESS)
    dataset=database[database.datasets[0].name]
    save_positive_rate_dict={}
    for i,key in tqdm(enumerate(dataset.object_keys),total=len(dataset.object_keys)):
        grasps=dataset.grasps(key)
        if len(grasps)==0:
            print("object key {} has no grasps".format(key))
            save_positive_rate_dict[key]=0
            continue
        grasp_metrics=dataset.grasp_metrics(key,grasps)
        if len(grasp_metrics)==0:
            print("object key {} has no grasps metric".format(key))
            save_positive_rate_dict[key]=0
            continue

        qualities=[]
        for g in grasps:
            qualities.append(grasp_metrics[g.id]['robust_ferrari_canny'])

        qualities_array=np.array(qualities)
        temp_count=np.where(qualities_array>0.002,1,0).astype(np.int)
        each_object_positive_rate=np.sum(temp_count)/float(temp_count.shape[0])
        save_positive_rate_dict[key]=each_object_positive_rate
    print("save dict len is:")
    print(len(save_positive_rate_dict))

    positive_rate_list=list(data_frame['positive_rate'])


    change_count=0
    for index,key in tqdm(enumerate(data_frame['key'])):
        if key in save_positive_rate_dict:
            positive_rate_list[index]=save_positive_rate_dict[key]
            change_count=change_count+1
    print("final change data {}".format(change_count))

    data_frame["positive_rate"]=positive_rate_list
    data_frame.to_csv(os.path.join(Abs_Path,"data/all_info_adversarial.csv"),index=False)
            

def select_adversarial_data():
    data_frame=pandas.read_csv(os.path.join(Abs_Path,"data/all_info_adversarial.csv"))
    positive_rate_list=np.array(list(data_frame['positive_rate']))
    train_split_list=list(data_frame['split'])
    key_list=list(data_frame['key'])
    print(positive_rate_list.shape[0])
    print(positive_rate_list[:30])


    sorted_positive_list=np.array(sorted(positive_rate_list)) 
    # print(sorted_positive_list[:300])

    print(sorted_positive_list.dtype)
    sorted_positive_list.astype(np.float)
    for index,data in enumerate(sorted_positive_list):
        if data>0:
            print("{} data bigger than 0,it is:{}".format(index,data))
            break
    
    print(np.where(sorted_positive_list==0.01))


    #generate all 0 key list
    generate_key_list=[]
    for index,positive_rate in enumerate(positive_rate_list):
        if positive_rate==0 and train_split_list[index]=='train':
            generate_key_list.append(key_list[index])

    print(len(generate_key_list))

    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/low_positive_rate_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/myself_train_database.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,generate_key_list)


def select_high_positive_rate_data():
    #1: load all positive rate
    txt_data=open(os.path.join(Abs_Path,"temp_record_200.txt")).read().splitlines()
    index_list=[]
    positive_rate_list=[]

    for i,data in enumerate(txt_data):
        index,positive_rate=data.split(":")
        index,positive_rate=int(index),float(positive_rate)

        index_list.append(index)
        positive_rate_list.append(positive_rate)

    #2: load threshold
    sort_positive_rate_list=sorted(positive_rate_list,reverse=True)
    print(sort_positive_rate_list[:30])
    threshold=sort_positive_rate_list[100]


    #3: get high positive_rate data
    generate_select_key_index_list=[]
    positive_rate_list=[]
    for data in txt_data:
        index,positive_rate=data.split(":")
        index,positive_rate=int(index),float(positive_rate)

        if positive_rate>threshold:
            generate_select_key_index_list.append(index)
            positive_rate_list.append(positive_rate)

    print("Generate high positive rate data are:")
    print(generate_select_key_index_list)
    print("There mean positive rate is:")
    print(np.sum(np.array(positive_rate_list)))

    ####Plot all add shapes####
    # meshes_path="/home/elevenjiang/Documents/Project/ObjectSynthesis/Code/code/data_augmentation/data/generate_select_200_data/meshes"
    # import open3d as o3d
    # show_meshes_list=[]
    # for i,index in enumerate(generate_select_key_index_list):
    #     o3d_mesh=o3d.io.read_triangle_mesh(os.path.join(meshes_path,"{}.obj".format(index)))
    #     scale_factor=0.08/np.max(o3d_mesh.get_max_bound()-o3d_mesh.get_min_bound())
    #     o3d_mesh.vertices=o3d.utility.Vector3dVector(np.asarray(o3d_mesh.vertices)*scale_factor) 
    #     o3d_mesh.translate(np.array([i*0.1,0,0]))
    #     o3d_mesh.compute_vertex_normals()

    #     show_meshes_list.append(o3d_mesh)

    # o3d.visualization.draw_geometries(show_meshes_list)
    # return
    ####Plot all add shapes####


    #4: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_in_200_high_positive_rate")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path,database_path,config_path,generate_select_key_index_list)
    print("Finish generate gqcnn data!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_in_200_high_positive_rate")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_aug_select_100_in_200_high_positive_rate')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

    
def select_low_positive_rate_data():
    #1: load all positive rate
    txt_data=open(os.path.join(Abs_Path,"temp_record_200.txt")).read().splitlines()
    index_list=[]
    positive_rate_list=[]

    for i,data in enumerate(txt_data):
        index,positive_rate=data.split(":")
        index,positive_rate=int(index),float(positive_rate)

        index_list.append(index)
        positive_rate_list.append(positive_rate)

    #2: load threshold
    sort_positive_rate_list=sorted(positive_rate_list)
    print(sort_positive_rate_list[:30])
    threshold=sort_positive_rate_list[100]


    #3: get high positive_rate data
    generate_select_key_index_list=[]
    positive_rate_list=[]
    for data in txt_data:
        index,positive_rate=data.split(":")
        index,positive_rate=int(index),float(positive_rate)

        if positive_rate<threshold:
            generate_select_key_index_list.append(index)
            positive_rate_list.append(positive_rate)


    print("Generate low positive rate data are:")
    print(generate_select_key_index_list)
    print("There mean positive rate is:")
    print(np.sum(np.array(positive_rate_list)))


    #4: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_in_200_low_positive_rate")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path,database_path,config_path,generate_select_key_index_list)
    print("Finish generate gqcnn data!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_in_200_low_positive_rate")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_aug_select_100_in_200_low_positive_rate')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_select_200_aug_200_data():
    """
    just random select data from 400 augmentation data
    """
    txt_data=open(os.path.join(Abs_Path,"temp_record_200.txt")).read().splitlines()
    index_list=[]
    positive_rate_list=[]

    for i,data in enumerate(txt_data):
        index,positive_rate=data.split(":")
        index,positive_rate=int(index),float(positive_rate)

        index_list.append(index)
    
    print("all index have {} data".format(len(index_list)))
    random.shuffle(index_list)
    print("new index list is:")
    print(index_list[:100])

    select_index_list=index_list[:200]
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_200_aug_200_data")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path,database_path,config_path,select_index_list)
    print("Finish generate gqcnn data!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/select_200_aug_200_data")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_aug_200_rate_1_1_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


def generate_select_high_lof_aug_200_data():
    #1: load all positive rate
    txt_data=open(os.path.join(Abs_Path,"temp_record_200.txt")).read().splitlines()
    index_list=[]
    positive_rate_list=[]

    for i,data in enumerate(txt_data):
        index,positive_rate=data.split(":")
        
        index,positive_rate=int(index),float(positive_rate)
        if index<191:
            index_list.append(index)
    

    print(index_list[:10])
    random.shuffle(index_list)
    select_high_lof_list=index_list[:100]
    print("Final select high lof data are:")
    print(select_high_lof_list)

    #4: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_highlof_200")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path,database_path,config_path,select_high_lof_list)
    print("Finish generate gqcnn data!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_highlof_200")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_aug_select_100_in_200_high_lof')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


def generate_select_high_positive_aug_200_data():
    """
    [395, 216, 212, 313, 397, 396, 302, 372, 285, 245, 229, 249, 246, 193, 228, 231, 371, 208, 334, 378, 358, 319, 240, 273, 252, 406, 243, 384, 205, 284, 401, 301, 257, 275, 312, 328, 311, 368, 392, 381, 318, 357, 297, 259, 256, 268, 202, 235, 330, 403, 340, 349, 329, 210, 265, 343, 298, 344, 209, 211, 244, 281, 375, 341, 400, 299, 379, 362, 369, 351, 399, 227, 262, 288, 290, 326, 239, 386, 266, 360, 214, 270, 294, 258, 367, 359, 306, 309, 373, 394, 196, 263, 296, 287, 267, 237, 232, 307, 322, 192]

    """
    #1: load all positive rate
    txt_data=open(os.path.join(Abs_Path,"temp_record_200.txt")).read().splitlines()
    index_list=[]
    positive_rate_list=[]

    for i,data in enumerate(txt_data):
        index,positive_rate=data.split(":")
        
        index,positive_rate=int(index),float(positive_rate)
        if index>191:
            index_list.append(index)
    

    print(index_list[:10])
    random.shuffle(index_list)
    select_high_lof_list=index_list[:100]
    print("Final select high lof data are:")
    print(select_high_lof_list)

    #4: generate gqcnn data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_highpositive_rate_200")
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path,database_path,config_path,select_high_lof_list)
    print("Finish generate gqcnn data!!!")

    dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_highpositive_rate_200")
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_aug_select_100_highpositive_rate_in_200')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def generate_middle_positiverate_aug_200_data():
    #1: load all positive rate
    txt_data=open(os.path.join(Abs_Path,"temp_record_200.txt")).read().splitlines()
    index_list=[]
    positive_rate_list=[]

    # for i,data in enumerate(txt_data):
    #     index,positive_rate=data.split(":")
        
    #     index,positive_rate=int(index),float(positive_rate)
    #     if index>191:
    #         index_list.append(index)
    

    # print(index_list[:10])
    # random.shuffle(index_list)
    # select_high_lof_list=index_list[:100]
    # print("Final select high lof data are:")
    # print(select_high_lof_list)

    # #4: generate gqcnn data
    # config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    # database_path=os.path.join(Abs_Path,"data/generate_database/aug_200_data.hdf5")
    # generate_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_select_100_highpositive_rate_200")
    # example_create_gqcnn_data.generate_gqcnn_dataset(generate_path,database_path,config_path,select_high_lof_list)
    # print("Finish generate gqcnn data!!!")

    dataset_path1=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_200_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")
    
    generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_aug_middlepositive_rate_in_200')
    example_merge_two_gqcnn.custom_add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


###############################Test generate single airplane data###############################
def generate_aug_airplane_data():
    objs_path="/home/Project/Code/code/data_augmentation/data/airplane_generate/meshes"
    database_path=os.path.join(Abs_Path,"data/generate_database/aug_airplane_database.hdf5")
    gqcnn_path=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_airplane_gqcnn_data")
    generate_gqcnn_data_from_meshes(objs_path,database_path,gqcnn_path)
    print("Finish generate all shapes!!!")

    # dataset_path1=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_50_gqcnn_data")
    # dataset_path2=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_airplane_gqcnn_data")
    # generate_dataset_path=os.path.join(Abs_Path,'data/merge_gqcnn_data/merge_select_50_aug_25_gqcnn_data')
    # example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)


def generate_airplane_test_data():
    #1: load train data key
    generate_path="/home/Project/Code/code/data_augmentation/data/airplane_generate"
    data_frame=pandas.read_csv(os.path.join(generate_path,"all_info.csv"))
    select_train_airplane=np.array(list(data_frame['select_airplane_split']))
    key_list=list(data_frame['key'])
    split_list=list(data_frame['split'])

    select_train_airplane_list=[]
    generate_count=0
    for index,data in enumerate(select_train_airplane):
        if data==2:
            select_train_airplane_list.append(key_list[index])
            if split_list[index]=='train':
                generate_count=generate_count+1

    print("Will generate all data are:{}".format(generate_count))
            
    print("select key are:{}".format(len(select_train_airplane_list)))
    print(select_train_airplane_list)

    #2: begin to generate data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/airplane/test_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/airplane.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_train_airplane_list)


def generate_airplane_train_data():
    #1: load train data key
    generate_path="/home/Project/Code/code/data_augmentation/data/airplane_generate"
    data_frame=pandas.read_csv(os.path.join(generate_path,"all_info.csv"))
    select_train_airplane=np.array(list(data_frame['select_airplane_split']))
    key_list=list(data_frame['key'])
    split_list=list(data_frame['split'])

    select_train_airplane_list=[]
    generate_count=0
    for index,data in enumerate(select_train_airplane):
        if data==1:
            select_train_airplane_list.append(key_list[index])
            if split_list[index]=='train':
                generate_count=generate_count+1

    print("Will generate all data are:{}".format(generate_count))
            
    print("select key are:{}".format(len(select_train_airplane_list)))
    print(select_train_airplane_list)

    #2: begin to generate data
    config_path=os.path.join(Abs_Path,'data/cfg/generate_gqcnn_dataset.yaml')
    generate_path=os.path.join(Abs_Path,"data/airplane/train_gqcnn_data")
    database_path="/home/Project/Code/code/DexNet/dex-net/object_synthesis_code/data/myself_database/airplane.hdf5"
    example_create_gqcnn_data.generate_gqcnn_dataset(generate_path, database_path, config_path,select_train_airplane_list)


def merge_select_50_train_20_data():
    dataset_path1=os.path.join(Abs_Path,'data/airplane/train_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/3dnet_gqcnn_data/select_50_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/airplane/select_50_train_20_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

def merge_aug_50_train_20_data():
    dataset_path1=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_airplane_gqcnn_data")
    dataset_path2=os.path.join(Abs_Path,'data/airplane/train_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/airplane/aug_50_train_20_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)

    #For both
    dataset_path1=os.path.join(Abs_Path,'data/airplane/aug_50_train_20_gqcnn_data')
    dataset_path2=os.path.join(Abs_Path,'data/3dnet_gqcnn_data/select_50_gqcnn_data')
    generate_dataset_path=os.path.join(Abs_Path,'data/airplane/select_50_aug_50_train_20_gqcnn_data')
    example_merge_two_gqcnn.add_tensor_dataset(dataset_path1, dataset_path2, generate_dataset_path)



###############################Merge specify data###############################
def merge_select_200_aug_400():
    
    

    high_graspness_score_list=[]

    high_rarity_score_list=[]

    import json
    dataset_path1=os.path.join(Abs_Path,"data/generate_gqcnn_data/aug_200_gqcnn_data")
    json_path=os.path.join(dataset_path1,"object_category_map.json")
    with open(json_path,'r') as load_f:
        map_key_to_label=json.load(load_f)#{'key':label,'key':label...}

    for index,key in enumerate(map_key_to_label):
        if int(key)>190:
            high_graspness_score_list.append(map_key_to_label[key])
        else:
            high_rarity_score_list.append(map_key_to_label[key])

    print(len(high_graspness_score_list))
    print(len(high_rarity_score_list))
    print(high_graspness_score_list)
    print(high_rarity_score_list)

    dataset_path2=os.path.join(Abs_Path,"data/3dnet_gqcnn_data/select_200_gqcnn_data")

    generate_dataset_path_1=os.path.join(Abs_Path,"data/merge_gqcnn_data/high_positive_rate_select_200_aug_data")
    generate_dataset_path_2=os.path.join(Abs_Path,"data/merge_gqcnn_data/high_rarity_select_200_aug_data")

    example_merge_two_gqcnn.add_specify_label_tensor_dataset(dataset_path1,dataset_path2,generate_dataset_path_1,specify_label_list=high_graspness_score_list)
    
    example_merge_two_gqcnn.add_specify_label_tensor_dataset(dataset_path1,dataset_path2,generate_dataset_path_2,specify_label_list=high_rarity_score_list)


if __name__ == '__main__':
    #region
    # create_select_500_gqcnn_data()
    # create_generate_meshes_database()
    # check_generate_meshes_database()
    # create_generate_meshes_gqcnn_data()
    # merge_two_gqcnn_data()
    # create_ycb_meshes_database()
    # create_ycb_augmentation_database()
    # create_ycb_gqcnn_data()
    # check_ycb_gqcnn_data()
    # create_ycb_augmentation_gqcnn_data()
    # merge_ycb_augmentation_gqcnn_data()
    # generate_positive_gqcnn_data()
    # generate_500_positive_gqcnn_data()
    # generate_500_lof_gqcnn_data()
    # merge_ycb_positive_gqcnn_data()
    # create_select_100_gqcnn_data()
    # see_100_data()
    # merge_100_augmentation_gqcnn_data()
    # create_test_smooth_meshes_database()
    # create_augmentation_select_100_database()
    # create_augmentation_select_100_smooth_less_database()
    # create_augmentation_select_100_gqcnn_data()
    # merge_augmentation_select_100_gqcnn_data()
    # example_generate_one_object()
    # create_select_100_unsmooth_samesize_gqcnn()
    # merge_select_100_unsmooth_samessize()
    # generate_again_origin_shapes()
    # generate_test_gqcnn_data_from_mesh()
    # generate_select_100_gqcnn_data_from_mesh()
    # generate_train_gqcnn_data_from_mesh()
    # generate_select_100_aug_moredata_from_mesh()
    #endregion
    # generate_high_lof_gqcnn_data()
    # generate_easygrasp_gqcnn_data()
    # generate_high_critic_data()
    # generate_183_lowweight_aug_data()
    # generate_lowweight_aug100_87data()
    # generate_select_50_gqcnn_data()
    # generate_select_50_augmentation_data()
    # generate_high_lof_500_gqcnn_data()
    # generate_easygrasp_500_gqcnn_data()
    ########Begin to generate different ratio of data#########
    # generate_aug_50_data()
    # generate_different_ratio_aug_50()
    # generate_aug_200_data()
    # generate_select_200_gqcnn_data()
    # generate_select_50_aug_25_data()
    # generate_random_select_50_aug_25_data()
    generate_select_200_aug_low_rarity_data()
    ########Begin to generate adversarial data#########
    # store_positive_rate()
    # select_adversarial_data()
    # select_high_positive_rate_data()
    # select_low_positive_rate_data()
    # generate_select_200_aug_200_data()
    # generate_select_high_lof_aug_200_data()
    # generate_random_select_200_aug_200_data()
    # generate_select_high_positive_aug_200_data()
    # generate_middle_positiverate_aug_200_data()
    ########Begin to generate airplane data#########
    # generate_aug_airplane_data()
    # generate_airplane_test_data()
    # generate_airplane_train_data()]
    # merge_select_50_train_20_data()
    # merge_aug_50_train_20_data()
    #########Begin merge specify label data#########
    # merge_select_200_aug_400()


    



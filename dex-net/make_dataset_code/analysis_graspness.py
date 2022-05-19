import os
import sys
import numpy as np
import matplotlib.pyplot as plt

Abs_Path=os.path.dirname(os.path.abspath(__file__))

def check_distribution():
    data_split=43
    txt_data=open(os.path.join(Abs_Path,"temp_record_100.txt")).read().splitlines()

    index_list=[]
    positive_rate_list=[]

    for i,data in enumerate(txt_data):
        index,positive_rate=data.split(":")
        index,positive_rate=int(index),float(positive_rate)

        index_list.append(index)
        positive_rate_list.append(positive_rate)

    plt.figure()
    plt.scatter(index_list,positive_rate_list)
    

    new_positive_rate_list=[]
    save_dict={}
    for list_index,true_index in enumerate(index_list):
        save_dict[true_index]=positive_rate_list[list_index]

    change_order_list=[]
    number_list=[]
    for i in range(len(save_dict)):
        change_order_list.append(save_dict[i])
        number_list.append(i)

    plt.figure()
    plt.scatter(number_list,change_order_list)


    lof_graspness=np.sum(np.array(change_order_list[:data_split]))/len(change_order_list[:data_split])
    easy_graspness=np.sum(np.array(change_order_list[data_split:]))/len(change_order_list[data_split:])
    print("lof graspness is:{} easy graspnes is:{}".format(lof_graspness,easy_graspness))


    plt.show()
    
        




if __name__ == '__main__':
    check_distribution()
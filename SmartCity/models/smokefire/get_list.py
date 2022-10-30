import os
import random


def walk(root_dir):
    list = []
    for root2, dirs2, files2 in os.walk(root_dir):
        for f in range(len(files2)):
            path = os.path.join(root2, files2[f])
            list.append(path)
    return list


def write_list2txt(list, txt_name):
    with open(txt_name, 'w') as f:
        for i in range(len(list)):
            f.write(list[i] + '\n')
    print(txt_name, len(list))


if __name__ == '__main__':
    root_dir1 = '/data1/littlesc/Fire/smoke_fire_class/train_relabel/smoke'
    root_dir2 = '/data1/littlesc/Fire/smoke_fire_class/train_relabel/peace'
    root_dir3 = '/data1/littlesc/Fire/smoke_fire_class/train_relabel/fire'
    root_dir4 = '/data1/littlesc/Fire/smoke_fire_class/train_relabel/fire_and_smoke'
    list_fire = walk(root_dir1) + walk(root_dir2)+ walk(root_dir3) + walk(root_dir4)
    #list_smoke = walk(root_dir3) + walk(root_dir2)

    write_list2txt(list_fire,'lists/fire_and_smoke_train.txt')
    #write_list2txt(list_smoke,'lists/smoke_val.txt')

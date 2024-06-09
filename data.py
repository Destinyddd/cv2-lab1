
import os
import shutil


data_dir = '/Localize/lc/homework/work1/dataset/CUB_200_2011/'

def prepare_data_split(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    with open(os.path.join(data_dir, 'train_test_split.txt'), 'r') as f:
        lines = f.readlines()

    with open(os.path.join(data_dir, 'images.txt'), 'r') as f:
        image_files = f.readlines()

    for line in lines:
        image_id, is_train = line.strip().split()
        image_file = image_files[int(image_id) - 1].strip().split(' ')[1]
        image_class = image_file.split('/')[0]
        
        if is_train == '1':
            target_dir = os.path.join(train_dir, image_class)
        else:
            target_dir = os.path.join(val_dir, image_class)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        shutil.copy(os.path.join(data_dir, 'images', image_file), target_dir)


if __name__ == "__main__":

    prepare_data_split(data_dir)
    print('data yes!')
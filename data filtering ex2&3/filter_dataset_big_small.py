import os
from tqdm import tqdm
import config
import pdb
import pandas as pd
import shutil
import time

class_union = []
for pascal_class in config.PASCAL_CLASSES:
    if pascal_class in config.COCO_LABELS:
        class_union.append(pascal_class)


def get_classes_dict(dataset):
    if dataset == 'COCO':
        classes = {config.COCO_LABELS[i]: i for i in range(len(config.COCO_LABELS))}

    elif dataset == 'PASCAL_VOC':
        classes = {config.PASCAL_CLASSES[i]: i for i in range(len(config.PASCAL_CLASSES))}
    else:
        raise Exception('Specify which dataset! "COCO" or "PASCAL_VOC" in get_classes_dict')
    return classes

def get_images_containing_classes(dataset, classes, label_src, all_incl=True, max_length=20000):
    CLASSES = get_classes_dict(dataset)
    classes = {CLASSES[i]: i for i in classes}

    stats = {label: 0 for label in classes.keys()}  # dict to keep track of distribution in dataset
    path = label_src
    names = []

    for filename in tqdm(os.listdir(path)):
        with open(os.path.join(path, filename), 'r') as f:
            image = [int(line.split(None, 1)[0]) for line in f]  # list with all objects in current image

            if all_incl:
                if all(item in image for item in classes.keys()):
                    #name = filename.removesuffix('.txt')
                    name = remove_txt(filename)
                    names.append(name)
            else:
                if any(item in image for item in classes.keys()):
                    name = remove_txt(filename)
                    #name = filename.removesuffix('.txt') 
                    names.append(name)
                    # Keep statistics when all_incl=false as might be unbalanced
                    for obj in set(image):
                        # set(image) gets unique labels in image = doesnt count the NUMBER
                        # of objects in each image but just if the image contains said object
                        if obj in stats.keys():
                            stats[obj] += 1

            # Stopping criterion
            if len(names) == max_length:
                print("max length reached! breaking out of loop")
                break

    if not all_incl:
        for key in stats.keys():
            print("{}% images of {}".format(round(stats[key] / sum(stats.values()) * 100, 2), classes[key]))

    return names, stats

def get_big_small_images(dataset, classes, label_src, max_length=20000):
    """
    From this method we want to filter out images containing specific classes that we 
    assume to be either relative big or small. 
    These images will later on be used to evaluate if YOLOv3 detects big or small images better.
    """
    CLASSES = get_classes_dict(dataset)
    classes = {CLASSES[i]: i for i in classes}
    small_size_threshold =  0.065  # Fraction of total area of the picture
    big_size_threshold =    0.6    # Fraction of total area of the picture
    stats = {label: 0 for label in classes.keys()}  # dict to keep track of distribution in dataset
    path = label_src
    names = []
    for filename in tqdm(os.listdir(path)):
        add_name = False
        contents = open(os.path.join(path, filename), "r").read().split('\n')
        templist = []
        for line in contents:
            if len(line .split(" ")) != 5: # to make sure the labels are in the right format
                continue
            else:
                c, _, _, w, h = line .split(" ")
                c = int(c)
                w = float(w)
                h = float(h)
            if  not(c in classes.keys()): # if not a one of the wanted classes
                continue
            elif c == 2 or c == 11: # if one of the classes that are assumed to be small
                # CHANGE HERE ^
                if w*h < small_size_threshold: # if small
                    add_name = True
                    templist.append(c)
            elif c == 6 or c == 18: # if one of the classes that are assumed to be big
                # CHANGE HERE ^
                if w*h > big_size_threshold: #if big
                    add_name = True
                    templist.append(c)
            
        if add_name:
            #name = remove_txt(filename)
            name = filename[:-4]
            names.append(name)
            add_name = False
            for obj in set(templist): # set(image) gets unique labels in image 
                stats[obj] += 1

        # Stopping criterion
        if len(names) == max_length:
            print("max length reached! breaking out of loop")
            break      
    for key in stats.keys():
        print("{}% images of {}".format(round(stats[key] / sum(stats.values()) * 100, 2), classes[key]))
    raise Exception(stats) # this handy when you are tweeking the parameters
    return names, stats

def get_big_small_images2(dataset, classes, label_src, max_length=20000):
    """
    From this method we want to filter out images containing specific classes that we 
    assume to be either relative big or small. 
    These images will later on be used to evaluate if YOLOv3 detects big or small images better.
    """
    CLASSES = get_classes_dict(dataset)
    classes = {CLASSES[i]: i for i in classes}
    small_size_threshold =  0.065  # Fraction of total area of the picture
    big_size_threshold =    0.6    # Fraction of total area of the picture
    stats = {label: 0 for label in classes.keys()}  # dict to keep track of distribution in dataset
    path = label_src
    names = []
    for filename in tqdm(os.listdir(path)):
        add_name = False
        contents = open(os.path.join(path, filename), "r").read().split('\n')
        templist = []
        for line in contents:
            if len(line .split(" ")) != 5: # to make sure the labels are in the right format
                continue
            else:
                c, _, _, w, h = line .split(" ")
                c = int(c)
                w = float(w)
                h = float(h)
            if  not(c in classes.keys()): # if not a one of the wanted classes
                add_name = False
                break
            else:
                if c == 2 or c == 11: # if one of the classes that are assumed to be small
                    # CHANGE HERE ^
                    if w*h < small_size_threshold: # if small
                        add_name = True
                        templist.append(c)
                    else:
                        add_name = False
                        break
                elif c == 6 or c == 18: # if one of the classes that are assumed to be big
                    # CHANGE HERE ^
                    if w*h > big_size_threshold: #if big
                        add_name = True
                        templist.append(c)
                    else:
                        add_name = False
                        break
            
        if add_name:
            #name = remove_txt(filename)
            name = filename[:-4]
            names.append(name)
            add_name = False
            for obj in set(templist): # set(image) gets unique labels in image 
                stats[obj] += 1

        # Stopping criterion
        if len(names) == max_length:
            print("max length reached! breaking out of loop")
            break      
    for key in stats.keys():
        print("{}% images of {}".format(round(stats[key] / sum(stats.values()) * 100, 2), classes[key]))
    #raise Exception(stats) # this handy when you are tweeking the parameters
    return names, stats

def get_big_or_small_images(dataset, classes, label_src, bigORsmall):
    """
    From this method we want to filter out images containing specific classes that we 
    assume to be either relative big or small. 
    These images will later on be used to evaluate if YOLOv3 detects big or small images better.
    """
    all_labels = classes == config.PASCAL_CLASSES or classes == config.COCO_LABELS
    big = True if bigORsmall == "big" else False
    CLASSES = get_classes_dict(dataset)
    classes = {CLASSES[i]: i for i in classes}
    stats = {label: 0 for label in classes.keys()}  # dict to keep track of distribution in dataset
    path = label_src
    names = []
    size_threshold = 0.01 if (bigORsmall == 'small') else 0.3
    for filename in tqdm(os.listdir(path)):
        add_name = False
        contents = open(os.path.join(path, filename), "r").read().split('\n')
        templist = []
        for line in contents:
            if len(line .split(" ")) != 5: # to make sure the labels are in the right format
                continue
            else:
                c, _, _, w, h = line .split(" ")
                c = int(c)
                w = float(w)
                h = float(h)
            if  not(c in classes.keys()): # if not a one of the wanted classes
                add_name = False
                break
            else:
                if big: # if big
                    if  w*h > size_threshold: # if small
                        add_name = True
                        templist.append(c)
                    else:
                        add_name = False
                        break
                else: #if small
                    if  w*h < size_threshold: # if small
                        add_name = True
                        templist.append(c)
                    else:
                        add_name = False
                        break
            
        if add_name:
            name = filename[:-4]
            names.append(name)
            add_name = False
            for obj in set(templist): # set(image) gets unique labels in image 
                stats[obj] += 1

        # Stopping criterion
        if len(names) == 20000:
            print("max length reached! breaking out of loop")
            break      
    for key in stats.keys():
        print("{}% images of {}".format(round(stats[key] / sum(stats.values()) * 100, 2), classes[key]))
    #raise Exception(stats) # this handy when you are tweeking the parameters
    return names, stats

def make_somewhat_balanced(dataset, classes, names, stats, path):
    CLASSES = get_classes_dict(dataset)
    classes = {CLASSES[i]: i for i in classes}
    smallest_class = min(stats, key=stats.get)
    removed = {key: 0 for key in classes.keys() if key != smallest_class}
    diff = {key: stats[key]-stats[smallest_class] for key in classes.keys() if key != smallest_class}
    remove_names = []

    for name in names:
        with open(os.path.join(path, name + '.txt'), 'r') as f:
            image = [int(line.split(None, 1)[0]) for line in f]  # list with all objects in current image

            if smallest_class not in image:
                for obj in set(image):
                    if obj in classes.keys() and removed[obj] < diff[obj]:
                        remove_names.append(name)
                        removed[obj] += 1

    new_names = [name for name in names if name not in remove_names]
    stats_after_trim = check_balance(new_names, classes, path)
    return new_names, removed, diff, stats_after_trim 

def check_balance(names, classes, path):
    stats = {key: 0 for key in classes.keys()}
    for name in tqdm(names):
        with open(os.path.join(path, name + '.txt'), 'r') as f:
            image = [int(line.split(None, 1)[0]) for line in f]  # list with all objects in current image
            for obj in set(image):
                if obj in stats.keys():
                    stats[obj] += 1

    for key in stats.keys():
        print("{}% images of {}".format(round(stats[key] / sum(stats.values()) * 100, 2), classes[key]))
    return stats

def copy_images_and_labels(names, image_src, label_src, image_dest, label_dest):
    print("Copying images and labels..")
    for name in tqdm(names):
        shutil.copy2(image_src + name + '.jpg', image_dest)  # copy image
        shutil.copy2(label_src + name + '.txt', label_dest)  # copy label

def delete_content(dest_1, dest_2):
    for folder in [dest_1, dest_2]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

def create_csv_files(path, names, num_classes, test_size=0.1):
    train = names[:int((1 - test_size) * len(names))]
    test = names[int((1 - test_size) * len(names)):]

    for SET in [train, test]:
        fig = [name + '.jpg' for name in SET]
        label = [name + '.txt' for name in SET]
        df = pd.DataFrame({'fig': fig, 'label': label})
        if SET == train:
            df.to_csv(path + '{}classes_train.csv'.format(num_classes), index=False, header=False)
        else:
            df.to_csv(path + '{}classes_test.csv'.format(num_classes), index=False, header=False)

def convert_labels(dataset, classes, path):
    CLASSES = get_classes_dict(dataset)
    classes = [CLASSES[key] for key in classes]
    new_class_labels = {classes[i]: i for i in range(len(classes))}
    all_labels_file = path + "_all_labels"
    try:
        shutil.rmtree(all_labels_file)
    except:
        pass
    try:
        shutil.copytree(path, all_labels_file)
    except:
        pass
    for filename in tqdm(os.listdir(path)):

        with open(os.path.join(path, filename), 'r') as f:
            lines = f.readlines()

        with open(os.path.join(path, filename), 'w') as f:
            for line in lines:
                if int(line.strip("\n").split(None, 1)[0]) in classes:
                    old_label = line.split(None, 1)[0]
                    new_line = line.replace(old_label, str(new_class_labels[int(old_label)]), 1)
                    f.write(new_line)

def listToString(s): 
    str1 = "" 
    for ele in s: 
        str1 += ele + "-"   
    str1 = str1[:len(str1)-1] 
    if s == config.COCO_LABELS: 
        str1 = "all_COCO_classes"
    elif s == config.PASCAL_CLASSES:
        str1 = "all_PASCAL_classes"
    elif s == class_union:
        str1 = "union_classes"
    return str1  

def make_readme_file(main_folder, bigORsmall, classes_list, dataset, stats_before):
    f = open(main_folder+ "/" + bigORsmall + "-" + listToString(classes_list) + "_readme.txt","w+")
    f.write("Time: " + time.ctime() +"\n")
    f.write(dataset +"\n")
    f.write("Classes: "+ listToString(classes_list) +"\n")
    f.write("Number of pictures: {} \n".format(sum(stats_before.values())))
    f.write("\nComposition of classes: \n")
    CLASSES = get_classes_dict(dataset)
    classes = {CLASSES[i]: i for i in classes_list}
    for key in stats_before.keys():
        f.write("{}% images of {} \n".format(round(stats_before[key] / sum(stats_before.values()) * 100, 2), classes[key]))
    #f.write("\nComposition of classes after trimming: \n")
    # TODO: Change or delete this
    #for key in stats_after.keys():
    #    f.write("{}% images of {} \n".format(round(stats_after[key] / sum(stats_after.values()) * 100, 2), classes[key]))
    f.close() 

def main():
    dataset = 'COCO'
    data_path = 'C:/Users/lucas/Documents/Deep_Learning/COCO'
    classes = ['person', 'car']
    bigORsmall = "small"

    # Paths to data source and destination
    image_src = data_path + '/images/'
    label_src = data_path + '/labels/'
    main_folder = data_path + '/filtered_data/'+ bigORsmall +"-" + listToString(classes) 
    image_dest = main_folder + '/images_'+ bigORsmall +"-" + listToString(classes) 
    label_dest = main_folder + '/labels_'+ bigORsmall +"-" + listToString(classes) 
    csv_file_dest = main_folder + '/'

    # Create folders
    try:
        shutil.rmtree(main_folder)
    except:
        pass
    os.makedirs(main_folder, exist_ok=True)
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)

    # clear images and label folders before copying to folder
    delete_content(image_dest, label_dest)

    names, stats = get_big_or_small_images(dataset, classes, label_src, bigORsmall)

    copy_images_and_labels(names, image_src, label_src, image_dest, label_dest)

    create_csv_files(csv_file_dest, names, num_classes=len(classes), test_size=0.1)

    # Converts the labels to format: [0, .., num_classes-1] AND removes labels for unwanted classes
    convert_labels(dataset, classes, label_dest)

    make_readme_file(main_folder, bigORsmall, classes, dataset, stats)

if __name__ == '__main__':
    main()

import os
from tqdm import tqdm
import config
import pdb
import pandas as pd
import shutil


def get_classes_dict(dataset):
    if dataset == 'COCO':
        classes = {config.COCO_LABELS[i]: i for i in range(len(config.COCO_LABELS))}

    elif dataset == 'PASCAL_VOC':
        classes = {config.PASCAL_CLASSES[i]: i for i in range(len(config.PASCAL_CLASSES))}
    else:
        classes = None
        print('Specify which dataset! "COCO" or "PASCAL_VOC"')
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
                    name = filename.removesuffix('.txt')
                    names.append(name)
            else:
                if any(item in image for item in classes.keys()):

                    name = filename.removesuffix('.txt')
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
    check_balance(new_names, classes, path)
    return new_names, removed, diff


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


def copy_images_and_labels(names, image_src, label_src, image_dest, label_dest):
    print("Copying images and labels..")
    for name in tqdm(names):
        shutil.copy2(image_src + name + '.jpg', image_dest)  # copy image
        shutil.copy2(label_src + name + '.txt', label_dest)  # copy label


def delete_content(dest_1, dest_2):
    for folder in [dest_1, dest_2]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))


def create_csv_files(path, names, num_classes, test_size=0.2):
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

    print("Converting labels...")
    for filename in tqdm(os.listdir(path)):
        with open(os.path.join(path, filename), 'r') as f:
            lines = f.readlines()

        with open(os.path.join(path, filename), 'w') as f:
            for line in lines:
                if int(line.strip("\n").split(None, 1)[0]) in classes:
                    old_label = line.split(None, 1)[0]
                    new_line = line.replace(old_label, str(new_class_labels[int(old_label)]), 1)
                    f.write(new_line)


def main():
    dataset = 'COCO'  # 'PASCAL_VOC' or 'COCO'
    print(get_classes_dict(dataset))  # Prints dictionary with all classes in entire dataset
    classes = ['person', 'car']

    # Paths to data source and destination
    image_src = dataset + '/images/'
    label_src = dataset + '/labels/'
    image_dest = dataset + '/filtered_data/1000_images/images_{}classes'.format(len(classes))
    label_dest = dataset + '/filtered_data/1000_images/labels_{}classes'.format(len(classes))
    csv_file_dest = dataset + '/filtered_data/1000_images/'

    # Create folders
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(label_dest, exist_ok=True)

    # clear images and label folders before copying to folder
    delete_content(image_dest, label_dest)

    # all_incl=True means each image has to include all classes
    all_incl = True
    names, stats = get_images_containing_classes(dataset, classes, label_src, all_incl=all_incl, max_length=1200)

    if not all_incl:
        # Make somewhat balanced. Function not perfect
        names, removed, diff = make_somewhat_balanced(dataset, classes, names, stats, label_src)

    copy_images_and_labels(names, image_src, label_src, image_dest, label_dest)

    create_csv_files(csv_file_dest, names, num_classes=len(classes), test_size=0.1)

    # Converts the labels to format: [0, .., num_classes-1] AND removes labels for unwanted classes
    convert_labels(dataset, classes, label_dest)


if __name__ == '__main__':
    main()

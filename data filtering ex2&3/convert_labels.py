import os
from tqdm import tqdm
import config
import pdb
import pandas as pd
import shutil
import time

def coco_to_pascal_dict():
	class_union = []
	#class_union = [pascal_class if pascal_class in config.COCO_LABELS else pass for pascal_class in config.PASCAL_CLASSES]
	for pascal_class in config.PASCAL_CLASSES:
		if pascal_class in config.COCO_LABELS:
			class_union.append(pascal_class)	
	coco_dict = 	{config.COCO_LABELS[i]: i for i in range(len(config.COCO_LABELS))}
	pascal_dict = 	{config.PASCAL_CLASSES[i]: i for i in range(len(config.PASCAL_CLASSES))}
	dict_to_pass = {}
	for label in coco_dict.keys():
		if label in class_union:
			dict_to_pass[coco_dict[label]] = pascal_dict[label]
	return dict_to_pass


def convert_coco_to_pascal(path, dict_c_to_p):
	# Send in the file with filtered coco labels in order to convert the to pascal equivalent
	# The dictionary should map coco labels to pascal equivalent
	for filename in tqdm(os.listdir(path)):

		with open(os.path.join(path, filename), 'r') as f:
			lines = f.readlines()

		with open(os.path.join(path, filename), 'w') as f:
			for line in lines:
				old_label = line.split(None, 1)[0]
				new_line = line.replace(old_label, str(dict_c_to_p[int(old_label)]), 1)
				f.write(new_line)

def mark_txt_as_done(path):
	with open(path, 'a') as f:
		f.write("\n COCO labels converted to Pascal at: "+ time.ctime() +"\n")

def main():
	try:
		 open(path, 'a')
	except:
		print("wrong name for txt file")

	dict_c_to_p = coco_to_pascal_dict()
	path_labels = "C:/Users/lucas/Documents/Deep_Learning/COCO/filtered_data/small1-union_classes/labels_small-union_classes_all_labels"
	path_txt = "C:/Users/lucas/Documents/Deep_Learning/COCO/filtered_data/small1-union_classes/small-union_classes_readme.txt"
	convert_coco_to_pascal(path_labels,dict_c_to_p)
	mark_txt_as_done(path_txt )

if __name__ == '__main__':
    main()
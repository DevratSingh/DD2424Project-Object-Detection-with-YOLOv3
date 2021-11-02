"""
I made this script to check for invalid data in the labels file because I got
som unexpected errors while debugging.
To check if I perhaps had messed up the labels on my own.
"""
directory = r'C:\Users\lucas\Google Drive\Colab Notebooks\Datasets\Pascal voc\labels'

import os
from tqdm import tqdm

def search_for_invalid_data(file_name = "000001.txt"):
	contents = open(f"{file_name}", "r").read().split('\n')
	for line in contents:
		if len(line.split(" ")) != 5:
			continue
		else:
			c, x, y, w, h = line.split(" ")
			x = float(x)
			y = float(y)
			w = float(w)
			h = float(h)
			# Check if int
			if  not float(c) == float(int(c)):
				raise TypeError(f'Class label is not an int in file: {file_name}')
			# Check if value is in range
			if not (0 < x < 1):
				raise ValueError(f'x label is not between 0 and 1 in file: {file_name}\n'
								 f'x = {x}')
			# Check if value is in range
			if not (0 < y < 1):
				raise ValueError(f'y label is not between 0 and 1 in file: {file_name}\n'
								 f'y = {y}')

def search_all_files():
	for filename in tqdm(os.listdir(directory)):
		if filename.endswith(".txt"):
			search_for_invalid_data(filename)
		else:
			continue
	print("No invalid data found")


#search_for_invalid_data(file_name)
search_all_files()
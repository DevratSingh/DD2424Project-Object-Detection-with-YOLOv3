import os
from tqdm import tqdm
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def count(classes, label_src):
    count = {i: 0 for i in range(len(classes))}

    for filename in tqdm(os.listdir(label_src)):
        with open(os.path.join(label_src, filename), 'r') as f:
            image = [int(line.split(None, 1)[0]) for line in f]  # list with all objects in current image
            for obj in image:
                count[obj] += 1

    return {classes[i]: count[i] for i in range(len(classes))}


def get_bbox_data(data_paths):
    # keys [0, 1, 2, 3] = [x, y, w, h]
    coco_bbox = {i: [] for i in range(4)}
    pascal_bbox = {i: [] for i in range(4)}
    bbox_data = [coco_bbox, pascal_bbox]

    for i, path in enumerate(data_paths):
        for filename in tqdm(os.listdir(path)):
            contents = open(os.path.join(path, filename), "r").read().split('\n')

            for line in contents:
                for j, val in enumerate(line.split(" ")[1:]):
                    bbox_data[i][j].append(float(val))

    return bbox_data


def save_obj_dist_plots(bbox_data):
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(14, 7))
    fig.suptitle('Heatmap showing the distribution of the object center points (x, y) in both datasets',
                 fontsize=16)
    datasets = ['COCO', 'PASCAL_VOC']
    for i in range(len(datasets)):
        N_points = len(bbox_data[i][0])
        x = []
        y = []
        for p in tqdm(range(N_points)):
            x.append(bbox_data[i][0][p])
            y.append(bbox_data[i][1][p])

        N_bins = 50
        h, x, y, p = axs[i].hist2d(x, y, bins=N_bins, cmap='plasma')
        im = axs[i].imshow(h)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(im, cax=cax,  ax=axs[i])
        cb.set_label('Number of objects in pixel')

        axs[i].set_title('{}'.format(datasets[i]))
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')

        '''LEFTOVER CODE'''
        # plt.hist2d(x, y, bins=N_bins, cmap='plasma')
        #
        # # density = fig.scatter_density(x, y)
        # cb = plt.colorbar()
        # cb.set_label('Number of objects in pixel')
        #
        # plt.title('Heatmap showing how the center point (x, y) of each object\n is distributed in the {} dataset'.format(datasets[i]))
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.savefig('plots/{}_obj_distribution'.format(datasets[i]))
        # cb.remove()

    # plt.show()
    plt.savefig('plots/obj_distribution')


def save_obj_size_plots(bbox_data):
    datasets = ['COCO', 'PASCAL_VOC']
    fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(14, 7))
    fig.suptitle('Relative size of objects compared to the entire image for each dataset',
                 fontsize=16)
    for i in range(len(datasets)):
        N_points = len(bbox_data[i][0])
        w = []
        h = []
        areas = []
        for p in tqdm(range(N_points)):
            w.append(bbox_data[i][2][p])
            h.append(bbox_data[i][3][p])
            areas.append(100*w[p]*h[p])

        domains = np.linspace(0, 100, 11)

        d = {i: 0 for i in range(len(domains)-1)}  # dict of distributions between all domains
        for j in range(len(domains)-1):
            d[j] = 100*sum(map(lambda x: domains[j] <= x < domains[j+1], areas))/len(areas)  # % of obj in each domain
            # d[j] = sum(map(lambda x: domains[j] <= x < domains[j + 1], areas))

        n_bins = 10
        
        axs[i].hist(areas, bins=n_bins, density=True, edgecolor='black', linewidth=0.8, color='b')
        axs[i].set_yscale("log")
        axs[i].set_ylim([0.0001, 0.1])
        axs[i].set_yticks([0.001, 0.01, 0.1])
        axs[i].yaxis.set_major_formatter(PercentFormatter(0.1))

        axs[i].set_xlim([0, 100])
        axs[i].set_xticks(domains)
        axs[i].set_xticklabels(['{}%'.format(int(val)) for val in domains])

        axs[i].set_xlabel('Percent of image size')
        axs[i].set_ylabel('Percent of objects')

        axs[i].set_title('{}'.format(datasets[i]))
        for k, p in enumerate(axs[i].patches):
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            if k == 0 and i == 0:
                axs[i].annotate(f'{round(height * 1000, 2)}%', (x + width / 2, y + height * 0.8), ha='center')
            else:
                axs[i].annotate(f'{round(height*1000, 2)}%', (x + width / 2, y + height * 1.02), ha='center')

    plt.savefig('plots/relative_size_of_objects')


def main():

    classes = ['person', 'car']
    coco_src = 'COCO/filtered_data/4500_images/labels_{}classes'.format(len(classes))
    coco1200_src = 'COCO/filtered_data/1200_images/labels_{}classes'.format(len(classes))
    pasc_src = 'PASCAL_VOC/filtered_data/labels_{}classes'.format(len(classes))

    bbox_data = get_bbox_data([coco_src, pasc_src])  # keys [0, 1, 2, 3] = [x, y, w, h]
    # save_obj_dist_plots(bbox_data)
    # save_obj_size_plots(bbox_data)

    coco = count(classes, coco_src)
    coco1200 = count(classes, coco1200_src)
    pasc = count(classes, pasc_src)

    print("COCO count: ", coco)
    print("COCO1200 count: ", coco1200)
    print("PASCAL_VOC count: ", pasc)
    quota = {i: round(coco[i]/pasc[i], 2) for i in classes}
    print("Quota between COCO and PASCAL_VOC: ", quota)


if __name__ == '__main__':
    main()



'''LEFTOVER CODE'''
# fig, ax = plt.subplots(tight_layout=True)
# ax.hist(areas, bins=n_bins, density=True, edgecolor='black', linewidth=0.8, color='b')
# # plt.yscale("log")
# ax.set_yscale("log")
#
# ax.yaxis.set_major_formatter(PercentFormatter(0.1))
# ax.set_yticks([0.001, 0.01, 0.1])
#
# ax.set_xlim([0, 100])
# ax.set_xticks(domains)
# ax.set_xticklabels(['{}%'.format(int(val)) for val in domains])
#
# ax.set_xlabel('Percent of image size')
# ax.set_ylabel('Percent of objects')
# # plt.xlabel('Percent of image size')
# # plt.ylabel('Percent of objects')
# for p in ax.patches:
#     width = p.get_width()
#     height = p.get_height()
#     x, y = p.get_xy()
#     ax.annotate(f'{round(height*1000, 2)}%', (x + width / 2, y + height * 1.02), ha='center')
# plt.show()


# xs = np.linspace(5, 95, 10)
# fig, ax = plt.subplots()
#
# plt.xlim([0, 100])
# ax.set_xticks(domains)
# ax.set_xticklabels(['{}%'.format(int(val)) for val in domains])
#
# plt.yscale("log")
# ax.yaxis.set_major_formatter(ScalarFormatter())
# ax.set_yticks([0, 1, 10, 100])
# ax.set_yticklabels(['0%', '1%', '10%', '100%'])
# plt.xlabel('Percent of image size')
# plt.ylabel('Percent of instances')
# for i, val in enumerate(d.values()):
#     ax.plot(xs[i], val, '.')
#
# plt.show()
# pdb.set_trace()
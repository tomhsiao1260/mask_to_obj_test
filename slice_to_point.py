import os
import nrrd
import tifffile
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean

def find_endpoints(skel):
    points = np.column_stack(np.where(skel > 0))
    G = nx.Graph()
    for y, x in points:
        G.add_node((y, x))

        # find connection
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                sy, sx = y + dy, x + dx
                if 0 <= sy < skel.shape[0] and 0 <= sx < skel.shape[1]:
                    if skel[sy, sx] > 0:
                        G.add_edge((y, x), (sy, sx))

    endpoints = [node for node, degree in G.degree() if degree == 1]
    return endpoints, G

def find_distance(path_coords):
    distances = [0]
    for i in range(1, len(path_coords)):
        dist = euclidean(path_coords[i-1], path_coords[i])
        distances.append(distances[-1] + dist)
    return distances

# select points (interp via distant)
def find_points(path_coords, distances, interval):
    total_length = distances[-1]
    num_points = int(total_length // interval) + 1

    idx = 1
    current_distance = interval
    selected_points = [path_coords[0]]

    for i in range(1, num_points - 1):
        while idx < len(distances) and distances[idx] < current_distance:
            idx += 1
        if idx == len(distances):
            break
        if distances[idx] == current_distance:
            selected_points.append(path_coords[idx])
        else:
            prev_point = np.array(path_coords[idx -1])
            next_point = np.array(path_coords[idx])
            ratio = (current_distance - distances[idx -1]) / (distances[idx] - distances[idx -1])
            interp_point = prev_point + ratio * (next_point - prev_point)
            selected_points.append(tuple(interp_point))
        current_distance += interval

    selected_points.append(path_coords[-1])

    return selected_points

def process_slice_to_point(skeleton_image, interval, prev_start=None, prev_end=None):
    # find endpoints & shortest path
    endpoints, G = find_endpoints(skeleton_image)

    if (prev_start is None):
        start, end = endpoints[0], endpoints[1]
    else:
        # need to check start, end order
        (xs, ys), (xe, ye), (xps, yps) = endpoints[0], endpoints[1], prev_start
        p = (xs - xps) ** 2 + (ys - yps) ** 2
        q = (xe - xps) ** 2 + (ye - yps) ** 2
        if (p < q): start, end = endpoints[0], endpoints[1]
        if (p > q): end, start = endpoints[0], endpoints[1]
    # print(f"Start (y, x): {start}, End (y, x): {end}")

    path = nx.shortest_path(G, source=start, target=end)
    path_coords = [(x, y) for y, x in path]

    distances = find_distance(path_coords)
    total_length = distances[-1]
    # print(f"Total distance: {total_length}")

    selected_points = find_points(path_coords, distances, interval)
    # print("Selected Points:")
    # for point in selected_points: print(point)
    return selected_points, start, end

def main(output_dir, mask_dir, layer, label, interval, plot):
    # load mask
    data, header = nrrd.read(os.path.join(output_dir, mask_dir))
    data = np.asarray(data)

    # original
    image = np.zeros_like(data[layer], dtype=np.uint8)
    image[data[layer] == label] = 255
    tifffile.imwrite(os.path.join(output_dir, 'output.tif'), image)

    # skeletonize
    mask = skeletonize(image)
    skeleton_image = np.zeros_like(image)
    skeleton_image[mask] = 255
    tifffile.imwrite(os.path.join(output_dir, 'skeleton.tif'), skeleton_image)

    selected_points, _, _ = process_slice_to_point(skeleton_image, interval)

    if (plot):
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        x_coords, y_coords = zip(*selected_points)
        plt.scatter(x_coords, y_coords, c='red', s=int(interval // 3))
        plt.title('Selected Equidistant Points')
        plt.show()

# python slice_to_point.py --plot
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a series of points in a sliced mask (equal distance along the mask path).')
    parser.add_argument('--label', type=int, default=1, help='Selected label')
    parser.add_argument('--plot', action='store_true', help='Plot the result')
    parser.add_argument('--d', type=int, default=5, help='Interval between each points or layers')
    args = parser.parse_args()

    output_dir = '/Users/yao/Desktop/output'
    mask_dir = '/Users/yao/Desktop/mask_to_obj_test/10624_02304_02432_mask.nrrd'
    layer = 450
    label = args.label
    interval = args.d
    plot = args.plot

    main(output_dir, mask_dir, layer, label, interval, plot)





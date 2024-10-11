import nrrd
import open3d
import tifffile
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
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx] > 0:
                        G.add_edge((y, x), (ny, nx))

    endpoints = [node for node, degree in G.degree() if degree == 1]
    return endpoints, G

filename = '10624_02304_02432_mask.nrrd'

# x, y, z
data, header = nrrd.read(filename)
data = np.asarray(data)

# original
image = (data[50] * 255).astype(np.uint8)
tifffile.imwrite('output.tif', image)

# skeletonize
mask = skeletonize(image)
skeleton = np.zeros_like(image)
skeleton[mask] = 255
tifffile.imwrite('skeleton.tif', skeleton)

# find endpoints & shortest path
endpoints, G = find_endpoints(skeleton)
start, end = endpoints[0], endpoints[1]
print(f"Start (y, x): {start}, End (y, x): {end}")

path = nx.shortest_path(G, source=start, target=end)
path_coords = [(x, y) for y, x in path]

# calculate distance
distances = [0]
for i in range(1, len(path_coords)):
    dist = euclidean(path_coords[i-1], path_coords[i])
    distances.append(distances[-1] + dist)

total_length = distances[-1]
print(f"Total path length: {total_length}")

# select points (interp via distant)
num_points = 50
interval = total_length / (num_points - 1)

idx = 1
current_distance = interval
selected_points = [path_coords[0]]

for i in range(1, num_points -1):
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

print("Selected Points:")
for point in selected_points: print(point)

# plot
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
x_coords, y_coords = zip(*selected_points)
plt.scatter(x_coords, y_coords, c='red', s=50)
plt.title('Selected Equidistant Points')
plt.show()
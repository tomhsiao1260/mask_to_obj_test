import cv2
import nrrd
import random
import tifffile
import argparse
import numpy as np
import open3d as o3d
import pyvista as pv
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.distance import euclidean
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

# closed shape to .obj (original used for the fossils extraction)

def save_obj(filename, data):
    vertices = data.get('vertices', np.array([]))
    normals  = data.get('normals' , np.array([]))
    faces    = data.get('faces'   , np.array([]))

    with open(filename, 'w') as f:
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n")

        for i in range(len(vertices)):
            vertex = vertices[i]
            normal = normals[i]

            f.write('v ')
            f.write(f"{' '.join(str(round(x, 2)) for x in vertex)}")
            f.write('\n')

            f.write('vn ')
            f.write(f"{' '.join(str(round(x, 6)) for x in normal)}")
            f.write('\n')

        for face in faces:
            indices = ' '.join(['/'.join(map(str, vertex)) for vertex in face])
            f.write(f"f {indices}\n")

# update normals
def compute_normals(mesh):
  if not mesh.triangle_normals:
    mesh.compute_vertex_normals()
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
  else:
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

def find_endpoints(skel):
    points = np.column_stack(np.where(skel > 0))
    points = points[::1]

    G = nx.Graph()
    for z, y, x in points: G.add_node((x, y, z))

    return G

def find_closest_node(G, target_point):
    yp, xp = target_point
    return min(G.nodes, key=lambda node: (node[0] - yp)**2 + (node[1] - xp)**2)

def process_edge_to_point(edge_image, interval):
    # find endpoints & shortest path
    G = find_endpoints(edge_image)

    selected_points = G.nodes
    return selected_points

# python slice_to_point.py --plot
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a series of points in a sliced mask (equal distance along the mask path).')
    parser.add_argument('--label', type=int, default=1, help='Selected label')
    parser.add_argument('--plot', action='store_true', help='Plot the result')
    parser.add_argument('--d', type=int, default=1, help='Interval between each points or layers')
    args = parser.parse_args()

    # original
    label = args.label
    interval = args.d
    max_distance = 3 * interval

    # load mask
    filename = 't1_mask.nrrd'
    # filename = '10624_02304_02432_mask.nrrd'
    data, header = nrrd.read(filename)
    data = np.asarray(data)

    # smooth
    threshold = 0.5
    data = np.where(data == label, 1, 0).astype(float)
    smoothed = gaussian_filter(data, sigma=[1, 1, 1])
    data = (smoothed > threshold).astype(np.uint8)
    data *= label

    image = np.zeros_like(data, dtype=np.uint8)
    image[data == label] = 255
    # tifffile.imwrite('output.tif', image)
    # tifffile.imwrite('data.tif', data * 255)

    # edge detection
    edge_image = np.zeros_like(data, dtype=np.uint8)
    for layer in range(edge_image.shape[0]):
        edge_image[layer] = cv2.Canny(image[layer], 0, 255)

    selected_points = process_edge_to_point(edge_image, interval)
    vertices = np.array(selected_points, dtype=float)

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(vertices)
    
    # Find pairs of points within max_distance
    pairs = tree.query_pairs(r=max_distance)

    # Create edges from these pairs
    edges = np.array(list(pairs))
    edges = np.hstack([[2, edge[0], edge[1]] for edge in edges])
    
    # Create a PyVista PolyData object
    mesh = pv.PolyData(vertices, lines=edges)

    # Convert lines to surface
    surf = mesh.delaunay_2d(alpha=max_distance)

    vertices = surf.points
    faces = surf.faces.reshape(-1, 4)[:, 1:4] # [3, v1, v2, v3]

    # save the mesh as obj
    data = {}
    data['vertices'] = np.copy(vertices)
    data['faces'] = np.copy(faces)

    # use open3d to compute normals
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data['vertices'])
    mesh.triangles = o3d.utility.Vector3iVector(data['faces'])
    compute_normals(mesh)
    data['normals'] = np.asarray(mesh.vertex_normals)

    # switch faces format [1, 3, 2] -> [[1, 1, 1], [3, 3, 3], [2, 2, 2]]
    data['faces'] += 1
    data['faces'] = np.repeat(data['faces'], 3).reshape(-1, 3, 3)
    data['vertices'] += np.array([0, 0, 0])
    # data['vertices'] += np.array([2432, 2304, 10624])

    # save_obj('10624_02304_02432.obj', data)
    save_obj('t1.obj', data)

    if (args.plot):
        plt.figure(figsize=(8, 8))
        plt.imshow(edge_image[50], cmap='gray')
        x_coords, y_coords, z_coords = zip(*selected_points)
        selected_points = [(x, y, z) for x, y, z in zip(x_coords, y_coords, z_coords) if 49.5 <= z <= 50.5]
        x_coords, y_coords, z_coords = zip(*selected_points)
        plt.scatter(x_coords, y_coords, c='red', s=3)
        # plt.scatter(x_coords, y_coords, c='red', s=int(interval // 3))
        plt.title('Selected Equidistant Points')
        plt.show()





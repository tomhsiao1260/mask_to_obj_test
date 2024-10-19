import os
import nrrd
import open3d
import tifffile
import argparse
import numpy as np
import open3d as o3d

from scipy.spatial import Delaunay
from skimage.morphology import skeletonize
from slice_to_point import process_slice_to_point

z, y, x = 3513, 1900, 3400
# z, y, x = 10624, 2304, 2432

def save_obj(filename, data):
    vertices = data.get('vertices', np.array([]))
    normals  = data.get('normals' , np.array([]))
    uvs      = data.get('uvs'     , np.array([]))
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

        for uv in uvs:
            f.write(f"vt {' '.join(str(round(x, 6)) for x in uv)}\n")

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

def main(output_dir, mask_dir, label, interval):
    # load mask
    data, header = nrrd.read(mask_dir)
    data = np.asarray(data)

    selected_points_list = []
    prev_start, prev_end = None, None

    for layer in range(0, data.shape[0], interval):
        # original
        image = np.zeros_like(data[layer], dtype=np.uint8)
        image[data[layer] == label] = 255

        # skeletonize
        mask = skeletonize(image)
        skeleton_image = np.zeros_like(image, dtype=np.uint8)
        skeleton_image[mask] = 255

        if (np.sum(mask) < 50): continue

        # selected points for each slices
        if (prev_start is None):
            selected_points, start, end = process_slice_to_point(skeleton_image, interval)
            prev_start, prev_end = start, prev_end
        else:
            selected_points, _, _ = process_slice_to_point(skeleton_image, interval, prev_start, prev_end)

        selected_points = [(x, y, layer) for x, y in selected_points]
        selected_points_list.append(selected_points)

    points = []
    for selected_points in selected_points_list:
        for point in selected_points:
            points.append(point)

    num_u, num_v = 0, len(selected_points_list)
    for selected_points in selected_points_list:
        num_u = max(num_u, len(selected_points))

    uvs = []
    for index_v, selected_points in enumerate(selected_points_list):
        for index_u, point in enumerate(selected_points):
            shift = float(num_u - len(selected_points)) / 2
            u = (float(index_u) + shift) / float(num_u)
            v = float(index_v) / float(num_v)
            uvs.append((round(u, 6), round(v, 6)))

    tri = Delaunay(uvs)

    data = {}
    data['vertices'] = np.array(points)
    data['uvs'] = np.array(uvs)
    data['faces'] = np.array(tri.simplices)

    # filter out the triangle with too large distance
    triangles = data['vertices'][data['faces']]

    edge_0 = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    edge_1 = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    edge_2 = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)

    max_d = interval * 5
    mask = (edge_0 < max_d) & (edge_1 < max_d) & (edge_2 < max_d)
    data['faces'] = data['faces'][mask] 

    # use open3d to compute normals
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data['vertices'])
    mesh.triangles = o3d.utility.Vector3iVector(data['faces'])
    compute_normals(mesh)
    data['normals'] = np.asarray(mesh.vertex_normals)

    # switch faces format [1, 3, 2] -> [[1, 1, 1], [3, 3, 3], [2, 2, 2]]
    data['faces'] += 1
    data['faces'] = np.repeat(data['faces'], 3).reshape(-1, 3, 3)
    data['vertices'] += np.array([x, y, z])

    save_obj(os.path.join(output_dir, f'{z:05}_{y:05}_{x:05}.obj'), data)

# python mask_to_obj.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mesh from a given mask.')
    parser.add_argument('--label', type=int, default=1, help='Selected label')
    parser.add_argument('--d', type=int, default=5, help='Interval between each points or layers')
    args = parser.parse_args()

    output_dir = '/Users/yao/Desktop/output'
    mask_dir = f'/Users/yao/Desktop/ink-explorer/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}_mask_1.nrrd'
    label = args.label
    interval = args.d

    main(output_dir, mask_dir, label, interval)







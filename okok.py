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

def main(output_dir, mask_dir, label, interval):
    return

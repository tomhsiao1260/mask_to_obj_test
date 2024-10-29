from mask_to_obj import main
from pipeline_api import Node

def handler(inputs):
    z, y, x = 4281, 1765, 3380
    label, interval, sliceZ = 1, 5, False

    mask_dir = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}_mask.nrrd'
    obj_dir = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/label_{label:02}/{z:05}_{y:05}_{x:05}_{label:02}.obj'

    main(mask_dir, obj_dir, label, interval, sliceZ, (z, y, x))

    return {**inputs, "data": {"counter": 3}, "view": {"obj": obj_dir}}

if __name__ == "__main__":
    Node(handler)

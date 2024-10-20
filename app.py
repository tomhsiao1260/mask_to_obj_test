from flask import Flask, jsonify
import requests
from parse_args import parse_args
from mask_to_obj import main

app = Flask(__name__)

id_value, from_value = parse_args()


@app.route("/")
def process():
    # inputs = requests.get("http://127.0.0.1:" + from_value).json()

    handler()

    outputs = {"data": {"counter": 10}}
    
    return jsonify(outputs)

def handler():
    z, y, x = 3513, 1900, 3400
    label, interval = 1, 5

    mask_dir = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}_mask.nrrd'
    obj_dir = f'/Users/yao/Desktop/cubes/{z:05}_{y:05}_{x:05}/{z:05}_{y:05}_{x:05}.obj'

    main(mask_dir, obj_dir, label, interval, (z, y, x))

if __name__ == "__main__":
    app.run(port=id_value)

from flask import Flask, jsonify
import requests
from parse_args import parse_args

import argparse
# from okok import main
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
    parser = argparse.ArgumentParser(description='Extract a series of points in a sliced mask (equal distance along the mask path).')
    parser.add_argument('--label', type=int, default=1, help='Selected label')
    parser.add_argument('--plot', action='store_true', help='Plot the result')
    parser.add_argument('--d', type=int, default=5, help='Interval between each points or layers')
    args = parser.parse_args()

    output_dir = '/Users/yao/Desktop/output'
    mask_dir = '/Users/yao/Desktop/mask_to_obj_test/10624_02304_02432_mask.nrrd'
    label = args.label
    interval = args.d
    plot = args.plot

    # main(1, 2, 3, 4)
    main(output_dir, mask_dir, label, interval)

if __name__ == "__main__":
    app.run(port=id_value)

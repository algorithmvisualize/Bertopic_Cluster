import os
import json
from cluster.run import get


if __name__ == "__main__":
    data_path = "data"
    cluster_path = "cluster_data"
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    all_data_file = os.listdir(data_path)
    all_data_file = [os.path.join(data_path, i) for i in all_data_file]
    for data_file in all_data_file:
        with open(data_file, "r") as f:
            data = json.load(f)
        dumps = [get(d) for d in data]
        dumps_file = os.path.join(cluster_path, f'{os.path.basename(data_file).split(".")[0]}_cluster.json')

        with open(dumps_file, "w") as f:
            json.dump(dumps, f)




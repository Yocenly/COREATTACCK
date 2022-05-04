import numpy as np
import os
import os
# path1=os.path.abspath('.')   # 表示当前所处的文件夹的绝对路径
# print(path1)
# path2=os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
# print(path2)
import pandas as pd
import math
import networkx as nx
import random
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import time
import json
from collections import defaultdict
from collections import Counter
from tqdm import tqdm
import openpyxl
from itertools import combinations
import EfficiencyEvaluation

data_path = "./datasets/"
cache_path = "./cache/"


def get_cmd_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, default='karate', help='select dataset')
    parser.add_argument('-r', '--round', dest='round', type=int, default=30, help='attack rounds')
    parser.add_argument('-t', '--type', dest='type', type=str, default="HA", help='attack type')
    args = parser.parse_args()
    try:
        file = open(data_path + args.dataset + ".pkl", "rb")
    except:
        raise ValueError("Unexpected filename " + str(args.dataset) + " received.")
    else:
        return args, file


if __name__ == "__main__":
    # dirs = os.listdir("./graphml/")
    # for fullname in dirs:
    #     filename = fullname.split(".")[0]
    #     print(filename)
    #     g = nx.read_graphml(os.path.join("./graphml/", fullname))
    #     print(nx.neighbors(g, "1"))
    #     # edges = list(g.edges)
    #     # file = open("./datasets/" + filename + ".pkl", "wb")
    #     # pkl.dump(edges, file)
    #     # file.close()
    #     break
    ticks = time.process_time()
    print(time.process_time() - ticks)
    pass

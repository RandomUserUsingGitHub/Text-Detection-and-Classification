import os
import pandas as pd
import random
import time

def initial_random_seed():
    random.seed(time.time())

def get_current_folder_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_TextDataset_file_path(current_folder_path):
    return os.path.join(current_folder_path, 'TextDataset', 'EnFa.csv')


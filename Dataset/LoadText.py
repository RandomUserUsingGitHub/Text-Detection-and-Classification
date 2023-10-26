from Utils.fileUtils import get_TextDataset_file_path
from Utils.RandomUtils import initial_random_seed

import pandas as pd


classes = ['eng','pes']

def loadText():
    initial_random_seed()
    TextDataset_file_path = get_TextDataset_file_path()
    df = pd.read_csv(TextDataset_file_path)
    eng_rows = df[df["lan_code"].str.contains(classes[0], case=False)]
    pes_rows = df[df["lan_code"].str.contains(classes[1], case=False)]






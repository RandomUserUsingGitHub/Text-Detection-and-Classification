from Utils.fileUtils import get_TextDataset_file_path
from Utils.RandomUtils import initial_random_seed
from Utils.PandasUtils import read_file

classes = ['eng','pes']

def LoadText():
    initial_random_seed()
    TextDataset_file_path = get_TextDataset_file_path()
    df = read_file(TextDataset_file_path)
    eng_rows = df[df["lan_code"].str.contains(classes[0], case=False)]
    pes_rows = df[df["lan_code"].str.contains(classes[1], case=False)]
from 

def LoadText():
    current_folder_path = get_current_folder_path()
    TextDataset_file_path = get_TextDataset_file_path(current_folder_path)
    df = pd.read_csv(TextDataset_file_path)
    eng_rows = df[df["lan_code"].str.contains(classes[0], case=False)]
    pes_rows = df[df["lan_code"].str.contains(classes[1], case=False)]
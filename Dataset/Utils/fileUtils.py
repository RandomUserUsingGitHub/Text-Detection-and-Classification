import os


def get_current_folder_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_TextDataset_file_path():
    current_folder_path = get_current_folder_path()
    return os.path.join(current_folder_path, '..', 'TextDataset', 'EnFa.csv')

def get_font_file_path(language):
    current_folder_path = get_current_folder_path()
    return os.path.join(current_folder_path, '..', 'Fonts', language)



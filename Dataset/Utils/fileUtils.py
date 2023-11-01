import os


def get_current_folder_path():
    return os.path.dirname(os.path.abspath(__file__))

def get_TextDataset_file_path():
    current_folder_path = get_current_folder_path()
    return os.path.join(current_folder_path, '..', 'TextDataset', 'EnFa.csv')

def get_font_file_path(language):
    current_folder_path = get_current_folder_path()
    return os.path.join(current_folder_path, '..', 'Fonts', language)

def get_backgrounds_file_path():
    current_folder_path = get_current_folder_path()
    return os.path.join(current_folder_path, '..', 'Backgrounds')

def find_files(path, file_extensions):
    file_list = []
    try:
        files = [f for f in os.listdir(path) if any(f.endswith(ext) for ext in file_extensions)]
        if not files:
            print(f"No files with {', '.join(file_extensions)} extension(s) found in {path}")
        else:
            for file in files:
                file_list.append(os.path.join(path, file))
    except FileNotFoundError:
        print(f"path '{path}' not found.")

    return file_list

def define_image_name(label):
    current_folder_path = get_current_folder_path()
    return os.path.join(current_folder_path, '..', 'Dataset', f'{label+1}.png')

def delete_old_dataset():
    current_folder_path = get_current_folder_path()
    generated_path = os.path.join(current_folder_path, '..', 'Dataset')
    files = os.listdir(generated_path)

    for file in files:
        path = os.path.join(generated_path, file)
        if os.path.isfile(path):
            os.remove(path)



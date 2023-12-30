from Generator.LoadText import loadText
from Generator.Generate import generator
from Generator.Utils.FileUtils import delete_old_dataset, get_model_path, list_models
from train import train_model
# from DataLoader import parse_csv_row

import warnings


def check_answer(inputt):
    if('y' in inputt.lower()):
        return True
    else:
        return False

def create_new_dataset():
    user_input = input("Create new dataset? (y/n) ")
    if(not check_answer(user_input)):
        return
    delete_dataset = input("DELETE ALL PREVIOUS FILES? (y/n) ")
    if(check_answer(delete_dataset)):
        delete_old_dataset()
    else:
        raise SystemExit
    # delete_old_dataset()
    print("Loading the texts...")
    text = loadText()
    amount = int(input("How many images? "))
    generator(text, amount=amount)


def main():      
    # create_new_dataset()
    train_model()





if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
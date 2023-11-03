from Generator.LoadText import loadText
from Generator.Generate import generator
from Generator.Utils.FileUtils import delete_old_dataset
from train import train_model
# from DataLoader import parse_csv_row

import warnings


def create_new_dataset():
    delete_dataset = input("DELETE ALL PREVIOUS FILES? (y/n) ")
    if('y' in delete_dataset.lower()):
        delete_old_dataset()
    else:
        raise SystemExit
    # delete_old_dataset()
    print("Loading the texts...")
    text = loadText()
    amount = int(input("How many images? "))
    generator(text, amount=amount)

    


def main():
    create_dataset = input("Create new dataset? (y/n) ")
    if('y' in create_dataset.lower()):
        
        create_new_dataset()

       
    # create_new_dataset()
    train_model()





if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
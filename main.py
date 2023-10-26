from Dataset.LoadText import loadText
from Dataset.Generate import generator
from Dataset.Utils.FileUtils import

import warnings


def create_new_dataset():
    print("Loading the texts...")
    text = loadText()
    # amount = int(input("How many images? "))
    generator(text, amount=5)

    


def main():
    # create_dataset = input("Create new dataset? (y/n) ")
    # if('y' in create_dataset.lower()):
    #     create_new_dataset()

    create_new_dataset()



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
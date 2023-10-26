from Dataset.LoadText import loadText
from Dataset.Generate import generator

import warnings


def create_new_dataset():
    print("Loading the texts...")
    text = loadText()
    amount = input("How many images? ")
    generator(text, amount)

    


def main():
    create_dataset = input("Create new dataset? (y/n)")
    if('y' in create_dataset.lower()):
        create_new_dataset()




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
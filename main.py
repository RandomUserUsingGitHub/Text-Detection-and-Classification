from Generator.LoadText import loadText
from Generator.Generate import generator
from Generator.Utils.FileUtils import delete_old_dataset
from train import train_model
from eval import evaluate_model
from ScreenShotTester import test
# from DataLoader import parse_csv_row

import warnings


def check_answer(message):
    user_input = input(message)
    if('y' in user_input.lower()):
        return True
    else:
        return False

def create_new_dataset():
    if(not check_answer("Create new dataset? (y/n) ")):
        return
    if(check_answer("DELETE ALL PREVIOUS FILES? (y/n) ")):
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

    if(check_answer("Train a model? (y/n) ")):
        train_model()

    if(check_answer("Evaluate a model? (y/n) ")):
        evaluate_model()

    if(check_answer("Test a model? (y/n) ")):
        test()





if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
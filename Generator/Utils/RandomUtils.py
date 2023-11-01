import random
import time

def initial_random_seed():
    random.seed(time.time())

def random_choice(choices):
    return random.choice(choices)

def random_number(min, max):
    return random.randint(min, max)
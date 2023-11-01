from .TextDataset.Languages import classes
from .Utils.FileUtils import get_font_file_path, get_backgrounds_file_path, find_files, define_image_name
from .Utils.RandomUtils import random_choice, random_number

from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display

font_min = 20
font_max = 70
gray_level_max = 50
image_ratio = [400,300]
x_ofset = 100
y_ofset = 50
outer_left_text_spawn = -150
outer_up_text_spawn = -50
bounding_box = [0, 0, 0, 0]
color: int


def calculate_true_box(x_min, y_min, x_max, y_max):
    global bounding_box
    offset = 20 # in pixels
    bounding_box = [x_min, y_min, x_max, y_max]

    if(x_min > image_ratio[0] - offset):
        return False
    if(x_max < 0 + offset):
        return False
    if(y_min > image_ratio[1] - offset):
        return False
    if(y_max < 0 + offset):
        return False

    if(x_min < 0):
        bounding_box[0] = 0
    if(x_max > image_ratio[0]):
        bounding_box[2] = image_ratio[0]
    if(y_min < 0):
        bounding_box[1] = 0
    if(y_max > image_ratio[1]):
        bounding_box[3] = image_ratio[1]

    return True

def random_initial_position():
    return random_number(outer_left_text_spawn, image_ratio[0] - x_ofset), random_number(outer_up_text_spawn, image_ratio[1] - y_ofset)



def generate_image(text_to_generate, font_folder):
    global image, bool_text_on_screen, text
    text = get_display(reshape(text_to_generate))
    font_path = random_choice(find_files(font_folder, ([".ttf", ".otf"])))
    font_size = random_number(font_min, font_max)
    font = ImageFont.truetype(font_path, font_size)
    color = random_number(0, gray_level_max)
    text_color = (color,color,color)
    background_path = find_files(get_backgrounds_file_path(), ([".png"]))
    background_path = random_choice(background_path)
    image = Image.open(background_path)
    image = image.resize((image_ratio[0],image_ratio[1]))
    draw = ImageDraw.Draw(image)
    x, y = random_initial_position()
    text_width, text_height = draw.textsize(text, font)
    bool_text_on_screen = calculate_true_box(x, y, x+text_width, y+text_height)
    draw.text((x, y), text, font=font, fill=text_color)
    image = image.convert("L")
    return image

def image_viewer(image):
    import cv2
    import numpy as np
    outline_color = (0,255,0)
    width = 2
    pil_array = np.array(image)
    cv2_image = cv2.cvtColor(pil_array, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(cv2_image, bounding_box, outline_color, width)
    cv2.imshow("Image Viewer", cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def creare_dataset(generated_image, image_ID):
    generated_image.save(define_image_name(image_ID))


def generator(text, amount):
    global bool_text_on_screen, counter
    eng_texts, pes_texts = text

    selected_eng_texts = eng_texts.sample(n=amount)
    selected_pes_texts = pes_texts.sample(n=amount)

    images = []
    label_and_bbox = []
    labels = []
    image_ID = 0
    for index in selected_eng_texts.index:
        # index = random.choice(rows.index)
        text_to_generate = selected_eng_texts.loc[index, 'sentence']
        font_folder_path = get_font_file_path('eng')
        bool_text_on_screen = False

        while(not bool_text_on_screen):
            generated_image = generate_image(text_to_generate, font_folder_path)
        # generated_image.show()
        image_viewer(generated_image)
        creare_dataset(generated_image, image_ID)
        image_ID += 1


    for index in selected_pes_texts.index:
        # index = random.choice(rows.index)
        text_to_generate = selected_pes_texts.loc[index, 'sentence']
        font_folder_path = get_font_file_path('pes')
        bool_text_on_screen = False

        while(not bool_text_on_screen):
            generated_image = generate_image(text_to_generate, font_folder_path)
        # generated_image.show()
        image_viewer(generated_image)
        creare_dataset(generated_image, image_ID)
        image_ID += 1

    return images, labels

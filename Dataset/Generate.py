from TextDataset.Languages import classes
from Utils.fileUtils import get_font_file_path

from arabic_reshaper import reshape
from bidi.algorithm import get_display





def generator(text, amount):
    global bool_text_on_screen, counter
    eng_texts, pes_texts = text

    selected_eng_texts = eng_texts.sample(n=amount)
    selected_pes_texts = pes_texts.sample(n=amount)

    images = []
    label_and_bbox = []
    labels = []

    for index in selected_eng_texts.index:
        # index = random.choice(rows.index)
        text_to_generate = selected_eng_texts.loc[index, 'sentence']
        font_folder_path = get_font_file_path('eng')
        bool_text_on_screen = False

        while(not bool_text_on_screen):
            generated_image = generate_image(text_to_generate, font_folder_path)


    return images, labels

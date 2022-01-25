from tensorflow import keras
import cv2
from PIL import ImageOps, Image
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from train import (
    preprocess_category,
    preprocess_card,
    preprocess_national_id_new,
    preprocess_national_id_old,
    num_to_label,
)


def load(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def infer(model, img_pre, alphabet, concat=16):
    preds = model.predict(img_pre)
    decoded = K.get_value(
        K.ctc_decode(
            preds,
            input_length=np.ones(preds.shape[0]) * preds.shape[1],
            greedy=True,
        )[0][0]
    )
    prediction = num_to_label(decoded[0], alphabets=alphabet)
    return prediction[:concat]


def predict(
    image_path: str,
    category_model_path,
    card_model_path,
    national_id_new_model_path,
    national_id_old_model_path,
    category,
):
    """predict image number

    Args:
        image_path (str): image path
        category_model_path (str): category model path
        card_model_path (str): card model path
        national_id_new_model_path (str): new national id model path
        national_id_old_model_path (str): old national id model path

    Returns:
        [type]: [description]
    """
    img = Image.open(image_path)

    # feed to category modl
    # img_cat = preprocess_category(img)

    # model_cat = load(category_model_path)
    # category = np.argmax(model_cat.predict(img_cat))
    # category = 1
    print("category is ", category)
    if category == 0:  # card
        model = load(card_model_path)
        img_pre = preprocess_card(img)
        alphabet = u"0123456789"
        concat = 16
    elif category == 1:  # new national id
        model = load(national_id_new_model_path)
        img_pre = preprocess_national_id_new(img)
        print(national_id_new_model_path)
        print(img_pre.shape)
        plt.imshow(img_pre[0, :, :, 0])
        plt.show()

        alphabet = u"۰۱۲۳۴۵۶۷۸۹"
        # alphabet = u"0123456789"
        concat = 10

    else:  # old national id
        model = load(national_id_old_model_path)
        img_pre = preprocess_national_id_old(img)
        alphabet = u"۰۱۲۳۴۵۶۷۸۹"
        concat = 10
    print(np.mean(img_pre))
    return infer(model, img_pre, alphabet, concat)


def predict_card(image_path, card_model_path):
    img = Image.open(image_path)
    model = load(card_model_path)
    img_pre = preprocess_card(img)
    alphabet = u"0123456789"
    concat = 16

    return infer(model, img_pre, alphabet, concat)


# a = predict_card(
#     "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\data\\data2.PNG",
#     "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\savedmodels\\modelcard.h5",
# )
for i in range(3, 8):
    a = predict(
        "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\data\\data{}.PNG".format(i),
        "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\savedmodels\\init_model.h5",
        "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\savedmodels\\modelcard.h5",
        "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\savedmodels\\model_new_national_card.h5",
        "C:\\Users\\Lion\\Documents\\Repos\\DeepOCR\\savedmodels\\model_new_national_card.h5",
    )
    print("data {} is {}".format(i, a))

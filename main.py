from tensorflow import keras
import cv2
from PIL import ImageOps, Image
import numpy as np
from tensorflow.keras import backend as K
from train import (
    preprocess_category,
    preprocess_card,
    preprocess_national_id_new,
    preprocess_national_id_old,
    num_to_label,
)


def load(model_path):
    model = keras.models.load_model(model_path)
    return model


def infer(model, img_pre, alphabet):
    preds = model.predict(img_pre)
    decoded = K.get_value(
        K.ctc_decode(
            preds,
            input_length=np.ones(preds.shape[0]) * preds.shape[1],
            greedy=True,
        )[0][0]
    )
    prediction = num_to_label(decoded[0], alphabets=alphabet)
    return prediction


def predict(
    image_path: str,
    category_model_path,
    card_model_path,
    national_id_new_model_path,
    national_id_old_model_path,
):
    img = Image.open(image_path)

    # feed to category modl
    img_cat = preprocess_category(img)

    model_cat = load(category_model_path)
    category = np.argmax(model_cat.predict(img_cat))

    if category == 0:  # card
        model = load(card_model_path)
        img_pre = preprocess_card(img)
        alphabet = u"0123456789"
    elif category == 1:  # new national id
        model = load(national_id_new_model_path)
        img_pre = preprocess_national_id_new(img)
        alphabet = u"۰۱۲۳۴۵۶۷۸۹"

    else:  # old national id
        model = load(national_id_old_model_path)
        img_pre = preprocess_national_id_old(img)
        alphabet = u"۰۱۲۳۴۵۶۷۸۹"

    return infer(model, img_pre, alphabet)

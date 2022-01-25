from tensorflow import keras
import cv2
from PIL import ImageOps, Image, ImageDraw
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Bidirectional,
    LSTM,
    Dense,
    Lambda,
    Activation,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

# params
img_shape = (400, 100, 1)
alphabets = u"0123456789"
num_of_characters = len(alphabets) + 1
max_str_len = 16
num_of_timestamps = 64


def load_image(train_data_path):
    train = []
    label = []
    for ei in os.listdir(train_data_path):
        train.append(Image.open(train_data_path + ei))
        label.append(ei.split(".")[0])
    return train, label


def preprocess(img):
    img = ImageOps.grayscale(img)
    img = np.array(img)
    (h, w) = img.shape
    final_img = np.ones([100, 400]) * 255  # blank white image

    # crop
    if w > 400:
        img = img[:, :400]
    if h > 100:
        img = img[:100, :]

    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)


def preprocess_image(img):
    return cv2.rotate(
        np.array(img.resize((400, 100)).convert("L")), cv2.ROTATE_90_CLOCKWISE
    )


def preprocess_train(data, shape=(400, 100)):
    res_data = []
    for i, d in enumerate(data):
        img = preprocess_image(d)
        # img = img/255
        res_data.append(img)
    return np.array(res_data).reshape(-1, shape[0], shape[1], 1)


def label_to_num(label, alphabets=u"0123456789"):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))

    return np.array(label_num)


def num_to_label(num, alphabets=u"0123456789"):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret


def preprocess_label(
    data, alphabets=u"0123456789", max_str_len=16, num_of_timestamps=64
):
    num_of_characters = len(alphabets) + 1
    train_size = len(data)
    train_y = np.ones([train_size, max_str_len]) * -1
    train_label_len = np.zeros([train_size, 1])
    train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
    train_output = np.zeros([train_size])
    for i in range(train_size):
        train_label_len[i] = len(data[i])
        train_y[i, 0 : len(data[i])] = label_to_num(data[i])
    return train_y, train_input_len, train_label_len, train_output


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_card_model():
    input_data = Input(shape=img_shape, name="input")

    inner = Conv2D(
        32, (3, 3), padding="same", name="conv1", kernel_initializer="he_normal"
    )(input_data)
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name="max1")(inner)

    inner = Conv2D(
        64, (3, 3), padding="same", name="conv2", kernel_initializer="he_normal"
    )(inner)
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name="max2")(inner)
    inner = Dropout(0.3)(inner)

    inner = Conv2D(
        128, (3, 3), padding="same", name="conv3", kernel_initializer="he_normal"
    )(inner)
    inner = BatchNormalization()(inner)
    inner = Activation("relu")(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name="max3")(inner)
    inner = Dropout(0.3)(inner)

    # CNN to RNN
    inner = Reshape(target_shape=((100, 1536)), name="reshape")(inner)
    inner = Dense(64, activation="relu", kernel_initializer="he_normal", name="dense1")(
        inner
    )

    ## RNN
    inner = Bidirectional(LSTM(256, return_sequences=True), name="lstm1")(inner)
    inner = Bidirectional(LSTM(256, return_sequences=True), name="lstm2")(inner)

    ## OUTPUT
    inner = Dense(num_of_characters, kernel_initializer="he_normal", name="dense2")(
        inner
    )
    y_pred = Activation("softmax", name="softmax")(inner)

    model = Model(inputs=input_data, outputs=y_pred)

    labels = Input(name="gtruth_labels", shape=[max_str_len], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )
    model_final = Model(
        inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss
    )

    return model_final, model, labels, input_length, label_length


def prepare_card_data(
    train_data_path,
    val_percent=0.1,
):

    train, label = load_image(train_data_path)
    N = int(len(train) * (1 - val_percent))
    X = train[:N]
    l_x = label[:N]
    V = train[N:]
    l_v = label[N:]
    valid_x = preprocess_train(V)
    train_x = preprocess_train(X)
    train_y, train_input_len, train_label_len, train_output = preprocess_label(l_x)

    return (
        train_x,
        train_y,
        train_input_len,
        train_label_len,
        valid_x,
        l_v,
        l_x,
        train_output,
    )


def train_card_model(train_data_path, epochs=40, batch_size=128, save_dir="model/"):
    (
        train_x,
        train_y,
        train_input_len,
        train_label_len,
        valid_x,
        l_v,
        l_x,
        train_output,
    ) = prepare_card_data(train_data_path)

    model_final, model, labels, input_length, label_length = get_card_model()

    model_final.compile(
        loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=0.001)
    )

    hist = model_final.fit(
        x=[train_x, train_y, train_input_len, train_label_len],
        y=train_output,
        epochs=epochs,
        batch_size=batch_size,
    )

    model.save(save_dir + "model.h5")
    model_final.save(save_dir + "modelfinal.h5")

    return model_final, model


def preprocess_category(img):
    pass


def preprocess_card(img):
    pass


def preprocess_national_id_new(img):
    pass


def preprocess_national_id_old(img):
    pass

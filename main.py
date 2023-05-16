# Импортируем сам keras
import keras
# Последовательный тип модели
from keras.models import Sequential
# Импортируем полносвязный слой, слои активации и слой,
# превращающий картинку в вектор
from keras.layers import Dense, Activation, Flatten
# Импортируем сверточный слой, слои, фильтрующий максимальные значения из
# входных данных, слой "выключающий часть нейронов"
from keras.layers import Conv2D, MaxPooling2D, Dropout

# Импортируем датасеты, чтобы вытащить оттуда нужные нам данные
import keras.datasets

import numpy as np
from keras.utils import load_img
from matplotlib import pyplot as plt

# Эти библиотеки отключают лишние предупреждения от библиотек, в частности,
# tensorflow, чтобы не засорять вывод наших результатов
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf


def plot_dataset_samples_grid(image_data, dataset_name='', N=8):
    """
    Эта функция строит NxN сэмплов из датасета image_data

    Параметры
    ----------
    image_data : array
        Array of shape
        (number_of_samples, image_width, image_height, number of channels)
        with images
    dataset_name : str
        Name of dataset to write in the title
    N : int
        Size of grid of samples
  """
    plt.figure(figsize=(10, 10))
    data1 = image_data[:N * N]

    image_width = image_data.shape[1]
    image_heigth = image_data.shape[2]

    if len(data1.shape) == 4:
        image_channels = image_data.shape[3]
        data1 = data1.reshape(N, N, image_width, image_heigth, image_channels)
        data1 = np.transpose(data1, (0, 2, 1, 3, 4))
        data1 = data1.reshape(N * image_width, N * image_heigth, image_channels)
        plt.imshow(data1)

    elif len(data1.shape) == 3:
        data1 = data1.reshape(N, N, image_width, image_heigth)
        data1 = np.transpose(data1, (0, 2, 1, 3))
        data1 = data1.reshape(N * image_width, N * image_heigth)
        plt.imshow(data1, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('First ' + str(N * N) + ' ' + dataset_name + ' samples of training set')
    plt.show()


def plot_CIFAR_samples(image_data, label_data, classes, N=8):
    """
    Эта функция строит N сэмплов каждого класса из датасета image_data

    Параметры
    ----------
    image_data : array
        Array of shape
        (number_of_samples, image_width, image_height, number of channels)
        with images
    label_data : array
        Array of shape
        (number_of_samples, )
        with labels
    classes : dict
        Dictionary {class_number:class_name}
    dataset_name : str
        Name of dataset to write in the title
    N : int
        Number of samples for each class
  """
    plt.figure(figsize=(10, N))
    num_classes = len(classes.keys())
    for i, key in enumerate(classes.keys()):
        idxs = np.flatnonzero(label_data == key)
        idxs = np.random.choice(idxs, N, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + key + 1
            plt.subplot(N, num_classes, plt_idx)
            plt.imshow(image_data[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(classes[key])
    plt.show()


from os import listdir, sep
from os.path import abspath, basename, isdir


def tree(dir, padding='  ', print_files=False):
    """
    Эта функция строит дерево поддиректорий и файлов для заданной директории

    Параметры
    ----------
    dir : str
        Path to needed directory
    padding : str
        String that will be placed in print for separating files levels
    print_files : bool
        "Print or not to print" flag
    """
    cmd = "find '%s'" % dir
    files = os.popen(cmd).read().strip().split('\n')
    padding = '|  '
    for file in files:
        level = file.count(os.sep)
        pieces = file.split(os.sep)
        symbol = {0: '', 1: '/'}[isdir(file)]
        if not print_files and symbol != '/':
            continue
        print(padding * level + pieces[-1] + symbol)


def plot_cats_dogs_samples(train_dir, N=4):
    """
    Эта функция строит N сэмплов каждого класса из датасета Cats vs Dogs

    Параметры
    ----------
    train_dir : str
        Directory with train Cats vs Dogs dataset
    N : int
        Number of samples for each class
  """
    import random
    fig, ax = plt.subplots(2, N, figsize=(5 * N, 5 * 2))

    for i, name in enumerate(['cat', 'dog']):
        filenames = os.listdir(os.path.join(train_dir, name))

        for j in range(N):
            sample = random.choice(filenames)
            image = load_img(os.path.join(train_dir, name, sample))
            ax[i][j].imshow(image)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_title(name)
    plt.grid(False)
    plt.show()


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """
    Эта функция показывает 6 картинок с предсказанными и настоящими классами
    """
    label_dict = {0.: 'cat', 1.: 'dog'}
    n = 0
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 10))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((224, 224, 3)), cmap='gray')
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(label_dict[pred_errors[error]],
                                                                                label_dict[obs_errors[error]]))
            n += 1
    plt.tight_layout()


def image_to_batch(img, size=150):
    """
    Эта функция переводит картинку размером (img_width,img_height, 3) в батч
    размером (1,size,size,3)

    Parameters
    ----------
    img : array
        Image of size (img_width,img_height, 3)

    size : int
        Size of image in batch

    Returns
    ----------
    img_resized : array
        Batch of one image with shape (1,size,size,3)
  """
    import cv2
    img_resized = cv2.resize(img, (size, size)).reshape(1, size, size, img.shape[2])
    return img_resized


def get_test_predictions(test_generator, model, dataset_len=2500):
    """
    Эта функция вытаскивает из генератора все предсказания

    Параметры
    ----------
    test_generator  : ImageDataGenerator
        Generator, producing batches (img, label)
    model : keras.model
        Model for getting predictions
    dataset_len : int
        Number of samples in generator

    Returns
    ----------
    preds_labels  : array
        Predicted labels
    preds_vec : array
        Predicted probabilities
    labels_vec : array
        True labels
    datas_vec : array
        Array of images
  """
    labels = []
    preds = []
    datas = []

    samples = 0
    for i, batch in enumerate(test_generator):
        data, label = batch
        labels.append(label)
        preds.append(model.predict(data))
        datas.append(data)
        samples += len(data)
        if samples >= dataset_len:
            break

    labels_vec = np.hstack(labels)
    preds_vec = np.hstack([pred.reshape(-1, ) for pred in preds])
    datas_vec = np.vstack(datas)
    preds_labels = preds_vec.copy()
    preds_labels[preds_labels < 0.5] = 0
    preds_labels[preds_labels >= 0.5] = 1

    return preds_labels, preds_vec, labels_vec, datas_vec


# Загрузка данных
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
LABEL_TRANSLATION = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                     9: 'truck'}  # Создаем классы(столбцы) картинок
# plot_CIFAR_samples(X_train, y_train, LABEL_TRANSLATION, N=7) проверяем работоспосбность

# Предобработка данных
X_train = X_train / 255
X_test = X_test / 255

y_train = keras.utils.np_utils.to_categorical(y_train, 10)
y_test = keras.utils.np_utils.to_categorical(y_test, 10)
# норамализуем картирки путем деленеия их на 255

# Производим аугментацию данных
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.6, height_shift_range=0.3, horizontal_flip=True)
# Сдвигаем изображение на 0.6 по горизонтали и на 0.3 по вериткали а также случайно отражаем картинку
train_generator = datagen.flow(X_train, y_train, batch_size=128)




# при помощи метода flow получаем доступ к данным

# Строим сверхточную неронную сеть на основе трех блоков VGG
def define_model():
    # Создаем пустую модель
    model = Sequential()

    # VGG1-блок
    # Начинаем со сверточных слоя, указывая тип активации на выходе из него,
    # способ заполнения краев (padding) и способ инициализации весов
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # Здесь мы используем метод MaxPooling, который уменьшает размер обрабатываемого изображения,
    # выбирая из 4 пикселей 1 с максимальным значением, чтобы это быстрее считалось. (2,2) -> 1
    model.add(MaxPooling2D((2, 2)))

    # Слой dropout, который на каждом шаге "выключает" 20% случайно выбранных нейронов
    model.add(Dropout(0.2))

    # VGG2-блок
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # # VGG3-блок
    # Разворачиваем данные в вектор
    model.add(Flatten())
    # Добавляем полносвязные слои:
    # ReLU активация скрытого слоя
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    # Знакомый нам softmax для выходного полносвязного слоя
    model.add(Dense(10, activation='softmax'))

    # Компилируем модель с функцией ошибки categorical crossentropy, оптимизатором Адам
    # (оптимизатор, который со стандартным набором параметров может обучить эффективную
    # нейросеть), и метрикой - количеством правильно угаданных картинок.
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    return model

gen_model = define_model()
history_cnn = gen_model.fit_generator(train_generator,
                                      epochs=10,
                                      validation_data=(X_test, y_test),
                                      shuffle=True)
# обучаем модель с 10 эпохами дабы избежать переобучения


# построи график для того чтобы убучится что модель не преобучается
plt.plot(history_cnn.history['val_accuracy'], '-o', label='validation accuracy')
plt.plot(history_cnn.history['accuracy'], '--s', label='training accuracy')
plt.legend();
plt.show()

# Посмотрим на изменение точности на валидационной (val_acc) и обучающей (acc) выборках
# с каждой эпохой
gen_model.evaluate(X_test, y_test)

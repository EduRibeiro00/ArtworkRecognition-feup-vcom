import pandas as pd
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt

def load_train_dataset(dataset_dir):
    train = pd.read_csv(dataset_dir)
    train["id"] = train["id"].apply(lambda x:x+".png")
    train["attribute_ids"] = train["attribute_ids"].astype(str)
    train = shuffle(train)
    return train

def create_image_generators(train_dir, test_dir, train_dataframe, test_dataframe, batch_size, input_size, data_augmentation=False):
    if(data_augmentation):
        train_datagen=ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        train_datagen=ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
        )

    test_datagen=ImageDataGenerator(
        rescale=1./255,
    )

    train_generator=train_datagen.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=train_dir,
        x_col="id",
        y_col="attribute_ids",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=input_size,
        subset='training'
    )

    valid_generator=train_datagen.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=train_dir,
        x_col="id",
        y_col="attribute_ids",
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",    
        target_size=input_size,
        subset='validation'
    )

    test_generator=test_datagen.flow_from_dataframe(
        dataframe=test_dataframe,
        directory=test_dir,
        x_col="id",
        y_col="attribute_ids",
        batch_size=1,
        shuffle=False,
        class_mode="categorical",
        target_size=input_size,
    )

    return train_generator, valid_generator, test_generator

def plot_history(history, params=["loss, accuracy"]):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, len(params), sharex='col', figsize=(20,7))

    i = 0
    for axis in axes:
        axis.plot(history.history[params[i]], label=params[i])
        axis.plot(history.history['val_' + params[i]], label='val_'+params[i])
        axis.legend(loc='best')
        axis.set_title(params[i])
        i = i + 1

    plt.xlabel('Epochs')
    sns.despine()
    plt.show()
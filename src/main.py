import numpy as np
import tensorflow as tf
import sys
import cv2
import os
import requests
from pathlib import Path



# add parent directory to Python path to find modules
current_dir = Path(__file__).parent
project_root = current_dir.parent  # Go up one level to the project root
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from tools import visualiser
from image_processor import ImageProcessor
from neural_network import ConvolutionalNeuralNetwork, cross_entropy_loss, softmax_crossentropy_backward



labels = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "add",
    11: "sub",
    12: "mul",
    13: "div",
}


def process_dataset(dataset_select: str):

    dataset = None

    os.makedirs(f"../data/dataset/operators_processed/{dataset_select}", exist_ok=True)

    # iterate through samples
    for batch_idx, (sample_image_batch, sample_label_batch) in enumerate(dataset):

        # convert from one hot encoded vectors to int representations to fetch class
        sample_label_batch = tf.argmax(sample_label_batch, axis=1)

        for idx, image in enumerate(sample_image_batch):
            label = int(sample_label_batch.numpy()[idx])

            class_name = labels[label]

            os.makedirs(f"../data/dataset/operators_processed/{dataset_select}/{label}", exist_ok=True)

            processor = ImageProcessor(training=True)

            print(type(image))

            #cast to numpy array and process
            image = image.numpy().astype(np.uint8)

            print(type(image))

            try:
                processed_image = processor.process(image, True)

                print(type(processed_image))

                print(f"../data/dataset/operators_processed/{dataset_select}/{label}/{batch_idx}-{idx}.png")


                cv2.imwrite(
                    f"../data/dataset/operators_processed/{dataset_select}/{label}/{batch_idx}-{idx}.png", processed_image

                )

            except Exception as e:
                print(e)
                continue


def load_parameters(model):
    try:
        parameters = np.load("../data/parameters.npz")

    except EOFError:
        print("Parameters file is empty, weights pre-initialised")
        return

    for layer in model.layers:
        if layer.trainable:
            # load weights and biases
            layer.parameters["W"] = parameters[f"{layer.name}_W"]
            layer.parameters["b"] = parameters[f"{layer.name}_b"]


def build_cnn():
    # build all layers in the CNN
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(14, activation='softmax')
    ])

    return model


def train():
    # Define parameters
    BATCH_SIZE = 64
    IMG_SIZE = (28, 28)


    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/dataset/operators_processed/train',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/dataset/operators_processed/test',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '../data/dataset/operators_processed/val',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42
    )

    # initialise network
    model = build_cnn()

    weights_path = "../data/parameters.h5"
    # checkpointing
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weights_path,
        save_best_only=True,
        save_weights_only=False,
        monitor='loss',
        mode='min',
        verbose=1
    )

    if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
        try:
            model.load_weights(weights_path)
            print(f"weights loaded.")
        except Exception as e:
            print(f"could not load weights from {weights_path}: {e}\nTraining from scratch.")
    else:
        print("no valid weights file found, training from scratch.")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        validation_data= val_dataset,
        epochs=10,
        callbacks=[checkpoint_callback],

    )

    # evaluate model after training
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss:}")


def main():
    path = "data/dataset/expressions_processed/2.png"

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    visualiser.visualise(image, "segment")




if __name__ == "__main__":
    main()
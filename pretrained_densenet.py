import numpy as np
from sklearn.preprocessing import normalize
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import json
from keras.models import load_model
from tqdm import tqdm
from keras.applications import DenseNet169, MobileNetV2
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image as imagelib
import glob

def load_custom_model():
    model_file = "/home/apurva/work/hfme/visio/data/models/keras_model_f900f4000tuned_v1/embedder.last.txt"
    model = DenseNet169(include_top=False, pooling='avg', weights=None)
    model.load_weights(model_file, by_name=True)

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # model = load_model(model_file)
    return model


def create_train_index(model, create=False):
    if create:
        label_path = "data/train_labels.csv"
        train_labels = [i.strip().split(",") for i in open(label_path)][1:]
        class_sample_dict = {}

        for image, label in train_labels:
            class_sample_dict[label] = class_sample_dict.get(label, [])
            class_sample_dict[label].append(image)

        mean_embeddings = []
        class_labels = []

        for label in tqdm(class_sample_dict):
            image_paths = class_sample_dict[label]
            images = []
            for image_path in image_paths:
                image = imagelib.load_img("data/train_set/" + image_path, target_size=(256, 256))
                image = imagelib.img_to_array(image, dtype=np.float32)
                image = preprocess_input(image)
                images.append(image)

            images = np.asarray(images, dtype=np.float32)
            embeddings = model.predict(images, batch_size=32)
            embeddings = normalize(embeddings)
            mean_embedding = np.average(embeddings, axis=0)
            mean_embeddings.append(mean_embedding)
            class_labels.append(label)

        mean_embeddings = np.asarray(mean_embeddings)
        mean_embeddings = normalize(mean_embeddings)
        class_labels = np.asarray(class_labels)

        np.save("train_mean_embeddings.bin.npy", mean_embeddings)
        np.save("train_labels.bin.npy", class_labels)
    else:
        mean_embeddings = np.load("train_mean_embeddings.bin.npy")
        class_labels = np.load("train_labels.bin.npy")

    return mean_embeddings, class_labels


def create_val_embeddings(model, create=False):
    if create:
        label_path = "data/val_labels.csv"
        val_labels = [i.strip().split(",") for i in open(label_path)][1:]
        images = []
        labels = []

        for image_path, label in tqdm(val_labels):
            image = imagelib.load_img("data/val_set/" + image_path, target_size=(256, 256))
            image = imagelib.img_to_array(image, dtype=np.float32)
            image = preprocess_input(image)
            images.append(image)
            labels.append(label)

        images = np.asarray(images, dtype=np.float32)
        embeddings = model.predict(images, verbose=1, batch_size=32)
        embeddings = normalize(embeddings)
        labels = np.asarray(labels)

        np.save("val_embeddings.bin.npy", embeddings)
        np.save("val_labels.bin.npy", labels)

    else:
        embeddings = np.load("val_embeddings.bin.npy")
        labels = np.load("val_labels.bin.npy")

    return embeddings, labels


def create_test_embeddings(model, create=False):
    if create:
        image_names = []

        def generator():
            images = []
            for image_path in tqdm(glob.glob("data/test_set/*.jpg")):
                image = imagelib.load_img(image_path, target_size=(256, 256))
                image = imagelib.img_to_array(image, dtype=np.float32)
                image = preprocess_input(image)
                images.append(image)
                image_names.append(image_path)

                if len(images) == 32:
                    images = np.asarray(images, dtype=np.float32)
                    yield images
                    images = []

            images = np.asarray(images, dtype=np.float32)
            yield images

        embeddings = model.predict_generator(generator(), steps=887, verbose=1)
        embeddings = normalize(embeddings)

        np.save("test_embeddings.bin.npy", embeddings)
        json.dump(image_names, open("test_names.json", 'w'))

    else:
        embeddings = np.load("test_embeddings.bin.npy")
        image_names = json.load(open("test_names.json"))

    return embeddings, image_names


def compute_accuracy(train_index, train_labels, test_index, test_labels):
    mean_dists = euclidean_distances(test_index, train_index)
    # a[i] = j where j is the index of element
    # which comes at i after a is sorted
    ranked_args = np.argsort(np.argsort(mean_dists))
    top_3_accuracy = 0

    training_label_indices = {}
    for label in train_labels:
        training_label_indices[label] = len(training_label_indices)

    for i in range(len(test_labels)):
        test_label = test_labels[i]
        index = training_label_indices[test_label]
        rank = ranked_args[i][index]
        if rank <= 2:
            top_3_accuracy += 1

    return top_3_accuracy / len(test_labels)


def get_top_k_labels(train_index, train_labels, test_index):
    mean_dists = euclidean_distances(test_index, train_index)
    ranking_indices = np.argsort(mean_dists)
    top_3 = []
    for i in range(ranking_indices.shape[0]):
        top_3.append(train_labels[ranking_indices[i]][:3])

    return top_3


def test():
    model = load_custom_model()
    mean_embeddings, class_labels = create_train_index(model, create=True)
    val_embeddings, val_labels = create_val_embeddings(model, create=True)
    test_embeddings, test_names = create_test_embeddings(model, create=True)
    print (compute_accuracy(mean_embeddings, class_labels, val_embeddings, val_labels))
    top_k = get_top_k_labels(mean_embeddings, class_labels, test_embeddings)

    output = open("submission.csv", 'w')
    output.write("img_name,label\n")

    for name, results in zip(test_names, top_k):
        outstr = (name.split("/")[-1] + "," + " ".join(list(results))) + "\n"
        output.write(outstr)

    output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    test()

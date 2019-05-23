import tensorflow as tf
from keras.applications import DenseNet169 as BackBoneNet
from keras.applications import densenet as backbonelib
import argparse
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import Model
from keras.models import load_model, Sequential, save_model
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, InputLayer, Input, Lambda
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.regularizers import L1L2
from keras import callbacks
import json
from keras.preprocessing import image as imagelib
from tqdm import tqdm
import glob


image_size = (256, 256)


def embed(data_dir):
    batch_size = 32
    datagen = ImageDataGenerator(preprocessing_function=backbonelib.preprocess_input)

    generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_samples = generator.samples
    num_batches = num_samples / batch_size

    model_file = "/home/apurva/work/hfme/visio/data/models/keras_model_f900f4000tuned_v1/embedder.last.txt"
    backbone_model = BackBoneNet(include_top=False, pooling='avg', weights=None, input_shape=(image_size[0], image_size[1], 3))
    backbone_model.load_weights(model_file, by_name=True)

    bottleneck_features_train = backbone_model.predict_generator(generator, num_batches, verbose=1)
    return bottleneck_features_train, generator


def lr_schedule(epoch, lr):
    if epoch == 0 :
        lr = 0.002

    if epoch % 20 == 0:
        lr = lr / 2
    return lr


def train(train_data, train_labels, val_data, val_labels):
    batch_size = 32
    num_classes = len(set(train_labels))
    # train_labels = to_categorical(train_labels)
    # val_labels = to_categorical(val_labels)

    input1 = Input(shape=train_data.shape[1:])
    dense1 = Dense(num_classes, activation='softmax', use_bias=True)(input1)

    model = Model(inputs=[input1], output=dense1)

    model.compile(optimizer=SGD(lr=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.summary()

    # state = np.random.get_state()
    # np.random.set_state(state)
    # np.random.shuffle(train_data)
    # np.random.set_state(state)
    # np.random.shuffle(train_labels)

    lr_scheduler = callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    model.fit(train_data, train_labels,
              epochs=52,
              batch_size=batch_size,
              validation_data=(val_data, val_labels),
              shuffle=True,
              callbacks=[lr_scheduler])

    return model


def main(train_dir, val_dir, test_dir):
    # train_data, train_generator = embed(train_dir)
    # np.save("train.npy", train_data)
    # np.save("train_labels.npy", train_generator.classes)
    # json.dump(train_generator.class_indices, open("class_indices.json", 'w'))
    #
    # val_data, val_generator = embed(val_dir)
    # np.save("val.npy", val_data)
    # np.save("val_labels.npy", val_generator.classes)
    #
    # test_data, test_generator = embed(test_dir)
    # np.save("test.npy", test_data)
    # json.dump(test_generator.filenames, open("test_filenames.json", 'w'))
    #
    # train_data = np.load("train.npy")
    # val_data = np.load("val.npy")
    #
    # train_labels = np.load("train_labels.npy")
    # val_labels = np.load("val_labels.npy")
    #
    # print(train_data.shape, train_labels.shape, val_data.shape, val_labels.shape)
    # model = train(train_data, train_labels, val_data, val_labels)
    # save_model(model, 'dense_model_retrain.hd5')

    model = load_model('dense_model_retrain.hd5')
    test_data = np.load("test.npy")
    test_predict = model.predict(test_data, verbose=1)
    argsorts = np.argsort(test_predict)

    class_indices = json.load(open("class_indices.json", 'r'))
    label_to_class = {class_indices[i]:i for i in class_indices}
    test_filenames = json.load(open("test_filenames.json", 'r'))

    print ("img_name,label")

    for i, fname in enumerate(test_filenames):
        fname = fname.split("/")[1]
        top_3_classes = [label_to_class[j] for j in argsorts[i][::-1][:3]]
        print (fname + "," + " ".join(top_3_classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)


    args = parser.parse_args()
    main(args.train, args.val, args.test)


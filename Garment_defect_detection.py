from sklearn.preprocessing import LabelBinarizer  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report  
from keras.preprocessing.image import ImageDataGenerator  
from keras.preprocessing.image import img_to_array  
from keras.optimizers import RMSprop  
from keras.optimizers import SGD  
from keras.applications import ResNet50  
from keras.layers import Input  
from keras.models import Model  
from imutils import paths  
import numpy as np  
import argparse  
import os  
import imutils  
import cv2  
from keras.layers.core import Dropout  
from keras.layers.core import Flatten  
from keras.layers.core import Dense  
  
#  
argpaser = argparse.ArgumentParser()  
argpaserg.add_argument("-d", "--dataset", required=True,  
    help="path to input dataset")  
argpaser.add_argument("-m", "--model", required=True,  
    help="path to output model")  
args = vars(argpaser.parse_args())  
  
  
image_augmentation = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,  
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,  
    horizontal_flip=True, fill_mode="nearest")  
print("[INFO] loading images...")  
imagePaths = list(paths.list_images(args["dataset"]))  
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]  
classNames = [str(x) for x in np.unique(classNames)]  
print("log1"  
  
array_preprocessor = ImageToArrayPreprocessor()  
DL = DataLoader(preprocessors=[array_preprocessor])  
(data, labels) = DL.load(imagePaths, verbose=500)  
data = data.astype("float") / 255.0  
  
(trainX, testX, trainY, testY) = train_test_split(data, labels,  
    test_size=0.25, random_state=42)  
  
trainY = LabelBinarizer().fit_transform(trainY)  
testY = LabelBinarizer().fit_transform(testY)  
baseModel = ResNet50(weights="imagenet", include_top=False,  
    input_tensor=Input(shape=(224, 224, 3)))  
headModel = FCHeadNet.build(baseModel, len(classNames), 256)  
model = Model(inputs=baseModel.input, outputs=headModel)  
for layer in baseModel.layers:  
    layer.trainable = False  
  
print("[INFO] compiling model...")  
opt = RMSprop(lr=0.001)  
model.compile(loss="categorical_crossentropy", optimizer=opt,  
    metrics=["accuracy"])  
  
print("[INFO] training head...")  
model.fit_generator(image_augmentation.flow(trainX, trainY, batch_size=32),  
    validation_data=(testX, testY), epochs=1,  
    steps_per_epoch=len(trainX) // 32, verbose=1)  
  
print("[INFO] evaluating after initialization...")  
predictions = model.predict(testX, batch_size=32)  
print(classification_report(testY.argmax(axis=1),  
    predictions.argmax(axis=1), target_names=classNames))  
  
for layer in baseModel.layers[15:]:  
    layer.trainable = True  
  
print("[INFO] re-compiling model...")  
opt = SGD(lr=0.001)  
model.compile(loss="categorical_crossentropy", optimizer=opt,  
    metrics=["accuracy"])  
  
print("[INFO] fine-tuning model...")  
model.fit_generator(image_augmentation.flow(trainX, trainY, batch_size=32),  
    validation_data=(testX, testY), epochs=1,  
    steps_per_epoch=len(trainX) // 32, verbose=1)  
  
print("[INFO] evaluating after fine-tuning...")  
predictions = model.predict(testX, batch_size=32)  
print(classification_report(testY.argmax(axis=1),  
    predictions.argmax(axis=1), target_names=classNames))  
  
print("[INFO] serializing model...")  
model.save(args["model"])  
  
class DataLoader:  
    def __init__(self, preprocessors=None):  
        self.preprocessors = preprocessors  
  
        if self.preprocessors is None:  
            self.preprocessors = []  
  
      def load(self, imagePaths, verbose=-1):  
          data = []  
          labels = []  
    
          for (i, imagePath) in enumerate(imagePaths):  
              image = cv2.imread(imagePath)  
              label = imagePath.split(os.path.sep)[-2]  
    
              if self.preprocessors is not None:  
                  for p in self.preprocessors:  
                      image = p.preprocess(image)  
    
              data.append(image)  
            labels.append(label)  
  
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:  
                print("[INFO] processed {}/{}".format(i + 1,  
                    len(imagePaths)))  
  
        return (np.array(data), np.array(labels))  
  
class AspectAwarePreprocessor:  
    def __init__(self, width, height, inter=cv2.INTER_AREA):  
        self.width = width  
        self.height = height  
        self.inter = inter  
  
    def preprocess(self, image):  
        (h, w) = image.shape[:2]  
        dW = 0  
        dH = 0  
  
        if w < h:  
            image = imutils.resize(image, width=self.width,  
                inter=self.inter)  
            dH = int((image.shape[0] - self.height) / 2.0)  
        else:  
            image = imutils.resize(image, height=self.height,  
                inter=self.inter)  
            dW = int((image.shape[1] - self.width) / 2.0)  
  
        (h, w) = image.shape[:2]  
        image = image[dH:h - dH, dW:w - dW]  
  
        return cv2.resize(image, (self.width, self.height),  
            interpolation=self.inter)  
  
class ImageToArrayPreprocessor:  
    def __init__(self, dataFormat=None):  
        self.dataFormat = dataFormat  
  
    def preprocess(self, image):  
        return img_to_array(image, data_format=self.dataFormat)  

class FCHeadNet:  
    @staticmethod  
    def build(baseModel, classes, D):  
        headModel = baseModel.output  
        headModel = Flatten(name="flatten")(headModel)  
        headModel = Dense(D, activation="relu")(headModel)  
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")(headModel)
    return headMo
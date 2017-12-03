# import the necessary packages
from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization


class LeNet:
	@staticmethod
	def build(width, height, depth, classes, weightsPath=None):
		# initialize the model
		model = Sequential()

		# first set of CONV => RELU => POOL
		model.add(Convolution2D(20, 5, 5, border_mode="same",
			input_shape=(depth, height, width)))
		model.add(BatchNormalization())
		model.add(Activation("sigmoid"))
		#model.add(Dropout(0.5))
		model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

		# second set of CONV => RELU => POOL
		#model.add(Dropout(0.5))
		model.add(Convolution2D(50, 5, 5, border_mode="same"))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		#model.add(Dropout(0.5))
		model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

		# set of FC => RELU layers
		model.add(Flatten())
		model.add(BatchNormalization())
		#model.add(Dropout(0.5))
		model.add(Dense(500))
		model.add(BatchNormalization())
		model.add(Activation("relu"))

		# softmax classifier
		#model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation("relu"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)

		# return the constructed network architecture
		return model





# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


np.random.seed(7)


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# reshape the MNIST dataset from a flat list of 784-dim vectors, to
# 28 x 28 pixel images, then scale the data to the range [0, 1.0]
# and construct the training and testing splits

data = dataset.data.reshape((dataset.data.shape[0], 28, 28))  # dataset.data.shape=(70000,784)
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=10000)

print trainData.shape
print testData.shape
#print trainData[0,0,]
print trainData.max()
print trainData.min()
#print testData[0,0,]
print testData.max()
print testData.min()

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print trainLabels.shape
print testLabels.shape
print trainLabels[1:5,]

Data_train, Data_validation, Labels_train, Labels_validation = train_test_split(trainData,trainLabels,test_size=10000)


# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(Data_train[0][0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(Data_train[1][0], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(Data_train[2][0], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(Data_train[3][0], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()



# initialize the optimizer and model
print("[INFO] compiling model...")
opt=RMSprop()   #1: SGD(lr=0.01); 2: SGD(lr=0.01, momentum=0.9) ; 3(if RMS): RMSprop()
model = LeNet.build(width=28, height=28, depth=1, classes=10,
    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
	print("[INFO] training...")
	#early_stopping = EarlyStopping(monitor='val_loss', patience=1, mode='min')
	history=model.fit(Data_train, Labels_train, validation_data=(Data_validation,Labels_validation),
			  batch_size=128, nb_epoch=20, verbose=1,shuffle=True)  ########## callbacks=[early_stopping],

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()


	# show the accuracy on the testing set
	print("[INFO] evaluating...")
	(loss, accuracy) = model.evaluate(testData, testLabels,
		batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# check to see if the model should be saved to file
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# randomly select a few testing digits
#for i in np.random.choice(np.arange(0, len(testLabels)), size=(1,)):  #random choose 10 samples from 10000 test samples
	# classify the digit
	#probs = model.predict(testData[np.newaxis, i])  # testData[np.newaxis, i].shape=(1, 10, 1, 28, 28)
	#prediction = probs.argmax(axis=1)

	# resize the image from a 28 x 28 image to a 96 x 96 image so we can better see it
	#image = (testData[i][0] * 255).astype("uint8")  # testData[i][0].shape=(1,28,28)
	#image = cv2.merge([image] * 3)
	#image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	#cv2.putText(image, str(prediction[0]), (5, 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# show the image and prediction
	#print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
		#np.argmax(testLabels[i])))
	#cv2.imshow("Digit", image)
	#cv2.waitKey(0)

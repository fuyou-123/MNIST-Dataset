import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras import regularizers




def build_multilayer_perceptron(dim):
    model = Sequential()

    model.add(Dense(120, input_shape=(dim,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model






np.random.seed(7)

print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")


data = dataset.data     # dataset.data.shape=(70000,784)
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data / 255.0, dataset.target.astype("int"), test_size=10000)

print trainData.shape
print testData.shape


trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print trainLabels.shape
print testLabels.shape


Data_train, Data_validation, Labels_train, Labels_validation = train_test_split(trainData,trainLabels,test_size=10000)


# PCA
mean=np.mean(Data_train, axis = 0)
Data_train -= mean # zero-center the data (important)
cov = np.dot(Data_train.T, Data_train)/ Data_train.shape[0] # get the data covariance matrix   dim(cov)=784*784







eig_vals, eig_vecs = np.linalg.eig(cov)

## check PCA ###

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)



plt.style.use('seaborn-whitegrid')

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(784), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(784), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()
    #plt.tight_layout()






n=512  # dimension reduce from 784 to n
U,S,V = np.linalg.svd(cov) # dim(U)=784*784
Data_train = np.dot(Data_train, U[:,:n])
print Data_train.shape

Data_validation -= mean
Data_validation = np.dot(Data_validation, U[:,:n])
print Data_validation.shape

testData-= mean
testData = np.dot(testData, U[:,:n])

print testData.shape



# plot the image after PCA
image=np.dot(Data_train,(np.transpose(U)[:n,:]))
image = (image * 255).astype("uint8")
image=image.reshape(50000,28,28)

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(image[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(image[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(image[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(image[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()




# initialize the optimizer and model
print("[INFO] compiling model...")
opt=SGD(lr=0.001)   #1: SGD(lr=0.01); 2: SGD(lr=0.01, momentum=0.9) ; 3(if RMS): RMSprop()

model = build_multilayer_perceptron(n)

model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print("[INFO] training...")
#early_stopping = EarlyStopping(monitor='val_loss', patience=1,mode='min')
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




# confusion matrix
probs = model.predict(testData)  # testData[np.newaxis, i].shape=(1, 10, 1, 28, 28)
prediction = probs.argmax(axis=1)
cm = confusion_matrix(testLabels.argmax(axis=1), prediction)
plt.matshow(cm)
plt.title('Confusion matrix for MLP')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import InputLayer, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import os

batch_size = 128 #
num_classes = 10
# epochs = 1 # you can try more epochs
#change the more epochs
epochs = 36

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# different backend saves data in diffferent format
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
else: # for Tensorflow, the format is [N_image, image_row, image_col, channel]
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize the pixels to [0, 1]
X_train /= 255
X_test /= 255

# subset 5000 samples from X_train as validation_set
X_train, X_val = X_train[0:55000,:,:,:], X_train[-5000:,:,:,:]
y_train, y_val = y_train[0:55000], y_train[-5000:]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_val.shape[0], 'validation samples')


# convert class vectors to one-hot class matrices
y_train_vec = y_train
y_test_vec = y_test
y_val_vec = y_val
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# check some of images to see if the data is loaded correctly
def plot_images(images, label, pred=None):
    if pred is None:
        try:
            assert images.shape[0] == len(label) == 9
        except AssertionError:
            images = images[0:9,:,:,:]
            label = label[0:9]
            print('Select the first 9 images')
    else:
        try:
            assert images.shape[0] == len(label) == len(pred) == 9
        except AssertionError:
            images = images[0:9,:,:,:]
            label = label[0:9]
            pred = pred[0:9]
            print('Select the first 9 images')
            
    fig, axes = plt.subplots(3,3)
    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape([img_rows, img_cols]), cmap='binary')
        # Show true and predicted classes
        if pred is None:
            xlabel = "True: %d" % label[i]
        else:
            xlabel = "True: %d, Pred: %d" % (label[i], pred[i])
        ax.set_xlabel(xlabel)
        # Show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])   
    plt.show()
    
plot_images(X_train, y_train_vec)

# plot the incorrectly predicted images 
def plot_example_errors(images, label, pred):
    # Boolean array whether the predicted class is incorrect.
    incorrect = (label != pred)
    # subset the images and labels
    images = images[incorrect,:,:,:]
    label = label[incorrect]
    pred = pred[incorrect]
    plot_images(images, label, pred)
    
        
# define the save path for the trained_model
save_path = './trained_model'
save_file = 'cnn_mnist.keras'
# check if the save_path exists
if not os.path.exists(save_path):
    os.mkdir(save_path)


learning_rate = 1e-3
model = Sequential()
# input layer
model.add(InputLayer(input_shape=(img_rows, img_cols, 1, )))
from keras.layers.normalization import BatchNormalization

model.add(Conv2D(32, kernel_size=3,
                 activation='relu',
                 padding='same',
                 name = 'layer_conv_1'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3,
                 activation='relu',
                 padding='same',
                 name = 'layer_conv_2'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#dropout it
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
#dropout it
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=Adam(lr=learning_rate),
              metrics=['accuracy'])

model.summary()
def scheduler(epoch):
    # 每隔12个epoch，学习率减小为原来的1/10
    if epoch % 12 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val),callbacks=[reduce_lr])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))



#you can change the data_aug according to your need
# datagen = ImageDataGenerator(
#                               rotation_range=0.2,
#                               zoom_range=0.2,
#                               width_shift_range=0.2,
#                               height_shift_range=0.2,
#                               zca_epsilon=1e-6,
#                               horizontal_flip=True,
# )
# model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=X_train.shape[0] // batch_size,
#                     epochs=epochs,
#                     validation_data=(X_test, y_test),workers=4,callbacks=[reduce_lr])


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save the model
model_path = os.path.join(save_path, save_file)
model.save(model_path)
print('Model saved')
# for model reuse in the future, use the following commands
#from keras.models import load_model
#model = load_model(model_path)

# for further investigation, return the label of the test set
y_pred = model.predict_classes(x=X_test)
plot_example_errors(X_test, y_test_vec, y_pred)

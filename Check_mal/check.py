from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class CNN(object):
    def create_model(self):
        self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, (3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    def __init__(self):
        self.model = Sequential()

        #train & test 데이터 생성
        train_datagen = ImageDataGenerator(
            rescale=1/255, #데이터 정규화
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.2,1.0],
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(
            rescale=1/255, #데이터 정규화
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.2,1.0],
            horizontal_flip=True) 

        self.train_generator = train_datagen.flow_from_directory(
            './dataset/train',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='binary')

        self.validation_generator = test_datagen.flow_from_directory(
            './dataset/validation',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='binary')

        #Model 생성
        self.create_model()

        #Model 컴파일
        self.compile_model()

        #model fitting
        self.model.fit(
            self.train_generator,
            steps_per_epoch=200,
            epochs=5,
            validation_data = self.validation_generator,
            validation_steps=80)

        #Model 학습과정 살펴보기
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
        loss_ax.set_ylim([0.0, 0.5])

        acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
        acc_ax.set_ylim([0.8, 1.0])

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.show()

        loss_and_metrics = self.model.evaluate(self.train_generator, self.validation_generator, batch_size = 32)
        print(loss_and_metrics)

        self.model.save('./model.h5')
        


if __name__=='__main__':
    img_size = 256
    model = CNN()

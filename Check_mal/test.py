from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def __init__(self):
    training_set = None
    test_set = None


def create_model(model):
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def main():
    if os.path.isfile("./model.h5"):
        pass
    else:
        #모델 생성
        model = Sequential()
        model = create_model(model)

        hist = model.fit(
            train_generator,
            steps_per_epoch=200,
            epochs=5,
            validation_data = validation_generator,
            validation_steps=80)

        #모델 학습과정 보기
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

    #모델 평가
    loss_and_metrics = model.evaluate(train_generator, validation_generator, batch_size = 32)
    print(loss_and_metrics)

    #모델 저장
    model.save('./model.h5')



if __name__=='__main__':
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

    train_generator = train_datagen.flow_from_directory(
        './dataset/train',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        './dataset/validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='binary')

    start = main()

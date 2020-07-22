from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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


def __init__(self):
    model = Sequential()

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
        'dataset/train',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        './dataset/validation',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='binary')

    #Model 생성
    # create_model()

    # #Model 컴파일
    # compile_model()

    #model fitting
    model.fit(
        train_generator,
        steps_per_epoch=200,
        epochs=5,
        validation_data = validation_generator,
        validation_steps=80)

    loss_and_metrics = model.evaluate(train_generator, validation_generator, batch_size = 32)
    print(loss_and_metrics)

    model.save('./model.h5')
        

def main():
    if os.path.isfile("./model.h5"):
        pass
    else:
        model = Sequential()
        create_model(model)


if __name__=='__main__':
    model = main()

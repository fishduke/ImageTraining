import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import matplotlib.pyplot as plt


class KCNN(object):
        def __init__(self):
            self.classifier = Sequential()
            self.training_set = None
            self.test_set = None

        #모델 구성
        def create_model(self):
            self.classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
            self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
            self.classifier.add(Flatten())
            self.classifier.add(Dense(units = 128, activation = 'relu'))
            self.classifier.add(Dense(units = 1, activation = 'sigmoid'))
            self.classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
        #모델 학습과정 설정
        def fit(self, train_path, test_path, epochs):
            if os.path.isfile('./model.h5'):
                print("학습된 모델이 있습니다. 모델을 로드합니다.")
                self.classifier = load_model('model.h5')
            else:
                print('학습된 모델이 없습니다. 학습을 시작합니다.')
                self.create_model()
                
                train_datagen = ImageDataGenerator(rescale = 1./255,
                                                   shear_range=0.2,
                                                   zoom_range=0.2,
                                                   horizontal_flip=True
                                                   )

                test_datagen = ImageDataGenerator(rescale = 1./255,
                                                  shear_range=0.2,
                                                  zoom_range=0.2,
                                                  horizontal_flip=True
                                                  )
                
                self.training_set = train_datagen.flow_from_directory(train_path,
                                                                 target_size = (128,128),
                                                                 batch_size = 4,
                                                                 class_mode = 'binary')

                self.test_set = test_datagen.flow_from_directory(test_path,
                                                            target_size = (128,128),
                                                            batch_size = 3,
                                                            class_mode = 'binary')

                hist = self.classifier.fit_generator(self.training_set,
                                         steps_per_epoch = 13,
                                         epochs = epochs,
                                         validation_data = self.test_set,
                                         validation_steps = 7)
                
                # matplotlib 생성
                fig, loss_ax = plt.subplots()
                acc_ax = loss_ax.twinx()
                
                loss_ax.plot(hist.history['loss'], 'y', label='train loss')
                loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
                
                acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
                acc_ax.plot(hist.history['val_accuracy'], 'g', label='val accuracy')
                
                loss_ax.set_xlabel('epoch')
                loss_ax.set_ylabel('loss')
                acc_ax.set_ylabel('accuracy')
                
                loss_ax.legend(loc='upper left')
                acc_ax.legend(loc='lower left')
                
                plt.show()

                self.classifier.save('./model.h5')
        
        #모델 평가
        def predict(self,test_path):
            test_datagen = ImageDataGenerator(rescale=1. / 255)
            test_set = test_datagen.flow_from_directory(test_path,
                                                             target_size=(128,128),
                                                             batch_size=3,
                                                             class_mode='binary')

            if self.classifier is not None:
                output = self.classifier.predict_generator(test_set, steps=1)
                return test_set.class_indices, output
            else:
                print('학습모델이 없습니다.')


if __name__=='__main__':
    # 모델을 생성합니다.
    model = KCNN()

    # args1 : 훈련데이터 셋위치(디렉토리형태),
    # args2 : 테스트 데이터위치(디렉토리형태),
    # args3 : 훈련반복횟수
    model.fit('./dataset/training_set', './dataset/test_set', 50)

    label, result = model.predict('./dataset/test')
    for i in range(len(result)):
        if result[i] > 0.8:
            print("등록된 사진은 김건우.", "확률:", result[i]*100)
        else:
            print("등록된 사람과 일치하지 않으므로 재 등록 바람.", "확률:", (1-result[i])*100)
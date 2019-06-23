# -*- coding: utf-8 -*-

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class build_CNN_Model():

    def __init__(self, Config):

        self.Config = Config

    def build_model(self):
        '''
        建置產生CNN模型實體。
        '''

        # 輸入層維度from keras.models import load_model
        input_shape = (
            self.Config.Train_DataSet.shape[1],
            self.Config.Train_DataSet.shape[2],
            self.Config.channel
        )

        model = Sequential()

        # 輸入層
        model.add(
            Conv2D(
                self.Config.CNN_inputlayer_conv2D_hidden_unit,
                kernel_size=self.Config.CNN_inputlayer_conv2D_kernel_size,
                padding=self.Config.CNN_inputlayer_conv2D_padding,
                input_shape=input_shape,
                activation=self.Config.CNN_inputlayer_Activation
            )
        )

        # 池化層
        model.add(
            MaxPooling2D(
                pool_size=self.Config.CNN_MaxPooling2D_pool_size
            )
        )

        # 遮罩
        model.add(
            Dropout(
                self.Config.CNN_Dropout
            )
        )

        # 平坦層
        model.add(
            Flatten()
        )

        # 全連接層
        model.add(
            Dense(
                self.Config.CNN_full_connectionlayer_Dense,
                activation=self.Config.CNN_full_connectionlayer_Activation
            )
        )

        model.add(
            Dropout(
                self.Config.CNN_full_connectionlayer_Dropout
            )
        )

        # 輸出層
        model.add(
            Dense(
                self.Config.class_num,
                activation=self.Config.CNN_ouputlayer_Activation
            )
        )

        model.compile(
            loss=self.Config.CNN_loss,
            optimizer=self.Config.CNN_optimizer,
            metrics=['accuracy']
        )

        # 輸出模型架構總體資訊
        model.summary()

        return model

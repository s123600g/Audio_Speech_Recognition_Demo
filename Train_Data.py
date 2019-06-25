# -*- coding:utf-8 -*-

'''
MIT License

Copyright (c) 2019 李俊諭 JYUN-YU LI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

from tensorflow.python.client import device_lib
from keras import backend as K
from keras.models import Model
from Config import Config
from Gen_DataSet import Gen_DataSet
from Model import build_CNN_Model

import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras
import time
import numpy as np
import history_plot

if __name__ == "__main__":

    try:

        Start_Time = time.time()

        print(device_lib.list_local_devices(), end="\n\n")

        ''' 產生訓練、測試、驗證資料集 '''
        Gen_DataSet = Gen_DataSet(Config)
        Gen_DataSet.DataSet_Process()

        build_CNN_Model = build_CNN_Model(Config)

        ''' 建置模型架構實體 '''
        net_model = build_CNN_Model.build_model()

        ''' 分配每一回合訓練資料量 '''
        steps_per_epoch = int((len(Config.Train_Labels) / Config.batch_size))

        print("steps_per_epoch：{}".format(
            int((len(Config.Train_Labels) / Config.batch_size))
        ))

        ''' 訓練模型 '''
        history = net_model.fit(
            Config.Train_DataSet,
            Config.Train_Labels,
            # 設置每一回合訓練資料量
            batch_size=steps_per_epoch,
            # 設置訓練幾回合
            epochs=Config.epochs,
            # 是否觀察訓練過程，設值為1代表要顯示觀察
            verbose=Config.verbose,
            # 設置驗證資料集
            validation_data=(Config.Valid_DataSet, Config.Valid_Labels)
        )

        ''' 驗證訓練後模型 '''
        score = net_model.evaluate(
            Config.Test_DataSet,
            Config.Test_Labels,
            verbose=Config.verbose
        )

        print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Keras CNN - accuracy: {:.2f}".format(score[1]))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        ''' 建立訓練過程之準確度與損失函數變化圖片 '''
        history_plot.plot_figure(
            history,
            os.path.join(os.getcwd(), Config.Plot_Figure_DirectoryName),
            "Audio_Speech_Training"
        )

        ''' 儲存訓練後模型和權重 '''
        net_model.save_weights(Config.Model_Weight_Path)
        net_model.save(Config.Model_Path)

        end_time = '{:.2f}'.format((time.time() - Start_Time))

        print("\nSpeed time: {}s".format(end_time))

    except Exception as err:

        print("\n>>> {} <<<".format(err))

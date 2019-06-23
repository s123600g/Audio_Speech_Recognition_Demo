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


Exporting a tf.keras File
https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_

How to convert keras(h5) file to a tflite file?
https://stackoverflow.com/questions/53256877/how-to-convert-kerash5-file-to-a-tflite-file

需注意！！！
'lite.TFLiteConverter.from_keras_model_file()' 要求tensorflow版本最低為 1.12
針對 1.9-1.11 版本 使用 'lite.TocoConverter'
針對 1.7-1.8 版本 使用 'lite.toco_convert'
'''

from tensorflow.contrib import lite
from Config import Config

import argparse
import os
import time

''' 設置CLI執行程式時，參數項目配置 '''
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str,
                    help="Set a model sources path for input.", default="None")
parser.add_argument("--output", type=str,
                    help="Set a save path for output tflite model", default="None")

args = parser.parse_args()

input_model_path = ""
output_model_path = ""


def gen_Path_DetailInfo():

    global input_model_path, output_model_path

    if args.input == "None":

        print("[Model_to_TFLite] No set argument ['{}'] value, automated use default {} value.".format(
            "--input",
            "input"
        ))

        input_model_path = os.path.join(
            Config.Input_Model_Path, Config.Model_Name
        )

    else:

        input_model_path = args.input

    if args.output == "None":

        print("[Model_to_TFLite] No set argument ['{}'] value, automated use default {} value.".format(
            "--output",
            "output"
        ))

        output_model_path = os.path.join(
            Config.Output_Model_Path
        )

    else:

        output_model_path = args.output

    return input_model_path, output_model_path


if __name__ == "__main__":

    try:

        Start_Time = time.time()

        ''' 產生輸入與輸出模型之路徑配置 '''
        gen_Path_DetailInfo()

        print("[Model_to_TFLite] Input path：[ {} ]".format(input_model_path))
        print("[Model_to_TFLite] Output path：[ {} ]".format(output_model_path))

        print()

        ''' 檢查模型來源是否不存在 '''
        if not os.path.exists(input_model_path):

            raise FileNotFoundError(
                "The ['{}'] can't found input model file.".format(
                    input_model_path
                )
            )

        ''' 檢查放置轉換後模型位置是否不存在 '''
        if not os.path.exists(output_model_path):

            raise FileNotFoundError(
                "The ['{}'] can't found output path.".format(
                    output_model_path
                )
            )

        ''' 讀取來源模型，進行轉換模型為TFLite格式模型 '''
        converter = lite.TFLiteConverter.from_keras_model_file(
            input_model_path
        )
        tflite_model = converter.convert()

        ''' 輸出轉換TFLite格式模型 '''
        with open(output_model_path, "wb") as TFM_file:

            TFM_file.write(tflite_model)

        print("\nSpeed time: {:.2f}s".format((time.time() - Start_Time)))

    except FileNotFoundError as err:

        print("\n>>> {} <<<".format(err))

    except Exception as err:

        print("\n>>> {} <<<".format(err))

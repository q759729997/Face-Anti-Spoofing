# Face-Anti-Spoofing

- A simple face anti-spoofing based on CNN trained with HSV + YCrCb color feature.
- HSV即色相(Hue)，饱和度(Saturation)，明亮度（Lightness），又称为HSB
- YCbCr有的时候会被写作：YCBCR，是色彩空间的一种，通常会用于影片中的影像连续处理，或是数字摄影系统中。Y为颜色的亮度（luma）成分、而CB和CR则为蓝色和红色的浓度偏移量成分。

- ![Network Architecture](https://github.com/Oreobird/Face-Anti-Spoofing/raw/master/model.png) 

## Dependencies

~~~shell
dlib >= 19.17.0
tensorflow >= 1.12.0
keras >= 2.2.4
numpy >= 1.11.1
scipy >= 0.14
opencv-python >= 3.4.3
~~~

## Usage

~~~wiki
* Data prepare:<br>
 1) Download train data and model file from:<br>
 [https://pan.baidu.com/s/1izOKKs8-CRy6Ykfa13r0Ew](https://pan.baidu.com/s/1izOKKs8-CRy6Ykfa13r0Ew)  code: u99k<br>
 2) untar data and model to your project dir
* Train model:<br>
 python data/NUAA/gen_label.py
 python main.py --train=True
* Test online via camera:<br>
 python main.py --online=True
~~~

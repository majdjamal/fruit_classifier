# Real-Time Image Classification
 

This is a project in the Deep Learning course [DD2424](https://www.kth.se/student/kurser/kurs/DD2424?l=en) at the Royal Institute of Technology. 

## Abstract


Deep networks are growing deeper and require more energy resources for training. Previous research has shown that the deep learning field will stop progressing if contributors do not produce lighter and efficient network architecture. This study utilizes MobileNet, which is the state-of-the-art network for mobile devices, to classify images of fruits and vegetables, and compares its performances to networks with more extensive architectures. Four networks were trained from randomly initialized weights and Transfer Learning using ImageNet weights. MobileNet trained with Transfer Learning produced a top-1 accuracy of 97.5% and performed like the more extensive network architectures. The larger networks required longer training sessions to converge. This study uses the top-1 MobileNet model to conclude a real-time image classification application that classifies fruits and vegetables and can be used by any computer that has a webcam.

## Test The Program

* Navigate to the repository

* Install Required Utility Packages
```bash
pip3 install -r requirements.txt
```

### Test the Real-Time Image Classification App

* Run,

```bash
python3 main.py --realtime 
```
----
### Make a prediction

* Run,

```bash
python3 main.py --predict --img_path 'MyPath/ToThe/Image'
```
-----
> **_NOTE:_** Replace MyPath/ToThe/Image with the path to your image to classify. 

### Train a network
This part requires an NVIDIA GPU.
#### With randomly initialized weight

* Run,

```bash
python3 main.py --train --model "mobilenet" 
```
#### With Transfer Learning

* Run,

```bash
python3 main.py --train --model "mobilenet" --transfer_learning
```
> **_NOTE:_**  --model takes the following arguments: ['mobilenet', 'efficientnetb0', 'efficientnetb5', 'resnet']

----

### Evaluate a network

* Run,

```bash
python3 main.py --evaluate --model "mobilenet" --path 'MyPath/To/Weights'
```
> **_NOTE:_**  If the model was trained with Transfer Learning, pass the --transfer_learning tag.

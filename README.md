# Real-Time Image Classification
 

This is a project in the Deep Learning course [DD2424](https://www.kth.se/student/kurser/kurs/DD2424?l=en) at the Royal Institute of Technology. 

## Abstract

Deep networks are growing deeper and require more energy resources for training. Previous research has shown that the deep learning field will stop progressing if contributors do not produce lighter and efficient network architectures. This study utilizes MobileNet, the state-of-the-art network for mobile devices, to classify images of fruits and vegetables, and compares its performances to networks with more extensive architectures. Networks were trained from randomly initialized weights and Transfer Learning using ImageNet weights. MobileNet trained with Transfer Learning produced a top-1 accuracy of 96.8% and performed like the more extensive network architectures. This study uses the top-1 MobileNet model to conclude a production-grade real-time image classification app of fruits and vegetables.

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
> **_NOTE:_**  Training supplements: scheduler, pass --scheduler; fine tune a pre-trained model, pass --fine_tune 

----

### Evaluate a network

* Run,

```bash
python3 main.py --evaluate --model "mobilenet" --path 'MyPath/To/Weights'
```
> **_NOTE:_**  If the model was trained with Transfer Learning, pass the --transfer_learning tag.

----

### Graphical User Interface
<img src="https://i.ibb.co/McKnL80/GUI.png" width="300" height="400">
(Figure 1. Graphical User Interface. The first window shows captured photos live from the user’s webcam. The second window includes prediction bars. The demonstration shows 99% Green Apple. At the bottom, there is a purple button to close the app.)

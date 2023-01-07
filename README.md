# Title: Detection of emotions from facial features using Image processing and Deep learning

### Objectives:
Process training images into pixel data and label them.
Create deep learning model to detect emotions.

#### Language used:
Python

#### Libraries used:
PIL, TensorFlow, Keras, Pandas, NumPy, Matplotlib

#### Dataset:
FER image dataset from Kaggle


### Figures and discussions

The very first of the preprocessing tasks was image convertion to pixel data. This was accomplished by using PIL library functions. Each array of pixel data was labelled according to the emotions they represent. It was found that the sample size varied greatly with emotions.

![](https://github.com/rud-ninja/emotion_detection/blob/main/images/sample_size.png)

**Fig 1: Sample size of each emotion from the training dataset**

Due to the difference in representation from each category of emotions, RandomOverSampler was used to sample the minority categories with replacement.

A little bit more preprocessing was performed, such as converting the data into applicable datatypes and shapes. Next, the training dataset was split into train and test in the ratio 4:1 and the neural network was built.

The Neural network comprises of multiple 2D Convolutional layers with 16, 32 and 64 neurons, Batch normalisation, Activation and Max Pooling layers. After flattening the layers, a dropout layer with rate 0.7 has been used. The total size of the training dataset was 40404. 


### Emotion detection

![](https://github.com/rud-ninja/emotion_detection/blob/main/images/angry_predictions.png)
![](https://github.com/rud-ninja/emotion_detection/blob/main/images/disgusted_predictions.png)
![](https://github.com/rud-ninja/emotion_detection/blob/main/images/fearful_predictions.png)
![](https://github.com/rud-ninja/emotion_detection/blob/main/images/happy_predictions.png)
![](https://github.com/rud-ninja/emotion_detection/blob/main/images/neutral_predictions.png)
![](https://github.com/rud-ninja/emotion_detection/blob/main/images/sad_predictions.png)
![](https://github.com/rud-ninja/emotion_detection/blob/main/images/surprised_predictions.png)


### Conclusion
The model does quite well with an accuracy of over 60% for the well represented classes in the dataset, such as happiness and neutral. Random over sampling does not seem to help misclassification of the less represented ones, such as disgust and fear, with the given neural network parameters. More complex CNN models can be built involving more number of layers which will consequently increase the computational and hardware requirements as well.

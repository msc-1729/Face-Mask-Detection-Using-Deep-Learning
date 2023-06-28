<h2 align = "center">Face-Mask-Detection-Using-Deep-Learning</h2>
This project aims to detect whether a person is wearing a face mask in an image using Deep Learning Techniques. Transfer Learning is used for prediction using MobileNet, Alex Net, and Yolov5 with different variations. A custom CNN is also built with 128 layers and a comparison of all the performances is made to choose the best-fitting model for the 

### Why wearing Masks and What is the need for detecting them in images?
A significant health issue that has been present for the past two years around the globe is the
rapid spread of the COVID-19 coronavirus illness. Direct human contact is one of the key factors
contributing to the virus's rapid transmission. Wearing face masks in public areas is one of
the numerous precautions that may be taken to stop the spread of this infection. To lessen the chance
of the virus spreading, it is important to find ways to detect face masks in public locations.
An automated system for face mask detection utilizing deep learning (DL) algorithms can be
Used to address these issues and effectively stop the spread of this contagious disease. For
For example, This Model can be deployed on edge devices like CCTV cameras to alert the Security
personnel if any person is not wearing a mask. This is very helpful in containing the spread of
the virus, as according to the World Health Organization, wearing masks can significantly reduce the
chance of COVID-19 Hotspots.

### Description of the acquired Dataset
The dataset that has been chosen for this project is provided in Face Mask Detection Dataset
from Kaggle. The dataset was scrapped and formed from various sources such as Github repositories, and by web scraping the Google Search Engine using Python scripts. There are a total of 7553 RGB photos in 2 folders—one with a mask and the other without a mask in the data set. Labels with masks and without masks are used to identify images. Out of these 3828 images of the faces are without masks and the other 3725 images are with face masks.

Number of Training Instances = 5287 </br>
Number of Validation Instances = 454 </br>
Number of Test Instances = 1812 </br>
Number of Labels= 2 (with_mask/without_mask) </br>
Source = [Face Mask Detection Dataset from Kaggle](https://www.kaggle.com/datasets/21faa9e463f87c2500de415965f97074cc83502d0f10766fb62a2e1c2bc6b512)

### Working Models

We have used and tested several different neural network variants. From the basic CNN models
to the more sophisticated MobileNet. As anticipated, the more complicated Mobile Net operated
more effectively and performed better than CNN, AlexNet, and YOLO. To train MobileNetv2,
we used a transfer learning technique. The model's accuracy increased because it was able to
perform pattern recognition with accuracy due to its enormous size. successfully categorizing
the photos as a result. Transfer learning enabled us to train this sizable model. Due to transfer
learning, we just needed to train the model's head and not the entire model as we were going to
freeze the weights from the original model. This made it possible for us to test out different
iterations of the MobileNet without needing a lot of computing power. However, we discovered
that having two thick layers and two dropout layers was the optimal number of layers. For
MobileNet, we show unexpectedly excellent transfer learning accuracy.
A convolutional neural network of 53 layers deep is called MobileNetV2. A pre-trained version
of the network that has been trained on more than a million photos is present in the ImageNet
database. A number of animals, a keyboard, a mouse, and a pencil are among the 1000 different
object categories that the pre-trained network can classify images into. As a result, the network
now has comprehensive feature representations for a range of photos. Images having a resolution
of 224 by 224 are supported by the network. The MobileNetv2 design makes use of a brand-new
convolutional block called Inverted Residuals with Linear Bottlenecks. In this block, the features
from a depiction in a lower dimension are magnified. After using a Depthwise convolution, the
features are subsequently compressed back into the original lower dimensional representation.
MobileNetV2 requires three color channels and 224 224 pixel images. As a result, it prepares for
the input of shape (224, 224, 3).
The predictions are once again returned as a two-dimensional array, with the first axis
corresponding to the observations (images, in this case). The second axis represents the model's
output neurons.
In the implementation of Face mask detection, another highly potential model that is working
highly accurately is the AlexNet model. A CNN model has been developed from scratch that is
identical to the AlexNet Architecture but has few modifications and other layers added into the
network making it more sophisticated and also suitable to the dataset chosen which has resulted
in higher accuracies. The developed CNN model has the following structure. Eight layers
with weights make up the Convolutional Neural Network design; the first five layers are
convolutional, while the latter three layers are fully linked. The first convolutional layer is an
input layer that applies 96 11x11x3 kernels with a 4-pixel stride to 224x224x3 input pictures.
Only the kernel maps of the preceding layer, which are located on the same GPU, are connected
to the kernels of the second, fourth, and fifth convolutional layers. The neurons in the
fully-connected layers are connected to all the neurons in the preceding layer, and the kernels of
the third convolutional layer are connected to all the kernel maps of the second convolutional
layer. So, here is how the AlexNet CNN architecture is set up altogether. The network achieved
an overall accuracy of 95.45%. But, the only visible and vital disadvantage that can be identified
is the running time that the model takes to learn is very high as that compared to the running
time of the MobileNet model implemented. The main reason for this can be that the AlexNet is
implemented from scratch and it has many hidden layers that may contribute to not only higher
time consumption but also very higher space consumption as well. This model can be proven to
be more efficient given higher computing power such as GPU, which can allow us to use more
epochs for running and obtain better results, and this model can be saved and used for further
applications using Transfer learning.

### Failure of the developed CNN:

We have also worked on face mask detection using CNN using the same dataset as provided
in the project proposal. While working on it finding the optimal parameter values for the
convolutional Neural network in order to identify the existence of the face mask accurately
without generating overfitting led to poor accuracy with a maximum accuracy percentage of 88.29.
In order to improve the accuracy, we tried changing the number of epochs, changing the
optimizers, changing the activation function, reducing the image input size, and increasing the batch size we also did the analysis and tested by varying the filter size. We have also tried adding more
layers to the model but that did not improve any accuracy.
In the Final analysis, we analyzed anyone of the below reasons that might have caused the
underperformance of the CNN model they are
1. Primary reason is because of underfitting which might be because of high bias.
2. The main disadvantage of the CNN is it did not encode the position and orientation of the
mask and as the CNN in general requires large data set to train optimally, the size of the dataset
might also be causing the CNN to underperform.
The above reasons might constitute the reasons behind CNN not performing to par with the other
models for face mask detection.

### Experiment and results
We have implemented multiple neural networks and compared the results and used the best-performing network. We have implemented more complex Convolutional neural networks by
adding more convolutional layers and also utilized MobileNet, YOLOv5s, and AlexNet with
the help of transfer learning. We expect to get a better understanding of the performance of the
aforementioned models on the dataset used and finally give a conclusion as to which model
performs the best based on several measures such as an F1-score, accuracy, Precision, recall, and
visual representations such as Confusion and Covariance matrices. This experiment can also be
used to give a proper understanding and insights on whether transfer learning works more
efficiently or whether a dedicated neural network that has been created from scratch for a particular dataset provides more accurate results. Finally, the project obtains better accuracy by using complex CNN and transfer learning than that of a basic CNN model.

Validation accuracies of all the different variants of the MobileNet CNN:

<img src= "https://github.com/msc-1729/Face-Mask-Detection-Using-Deep-Learning/blob/main/assets/Results%20of%20MobileNet.png" />

Training Accuracy of MobileNet CNN:

<img src="https://github.com/msc-1729/Face-Mask-Detection-Using-Deep-Learning/blob/main/assets/Mobile%20Training%20vs%20Validation.png"/>














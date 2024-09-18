# auto_colorization_of_gray_scale_images
This project focuses on automatic colorization of grayscale images using a Convolutional Neural Network (CNN) model. The model has been implemented using Python, Keras, TensorFlow, and NumPy, and achieves an accuracy of 65.5%.
## Project Overview
The aim of this project is to develop a deep learning model capable of converting grayscale images to color. The model is built using Convolutional Neural Networks (CNN) and utilizes multiple layers to extract meaningful features from the images to predict their color values.
## Technologies Used
* __Python__: Main programming language for data manipulation and model implementation.
* __Keras__: Used for building the deep learning model.
* __TensorFlow__: Backend for Keras and used for model training.
* __NumPy__: Utilized for numerical computations and array manipulations.
* __Matplotlib__: For visualizing model results and outputs.
## Model Architecture
The CNN-based model consists of multiple convolutional layers followed by upsampling layers. Here’s a summary of the model:
* Model: "sequential_5"
* Total params: 6,252,002
* Trainable params: 6,252,002
* Non-trainable params: 0
## Training and Results
The model was trained over 500 epochs using the Mean Squared Error(MSE) loss function and Adam Optimizer. The final results achieved are:
* __Accuracy__: 65.5%
* __Loss__: As observed during the training process
### Loss and Accuracy Trends
* The model shows improvement over epochs, with validation loss and training loss values converging.
## Conclusion
This project demonstrates how deep learning techniques, specifically Convolutional Neural Networks (CNN), can be applied to auto-colorize grayscale images.

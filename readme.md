# Landmark Classification and Tagging for Social Media

## Introduction
Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.  

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgment to classify these landmarks would not be feasible.  

This project is a step towards addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image. It goes through the machine learning design process end-to-end: performing data preprocessing, designing and training CNNs, comparing the accuracy of different CNNs, and deploying an app based on the best CNN trained.  

## Steps to run
1) Install the required packages for the project:  
`pip install -r requirements.txt`  
2) Test that the GPU is working (execute this only if you have a NVIDIA GPU on your machine, which Nvidia drivers properly installed)  
`python -c "import torch;print(torch.cuda.is_available())`  

## Methodology
The high level steps of the project include:

1) **Create a CNN to Classify Landmarks (from Scratch)** - Here, we'll visualize the dataset, process it for training, and then build a convolutional neural network from scratch to classify the landmarks. We also describe some decisions around data processing and how we chose the network architecture. we will then export the best network using Torch Script.  
2) **Create a CNN to Classify Landmarks (using Transfer Learning)** - Next, we'll investigate different pre-trained models and decide on one to use for this classification task. Along with training and testing this transfer-learned network, we'll explain how we arrived at the pre-trained network we chose. We will also export wer best transfer learning solution using Torch Script.  
3) **Deploy the algorithm in an app** - Finally, we will use our best model to create a simple app for others to be able to use the model to find the most likely landmarks depicted in an image. We'll also test out the model and reflect on its strengths and weaknesses.  
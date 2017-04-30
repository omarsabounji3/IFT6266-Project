## Methods : 

In the deep learning literature, there are many models able to complete task needed for this project. Among them, there are autoencoders, L2 regularization Network, GAN and its multiple version (Generative Adversial Network). 



(Pathak and al., Context Encoders: Feature Learning by Inpainting, 2016)


## Project description

  The objectives of this blog is to document the methodology and the results of the class project. This project is part of the deep learning course IFT 6266. 

  The purpose of the project is to to generate the middle region of images conditioned on the outside border of the image and a caption describing the image. Here is an ilustration of the concept.

![imagecaptionex](https://cloud.githubusercontent.com/assets/18235074/25565825/f42d7fec-2d9c-11e7-9361-89a9ff9b3dae.png)

  All the images used during this experimentation are taken from the MSCOCO dataset. This dataset contains high resolution images, roughly 500×500 pixels. For the actual project, all images are downsampled to 64×64 and the goal is to complete the middle 32×32 section.

  Many models will be tried to check what are the ones that gives the best results. The evaluation of the performance of those methods are essentially visual because it is quiet hard to evaluate quantitatively their abilities.
  
  The deep learning packages Theano and Lasagne will be used during this experimentation. 

## Project objectives

The objectives of this blog is to document the methodology and the results of the class project. This project is part of the deep learning course IFT 6266. 

The purpose of the project is to to generate the middle region of images conditioned on the outside border of the image and a caption describing the image. Here is an ilustration of the concept.

![imagecaptionex](https://cloud.githubusercontent.com/assets/18235074/25565825/f42d7fec-2d9c-11e7-9361-89a9ff9b3dae.png)

All the images used during this experimentation are taken from the MScoco dataset. This dataset contains high resolution images, roughly 500×500 pixels. For the actual project, all images are downsampled to 64×64 and the goal is to complete the middle 32×32 section.

Many models will be tried to check what are the ones that gives the best results.

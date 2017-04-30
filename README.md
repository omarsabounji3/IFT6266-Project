## The L2 regularization network

I tried partially using the L2 reconstruction network (Adversial loss) with some hyperparameters and it hasn't a significant impact on the result given by the this network. The results are still very blurry and the improvement isn't valuable

## Captions impact

The main difficulty of using captions is to integrate efficiently this advantage to the model. Obviously, we need to keep important words to improve the accuracy of the model. So we need to preprocess the words in the captions by extracting only meaningful words that can help in the prediction. The second issue is the way of integrating the useful information in the model. The transformation of all the meaningful information in a shape integrable for the model is to be done quantitatively. One way to do this is to use a matrix with binary factor corresponding to the presence of the word. This matrix can be integrated in one of the several layers of the CNN. 

From the other blog posts, we can see that the captions hasn't significant impact on the improvement of the model. The addition of the captions hasn't changed the blurry aspect of the CNN. 

## CNN autoencoders

CNN methods are reputed efficient for this kind of task. So, I chose to implement this method with basically many filters as described below. From an input, which is the image border, many operations where drawn to improve the learning capacity of the model. The loss function used was mean squared error, trained over 100 epochs with early stopping. Stochastic gradient descent was used with Adam method(RMSProp + momentum) and hyper-parameters of 0.01 for the learning rate. The non-linearities introduced in each layer were RELU. There wasn't any hyper-parameter search was performed. The upsampling used was used to replicate each value in the image as a 2×2 patch.All convolution layers use batch normalization and used L2 loss regularization.

<img width="287" alt="capture d ecran 2017-04-30 a 15 02 50" src="https://cloud.githubusercontent.com/assets/18235074/25567190/61749d24-2db6-11e7-8f63-bc5ab73f1fa7.png">

The results can be considered as correct results. Colors and lines are fairly respected based on the pixels of the background. A few shapes are still respected based on the neighbooring. Nevertheless, the images are blurred and not smooth. EXAMPLE. So, how can we explain this issues ? The different filter applied in the model could be responsible, especially the ones that eliminate or extend information such as pooling or upsampling. The repetition of those operations are may be excessive. Otherwise, we can explain this performance by a lack of filters to explain those patterns or the use of inapropriate loss function (mean squared error).Also, from the course we learned that Autoencoders blurry image generation problem can be explained by the fact that  optimizing p(x|z) puts heavy emphasis on the exact recovery of spatial information which is lost in the latent variables. We tried to overcome this problem using dense layer with large size but this doesn't have a huge impact. 

In this project, we have another advantage, it is the presence of captions that explains the image content. This can be used to help the model improve its accuracy. 

## Methods : 

In the deep learning literature, there are many models able to complete task needed for this project. Among them, there are CNNs(Convolutional neural network)and its multiple version, L2 reconstruction Network, GAN(Generative Adversial Network) and its multiple version. 

Convolutional neural network has shown their abilities on some datasets like the MNIST dataset. The most complex task when using those kind of networks is to choose the different filter and their sequences but also the loss function. Here are some clues to choose the best configuration (Pathak and al., Context Encoders: Feature Learning by Inpainting, 2016). L2 reconstruction network which is a sort of mix between GAN and VAE, has the ability to choose to captures long-range dependencies through use of a computational block based on weight-shared dilated convolutions, and improves generalization performance with orthogonal regularization (https://arxiv.org/pdf/1609.07093.pdf). Finally, introducted by Ian Goodfellow and al. (https://arxiv.org/pdf/1406.2661.pdf), propose a framework for estimating generative models via an adversarial process, in which they simultaneously train two models: a generative model that captures the data distribution, and a discriminative model that estimates the probability that a sample came from the training data rather than G.

## Project description

  The objectives of this blog is to document the methodology and the results of the class project. This project is part of the deep learning course IFT 6266. 

  The purpose of the project is to to generate the middle region of images conditioned on the outside border of the image and a caption describing the image. Here is an ilustration of the concept.

![imagecaptionex](https://cloud.githubusercontent.com/assets/18235074/25565825/f42d7fec-2d9c-11e7-9361-89a9ff9b3dae.png)

  All the images used during this experimentation are taken from the MSCOCO dataset. This dataset contains high resolution images, roughly 500×500 pixels. For the actual project, all images are downsampled to 64×64 and the goal is to complete the middle 32×32 section.

  Many models will be tried to check what are the ones that gives the best results. The evaluation of the performance of those methods are essentially visual because it is quiet hard to evaluate quantitatively their abilities.
  
  The deep learning packages Theano and Lasagne will be used during this experimentation. 

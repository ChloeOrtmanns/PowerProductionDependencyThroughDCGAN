# PowerProductionDependencyThroughStyleGAN
## Introduction
Repository of the Bachelor Thesis surrounding power production and their dependency through StyleGANs. Messy GitHub repos are unfortunatelly one of my strengths so hereby some further explanation.

# How to run
## Pre-processing
in the *Processing/* folder, you can find the **processing.py** file to convert the netCDF data into the power curves. By changing the function called in main, you can also choose to plot the histograms of the raw rsds data and raw windspeeds to see whether it variates over time.

## Config and Parameters
Most parameters that need a lot of tweaking can be found in the **config.py** file.

## Trainingloop
The trainingloop can be found in **Main.py**. It implements an early stop and you can decide whether you start anew or continue from a previously saved stage. The used dataset to train is split into an estimated 10-90 division, where a pseudorandom testbatch includes 9 random days of each season of each year (10%) and the remaining values are used for the trainbatch (90%).

Everything is also being logged to wandb which you should configure yourself.

## Plots and Evaluation
In the *Evaluation/* folder, you can find many different mathematical metrics to evaluate whether the StyleGAN model was doing good in general. The **mainEvaluation.py** file can run and will automatically log to wandb if configured right.

In the *StandalonePlots/* you can find a few seperate plotting files.
- **lowerTailResearch.py** plots the histograms and scatterplots of the chosen test batch, gan generated data and copula data. Though this code is mainly based on the PCA copula, I assume with a bit of tweaking you can use it for other copula generated data.
- **PixelCopulaGANKDE.py** plots the KDE of a single pixel in the test data, GAN generated data and t-copula/copula fitted data on that pixel. You can comment out which copula you want, but as the parameters of freedom are high enough in this case, the t-copula automatically reverts to a copula anyway.
- **spatialMapCorrCopulaGan.py** goes past every X pixels on the map, averages the data (test, GAN, t-copula) over that pixels and plots the spatial correlation for each. In the last two plots, it calculates the difference between that corelation of real-GAN and real-Copula.

# Copula generated data
## Coarsed Copula
The only way to approach my resolution is by lowering the resolution and performing copula functions on a smaller spatial grid instead of all the pixels spatially. These files can be found in the *CoarsedCopula/* folder.

- **coarsedCopulaGenerator.py** generates these data points. 
- **coarsedCopulaMap.py** maps them in the lower resolution.

## PCA Copula
Another way to retain the spatial dependency while performing a copula on my data is by implementing PCA. These files can be found in the *PCAcopulas/* folder.

- **PCAcopulaGenerator.py** generates the PCA components and the data.
- **inspectPCA.py** maps what these components are so you can understand what they actually mean
- **PCAcopulaKDE** maps the real test batch data, GAN generated data and PCA-Copula generated data and plots them in a KDE plot. It shows you three different images, A is a KDE in the PCA latent space, B is just pixel-histogram per regio, C is a KDE in the native space of the values.
- **PCAcopulaMeanMapPlot.py** plots the averaged real test data, GAN generated data and PCA-Copula generated data, and also plots the difference between real-GAN and real-Copula.

# Resources
Bunch of resources, data, checkpoints, ... that I saved but not everything because some were too big for GitHub so they might be irrelevant to you.

# Other branches
For the attentive readers, you will notice there are three other branches left.
- **Main**: the first branch that has primitive versions of the processing files
- **Discriminator**: the branch with my first attempt at a DCGAN
- **Wasserstein**: the branch with the WGAN-GP attempt

These are still here for reference, but they are not rightfully up to date. They also do not matter anymore since I changed my direction to StyleGAN anyway

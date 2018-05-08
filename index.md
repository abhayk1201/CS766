## Image Classification - Without Deep Nets
 Colin Butkus
 
 May 7, 2018

## Summary Statement

Image classification is the task of taking an image and producing a label that classifies the image. For example given a picture of dog we would like the image classification algorithm to produce the label ‘dog’. This is important problem in computer vision becuase it aids in image searching and retreval. Since 2011 image classification has been dominated by deep nets. The improvment in model precision was dramatic. For instance, before the deepnet era, state of the art image classification algorithms were achieving 70-80 percent accuracy on predicitions on well know test sets. On these same test sets, deep nets were achiveing precision figures above 90 percent.

With that said I choose to go back in time to learn and implement the models before the deep net revolution. I choose to do this for 3 reasons:

1. I wanted to learn something new. I previously worked with deep nets.
2. I wanted to run code on my own laptop. Deep nets require way too much computing time.
3. I wanted to learn the models that were state of the art pre-deep net era.

My goal was two fold. One, learn about and then implement the models that were commonly used before deep nets. Two, try a new technique to see if I can improve the precision of the older models.

## Dataset to Benchmark Model Performance
 
 For this project I made use of publicly available data set [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) of labeled images. The dataset contains 10 classes (shown in image below), 50,000 images for training (32x32 color images), 10,000 images for testing (32x32 color images).

![Image](https://cbutkus.github.com/CS766/CIFAR-10.png)
 
## State of the Art
 
 Deep Convolutional Neural Network has achieved over 95% accuracy on the CIFR-10 dataset. In comparison humans achieve classification accuracy of 95%. The reasons humans don't achieve 100% accuracy is becuase of the low resolution of the images in the data set. For pre-Deep Net models accuracy on this dataset was in the 80% range. The image below is high level schmatic of deep nets on image classification.
 
 ![Image](https://cbutkus.github.com/CS766/DeepNet.png)

## Image Classification Framework

The process works as follows. Collect a large dataset of images with labels attached to them. In this case we are using the CIFAR-10 dataset. You extract features from your dataset. That is for each image create a D-dimensional vector of numbers that "describes" the image. You then break up your data into a training set and test set. The trainning set is used to tune your model. The test set is only used at the end to measure how well the model you trainned actually predicts the correct label on a image that was never observed before by the model. The two parts to think about here are the algorithms used for Feature Creation and the algorithms used for Classifing. The algorithm that classifys the label of the images recieves as input, the output from the Feature Creation algoritm.

![Image](https://cbutkus.github.com/CS766/ClassificationOverview.png)

## The Classifiers

In this project I used the following classifaction algorithms:

1. Nearest Neighbor
2. Support Vector Machine (SVM)

### Nearest Neighbor

The nearest neighbor classifier is a very simple algorithm. At a highlevel, it works as follows. You provide the trainning set of D-dimensional feature vectors with labels. Then you query the algorithm by asking it to classify some D-dimensional feature vector. The algorithm finds the "closest" neighbor to this query. It then returns the majority class of K neighbors, where K is an input parameter. For instance if K = 1 you only return the closest neighbor. If K= 5 then those 5 neighbors get to vote with their label based on some weighting scheme of their votes. The image below depicts the nearest neighbor algorithm.

![Image](https://cbutkus.github.com/CS766/NN.png)

### Support Vector Machine (SVM)

SVM are also simple to understand. Like the nearest neighbor algorithm, the SVM algoritm receives a trainning set of D-dimensional feature vectors with labels. The algorithm then finds a set of hyperplanes, (those are lines when the feature fector is 2-dimensional) that maximizes the margin around labels of one class vs the other class. The image below depicts what the SVM is doing.

![Image](https://cbutkus.github.com/CS766/SVM.png)

## The Feature Extractors

In this project I used the following feature extractor algorithms:

1. Reduced Size Image
2. Scale-Invarient Feature Transform (SIFT)
3. Histogram of Oriented Gradients (HOG)

### Reduced Size Image

Couldn't be simplier. You take the image of 32x32 size and shrink it down. For my testing purposes I reduced the size to 16x16. Then created a feature vector by stacking each column into an array of size 256. Then I ran this through the two classifiers described above.

![Image](https://cbutkus.github.com/CS766/SmallImage.png)

### SIFT

The SIFT algorithm is bit more involved and little harder to explain at a high level. But here goes, the algoritm is looking for "interesting points" in the image. The algorithm first creates an image space of various blurring and shrinken. For each level of blur the algortim finds local extrema. These are the interesting points. It then sets a neighborhood around the keypoints to calculate the orientation and magnitude of each point in that neighbor and assigns it to bins of orientations, 10%, 20% etc. Finally, the algorithm produces feature fectors by taking a 16x16 neighborhood around the keypoint and then breaking it into 16 sub-blocks of 4x4 size. For each sub-block, 8 bin orientation histogram is created. This produces a total of 128 binds of orientation for each interesting point. Our feature vector is 128 dimensions. For more details check out D. Lowe's paper: Distinctive Image Features from Scale-Invariant Keypoints, 2004". The paper is well written and easy to follow.

![Image](https://cbutkus.github.com/CS766/SIFT.png)

### HOG

The HOG feature extractor works as follows. Break up image into cells. For each cell compute the orientations at each pixel. This is typically done at each pixel by calculating the gradient in the x direction by [-1 0 1] and in the y direction by [-1 0 1]’. With the gradient calculated you can calculate the orientation. Within each cell accumulate each bin, angle of orientation, by the magnitude gradient for each pixel. Typically 9 bins per cell: 0-20degrees, 20-40degrees, …, 160-180degrees. Combine histogram entries into one feature vector

![Image](https://cbutkus.github.com/CS766/HOG.png)

## Feature Enchancers



## Baseline Results

The results of the above described model is shown below. The accuracy column is showing % of the time the model is correctly predicting the label from the test set.

![Image](https://cbutkus.github.com/CS766/BaselineResults.png)

## Trying Something Different

After going through the standard algortims that existed back in 2010 I wanted to try and see if can improve upon the accuracy achieved on my test set. I was planning to do this by creating a novel feature vector. The vector vectors used so far are all local features and don't account for spatial realationship across the image. My thought was track a rough geometry of the image by using Gaussian Mixture Model (GMM). See picture below for example. If we can also encode a set of ellipses through the image with some of the above methods the classifier would better be able to disbriminate the objects in the picture.

![Image](https://cbutkus.github.com/CS766/DogAndCircles.png)

The idea is to use Gaussian Mixture Model (GMM) on each picture to obtain circle/ellipsoid shapes that capture a rough geometry of the image. Fitting a GMM to each image you can obtain the center location of the each gaussian along with its relative center location to other gaussian variables as well as its shape based on the covariance matrix. This is encoded into a feature vector by placing the mean vector of each Gaussian followed by the upper triangular covariance matrix of each Gaussian. The order of stacking was chosen by largest to smallest area covered by each Gaussian. To calculate the area covered I took the variance of x-coordinate multiplied by the variance of the y-coordinate. The Gaussian Mixture Model was fit for each image. The results presented are with 20 Gaussian variables estimated with data of size n x 3. Where n is the number of pixels in the image and 3 is the {x-coordinate, y-coordinate, pixel intensity} normalized.

![Image](https://cbutkus.github.com/CS766/shapeOfGaussian.png)

You can use the [editor on GitHub](https://github.com/cbutkus/CS766/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](https://cbutkus.github.com/CS766/CIFAR-10.png)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/cbutkus/CS766/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

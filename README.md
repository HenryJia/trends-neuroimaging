# Code for the Kaggle TReNDs Neuroimaging Challenge

Ultimately, during this competition, the best model I've found is support vector regression using only the tabular data without the images. This yielded a final leaderbaord score of approximately 0.16, placing me in the top 69% of the competition.

The code for this is relatively short and in svm\_loadings.py

I was unable to use the image datas as it seems that neural networks seem to overfit this data very easily. I'm not sufficiently familiar with biology to manually feature engineer using the image data.

I did try a number of other models. You can find the code for this inside the folder old/

These models include autoencoder based methods, neural networks, Gaussian processes, and random forests

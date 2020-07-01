# Code for the Kaggle TReNDs Neuroimaging Challenge

Ultimately, during this competition, the best model I've found is support vector regression using only the tabular data without the images. This yielded a final leaderbaord score of approximately 0.16.

The code for this is relatively short and in svm.py The submission generated is in svm_submission.csv

I was unable to use the image datas as it seems that neural networks seem to overfit this data very easily. I'm not sufficiently familiar with biology to manually feature engineer using the image data. I couldn't really think of another way to use the image data as it was too high dimensional for something like a forest of SVM. Any neural network I threw at the image data seemed to either overfit, or behave poorly with strong regularisation.

You can find the code for these neural networks and other models I tried inside the folder old/

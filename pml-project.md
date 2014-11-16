---
title: "PML-Project"
author: "BD"
date: "Wednesday, November 12, 2014"
output: html_document
---
**PROBLEM DESCRIPTION**  

Our objective is to use the activity monitor data collected during bicep curls in order to determine automatically if the excercise was performed true to form or if mistakes were made (excercise quality). The training data consists of accelerometer, gyroscope and magnetometer data mounted on forearm, arm, hip and dumbbell of several volunteer subjects. (The data consists of nearly 160 activity data parameters from these sensors and includes more than 19000 records). They perform one repitition correctly (as judged by a human expert) and 4 other repetitions improperly. The goal is to train a machine learning algorithm to identify the correctly and incorrectly performed reps of bicep curls. 

**MODELING STRATEGY**

The two main decisions to make are  
1) Which attributes (columns) from the training data to use for the model building?   
2) and do we need to use all 19000+ records to build a model?

To answer the first question, we can use principal component analysis (PCA), but the data set contains numerous missing values which may skew PCA results. Traditional correlation analysis is another option, but there are 153 numerical attributes, which result in far too many options to consider. 
To overcome these challenges, I decided to exclude all attributes which have values missing in the test sample dataset (20 samples, which need to be predicted). This brought down the feature set to 53.



**DATA/SAMPLES SELECTION**  

Now that we have chosen a feature selection strategy, we now need to decide if we need to use all samples to build the model. We have two options: use all 19k records to build a single model or as there are six subjects we can build a model which addresses each subject individually. The advantage of the second option is that it is computationally less expensive (although we need to run the process 6 separate times, which is not that bad). Furthermore, if this technology were to be implemented in practice, it would make sense to train an algorithm to be predictive for a given user rather than a general algorithm that works well for many users, but is not as accurate as a customized one. I decided to build 6 different models, one for each user. 


```r
#First read the entire training dataset
allData <- read.csv("pml-training.csv")
#Next read the 20 sample test dataset
testsamples <- read.csv("pml-testing1.csv")
#Remove the first 7 columns as they are not going to be used for machine learning
tsamp <- testsamples[,-(1:7)]
#Remove all columns that have only NAs
tsamp <- tsamp[,colSums(is.na(tsamp))<nrow(tsamp)]
#Add classe as the first column of the testing sample
classe <- vector(mode="character", length=nrow(tsamp))
tsamp <- cbind(classe, tsamp)
#Only choose attributes which have non-missing values in the testing sample
featSel <- colnames(tsamp)
#Raw dataset dimensions
nrow(allData)
```

```
## [1] 19622
```

```r
ncol(allData)
```

```
## [1] 160
```

```r
#Segment the training dataset by user or subject
subj <- "jeremy"
#select all rows for the subject
per_subj <- subset(allData, allData$user_name==subj)
#select all features which match our feature list identified earlier
per_subj <- subset(per_subj, select=featSel)
#Check the subject dataset dimensions
nrow(per_subj)
```

```
## [1] 3402
```

```r
ncol(per_subj)
```

```
## [1] 53
```

```r
#Check if the right columns have been selected from the testsample dataset and the subject dataset
colnames(tsamp)==colnames(per_subj)
```

```
##  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [15] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [29] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
## [43] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
```

```r
# As an added benefit, our feature selection strategy resulted in a dataset with no NA's,
# thus requiring no imputation etc.
```

**ERROR ESTIMATION AND CROSS VALIDATION**  

Because our model building strategy focuses on building a separate model for each subject, we expect our out of sample error to be very low, as long as we can identify if the unseen sample comes from the same subject. As it turns out, our 20 test samples each have the subject correctly identified and thus we expect our OOB error to be quite low. This was verified with the 10-fold cross validation process that is followed in the next few steps below.


```r
##Split into training and validation samples
inTrain <- createDataPartition(y=per_subj$classe, p=0.7, list=FALSE)
training <- per_subj[inTrain,]
validation <- per_subj[-inTrain,]
#Make sure the data is correctly split
nrow(training)
```

```
## [1] 2384
```

```r
ncol(training)
```

```
## [1] 53
```

```r
nrow(validation)
```

```
## [1] 1018
```

```r
ncol(validation)
```

```
## [1] 53
```

```r
#fit a model using RandomForest using all selected variables and
#with a 10-fold cross validation to estimate accuracy. For each
#subject, with approx 3000+ records, randomForest takes about 4-5
#min to completion (16GB RAM, 2.7GHz processor)
set.seed(123)
tc <- trainControl(method="cv", verboseIter=FALSE)
mod <- train(classe ~ ., method="rf", data=training, prox=TRUE, trControl=tc)
#Output the model
mod
```

```
## Random Forest 
## 
## 2384 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 2145, 2147, 2145, 2145, 2145, 2146, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1         1      0.008        0.010   
##   27    1         1      0.007        0.009   
##   52    1         1      0.009        0.012   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
#Output the predictions for the validation samples
preds <- predict(mod,validation)
confusionMatrix(validation$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 353   0   0   0   0
##          B   0 142   4   0   0
##          C   0   0 194   1   0
##          D   0   0   0 156   0
##          E   0   0   1   2 165
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.985, 0.997)
##     No Information Rate : 0.347         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.99          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    0.975    0.981    1.000
## Specificity             1.000    0.995    0.999    1.000    0.996
## Pos Pred Value          1.000    0.973    0.995    1.000    0.982
## Neg Pred Value          1.000    1.000    0.994    0.997    1.000
## Prevalence              0.347    0.139    0.195    0.156    0.162
## Detection Rate          0.347    0.139    0.191    0.153    0.162
## Detection Prevalence    0.347    0.143    0.192    0.153    0.165
## Balanced Accuracy       1.000    0.998    0.987    0.991    0.998
```
**ANALYZING RESULTS**  

As seen above, for the subject chosen, we get an **in-sample** accuracy of around 99%. The performance is very similar for the other subjects, ranging from 98% to 99.9% in accuracy. We can now apply this family of models to the unseen or out-of-sample test set.

**PREDICTING UNSEEN or OUT-OF-BAG (OOB) SAMPLES**  

We can now predict all the OOB samples using the model built for one subject, keeping in mind that the prediction will be most accurate only for the OOB sample of *that* subject. This has been verified: all the 20 OOB samples have been predicted accurately by using the corresponding models for each subject. Note that the function provided (for generating text files of each of the sample predictions) has been modified to append subject name to each file.


```r
pml_write_files = function(x,user){
  n = length(x)
  for(i in 1:n){
    filename = paste0(user,"problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

predSubj <- predict(mod, tsamp)
pml_write_files(predSubj,subj)
```

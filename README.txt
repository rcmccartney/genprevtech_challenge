Usage: "python project2.py <training datafile> <test datafile>"

You only need to run this command to execute all 4 programs, in the order of trainMLP.py executeMLP.py train_dt.py execute_dt.py.
For the dataset given for this project you don't have to set any parameters as we have already set them according to it. 
Our code runs for any classification dataset having its last row as the class numbers.  The only changes required to parameters are
the following variables: 

trainPlots = Set these according to the number of classes in training dataset
testPlots = Set these according to the number of classes in testing dataset
topology = Set the topology of the neural network (input layer is equal to the number of features of the dataset and output layer is
           equal to the number of classes.  For 2 classes, use a single output layer.)

If you are running a new dataset for the first time then you need to uncomment find_best_settings(sys.argv[1]) to get the best settings
for decision tree and set them accordingly that are: 

bestK = best number of splits
bestDepth = best depth you should go to
bestIt = best number of iterations 

Interpreting the output:

On running the code it produces all the plots of the Training and Testing datasets for MLP and Decision Tree which have been labeled 
and discussed in the write up. 
The code also gives the Number of classification errors, Recognition rate, Profit obtained and the Confusion matrix for the Training 
and Testing dataset after each iteration.
"Bad row, skipping line" is printed if there is any row in the data that does not contain proper data (eg: 2 blank lines in the given 
Test dataset)

Output:

Training Set
Number of classification errors: 3
Recognition rate: 95.95%
Profit obtained: $ 6.68
Confusion matrix:
     1    2    3    4 
1   14    0    0    1 
2    0   22    0    0 
3    0    0   22    0 
4    0    1    1   13
Test Set
Bad row, skipping line
Bad row, skipping line
Number of classification errors: 0
Recognition rate: 100.00%
Profit obtained: $ 2.03
Confusion matrix:
     1    2    3    4 
1    5    0    0    0 
2    0    6    0    0 
3    0    0    5    0 
4    0    0    0    4

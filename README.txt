This project is in reference to the top coder challenge located here: 
http://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=15761&pm=12634
put on by NASA's Center of Excellence for Collaborative Innovation and Harvard Business 
School in association with the Institute of Quantitative Social Sciences.  For more information
and the winners of the contest, see this website on the Tech Challenge for Atrocity Prevention:
http://thetechchallenge.org/.

It was related to the search by USAID (http://www.usaid.gov/) and Humanity United (http://www.humanityunited.org/)
for a computational model to detect and prevent atrocities.  

This project uses the GDELT dataset of world events (http://gdeltproject.org/), the geographical region data located
here http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/, and 
atrocity data from http://crisis.net/.

The aim was to build a mass atrocity predictor that could warn of impending atrocities, based off
of news reports and other features built from a particular region.  The winners of the challenge
are located here, along with all of the data used:
https://github.com/nasa/CoECI-USAID-Atrocity-Prevention-Model

The winner won the contest with a score of 213.66 on the provisional and 376.79 on the system tests,
with those tests and the scoring function described in the topcoder link above.  

My own implementation uses a random forest developed in Python.  I resued a lot of the features
that the winner of the contest used, but where he (magically) assigned weights to get rid of
certain features and promote other features I allowed the random forest to perform feeature selection.
It did this by first randomly bagging the data, then randomly choosing a certain number of 
attributes and split points for those attributes, and then for all of those possible split points
used entropy calculations to select the best feature.  This became the split point of that
node in the tree, with many different trees built on each iteration and new forests built every 
pre-defined number of days (my scores rebuilt forests every 5 days).  

Here are my results on the system test, which significantly surpassed the winner of the challenge:
#############################
Final score 1310.033721111152
Confusion matrix with threshold 0.2
        0         1
 0   4556952      6288
 1     10077      4420
#############################
From the confusion matrix (with 1 meaning an atrocity did in fact occur sometime in the following 30 days from
the date of the prediction) you can see that we correctly identified over 4,000 atrocities ahead of when they
occurred.  Further tests are currently being performed to increase the accuracy on this sparse class problem
even further (to push more of the false negatives into positive classification).

Note that the program outputs files for every day of testing with an estimate of the likelihood of an atrocity
in every region of the globe within the next 30 days.  

Usage: python3 main.py </data_path/> </output_directory/> <start_training_day> <start_testing> <end_testing>


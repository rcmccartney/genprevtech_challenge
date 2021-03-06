NOTE: Code has been removed pending the publishing of results in a paper, at 
which point it will be re-uploaded.

Usage: python3 main.py </data_path/> </output_directory/> <start_training_day> <start_testing> <end_testing>

This project is in reference to the top coder challenge located here: 
http://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=15761&pm=12634
put on by NASA's Center of Excellence for Collaborative Innovation and Harvard Business 
School in association with the Institute of Quantitative Social Sciences.  For more 
information and the winners of the contest, see this website on the Tech Challenge 
for Atrocity Prevention:
http://thetechchallenge.org/.

It was related to the search by USAID (http://www.usaid.gov/) and Humanity United 
(http://www.humanityunited.org/) for a computational model to detect and prevent 
atrocities.  

This project uses the GDELT dataset of world events (http://gdeltproject.org/), the 
geographical region data located here 
http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/ 
and atrocity data from http://crisis.net/.

The aim was to build a mass atrocity predictor that could warn of impending 
atrocities, based off of news reports and other features built from a particular 
region. The winners of the challenge are located here, along with all of the data used:
https://github.com/nasa/CoECI-USAID-Atrocity-Prevention-Model

The winner won the contest with a score of 213.66 on the provisional and 376.79 
on the system tests, with those tests and the scoring function described in the 
topcoder link above.  

Here are my results on the system test, which surpassed the winner of the challenge:
#############################
Final score 1310.033721111152
Confusion matrix with threshold 0.2
        0         1
 0   4556952      6288
 1     10077      4420
#############################

Using a lower threshold, the following results were achieved:
#############################
Final score 1316.3656965067314
Confusion matrix with threshold 0.1
        0         1 
 0   4546627     16613 
 1      7793      6704 
#############################

From the first confusion matrix (with 1 meaning an atrocity did in fact occur sometime 
in the following 30 days from the date of the prediction) you can see that we correctly 
identified over 4,000 atrocities ahead of when they occurred, but had a good number
of false negatives (~10,000). This is due to the difficulty of classifying unbalanced 
classes.  While the second run did in fact result in fewer false negatives, it greatly 
increased the number of false positives. Further tests are currently being performed to 
increase the accuracy on this sparse class problem even further.

This same program works for any input timeframes, not just the ones in the topcoder 
challenge.  For instance, over the time period of 14800 start training, 15000 start 
testing, and 16000 end testing we have the following results:
#############################
Final score 1112.008565434813
Confusion matrix with threshold 0.2
        0         1 
 0   3657777      6885 
 1      5666      4343 
#############################

Note that the program outputs files for every day of testing with an estimate of the 
likelihood of an atrocity in every region of the globe within the next 30 days. My 
results can be found in results/System and results/Provisional.

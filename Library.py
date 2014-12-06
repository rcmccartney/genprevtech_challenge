__author__ = 'mccar_000'

from rand_forest.forest import Forest
from rand_forest.AtrocityEntropyFn import *

FIRST_DAY = 11284
START_DT = 15000 - FIRST_DAY
INTERVAL = 1
LIB_DAYS = 180
DEPTH = 3
# use bagging on the samples when making the tree?
BAG = True
#random number of splits to use
K = 100
# Attr is the number of random attributes to use for the K split
# -1 would mean test every attribute
ATTR = 50


class Library():

    def __init__(self, regions, countries, periods):
        self.forests = []
        self.region_count = regions
        self.country_count = countries
        self.world_index = countries
        self.periods = periods
        self.position = 0
        # train_data[region][day][feature]
        # train on data from 30 days back and make predictions from data today to be learning 30 days future
        # this is the dataset, one row for every region/day in current sliding window
        self.train_data = [[] for _ in range(LIB_DAYS*regions)]
        # classf[region][day] is 0/1, the class of the train data
        self.classf = [0 for _ in range(LIB_DAYS*regions)]
        # predict[region][feature] is used to make the attrcoity prediction for the next 30 days
        self.predict_data = [[] for _ in range(regions)]

    def create_data(self, buffer, day):
        for reg in range(self.region_count):
            country_id = buffer.region_to_country[reg]  # country to which this region belongs
            # >0 if there has been an atrocity in the last 30 days
            # so this becomes a class 1 piece of data
            if buffer.region_atro_decision[reg] > 0:
                self.classf[self.position] = 1

            reg_to_cnt_ratio = [0.0, 0.0]  # [0] is trailing, [1] is current
            atroc_ratio = [0.0, 0.0]
            #this country has never seen an atrocity if == 0
            if buffer.trailing30_country_atroc[country_id][len(self.periods)-1] != 0:
                reg_to_cnt_ratio[0] = buffer.trailing30_region_atroc[reg][len(self.periods)-1] / \
                                        buffer.trailing30_country_atroc[country_id][len(self.periods)-1]
            if buffer.curr_country_atroc[country_id][len(self.periods)-1] != 0:
                reg_to_cnt_ratio[1] = buffer.curr_region_atroc[reg][len(self.periods)-1] / \
                                        buffer.curr_country_atroc[country_id][len(self.periods)-1]
            # weighted average of the country average of atrocities plus (region_atr/country_atr)
            atroc_ratio[0] = reg_to_cnt_ratio[0]*0.7 + 0.3/len(buffer.country_to_region[country_id])
            atroc_ratio[1] = reg_to_cnt_ratio[1]*0.7 + 0.3/len(buffer.country_to_region[country_id])

            #trailing30 contains counts of atrocities for every period, 14 in all
            self.train_data[self.position] = buffer.trailing30_region_atroc[reg]
            # each region gets a share of the countries total atrocities by period as a feature, 14 in all
            self.train_data[self.position]. \
                extend([atroc_ratio[0]*x for x in buffer.trailing30_country_atroc[country_id]])
            # add in the 9 additional features
            self.train_data[self.position] += [reg, country_id]
            self.train_data[self.position] += len(buffer.country_to_region[country_id])
            self.train_data[self.position] += reg_to_cnt_ratio[0]
            self.train_data[self.position] += buffer.trailing30_country_atroc[self.world_index][5]  # 35 day trail
            self.train_data[self.position] += buffer.last_atrocity(buffer.reg_atroc_dates[reg], day, tr=30)
            self.train_data[self.position] += buffer.last_atrocity(buffer.cnt_atroc_dates[country_id], day, tr=30)
            self.train_data[self.position].extend(buffer.region_geo[reg])  # two features
            # now add all the news features, which are aggregated over EVENT_AGGR_TIME
            self.train_data[self.position].extend(self.world_avg(buffer.trailing30_region_news[reg],
                                                                 buffer.trailing30_country_news[self.world_index]))
            #Create the current data same way as above
            self.predict_data[reg] = buffer.curr_region_atroc[reg]
            self.predict_data[reg].extend([atroc_ratio[1]*x for x in buffer.curr_country_atroc[country_id]])
            self.predict_data[reg] += [reg, country_id]
            self.predict_data[reg] += len(buffer.country_to_region[country_id])
            self.predict_data[reg] += reg_to_cnt_ratio[1]
            self.predict_data[reg] += buffer.curr_country_atroc[self.world_index][5]  # 35 day trail
            self.predict_data[reg] += buffer.last_atrocity(buffer.reg_atroc_dates[reg], day)
            self.predict_data[reg] += buffer.last_atrocity(buffer.cnt_atroc_dates[country_id], day)
            self.predict_data[reg].extend(buffer.region_geo[reg])
            self.predict_data[reg].extend(self.world_avg(buffer.recent_region_news[reg],
                                                         buffer.recent_country_news[self.world_index]))

        # wrap the buffer around to save space, only keep LIB_DAYS of data
        self.position += 1
        if self.position >= LIB_DAYS*self.region_count:
            self.position = 0

        # build dt only every INTERVAL days after 15,000 dayID
        if day >= START_DT and day % INTERVAL == 0:
            self.add_forest(day)

    # each feature divided by world average
    def world_avg(self, soc, world):
        temp = []
        for index in range(len(soc)):
            if world[index] != 0:
                temp.append(soc[index] / (world[index]/self.region_count))
            else:
                temp.append(1.0)
        return temp

    def add_forest(self, day):
        forest = Forest(DEPTH, weak_learner=AtrocityEntropyFn(K, ATTR), bagging=BAG)
        forest.set_train_delete(self.train_data, self.classf, 2)
        self.forests[day] = forest
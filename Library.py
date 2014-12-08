__author__ = 'Robert McCartney'

from rand_forest.forest import Forest
from rand_forest.AtrocityEntropyFn import *


class Library():

    def __init__(self, regions, countries, periods, lib_days, start_dt, interval, depth, k, attr, bag, bag_rat, trees):
        """
        The Library class is responsible for creating the data that actual goes into the classifier
        :param regions: int number of regions
        :param countries: int number of countries
        :param periods: the number of periods to aggregate data over
        :param lib_days: int the number of days of data to keep in the library
        :param start_dt: int first day to build a forest
        :param interval: int days between forests
        :param depth: int maximum depth each tree in a forest can be
        :param k: int number of split points to use
        :param attr: number of random attributes to use at a node of the tree
        :param bag: boolean to bag data or not
        :param bag_rat: percentage of data to use in bagging
        :param trees: int number of trees to use in a forest
        :return: None
        """
        self.forests = []
        self.lib_days = lib_days
        self.region_count = regions
        self.country_count = countries
        self.world_index = countries
        self.periods = periods
        self.start_dt = start_dt
        self.interval = interval
        self.depth = depth
        self.k = k
        self.attr = attr
        self.bag = bag
        self.bag_rat = bag_rat
        self.trees = trees
        self.position = 0
        # train on data from 30 days back and make predictions from data today to be learning 30 days future
        # this is the dataset, one row for every region/day in current sliding window: train_data[region/day][feature]
        self.train_data = [[] for _ in range(lib_days*regions)]
        # classf[region][day] is 0/1, the class of the train data
        self.classf = [0 for _ in range(lib_days*regions)]
        # predict[region][feature] is used to make the atrocity prediction for the next 30 days
        self.predict_data = [[] for _ in range(regions)]
        self.RECENT_DT = start_dt

    def create_data(self, buffer, day):
        """
        This function takes the data stored in the DataBuffer and turns it into
        a 2-D list of data that can be used for classification, with a classification
        class of 1 for an atrocity being committed within the last 30 days.  Note
        that training occurs on data that is 30 days trailing, with a class of 1
        if in the 30 days since there has been an atrocity.  Meanwhile, testing
        occurs on data from today to predict 30 days into the future
        :param buffer:
        :param day:
        :return:
        """
        # clear buffers for the next <region_count> of rows before adding the data in
        for i in range(self.region_count):
            self.classf[self.position+i] = 0
            self.train_data[self.position+i] = []
            self.predict_data[i] = []

        for reg in range(self.region_count):
            # country to which this region belongs, need to index at 0 since it inserted into map as a list
            country_id = buffer.region_to_country[reg][0]
            # >0 if there has been an atrocity in the last 30 days, so this becomes a class 1 piece of data
            # finding this class 1 of atrocities in last 30 days is what we want to predict
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
            atroc_ratio[0] = reg_to_cnt_ratio[0]*0.7 + 0.3/len(buffer.country_to_regions[country_id])
            atroc_ratio[1] = reg_to_cnt_ratio[1]*0.7 + 0.3/len(buffer.country_to_regions[country_id])

            #trailing30 contains counts of atrocities for every period, 14 in all
            self.train_data[self.position] = buffer.trailing30_region_atroc[reg]
            # each region gets a share of the countries total atrocities by period as a feature, 14 in all
            self.train_data[self.position]. \
                extend([atroc_ratio[0]*x for x in buffer.trailing30_country_atroc[country_id]])
            # add in the 9 additional features
            self.train_data[self.position].extend([reg, country_id])
            self.train_data[self.position].extend([len(buffer.country_to_regions[country_id])])
            self.train_data[self.position].extend([reg_to_cnt_ratio[0]])
            self.train_data[self.position].extend([buffer.trailing30_country_atroc[self.world_index][5]])  # [5]=35 days
            self.train_data[self.position].extend([self.last_atrocity(buffer.reg_atroc_dates, reg, day, tr=30)])
            self.train_data[self.position].extend([self.last_atrocity(buffer.cnt_atroc_dates, country_id, day, tr=30)])
            self.train_data[self.position].extend(buffer.region_geo[reg])  # two features
            # now add all the news features, which are aggregated over EVENT_AGGR_TIME
            self.train_data[self.position].extend(self.world_avg(buffer.trailing30_region_news[reg],
                                                                 buffer.trailing30_country_news[self.world_index]))
            #Create the current data same way as above
            self.predict_data[reg] = buffer.curr_region_atroc[reg]
            self.predict_data[reg].extend([atroc_ratio[1]*x for x in buffer.curr_country_atroc[country_id]])
            self.predict_data[reg].extend([reg, country_id])
            self.predict_data[reg].extend([len(buffer.country_to_regions[country_id])])
            self.predict_data[reg].extend([reg_to_cnt_ratio[1]])
            self.predict_data[reg].extend([buffer.curr_country_atroc[self.world_index][5]])  # [5] is 35 days trailing
            self.predict_data[reg].extend([self.last_atrocity(buffer.reg_atroc_dates, reg, day)])
            self.predict_data[reg].extend([self.last_atrocity(buffer.cnt_atroc_dates, country_id, day)])
            self.predict_data[reg].extend(buffer.region_geo[reg])
            self.predict_data[reg].extend(self.world_avg(buffer.recent_region_news[reg],
                                                         buffer.recent_country_news[self.world_index]))

            # wrap the buffer around to save space, only keep LIB_DAYS of data
            self.position += 1
            if self.position >= self.lib_days*self.region_count:
                self.position = 0

        # build dt only every INTERVAL days after 15,000 dayID
        if day >= self.start_dt and day % self.interval == 0:
            self.RECENT_DT = day		
            self.add_forest(day)

    # each feature divided by world average
    def world_avg(self, soc, world):
        """
        This function averages the values a country has for news data over the world total,
        normalized by number of regions in the world.  Thus, it can be greater or less than 1
        :param soc: the region feature vector
        :param world: the world total feature vector
        :return: the region features divided by the world total features to num of regions ratio.
        """
        temp = []
        for index in range(len(soc)):
            if world[index] != 0:
                temp.append(soc[index] / (world[index]/self.region_count))
            else:
                temp.append(1.0)
        return temp

    def add_forest(self, day):
        """
        Add a forest to the model
        :param day: day to add
        :return: None
        """
        forest = Forest(depthlimit=self.depth, weak_learner=AtrocityEntropyFn(self.k, self.attr),
                        bagging=self.bag, bag_ratio=self.bag_rat, default_tree_count=self.trees)
        print("In add_forest")
        forest.set_train_delete(self.train_data, self.classf, 2)
        self.forests.append(forest)

    #
    @staticmethod
    def last_atrocity(buffer, key, day, tr=0):
        """
        Returns the days since last atrocity occurred
        If provided, the day last atrocity occurred must be before the trailing factor
        First tests if key exists in the buffer, if not uses the last day of the period
        (which represents all time).
        :param buffer: buffer of atrocity days that occurred in given region,
                such that buf[key] = [day1, day2,...]
        :param day: the atrocity must occur before this day
        :param tr: trailing time that the atrocity must occur before
        :return: days ago that it occurred
        """
        if key in buffer:
            day -= tr
            # index points to last day an atrocity occurred in this location
            index = len(buffer[key]) - 1
            # buffer[key][index] is an atrocity date, want to find first <= day
            while index >= 0 and buffer[key][index] > day:
                index -= 1
            if index >= 0:
                return day - buffer[key][index]
        return 10000  # used for infinite into the past
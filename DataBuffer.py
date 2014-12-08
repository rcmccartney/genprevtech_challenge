from Library import Library

###################Don't change these, set by the data########################
FIRST_DAY = 11284  # start of data, normalize this to zero for entire program
LAST_DAY = 17644
REGIONS = 3671  # number of regions in dataset
COUNTRIES = 254  # number of countries in dataset
WORLD_INDEX = COUNTRIES
FEATURES = 33
##############################################################################
###################Can change these to affect outcome#########################
 # sliding windows to aggregate atrocity counts over
PERIODS = [3, 7, 14, 21, 28, 35, 42, 91, 182, 365, 730, 1460, 2920, 10000]
EVENT_AGGR_TIME = 90  # the window to aggregate news events over, can change this
START = 15000 - FIRST_DAY
INTERVAL = 30  # how often we make a new forest
LIB_DAYS = 180  # how many days of data we use for training a forest
BAG = False  # use bagging on the samples when making the tree?
BAG_RAT = .4  # how much of the data is bagged to make a forest
TREES = 3  # how many treees in a forest
DEPTH = 3  # depth of each tree in the forest
K = 30  # random number of splits to use
# Attr is the number of random attributes to use for the K split
# -1 would mean test every attribute
ATTR = 18
config_v = [220260, 233, 0.0009, 0.01]
##############################################################################
"""
The data used in this contest starts at the day 11284. There is 1 example, 1 provisional, and 1 system test case.
Each test case has a learning period and a testing period:

Test case	start-end day IDs of the learning period	start-end day IDs of the testing period
=======================================================================================================
Example		11284-15301 (44,904,851 lines*)			15302-15636 ( 6,345,731 lines)
Provisional	11284-15666 (51,250,582 lines)			15667-16367 (25,574,119 lines)
System		11284-16397 (76,824,701 lines)			16398-17644 (92,738,537 lines)
"""


class DataBuffer():
    def __init__(self, library):
        """
        Initialize the buffer that reads the data files and stores aggregate counts of various metrics
        :param library: creates the dataset to be used in training/testing the forest
        :return: None
        """
        # different time periods to make features from
        self.forest_library = library
        self.buffsize = EVENT_AGGR_TIME + 31  # store enough to look back that far with 30 days trailing
        self.cnt_atroc_dates = {}  # stores days of atrocities for this country
        self.reg_atroc_dates = {}  # stores days of atrocities for this region
        # stores a count of atrocities for all [countries][days] and [regions][days]
        self.all_cnt_atro = [[0 for _ in range(LAST_DAY - FIRST_DAY + 31)]
                             for _ in range(COUNTRIES)]
        self.all_region_atro = [[0 for _ in range(LAST_DAY - FIRST_DAY + 31)]
                                for _ in range(REGIONS)]
        #this counts the last month of atrocities in region, gives us the class decision
        self.region_atro_decision = [0 for _ in range(REGIONS)]
        # map for both the current time and 30 days prior for the counts of atrocities during different windows
        # formatted as [region][period] or [country][period]
        self.curr_region_atroc = [[0 for _ in range(len(PERIODS))] for _ in range(REGIONS)]
        self.trailing30_region_atroc = [[0 for _ in range(len(PERIODS))] for _ in range(REGIONS)]
        self.curr_country_atroc = [[0 for _ in range(len(PERIODS))] for _ in range(COUNTRIES+1)]
        self.trailing30_country_atroc = [[0 for _ in range(len(PERIODS))] for _ in range(COUNTRIES+1)]
        # here are the news feature buffers
        # region_news_buffer[region][event_flag_no][day_id] => true count for region news events
        self.region_news_buf = [[[0 for _ in range(self.buffsize)] for _ in range(FEATURES)] for _ in range(REGIONS)]
        # country_news_buffer[country][flag_no][day_id] => true count for country news events
        # last position is for WORLD aggregation
        self.cntry_news_buf = [[[0 for _ in range(self.buffsize)] for _ in range(FEATURES)] for _ in range(COUNTRIES+1)]
        self.recent_region_news = [[0 for _ in range(FEATURES)] for _ in range(REGIONS)]
        self.trailing30_region_news = [[0 for _ in range(FEATURES)] for _ in range(REGIONS)]
        self.recent_country_news = [[0 for _ in range(FEATURES)] for _ in range(COUNTRIES+1)]
        self.trailing30_country_news = [[0 for _ in range(FEATURES)] for _ in range(COUNTRIES+1)]
        # here are the geographic buffers
        self.region_to_country = {}
        self.country_to_regions = {}
        self.region_geo = {}

    def read_atrocities(self, day, data):
        """
        Read the atrocity data and store it into buffers.  Also, form the class labels for the trees to use
        Data is pre-split, so we don't split it here
        :param day: the day of atrocity data
        :param data: passed in as a two-dimensional list of lists of strings, where each inner list is formatted as
                    [...,["LATITUDE", "LONGITUDE", "COUNTRY_ID", "REGION_ID"],...]
        """
        for line in data:
            country = int(line[2])
            region = int(line[3])
            # fill in the buffers for atrocity data
            # as the data is read in
            self.map_insert(self.reg_atroc_dates, region, day)
            self.map_insert(self.cnt_atroc_dates, country, day)
            self.all_region_atro[region][day] += 1
            self.all_cnt_atro[country][day] += 1

        # create the buffers that will become features and classes
        for region in range(REGIONS):
            # keep last 30 days of data inside atrocity decision
            self.rolling_add(self.region_atro_decision, region, self.all_region_atro[region], day, 30)
            #aggregate counts over different snapshots of the data
            for pid in range(len(PERIODS)):
                period = PERIODS[pid]
                self.rolling_add(self.curr_region_atroc[region], pid, self.all_region_atro[region], day, period)
                self.rolling_add(self.trailing30_region_atroc[region], pid, self.all_region_atro[region],
                                 day, period, trailing=30)
        # do the same thing on the country level
        for country in range(COUNTRIES):
            for pid in range(len(PERIODS)):
                period = PERIODS[pid]
                self.rolling_add(self.curr_country_atroc[country], pid, self.all_cnt_atro[country], day, period)
                # store the count of all atrocities in the world in the final index
                self.rolling_add(self.curr_country_atroc[WORLD_INDEX], pid, self.all_cnt_atro[country], day, period)
                self.rolling_add(self.trailing30_country_atroc[country], pid, self.all_cnt_atro[country],
                                 day, period, trailing=30)
                self.rolling_add(self.trailing30_country_atroc[WORLD_INDEX], pid, self.all_cnt_atro[country],
                                 day, period, trailing=30)
        #now we can create labeled data thanks to known atrocities, so call the create function
        self.forest_library.create_data(self, day)

    @staticmethod
    def rolling_add(array, pid, buffer, day, period, trailing=0, wrap_amount=0):
        """
        For use when using a buffer to hold counts and each day increment adding the latest data to the counts
        while subtracting the data that just goes out of range
        :param array: that holds aggregate counts
        :param pid: index into the array
        :param buffer: holds counts for every day
        :param day: today's date
        :param period: the amount of time you're aggregating the count over
        :param trailing: if you are not counting from day, this is the amount of time to trail behind today
        :param wrap_amount: how far to wrap in the buffer
        :return: None
        """
        if day >= trailing:
            lookback = day - trailing
            if lookback < 0:
                lookback += wrap_amount  # need to look at end of buffer for earlier data
            array[pid] += buffer[lookback]
            if day >= period + trailing:
                lookback = day - period - trailing
                if lookback < 0:
                    lookback += wrap_amount
                array[pid] -= buffer[lookback]

    def read_news_data(self, day, data):
        """
        Reads a file of news data and stores it into a buffer, then aggregates over the time period specified

        Layout of the 18 fields space separated:
        1. participant1 ID, 2. precision (1=country, 2=region, 3=town, _=unknown location)
        3. P1 location description (centroid coords) 4. P1 countryID 5. P1 RegionID
        6. _ or Participant 2 7-10. Same as P1   11. 'a'-'t' action types
        12-15. Same as 2-5. 16. Importance: 't' or 'f' boolean 17. Media coverage (1,100)
        18. Media sentiment (0,50) - (neutral to very positive)
        :param day: day of this news data
        :param data: list of string lines from a file
        :return: None
        """
        day_index = day % self.buffsize
        # clear buffer to aggregate today's events into
        for feat in range(FEATURES):
            for reg in range(REGIONS):
                self.region_news_buf[reg][feat][day_index] = 0
            for cntry in range(COUNTRIES+1):
                self.cntry_news_buf[cntry][feat][day_index] = 0
        # now read the data and aggregate features over a day of data
        for line in data:
            line = line.strip().split()
            # convert 1 line of data into features
            features, region, country = self.create_features(line)
            # aggregate the features over this entire day of data
            for feat in range(FEATURES):
                # if it is True in the feature vector then aggregate the counts into the buffer
                if features[feat]:
                    self.cntry_news_buf[WORLD_INDEX][feat][day_index] += 1
                    if region >= 0:
                        # add this event to the total counts for this region on this day
                        self.region_news_buf[region][feat][day_index] += 1
                    if country >= 0:
                        self.cntry_news_buf[country][feat][day_index] += 1
        # this day has been read into the aggregate buffers
        # now aggregate over the event time period defined as a global variable
        # 0 and 30 are the current/trailing timeframes for testing/training, respectively
        for feat in range(FEATURES):
            for reg in range(REGIONS):
                self.rolling_add(self.recent_region_news[reg], feat, self.region_news_buf[reg][feat], day_index,
                                 EVENT_AGGR_TIME, trailing=0, wrap_amount=self.buffsize)
                self.rolling_add(self.trailing30_region_news[reg], feat, self.region_news_buf[reg][feat], day_index,
                                 EVENT_AGGR_TIME, trailing=30, wrap_amount=self.buffsize)
            for cntry in range(COUNTRIES+1):
                self.rolling_add(self.recent_country_news[cntry], feat, self.cntry_news_buf[cntry][feat], day_index,
                                 EVENT_AGGR_TIME, trailing=0, wrap_amount=self.buffsize)
                self.rolling_add(self.trailing30_country_news[cntry], feat, self.cntry_news_buf[cntry][feat], day_index,
                                 EVENT_AGGR_TIME, trailing=30, wrap_amount=self.buffsize)

    @staticmethod
    def create_features(line):
        """
        Turns a single line of data into a vector of boolean features
        :param line: line of data from the news files
        :return: feature vector, action country ID, and action region ID
        """
        features = [False for _ in range(FEATURES)]
        # participant details
        p1, p1_precision, p1_country, p1_region = DataBuffer.get_news_tuple(line, 0)
        p2, p2_precision, p2_country, p2_region = DataBuffer.get_news_tuple(line, 5)
        a_id, a_precision, a_country, a_region = DataBuffer.get_news_tuple(line, 10)
        a_id = ord(line[10]) - ord('a')  # convert to int between 'a' and 't'
        media_coverage = DataBuffer.convert_int(line[16])
        media_sentiment = DataBuffer.convert_int(line[17])
        # setup feature vector
        features[0] = True  # serves as a count of all events
        features[1] = line[15] == 't'  # event importance
        features[2] = not(features[1])
        features[3] = (p1 > 0 or p2 > 0) and not(p1 > 0 and p2 > 0)
        features[4] = p1 > 0 and p2 > 0
        features[5] = DataBuffer.not_same_geography(a_country, p1_country, p2_country)
        features[6] = not(features[5])
        features[7] = DataBuffer.not_same_geography(a_region, p1_region, p2_region)
        features[8] = not(features[7])
        features[9] = (media_coverage > 15)
        features[10] = not(features[9])
        features[11] = media_sentiment > 35 or media_sentiment < 15
        features[12] = not(features[11])
        if 0 <= a_id <= 19:  # represent 'a' through 't' action types
            features[a_id + 13] = True
        return features, a_region, a_country

    @staticmethod
    def get_news_tuple(line, index):
        """
        Extracts the common operation of grabbing four ints from a spot in the data
        :param line: the line of data in a list, split on whitespace
        :param index: where to start the parsing from
        :return: parsed integer values for identity, precision, country, and region on this part of the data
        """
        ident = DataBuffer.convert_int(line[index])
        precision = DataBuffer.convert_int(line[index+1])
        country = DataBuffer.convert_int(line[index+3])
        region = DataBuffer.convert_int(line[index+4])
        return ident, precision, country, region

    # checks whether the action occurs in the same geography of any participant
    @staticmethod
    def not_same_geography(action, p1, p2):
        """
        Tests if the action and the participants are the same geography,
        either based on country or region ID
        :param action: action location
        :param p1: Person1 location
        :param p2: Person2 location
        :return: boolean true if not located in same place
        """
        if action >= 0:
            if (p1 >= 0 and p1 != action) or (p2 >= 0 and p2 != action):
                return True
        return False

    # Converts given input string to int, otherwise to -1
    @staticmethod
    def convert_int(number):
        """
        Parse a string into an integer, or -1 if it is not a valid int
        :param number: string number to parse
        :return: number value, or -1 if not a number
        """
        try:
            num = int(number)
        except ValueError:
            num = -1
        return num


    @staticmethod
    def map_insert(input_map, key, value):
        """
        Inserts the given value into a map.  If the map already has the key, it
        appends the value to the list.  If it is a new key, it starts a new list
        with the given value so the map becomes { key : [value1, value2,...] }
        :param input_map: the map you are modifying
        :param key: the key of any type
        :param value: the value of any type, added to a list in the map
        :return: None
        """
        if key in input_map:
            input_map[key].append(value)
        else:
            input_map[key] = [value]

    def read_geography(self, inputdata):
        """
        Takes in the regions.txt file on the first day of training and stores the basic
        geographic info on each region
        :param inputdata: a file that has been read into a list of strings
        :return: None
        """
        i = 0
        while i < len(inputdata):
            country_id = int(inputdata[i].split(" ")[0])
            i += 1
            region_id = int(inputdata[i].split(" ")[0])
            i += 1
            self.map_insert(self.country_to_regions, country_id, region_id)
            self.map_insert(self.region_to_country, region_id, country_id)
            outer = False
            while i < len(inputdata) and (inputdata[i].rstrip('\n') == 'outer'
                                          or inputdata[i].rstrip('\n') == 'inner'):
                if outer is False and inputdata[i].rstrip('\n') == 'outer':
                    i += 1
                    coordinates = inputdata[i].split(" ")[0]
                    coordinates = coordinates.split(",")
                    #coordinates are a list [long, lat]
                    self.region_geo[region_id] = coordinates
                    outer = True
                    i += 1
                else:
                    i += 2


class Receiver():
    def __init__(self):
        self.buf = None
        self.library = None

    def receive_data(self, source_type, day, data):
        """
        Called by main.py to pass in data from the files as it is read
        dataSourceId = 0 means information about atrocity events that happened in the current day.
        dataSourceId = 1 corresponds to information about sociopolitical activities.
        dataSourceId = 2 means geographical data (supplied only at the first day of the learning period
        :param source_type: see above
        :param day: day of data passed in
        :param data: all data from that day, either atrocity or sociopolitical
        :return: None
        """
        if source_type == 0:  # this is atrocity data
            self.buf.read_atrocities(day - FIRST_DAY, data)
        elif source_type == 1:
            self.buf.read_news_data(day - FIRST_DAY, data)
        elif source_type == 2:  # this source will be called before the other two
            self.library = Library(REGIONS, COUNTRIES, PERIODS, LIB_DAYS, START, INTERVAL, DEPTH, K,
                                   ATTR, BAG, BAG_RAT, TREES)
            self.buf = DataBuffer(self.library)
            self.buf.read_geography(data)
        else:
            raise Exception("unknown data type " + str(source_type))

    def predict_atrocities(self, day):
        day -= FIRST_DAY
        rec_dt = self.library.RECENT_DT
        start_dt = START
        intrvl_dt = INTERVAL
        result = [0.0 for _ in range(REGIONS)]
        features = [0 for _ in range(len(self.library.predict_data[0]))]
        for i in range(REGIONS):
            for j in range(len(features)):
                features[j] = self.library.predict_data[i][j]
            weight = 0.0
            for j in range(start_dt, rec_dt+1, intrvl_dt):
                day_diff = day - j
                tv = self.library.forests[day].get_forest_distr(features)
                print(tv)
                wt = max(1.0 - day_diff * config_v[2], config_v[3])
                weight += wt
                result[i] += tv*wt
            result[i] = result[i]/weight
        return result
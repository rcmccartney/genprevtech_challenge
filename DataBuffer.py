from Library import Library

FIRST_DAY = 11284  # start of data, normalize this to zero for entire program
LAST_DAY = 17644
REGIONS = 3671  # number of regions in dataset
COUNTRIES = 254  # number of countries in dataset
WORLD_INDEX = COUNTRIES
EVENT_AGGR_TIME = 90  # the window to aggregate news events over, can change this
FEATURES = 33
PERIODS = [3, 7, 14, 21, 28, 35, 42, 91, 182, 365, 730, 1460, 2920, 10000]

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
        self.curr_country_atroc = [[0 for _ in range(len(PERIODS))] for _ in range(COUNTRIES)]
        self.trailing30_country_atroc = [[0 for _ in range(len(PERIODS))] for _ in range(COUNTRIES)]
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
        data is formatted as "LATITUDE LONGITUDE COUNTRY_ID REGION_ID"
        """
        for line in data:
            line = line.strip().split()
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
                self.rolling_add(self.trailing30_country_atroc[country], pid, self.all_cnt_atro[country],
                                 day, period, trailing=30)
        #now we can create labeled data thanks to known atrocities, so call the create function
        self.forest_library.create_data(self, day)

    @staticmethod
    def rolling_add(array, pid, buffer, day, period, trailing=0, wrap_amount=0):
        if day >= trailing:
            lookback = day - trailing
            if lookback < 0:
                lookback += wrap_amount
            array[pid] += buffer[lookback]
            if day >= period + trailing:
                lookback = day - period - trailing
                if lookback < 0:
                    lookback += wrap_amount
                array[pid] -= buffer[lookback]

    def read_news_data(self, day, data):
        """
        layout of the 18 fields space separated
        1. participant1 ID, 2. precision (1=country, 2=region, 3=town, _=unknown location)
        3. P1 location description (centroid coords) 4. P1 countryID 5. P1 RegionID
        6. _ or Participant 2 7-10. Same as P1   11. 'a'-'t' action types
        12-15. Same as 2-5. 16. Importance: 't' or 'f' boolean 17. Media coverage (1,100)
        18. Media sentiment (0,50) - (neutral to very positive)
        :param day:
        :param data:
        :return:
        """
        day_index = day % self.buffsize
        for line in data:
            line = line.strip().split()
            features, region, country = self.create_features(line)
            for feat in range(FEATURES):
                if features[feat]:
                    self.cntry_news_buf[WORLD_INDEX][feat][day_index] += 1
                    if region >= 0:
                        # add this event to the total counts for this region on this day
                        self.region_news_buf[region][feat][day_index] += 1
                    if country >= 0:
                        self.cntry_news_buf[country][feat][day_index] += 1
        # this day has been read
        # now aggregate over the event time period defined as a global variable
        for trailing in [0, 30]:  # these are the current/trailing dates
            for feat in range(FEATURES):
                for reg in range(REGIONS):
                    if trailing == 0:
                        self.rolling_add(self.recent_region_news[reg], feat, self.region_news_buf[reg][feat], day_index,
                                         EVENT_AGGR_TIME, trailing=trailing, wrap_amount=self.buffsize)
                    else:
                        self.rolling_add(self.trailing30_region_news[reg], feat, self.region_news_buf[reg][feat],
                                         day_index, EVENT_AGGR_TIME, trailing=trailing, wrap_amount=self.buffsize)
                for cntry in range(COUNTRIES+1):
                    if trailing == 0:
                        self.rolling_add(self.recent_country_news[cntry], feat, self.cntry_news_buf[cntry][feat],
                                         day_index, EVENT_AGGR_TIME, trailing=trailing, wrap_amount=self.buffsize)
                    else:
                        self.rolling_add(self.trailing30_country_news[cntry], feat, self.cntry_news_buf[cntry][feat],
                                         day_index, EVENT_AGGR_TIME, trailing=trailing, wrap_amount=self.buffsize)
        #Clear old buffer for tomorrow's events
        day_index = (day_index + 1) % self.buffsize
        for feat in range(FEATURES):
            for reg in range(REGIONS):
                self.region_news_buf[reg][feat][day_index] = 0
            for cntry in range(COUNTRIES+1):
                self.cntry_news_buf[cntry][feat][day_index] = 0

    @staticmethod
    def create_features(line):
        line = line.strip().split()
        features = [False for _ in range(FEATURES)]
        # participant details
        p1, p1_precision, p1_country, p1_region = DataBuffer.get_news_tuple(line, 0)
        p2, p2_precision, p2_country, p2_region = DataBuffer.get_news_tuple(line, 5)
        line[10] = ord(line[10]) - ord('a')
        a_id, a_precision, a_country, a_region = DataBuffer.get_news_tuple(line, 10)
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
        ident = DataBuffer.convert_int(line[index])
        precision = DataBuffer.convert_int(line[index+1])
        country = DataBuffer.convert_int(line[index+3])
        region = DataBuffer.convert_int(line[index+4])
        return ident, precision, country, region

    # checks whether the action occurs in the same geography of any participant
    @staticmethod
    def not_same_geography(action, p1, p2):
        if action >= 0:
            if (p1 >= 0 and p1 != action) or (p2 >= 0 and p2 != action):
                return True
        return False

    # Returns the days since last atrocity occurred, minus trailing factor
    @staticmethod
    def last_atrocity(buffer, day, tr=0):
        """

        :param buffer:
        :param day:
        :param tr: trailing time
        :return:
        """
        if buffer is not None:
            day -= tr
            index = len(buffer)
            # buffer[index] is an atrocity date
            while index >= 0 and buffer[index] > day:
                index -= 1
            if index >= 0:
                return day - buffer[index]
        return 10000  # used for infinite into the past

    # Converts given input string to int, otherwise to -1
    @staticmethod
    def convert_int(number):
        try:
            num = int(number)
        except ValueError:
            num = -1
        return num

    # Inserts the given value into map with existing key or inserts a new key.
    # Values are stored in list. key ---> [value1, value2]
    @staticmethod
    def map_insert(input_map, key, value):
        if key in input_map:
            input_map[key].append(value)
        else:
            input_map[key] = [value]

    def read_geography(self, data):
        """
        reads the file regions.txt, which consists of polygons that are labeled
        as 'outer' or 'inner' followed by 'Longitude,Lattitude'
        Only first Lat/Long for a region is stored
        """
        for line in data:
            print("here")
            #set up the maps from country to region and vice versa
            country = line.strip().split()[0]
            region = data.readline().strip().split()[0]
            print(country, region)

            self.region_to_country[region] = country
            self.map_insert(self.country_to_regions, country, region)
            #just grab and store the first instance of a lat/long that you see per region
            first = True
            line = data.readline().strip().lower()
            #grab the first lat/long for this region
            while line == 'outer' or line == 'inner':
                if first and line == 'outer':
                    first = False
                    line = next(data).strip().split()
                    line = line[0].split(",")
                    self.region_geo[region] = (float(line[0]), float(line[1]))
                else:
                    data.readline()
        print(self.region_geo[region])


class Receiver():
    def __init__(self):
        self.buf = None

    def receive_data(self, source_type, day, data):
        """
        dataSourceId = 0 means information about atrocity events that happened in the current day.
        dataSourceId = 1 corresponds to information about sociopolitical activities.
        dataSourceId = 2 means geographical data (supplied only at the first day of the learning period

        :param source_type:
        :param day: day of data passed in
        :param data: all data from that day, either atrocity or sociopolitical
        :return:
        """
        if source_type == 0:  # this is atrocity data
            self.buf.read_atrocities(day - FIRST_DAY, data)
        elif source_type == 1:
            self.buf.read_news_data(day - FIRST_DAY, data)
        elif source_type == 2:
            library = Library(REGIONS, COUNTRIES, PERIODS)
            self.buf = DataBuffer(library)  # this will be called before the other two
            self.buf.read_geography(data)
        else:
            raise Exception("unknown data type " + str(source_type))


    def predict_atrocities(self, day):
        pass


def main():
    rec = Receiver()
    rec.receive_data(2, 0, open(input("Enter file: ")))


if __name__ == '__main__':
    main()
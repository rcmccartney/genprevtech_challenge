
region_soc_0day = [ [ 0 for x in range(MAX_FEATURES) ] for x in range(total_Regions) ]
region_soc_30day = [ [ 0 for x in range(MAX_FEATURES) ] for x in range(total_Regions) ]

country_soc_0day = [ [ 0 for x in range(MAX_FEATURES) ] for x in range(total_Country+1) ]
country_soc_30day = [ [ 0 for x in range(MAX_FEATURES) ] for x in range(total_Country+1) ]


region_past_0day = [ [0 for x in range(len(periods)) ] for x in range(total_Regions) ]

region_past_30day = [ [0 for x in range(len(periods)) ] for x in range(total_Regions) ]

country_past_0day = [ [0 for x in range(len(periods)) ] for x in range(total_Country+1) ]

country_past_30day = [ [0 for x in range(len(periods)) ] for x in range(total_Country+1) ]

#Decision tree inputs
dt_pro_count = 2*len(periods) + 7 + 31 #62
lib_days = 180
dt_lib_position = 0
dt_lib_pro = [ [ [ 0.0 for x in range(dt_pro_count) ] for x in range(total_Regions) ] for x in range(lib_days) ]
dt_lib_ans = [ 0 for x in range(dt_pro_count) ]
dt_prd_pro = [ [ 0.0 for x in range(dt_pro_count) ] for x in range(total_Regions) ]







#####atrocity data parsing starts
def updateLibrary(dayID):
	global dt_lib_ans,dt_lib_position,dt_lib_pro,dt_prd_pro
	print 'Library Updated'
	
	for j in range(total_Regions):
		country = in_country[j]
		#country = 1
		if(recent_region_atrocity[j] >0):
			dt_lib_ans[dt_lib_position] = 1
		lib_index = 0
		for i in range(len(periods)):
			dt_lib_pro[dt_lib_position][j][lib_index] = region_past_30day[j][i]
			dt_prd_pro[j][lib_index] = region_past_0day[j][i]
			lib_index += 1
		
		ratio1 = [0.0,0.0]
		ratio2 = [0.0,0.0]
		ratio  = [0.0,0.0]
		
		if(country_past_30day[country][len(periods)-1] != 0):
			ratio1[0] = region_past_30day[j][len(periods)-1]*1.0 / country_past_30day[country][len(periods)-1]
		ratio2[0] = 1.0 / len(inner_regs[country])
		ratio[0] = ratio1[0]*0.7 + ratio2[0]*0.3
		
		if(country_past_0day[country][len(periods)-1] != 0):
			ratio1[1] = region_past_0day[j][len(periods)-1]*1.0 / country_past_0day[country][len(periods)-1]
		ratio2[1] = 1.0 / len(inner_regs[country])
		ratio[1] = ratio1[1]*0.7 + ratio2[1]*0.3
		
		for i in range(len(periods)):
			dt_lib_pro[dt_lib_position][j][lib_index] = country_past_30day[j][i]*1.0*ratio[0]
			dt_prd_pro[j][lib_index] = country_past_0day[j][i]*1.0*ratio[1]
			lib_index += 1
		
		#7 SEA features begin
		#1 no. of regions
		dt_lib_pro[dt_lib_position][j][lib_index] = len(inner_regs[country])
		dt_prd_pro[j][lib_index] = len(inner_regs[country])
		lib_index += 1
		#2 ratio1
		dt_lib_pro[dt_lib_position][j][lib_index] = ratio1[0]
		dt_prd_pro[j][lib_index] = ratio1[1]
		lib_index += 1
		#3 Count of world atrocity, 5->35 days
		dt_lib_pro[dt_lib_position][j][lib_index] = country_past_30day[total_Country][5]
		dt_prd_pro[j][lib_index] = country_past_0day[total_Country][5]
		lib_index += 1
		#4 changing features
		#lastt1 = days before last atrocity 30 days prior to this day
		#lastt2 = days before last atrocity prior to this day
		lastt1 = 10000
		lastt2 = 10000
		if (len(region_past_history[j]) > 0 ):
			k = 0
			while ( (dayID - region_past_history[j][len(region_past_history[j])-1-k]) < 30 and k < len(region_past_history[j] )):
				k+= 1
			lastt1 = dayID -first_DayID - region_past_history[j][len(region_past_history[j])-1-k]	
			lastt2 = dayID -first_DayID - region_past_history[j][len(region_past_history[j])-1]
		dt_lib_pro[dt_lib_position][j][lib_index] = lastt1
		dt_prd_pro[j][lib_index] = lastt2
		lib_index += 1
		
		#5 same for country
		lastt1 = 10000
		lastt2 = 10000
		if (len(country_past_history[country]) > 0 ):
			k = 0
			while ( (dayID - country_past_history[country][len(region_past_history[country])-1-k]) < 30 and k < len(country_past_history[country] )):
				k+= 1
			lastt1 = dayID -first_DayID - country_past_history[country][len(region_past_history[country])-1-k]	
			lastt2 = dayID -first_DayID - country_past_history[country][len(region_past_history[country])-1]
		dt_lib_pro[dt_lib_position][j][lib_index] = lastt1
		dt_prd_pro[j][lib_index] = lastt2
		lib_index += 1
		
		#6,7 longitudes latitude
		dt_lib_pro[dt_lib_position][j][lib_index] = region_long[j]
		dt_prd_pro[j][lib_index] = region_long[j]
		lib_index += 1
		dt_lib_pro[dt_lib_position][j][lib_index] = region_latt[j]
		dt_prd_pro[j][lib_index] = region_latt[j]
		lib_index += 1
		#7 SEA over
		
		for i in range(31):
			if(country_past_30day[total_Country][i] != 0):
				dt_lib_pro[dt_lib_position][j][lib_index] = region_soc_30day[j][i] / (country_past_30day[total_Country][i] * 1.0 / total_Regions)
			else :
				dt_lib_pro[dt_lib_position][j][lib_index] = 1.0
			lib_index+=1
			if(country_past_0day[total_Country][i] != 0):
				dt_lib_pro[dt_lib_position][j][lib_index] = region_soc_0day[j][i] / (country_past_0day[total_Country][i] * 1.0 / total_Regions)
			else :
				dt_lib_pro[dt_lib_position][j][lib_index] = 1.0
			lib_index+=1
		
	dt_lib_position += 1
	if ( dt_lib_position >= lib_days):
		dt_lib_position = 0


		

def readInputData(inputFile):
    #reader = open(inputFile, 'rU')
    reader = open(inputFile, 'rU')
    inputData = reader.readlines()
    return inputData


# Testing code
def testSocialData(input):
    result = []
    for line in input:
        result.append(createFeatureVector(line))
    return result

def test():
    inputData = readInputData('regions.txt')
    getGeographicData(inputData)
    inputData = readInputData('test.txt')
    result = testSocialData(inputData)
    pdb.set_trace()
    print "Stop Here"

#test()

receiveData(0,11623,[[1, 1, 11, 11]])
#inputData = readInputData('11630.txt')
receiveData(1,11630,'11630.txt')

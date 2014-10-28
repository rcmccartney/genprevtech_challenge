__author__ = 'mccar_000'

import math
import random


class EntropyFn():
    """
    This class represents a split at the node that chooses k random
    split points for every attribute of the data and
    chooses the optimal entropy among those split points
    This function is to be used for continuous data
    that is in the range (minsplit, maxsplit)
    """

    def __init__(self, k=100, minsplit=0, maxsplit=1):
        self.k = k  # random num of splits desired
        self.minsplit = minsplit
        self.maxsplit = maxsplit

    # returns the nK split points needed as dictionary that will become the split point for this node
    def randomize(self, samples):
        attr_dict = {}  # stores the k split points for each attr
        for attr_index in range(len(samples[0][0:-1])):
            splits = []
            for j in range(self.k):
                splits.append(random.uniform(self.minsplit, self.maxsplit))
            attr_dict[attr_index] = sorted(splits)
        return attr_dict

    @staticmethod
    def calc_distr(samples):
        tot = [0 for _ in range(len(samples[0][-1]))]
        tot_elem = len(samples)
        for row in samples:
            for i in range(len(tot)):  # tot is a zero vector for each class
                tot[i] += row[-1][i]  # row[-1] is a boolean vector for class
        return [float(count)/tot_elem for count in tot]  # this give prob dist for samples

    @staticmethod
    def calc_entropy(samples):
        if not samples:  # samples is empty, entropy is zero
            return 0
        tot = EntropyFn.calc_distr(samples)
        entrop = 0
        for prob in tot:  # loop calculates entropy for samples distribution
            if prob == 0.0:
                continue
            else:
                entrop += -prob * math.log(prob, 2)
        return entrop

    def calc_split(self, samples):
        min_ent = float('NaN')
        attr_dict = self.randomize(samples)
        bestsplit = 0
        bestattr = 0
        finaldata_l = []
        finaldata_r = []
        old_entr = self.calc_entropy(samples)
        for attrIndex, split_pts in attr_dict.items():
            left_data = []
            right_data = []
            for split in split_pts:  # we know the splits are sorted in ascending order
                if not left_data and not right_data:  # first time splitting
                    for row in samples:
                        if row[attrIndex] < split:
                            left_data.append(row)
                        else:  # row[attr] >= split_pt:
                            right_data.append(row)
                    right_data = sorted(right_data, key=lambda l: l[attrIndex])  # now its sorted by curr attr
                else:  # already split before, just move data from right list to left list
                    if right_data:  # make sure right isn't empty
                        while split > right_data[0][attrIndex]:  # need to move data into left bucket
                            left_data.append(right_data.pop(0))
                            if not right_data:  # just ran out of samples
                                break
                p1 = float(len(left_data)) / len(samples)
                p2 = float(len(right_data)) / len(samples)
                curr_ent = p1*self.calc_entropy(left_data) + p2*self.calc_entropy(right_data)
                assert old_entr - curr_ent >= 0, "Information gain cannot be negative"
                assert curr_ent >= 0, "Entropy cannot be negative"
                if math.isnan(min_ent) or curr_ent < min_ent:  # Min entropy for this data is MAX Inform Gain
                    min_ent = curr_ent
                    bestsplit = split
                    bestattr = attrIndex
                    finaldata_l = list(left_data)
                    finaldata_r = list(right_data)
        assert not math.isnan(min_ent), "Oops this data doesn't seem right"
        return Splitter(bestsplit, bestattr), finaldata_l, finaldata_r


# TODO: turn this into decorated function instead
class Splitter():

    def __init__(self, split_val, attr):
        self.attr = attr
        self.split_val = split_val

    def split(self, instance):
        return instance[self.attr] > self.split_val
__author__ = 'mccar_000'

import math
import random


class AtrocityEntropyFn():
    """
    This class represents a split at the node that chooses
    k random split points for every attribute of the data and
    chooses the optimal entropy among those split points
    The K split points are chosen from among the existing
    attribute values, so it doesn't need to be continuous data
    and a min or max value need not be defined
    """

    def __init__(self, k=100, attr=-1):
        self.k = k  # random num of splits desired
        self.attr = attr  # -1 means use all the atributes for training

    # returns the nK split points needed as dictionary that will become the split point for this node
    def randomize(self, samples):
        attr_dict = {}  # stores the k split points for each attr
        if self.attr == -1:
            indexes = [i for i in range(len(samples[0]))]
        else:
            indexes = [random.randint(0, len(samples[0][0:-1])) for _ in self.attr]
        for attr_index in indexes:
            splits = []
            for _ in range(self.k):
                row = random.randint(0, len(samples)-1)
                # store the value of the random sample for this attribute
                splits.append(samples[row][attr_index])
            attr_dict[attr_index] = sorted(splits)
            # repeat for the chosen attributes
        return attr_dict

    @staticmethod
    def get_class_counts(samples):
        """
        more efficient for only two classes
        :return:
        """
        tot = [0, 0]
        for row in samples:
            tot[0] += row[-1][0]  # row[-1] is a boolean vector for the two classes
            tot[1] += row[-1][1]
        return tot

    @staticmethod
    def calc_distr(samples):
        tot = [0 for _ in range(len(samples[0][-1]))]
        tot_elem = len(samples)
        for row in samples:
            for i in range(len(tot)):  # tot is a zero vector for each class
                tot[i] += row[-1][i]  # row[-1] is a boolean vector for class
        return [float(count)/tot_elem for count in tot]  # this give prob dist for samples

    @staticmethod
    def adjust_counts(left, right, class_v):
        """
        the parameters
        :param left:
        :param right:
        :param class_v:
        :return:
        """
        for i in range(len(class_v)):
            left[i] += class_v[i]
            right[i] -= class_v[i]

    @staticmethod
    def calc_entropy(samples):
        if not samples:  # samples is empty, entropy is zero
            return 0
        tot = AtrocityEntropyFn.calc_distr(samples)
        entrop = 0
        for prob in tot:  # loop calculates entropy for samples distribution
            if prob == 0.0:
                continue
            else:
                entrop += -prob * math.log(prob, 2)
        return entrop

    @staticmethod
    def entropy_split(l1, l2, r1, r2):
        # here the data is already split for us, unlike above
        totsum = 0.0
        if l1 != 0:
            totsum += -l1*math.log(l1/(l1+l2))
        if l2 != 0:
            totsum += -l2*math.log(l2/(l1+l2))
        if r1 != 0:
            totsum += -r1*math.log(r1/(r1+r2))
        if r2 != 0:
            totsum += -r2*math.log(r2/(r1+r2))
        if l1+l2+r1+r2 != 0:
            totsum /= l1+l2+r1+r2
        return totsum

    def calc_split(self, samples):
        min_ent = float('NaN')
        attr_dict = self.randomize(samples)
        bestsplit = 0
        bestattr = 0
        finaldata_l = []
        finaldata_r = []
        # more efficient to keep count than to recalculate distributions
        counts = self.get_class_counts(samples)
        for attrIndex, split_pts in attr_dict.items():
            r_count = counts[:]
            l_count = [0, 0]  # all the data starts in the right
            left_data = []
            right_data = []
            for split in split_pts:  # we know the splits are sorted in ascending order
                if not left_data and not right_data:  # first time splitting
                    for row in samples:
                        if row[attrIndex] < split:
                            left_data.append(row)
                            self.adjust_counts(left_data, right_data, row[-1])
                        else:  # row[attr] >= split_pt:
                            right_data.append(row)
                    right_data = sorted(right_data, key=lambda l: l[attrIndex])  # now its sorted by curr attr
                else:  # already split before, just move data from right list to left list
                    if right_data:  # make sure right isn't empty
                        # need to move data into left bucket,
                        while split >= right_data[0][attrIndex]:
                            row = right_data.pop(0)
                            left_data.append(row)
                            self.adjust_counts(left_data, right_data, row[-1])
                            if not right_data:  # just ran out of samples
                                break
                curr_ent = self.entropy_split(l_count[0], l_count[1], r_count[0], r_count[1])
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
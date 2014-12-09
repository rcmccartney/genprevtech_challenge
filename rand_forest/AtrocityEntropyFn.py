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
        """
        Initialize this entropy function used for the atrocity dataset
        :param k: the number of split points to try per attribute
        :param attr: the number of random attributes to try per node
        :return: None
        """
        self.k = k  # random num of splits desired
        self.attr = attr  # -1 means use all the atributes for training

    def randomize(self, samples):
        """
        Returns the nK split points needed as dictionary that will become the split point for this node
        The split point is the value a random instance in the dataset has for that attribute
        :param samples: data to randomize split points on
        :return: dictionary mapping attribute index to list of random sorted split points,
        that is d[attr] = [split1, split2,...]
        """
        attr_dict = {}  # stores the k split points for each attr
        if self.attr == -1:
            indexes = [i for i in range(len(samples[0])-1)]  # subtract one for the class attribute
        else:
            # choose attr random attributes to use.  subtract 2 because randint is inclusive of both (start, end)
            indexes = [random.randint(0, len(samples[0])-2) for _ in range(self.attr)]
        for attr_index in indexes:
            splits = []
            for _ in range(self.k):
                # the split point is a value from one of the random
                row = random.randint(0, len(samples)-1)
                # store the value of the random sample for this attribute
                splits.append(samples[row][attr_index])
            attr_dict[attr_index] = sorted(splits)  # this will overwrite the attr_index if chosen more than once
            # repeat for the chosen attributes
        return attr_dict

    @staticmethod
    def get_class_counts(samples):
        """
        Get the counts of the two classes of the samples
        :return: tot count vector
        """
        tot = [0, 0]
        for row in samples:
            tot[0] += row[-1][0]  # row[-1] is a boolean vector for the two classes
            tot[1] += row[-1][1]
        return tot

    @staticmethod
    def calc_distr(samples):
        """
        Calculates the distribution of the samples into the different classes
        Inefficient to use this instead of updating counts as you go
        :param samples: data to calc distribution over
        :return: calculated distribution
        """
        tot = [0 for _ in range(len(samples[0][-1]))]
        tot_elem = len(samples)
        for row in samples:
            for i in range(len(tot)):  # tot is a zero vector for each class
                tot[i] += row[-1][i]  # row[-1] is a boolean vector for class
        return [float(count)/tot_elem for count in tot]  # this give prob dist for samples

    @staticmethod
    def adjust_counts(left, right, class_v):
        """
        Use this to keep track of counts of the classes as you go, whenever you
        move an item from the right list into the left list since it is
        more efficient than continually recalculating the distribution
        :param left: list of counts for class 0,1 for left split
        :param right: list of counts for class 0,1 for right split
        :param class_v: the class vector for the data item getting moved
        :return: None
        """
        for i in range(len(class_v)):
            left[i] += class_v[i]
            right[i] -= class_v[i]

    @staticmethod
    def calc_entropy(samples):
        """
        Calculates the entropy of sample data using the calc distribution helper function
        Inefficient for large data since it must iterate through for each calculation
        :param samples: Data to calculate entropy with
        :return: entropy score
        """
        if not samples:  # samples is empty, entropy is zero
            return 0
        tot = AtrocityEntropyFn.calc_distr(samples)
        entrop = 0.0
        for prob in tot:  # loop calculates entropy for samples distribution
            if prob == 0.0:
                continue
            else:
                entrop += -prob * math.log(prob, 2)
        return entrop

    @staticmethod
    def entropy_split(l1, l2, r1, r2):
        """
        More efficient entropy calculation, uses the updated counts of left and right classes to
        calculate entropy without iterating through the dataset
        :param l1: number of class 0 instances in left data
        :param l2: number of class 1 instances in left data
        :param r1: number of class 0 instances in right data
        :param r2: number of class 1 instances in right data
        :return: entropy score
        """
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
        finaldist_l = []
        finaldist_r = []
        # more efficient to keep count than to recalculate distributions
        counts = self.get_class_counts(samples)
        for attrIndex, split_pts in attr_dict.items():
            r_count = counts[:]  # copy the starting counts
            right_data = sorted(samples, key=lambda l: l[attrIndex])  # now its sorted by the current attr
            l_count = [0, 0]  # all the data starts in the right
            left_data = []
            for split in split_pts:  # we know the splits are sorted in ascending order
                if not right_data:  # right ran out of data before this iteration, stop calculating entropy of splits
                    break
                # need to move data into left bucket based off of split point
                while split >= right_data[0][attrIndex]:
                    row = right_data.pop(0)
                    left_data.append(row)
                    self.adjust_counts(l_count, r_count, row[-1])
                    if not right_data:  # just ran out of samples
                        break
                curr_ent = self.entropy_split(l_count[0], l_count[1], r_count[0], r_count[1])
                assert curr_ent >= 0, "Entropy cannot be negative"
                if math.isnan(min_ent) or curr_ent < min_ent:  # Min entropy for this data is MAX Inform Gain
                    min_ent = curr_ent
                    bestsplit = split
                    bestattr = attrIndex
                    finaldata_l = left_data[:]
                    finaldata_r = right_data[:]
                    tot = l_count[0] + l_count[1]
                    if tot != 0:
                        finaldist_l = [x/tot for x in l_count]
                    else:
                        finaldist_l = None
                    tot = r_count[0] + r_count[1]
                    if tot != 0:
                        finaldist_r = [x/tot for x in r_count]
                    else:
                        finaldist_r = None
        assert not math.isnan(min_ent), "Oops this data doesn't seem right"
        return Splitter(bestsplit, bestattr), finaldata_l, finaldata_r, finaldist_l, finaldist_r


class Splitter():
    """
    This is a class that is parameterized by the best (lowest entropy) split
    point found for this given node in the tree.  It stores the values found
    and knows how to split an unseen instance based off of its stored value
    for its given attribute
    """
    def __init__(self, split_val, attr):
        self.split_val = split_val
        self.attr = attr

    def split(self, instance):
        return instance[self.attr] > self.split_val
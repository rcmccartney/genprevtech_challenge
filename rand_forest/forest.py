__author__ = 'Robert McCartney'

from rand_forest.tree import *
import math
import random
import pickle
from multiprocessing.pool import Pool

trainPlots = ["r<", "yv", "g^", "b>"]
testPlots = ["r8", "ys", "gp", "bh"]
# TODO: support more than 8 colors
coloropts = ('r', 'y', 'g', 'b', 'c', 'm', 'k', 'w')


def make_tree(tree_data):
    """
    Use this function to make a tree in parallel using all cores of the machine
    :param tree_data: Tuple of (self.data_copy(), self.bag, self.bag_ratio, self.depthlimit, self.weak_learner)
    :return: Tree made by this thread
    """
    return Tree(tree_data[0], tree_data[1], tree_data[2])


class Forest(object):

    def __init__(self, depthlimit=3, weak_learner=None, bagging=False, bag_ratio=.4, filename=None,
                 separator=',', class_idx=-1, default_tree_count=200):
        """
        Initialize all the variables required for this random forest
        :param depthlimit: the depth allowed in a tree of the forest
        :param weak_learner: how we split the data and traverse a tree, used as a Strategy design pattern
        :param bagging: boolean to use bagging or not
        :param bag_ratio: if using bagging, what percent of the data to use
        :param filename: file of data with a row of class values
        :param separator: the separator used in the data
        :param class_idx: the index of the class vale in a row of data
        :param default_tree_count: how many trees to train in this random forest
        :return: None
        """
        self.bagging = bagging
        self.bag_ratio = bag_ratio
        self.default_tree_count = default_tree_count
        self.data = []
        self.minclass = float('NaN')
        self.maxclass = float('NaN')
        self.numclasses = float('NaN')
        self.separator = separator
        self.class_idx = class_idx
        if filename is not None:
            self.prepare_and_add_data(filename, True)
        self.trees = []
        self.error = []
        self.depthlimit = depthlimit
        self.weak_learner = weak_learner

    def prepare_and_add_data(self, filename, first=False):
        """
        Used if this forest is initialized with a dataset
        :param filename: file of data to create forest on
        :param first: if this is the first time we have seen any data,
        then we need to do extra work for min/max and number of classes
        :return: None
        """
        # these variables will be initialized if they are unknown, or user requests
        # using first_time=True parameter
        if math.isnan(self.minclass) or math.isnan(self.numclasses):
            first = True
        # add the processed data to the data stored by this tree
        # and set the number of classes found, if required
        self.data += self.prepare_data(filename, first)
        if first:
            self.numclasses = len(self.data[0][-1])
        assert self.minclass != self.maxclass and self.numclasses != 1, \
            "Error: Only one class was found in the file, not suitable for classification"

    def prepare_data(self, filename, first_time=False):
        """
        Appends the filename passed in to the data stored in this forest
        Assume classes are labeled in order from min,min+1,min+2,...,max
        :param filename: file to create a Forest from
        :param first_time: if this is the first data parsed, need to set min and max classes
        """
        #TODO: Assumes the data can fit into memory, need to fix this
        samples = []
        lineno = 0
        with open(filename, 'r') as datafile:
            lineno += 1
            for line in datafile:
                try:
                    row = ([float(i.strip()) for i in line.split(self.separator)])
                    if first_time:  # need to find the smallest and largest class value
                        if math.isnan(self.minclass) or row[self.class_idx] < self.minclass:
                            self.minclass = int(row[self.class_idx])  # starting index of classes
                        if math.isnan(self.maxclass) or row[self.class_idx] > self.maxclass:
                            self.maxclass = int(row[self.class_idx])  # ending index of classes
                    # Move the class label to the last index of the row
                    if self.class_idx != len(row)-1:
                        temp = row[self.class_idx]
                        row[self.class_idx] = row[-1]
                        row[-1] = temp
                    samples.append(row)
                except Exception as e:
                    print("Bad row, skipping line: {0} Line {1}".format(e, lineno))
                    pass

        assert not math.isnan(self.minclass) and not math.isnan(self.maxclass), \
            "Error reading file: Please check the data separator and class index supplied"

        # go back through the data and turn a class index into a vector of booleans for each
        # class value.  i.e. if classes range from 0-2, a class of 0 would be the vector [1,0,0]
        for row in samples:
            # pop off the old class value that is an integer
            classification = int(row.pop())
            classification -= self.minclass  # normalize to a minclass of 0
            classvec = [0 for _ in range((self.maxclass-self.minclass)+1)]
            classvec[classification] = 1
            row.append(classvec)
        return samples

    def set_train_delete(self, instances, classes, numclass):
        """
        Used when there is a lot of data and you want to train a forest
        without saving it all
        :param instances: data to use, already processed into 2-D list of lists
        :param classes: list of the class decision for each row in instances
        :param numclass: number of classes in this dataset
        :return: None
        """
        self.numclasses = numclass
        newinstances = []
        for row_id in range(len(instances)):
            # append a class decision vector such as [0, 1, 0, 0] (here is a 4 class problem) to row of data
            classvec = [0 for _ in range(numclass)]
            # classvec will be appended as a sublist, not extended into the list itself
            classvec[classes[row_id]] = 1
            # make sure we don't have empty data
            if len(instances[row_id]) > 0:
                # pass in a copy of the Library, as it is re-used and we don't want to modify it with classvec
                newinstances.append(instances[row_id][:])
                newinstances[row_id].append(classvec)
        self.data = newinstances
        self.add_tree(iterations=self.default_tree_count)
        self.data = None

    def add_tree(self, iterations=-1, snapshot=False):
        """
        Multi-core, fully utilizes underlying CPU to create the trees
        of the forest and stores them into the forest's list of trees
        :param iterations: number of trees to make, -1 means use default setting
        :return: None
        """
        print("Adding trees:", iterations)
        if iterations == -1:
            iterations = self.default_tree_count
        #########################
        # MULTI THREADED
        ########################
        pool = Pool()  # creates multiple processes equal to cores in machine
        outputs = pool.map(make_tree, [(self.data_copy(), self.depthlimit, self.weak_learner)
                                       for _ in range(iterations)])
        pool.close()
        pool.join()
        self.trees.extend(outputs)  # get the trees created and store them
        #########################
        # SINGLE THREADED
        ########################
        #for i in range(iterations):
        #    tree = Tree(self.data, self.bagging, self.bag_ratio, self.depthlimit, self.weak_learner)
        #    self.trees.append(tree)  # get the trees created and store them
        if snapshot:
            self.sum_squares(len(self.trees))  # get error after each snapshot, if this command is run multiple times

    def data_copy(self):
        """
        Gives each thread its own copy of the data
        :return: thread-local copy of the data to make a tree from
        """
        # too slow
        #return deepcopy(self.data)
        if self.bagging:
            return [self.data[random.randint(0, len(self.data)-1)] for _ in range(int(self.bag_ratio*len(self.data)))]
        else:
            return self.data

    def sum_squares(self, iterations):
        sqerr = 0.0
        # TODO: get sum_squares over a random subset of the data instead of all of it
        for row in self.data:
            distr = self.get_forest_distr(row)
            for j in range(len(distr)):
                sqerr += (float(row[-1][j]) - distr[j]) ** 2
        self.error += [iterations, sqerr]

    def test(self, datafile=None):
        if datafile is None:  # use the training data
            data = self.data
        else:
            data = self.prepare_data(datafile)
        confusion = {}
        correct = 0
        error = 0
        # build the confusion matrix
        for i in range(self.numclasses):
            confusion.setdefault(i, {})  # dictionary of dictionaries where key is (class i, class j)
            for j in range(self.numclasses):
                confusion[i][j] = 0
        # put each instance into it's place in the matrix
        for instance in data:
            distr = self.get_forest_distr(instance)
            classpredict = distr.index(max(distr))  # this is the class chosen by the forest output
            actualclass = instance[-1].index(1)
            confusion[actualclass][classpredict] += 1
            if instance[-1][classpredict] == 1:  # the prediction matches the true class
                correct += 1
            else:
                error += 1
        print("Confusion matrix:")
        for i in sorted(confusion):
            print("%4d" % (i + self.minclass), end="")
        for i in sorted(confusion):
            print("\n", (i+self.minclass), end=" ")
            for j in sorted(confusion):
                print("%4d" % confusion[i][j], end="")
        print()
        print("Number of classf errors:", error)
        print("Recognition rate: %5.2f%%" % (100 * float(correct) / len(data)))
        # for skewed classes the recognition rate isn't very important
        # use the F1 score instead
        # skewed classes is when there is a zero and a one class and the one class is rare
        if self.numclasses == 2:
            accuracy, precision, recall = Forest.analyze_confusion_matrix(confusion)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            try:
                print("F1 Score: %5.2f" % ((2*precision*recall)/(precision+recall)))
            except ZeroDivisionError:
                print("F1 undefined")
        return confusion

    @staticmethod
    def analyze_confusion_matrix(confusionmat):
        """
        Calculates the accuracy, precision, and recall of a given confusion matrix
        :param confusionmat: a 2D matrix of class decisions for tested data
        :return: results for accuracy, precision, and recall
        """
        totalsize = confusionmat[0][0] + confusionmat[0][1] + confusionmat[1][0] + confusionmat[1][1]
        accuracy = (confusionmat[0][0] + confusionmat[1][1]) / totalsize
        # prec = true pos / tru pos + false positives
        try:
            precision = confusionmat[1][1] / (confusionmat[1][1] + confusionmat[1][0])
        except ZeroDivisionError:
            precision = 0
        # recall = true pos / tru pos + false negatives
        try:
            recall = confusionmat[1][1] / (confusionmat[1][1] + confusionmat[0][1])
        except ZeroDivisionError:
            recall = 0
        return accuracy, precision, recall

    def get_forest_distr(self, instance):
        """
        Averages the distribution of every tree in the forest to calc final distribution
        :param instance: that you want to classify using this forest
        :return: distribution of class decisions, representing a confidence percentage
        """
        # combine the distributions predicted by each tree
        # use a simple average to combine distributions
        # TODO: allow other combination options of distributions
        distr = [0 for _ in range(self.numclasses)]  # create the distribution container
        tot_trees = len(self.trees)
        for tree in self.trees:
            tree_distr = tree.get_instance_distr(instance)
            for i in range(len(distr)):
                distr[i] += tree_distr[i]
        return [prob/tot_trees for prob in distr]  # this gives avg prob dist for trees

    def learning_curve(self):
        """
        plots the learning curve found on this data
        Dependency for matplotlib
        :return: None
        """
        import matplotlib.pyplot as plt
        plt.figure(0)
        # plt.pause(1)  # use these when interactive plotting
        #plt.ioff()  # interactive graphics off
        #plt.clf()   # clear the plot
        #plt.hold(True) # overplot on
        plt.plot(self.error[::2], self.error[1::2])
        plt.ylabel('Sum of squared error')
        plt.xlabel('Epochs')
        plt.title("Learning Curve")
        plt.ylim([plt.ylim()[0] - 1, plt.ylim()[1] + 1])
        #plt.ion()   # interactive graphics on
        #plt.draw()  # update the plot
        plt.show()

    def region_plot(self, attr1=0, attr2=1, granularity=50, testfile=None):
        """
        This method graphs the decision boundaries.  Most useful for two featured data, as you
        can see exactly where the decision changes from one class to another
        Dependency for matplotlib and numpy
        :param attr1: which attribute to use, default is index 0 of the data
        :param attr2: which attribute to use for second attr, default is index 1 of the data
        :param granularity: how close to test points for their decision as a class, finer granularity
        gives more accurate coloring but takes longer
        :param testfile: if you want to graph test data instead of training data
        :return: None
        """
        import matplotlib.pyplot as plt
        from numpy import arange, meshgrid, array, reshape

        minval_x = minval_y = maxval_x = maxval_y = float("NaN")
        if testfile is not None:
            data = self.prepare_data(testfile)
            datatype = "Test Data"
        else:
            data = self.data
            datatype = "Train Data"

        # plot all of the data in the forest
        # TODO: this will be too much to plot for large data
        allpoints = {}
        for instance in data:
            allpoints.setdefault(instance[-1].index(1), [])
            x1 = instance[attr1]
            x2 = instance[attr2]
            if math.isnan(minval_x) or x1 < minval_x:
                minval_x = x1
            if math.isnan(maxval_x) or x1 > maxval_x:
                maxval_x = x1
            if math.isnan(minval_y) or x2 < minval_y:
                minval_y = x2
            if math.isnan(maxval_y) or x2 > maxval_y:
                maxval_y = x2
            allpoints[instance[-1].index(1)].append(x1)
            allpoints[instance[-1].index(1)].append(x2)
        # use the same number of ticks in each axis
        # set up a contour plot
        # TODO: this only works for 2-D data since otherwise the tree can't classify it
        ten_percent_x = (maxval_x-minval_x)/10
        ten_percent_y = (maxval_y-minval_y)/10
        ticks_x = arange(minval_x-ten_percent_x, maxval_x+ten_percent_x, (maxval_x-minval_x)/granularity)
        ticks_y = arange(minval_y-ten_percent_y, maxval_y+ten_percent_y, (maxval_y-minval_y)/granularity)
        X, Y = meshgrid(ticks_x, ticks_y)
        out = []
        for i in range(X.size):
            distr = self.get_forest_distr([X.ravel()[i], Y.ravel()[i]])
            out.append(distr.index(max(distr)))  # this is the class chosen by the forest output
        npout = reshape(array(out), X.shape)
        # print contour with training data
        plt.figure(1)
        plt.clf()  # clear the plot
        plt.hold(True)  # overplot on
        levels = arange(-0.5, self.numclasses+1, 1)
        plt.contourf(X, Y, npout, levels, colors=coloropts[0:self.numclasses])  # plot the contour
        # plot the actual instances on the same plot to see how accurate the model is
        for key, value in allpoints.items():
            plt.plot(allpoints[key][::2], allpoints[key][1::2], trainPlots[key], label="Class "+str(key+self.minclass))
        plt.ylabel('X2')
        plt.xlabel('X1')
        plt.legend(numpoints=1)
        plt.title("DT Classification Regions with " + datatype)
        plt.xlim([minval_x-ten_percent_x, maxval_x+ten_percent_x])
        plt.ylim([minval_y-ten_percent_y, maxval_y+ten_percent_y])
        plt.show()

    def print_to_file(self, label):
        """
        pickle this forest for later use
        :param label: name of file to print to
        :return: None
        """
        label += '.pkl'
        print("Printing to " + label)
        output = open(label, 'wb')
        pickle.Pickler(output, protocol=pickle.HIGHEST_PROTOCOL).dump(self)
        output.close()

    def __str__(self):
        """
        :return: string representation of this forest
        """
        name = ""
        for a_tree in self.trees:
            name += str(a_tree)
        return name
#end Forest class
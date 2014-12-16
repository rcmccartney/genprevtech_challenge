__author__ = 'Robert McCartney'


class Node(object):
    """A node of a tree inside a random forest"""

    def __init__(self, prob=None, splitter=None, left=None, right=None):
        """
        :param prob: Class decision if this node is a leaf
        :param splitter: Splitter class that knows how this node was trained
        :param left: node that is the left child
        :param right: node that is the right child
        :return: None
        """
        self.left = left
        self.right = right
        self.splitter = splitter
        self.prob = prob

    def get_direction(self, instance):
        """
        Determine where this one sample goes at this node, either left or right if
        this is not a leaf node
        :param instance: data to test
        :return: boolean for direction decision
        """
        # splitter is the parameterized function given to this node by the weak learner model
        #returns false for left, true for right
        return self.splitter.split(instance)

    def __str__(self):
        """
        :return: string representaion of this node and it's split function
        """
        return "Prob: " + str(self.prob) + " Splitter: " + str(self.splitter)
# end Node class


class Tree(object):
    """A tree of a random forest"""

    def __init__(self, samples, depth_limit, weak_learner):
        """
        Initialize the tree, and turn on bagging if required
        :param samples: data used by this tree
        :param depth_limit: limit of the size of each tree
        :param weak_learner: weak learner to use for splitting data
        :return: None
        """
        # data is not stored at the tree, so after the constructor finishes it is tossed
        self.depth_limit = depth_limit
        self.weak_learner = weak_learner
        self.root = self.add_node(samples, 0, None, self.weak_learner.calc_distr(samples))

    def get_instance_distr(self, instance):
        """
        This method is called by Forest to test one data instance
        :param instance: data to test
        :return: the distribution found for this instance in this tree
        """
        return self.get_tree_distr_recur(instance, self.root)

    def get_tree_distr_recur(self, instance, node):
        """
        Recursively find the leaf node that matches this instance and return
        that classification distribution. Each node has its own Splitter class
        that was given to it by the Split Strategy class when the node was created
        :param instance: data item to classify
        :param node: current location in the tree
        :return: distribution this tree decides after recursion ends
        """
        if node.prob is not None:  # then you are at a leaf
            return node.prob
        else:  # recurse down
            if node.get_direction(instance):  # true for go right
                return self.get_tree_distr_recur(instance, node.right)
            else:
                return self.get_tree_distr_recur(instance, node.left)

    def add_node(self, samples, depth_count, parentdistr, mydistr):
        """
        Recursively add nodes to the tree until reaching the depth limit
        this is the training portion of the random forest
        :param samples: data at this node
        :param depth_count: how deep into the recursion we have gone
        :param parentdistr: the distribution of the parent, used if we run out of data here
        :return: Node at this level of recursion
        """
        if not samples:  # no more data to split on
            return Node(prob=parentdistr)
        elif depth_count == self.depth_limit:  # this is a leaf node, stop recursing
            return Node(prob=mydistr)
        else:
            splitter, data_left, data_right, left_d, right_d = self.weak_learner.calc_split(samples)
            return Node(splitter=splitter,
                        left=self.add_node(data_left, depth_count+1, mydistr, left_d),
                        right=self.add_node(data_right, depth_count+1, mydistr, right_d))

    def __str__(self):
        """
        :return: string representation of this Tree
        """
        return "Root: " + self.print_dfs(self.root, "")

    def print_dfs(self, node, spaces):
        """
        :param node: node to be printed on tree
        :param spaces: for print formatting
        """
        name = str(node) + "\n"
        spaces += "  "
        if node.left is not None:
            name += spaces + "LC: " + self.print_dfs(node.left, spaces)
        if node.right is not None:
            name += spaces + "RC: " + self.print_dfs(node.right, spaces)
        return name
#end Tree class

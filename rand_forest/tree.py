__author__ = 'mccar_000'


class Node(object):
    """A node of a tree inside a random forest"""

    def __init__(self, prob=None, splitter=None, left=None, right=None):
        self.left = left
        self.right = right
        self.splitter = splitter
        self.prob = prob

    def get_direction(self, instance):  # where does this one sample go at this node
        # splitter is the parameterized function given to this node by the
        # weak learner model
        assert self.splitter is not None, "This node has not learned how to split, did you train it properly?"
        #returns 0 for left, 1 for right
        return self.splitter.split(instance)

    def __str__(self):
        return "Prob: " + str(self.prob) + " Splitter: " + str(self.splitter)
# end Node class


class Tree(object):
    """A tree of a random forest"""

    def __init__(self, samples, depth_limit, weak_learner):
        self.depth_limit = depth_limit
        self.weak_learner = weak_learner
        self.root = self.add_node(samples, 0, self.weak_learner.calc_distr(samples))

    def get_instance_distr(self, instance):  # called by Forest to test one data instance
        return self.get_tree_distr_recur(instance, self.root)

    # recursively does to leaf of tree & gets distribution for this piece of data
    def get_tree_distr_recur(self, instance, node):
        if node.prob is not None:  # then you are at a leaf
            return node.prob
        else:  # recurse down
            if node.get_direction(instance):  # true for go right
                return self.get_tree_distr_recur(instance, node.right)
            else:
                return self.get_tree_distr_recur(instance, node.left)

    def add_node(self, samples, depth_count, parentdistr):
        if not samples:  # no more data to split on
            return Node(prob=parentdistr)
        elif depth_count == self.depth_limit:  # this is a leaf node, stop recursing
            return Node(prob=self.weak_learner.calc_distr(samples))
        else:
            splitter, data_left, data_right = self.weak_learner.calc_split(samples)
            return Node(splitter=splitter,
                        left=self.add_node(data_left, depth_count+1, self.weak_learner.calc_distr(samples)),
                        right=self.add_node(data_right, depth_count+1, self.weak_learner.calc_distr(samples)))

    def __str__(self):
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

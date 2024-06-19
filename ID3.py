from math import log2
from collections import Counter
import numpy as np
import pandas as pd


class Node:
    '''
    Tree node class with nodes storing a probability vector
    which is the distribution of the classes of the response 
    variable that flow to that node.  If nodes are internal,
    they also have an attribute storing the attribute that 
    they split along and a dictionary with keys being the 
    various values of that attribute and values being the 
    corresponding children nodes.  
    '''

    def __init__(self, class_proba):
        self.class_proba = class_proba

    def set_children(self, split_attribute, attribute_value_to_vector):
        self.split_attribute = split_attribute
        self.children = {}
        for k, v in attribute_value_to_vector.items():
            self.children[k] = Node(v)


class ID3:
    '''
    Decision tree with methods:
        - fit
        - predict_proba
        - predict
    Fitting includes hyperparameters to limit the maximum depth
    of the tree and require a certain amount of data and 
    prediction class uncertainty in order to split leaves.  
    '''

    def _col_entropy(df, col):
        '''
        Computes the entropy of the column ``col`` of the 
        dataframe ``df``.
        '''
        class_counts = dict(df[col].value_counts())
        N = len(df[col])
        entropy = 0.0
        for v in class_counts.values():
            p = v / N
            entropy -= p * log2(p)
        return entropy

    def _classes_to_proba(v, K):
        '''
        Takes the array ``v`` consisting of classes and 
        returns the corresponding probability vector assuming
        that there are ``K`` classes in total and that the entries 
        of ``v`` are between 0 and K-1, inclusive.  
        '''
        N = len(v)
        v = Counter(v)
        res = [0.0] * K
        for k, v in v.items():
            res[k] += v/N
        return res

    def _count_to_proba(v):
        '''
        Takes an array with the i-th element being the count of 
        object i.  Returns the corresponding probability vector.   
        '''
        N = sum(v)
        return [x/N for x in v]

    def _find_split_leaf_attribute(self, cols, rows):
        '''
        Given the columns and rows of data that are under consideration,
        determine which attribute maximizes information gain when the 
        data is split at that attribute (this is equivalent to minimizing the 
        expected information required for the tree with that attribute split). 
        '''
        cols, rows = list(cols), list(rows)
        best = cols[0]
        lowest_expected = float('inf')
        N = len(rows)

        for col in cols:
            expected = 0.0
            df = self.X[col].iloc[rows]
            Y = self.Y.iloc[rows]
            value_counts = df.value_counts()

            for value, count in value_counts.items():
                Y_value = Y.loc[df[df == value].index]
                expected += count / N * ID3._col_entropy(Y_value,
                                                         Y_value.columns[0])

            if expected <= lowest_expected:
                best = col
                lowest_expected = expected

        return best

    # Uses the minimum entropy to split.
    # def _find_split_leaf_attribute(self, cols, rows):
    #    '''
    #    Return the attribute that ``split_leaf`` should be split along.
    #    '''
    #    return min(cols,
    #               key=lambda col : ID3._col_entropy(self.X.iloc[list(rows)], col))

    def _split_leaf(self, leaf, attribute, rows, K):
        '''
        Splits the leaf along ``attribute`` by calling the ``set_children``
        method of the leaf.  Initializes the children and returns a 
        dictionary from the new leaves of ``leaf`` mapping to the data that 
        flows to each by looking at rows.  ``X`` and ``Y`` is the data 
        flowing to ``leaf`` before the split and it is used to determine the 
        values of the returned dictionary. 
        '''
        rows = list(rows)
        # Values of are counts of the y values.
        attribute_value_to_counts = {}

        X = self.X[attribute].iloc[rows]
        Y = self.Y.loc[rows][0]
        for x, y in zip(X, Y):

            if x not in attribute_value_to_counts:
                attribute_value_to_counts[x] = [0] * K
                attribute_value_to_counts[x][y] += 1
            else:
                attribute_value_to_counts[x][y] += 1

        # Convert counts to probability vectors.
        for k, v in attribute_value_to_counts.items():
            attribute_value_to_counts[k] = ID3._count_to_proba(v)

        leaf.set_children(attribute, attribute_value_to_counts)

    def _push_data(self, leaf, cols, rows):
        '''
        Push the data from ``leaf`` down to the children.  
        Strip off the split attribute.
        '''
        res = {}
        cols_child = cols.copy()
        cols_child.remove(leaf.split_attribute)
        for attribute, child in leaf.children.items():
            X_child = self.X.loc[self.X[leaf.split_attribute] == attribute]
            rows_child = rows.intersection(set(X_child.index))
            res[child] = (cols_child, rows_child)
        return res

    def _is_certain(v, threshold):
        '''
        Returns if the vector ``v`` (for us a probability vector)
        has a value greater than or equal to ``threshold``
        '''
        return any(x >= threshold for x in v)

    def fit(self, X_train, Y_train,
            max_depth=5,
            min_split=2,
            certainty_threshold=0.95):
        '''
        Forms the decision tree using the ID3 algorithm for finding 
        the attributes to split the leaves along.  Only builds a tree
        of maximum depth ``max_depth``, only splits leaves with at 
        least ``min_split`` data points flowing to them that have a 
        prediction probability vector with no value greater than or 
        equal to ``certainty_threshold.
        '''

        if not len(X_train) == len(Y_train):
            raise Exception("Must have same number of samples \
                            for independent and dependent variables.")

        if not len(X_train) >= min_split:
            raise Exception("Too few datapoints.")

        X_train = pd.DataFrame(X_train)
        Y_train = pd.DataFrame(Y_train)

        # Number of classes in target.
        K = len(Y_train.value_counts())

        if K == 1:
            raise Exception("Only one class represented by the \
                            dependent variable.")

        self.X = X_train.copy().reset_index(drop=True)
        self.Y = Y_train.copy().reset_index(drop=True)

        # Initialize root.
        root_class_proba = ID3._classes_to_proba(
            Y_train.iloc[:, 0].to_list(), K)
        root = Node(root_class_proba)
        self.root = root

        # Do not split if there is not enough class diversity.
        if ID3._is_certain(root.class_proba, certainty_threshold):
            return

        # When a leaf is a key in ``leaves_to_data``, it will be split.
        # The value for that a tuple of the columns and rows that
        # correspond to that leaf.

        cols = set(self.X.columns)
        rows = set(self.X.index)
        leaves_to_data = {root: (cols, rows)}

        # Each iteration splits a leaf.
        while max_depth > 0 and len(leaves_to_data) > 0:
            max_depth -= 1
            new_leaves = {}
            # Split on the leaves.
            for leaf, (cols, rows) in leaves_to_data.items():
                # Find attribute to split.
                attribute = self._find_split_leaf_attribute(cols, rows)
                # Split the leaf.
                self._split_leaf(leaf, attribute, rows, K)
                # Push data to the new leaves.
                new_leaves.update(self._push_data(leaf, cols, rows))

            # Filter so there is at least ``min_split`` data points
            # in the leaves that we split and they have enough
            # class diversity as dictated by ``certainty_threshold``.
            leaves_to_data = {k: v for k, v in new_leaves.items()
                              if len(v[1]) >= min_split and
                              not ID3._is_certain(k.class_proba,
                                                  certainty_threshold)}

    def _predict_proba_row(row_data, root):
        '''
        Traverses the tree and finds the appropriate node.  
        Returns the class prediction probability.  
        '''
        # Check if you are a leaf.
        if "children" not in vars(root):
            return pd.Series(root.class_proba)

        # Or if we have an attribute that is not in the tree.
        if row_data[root.split_attribute] not in root.children:
            return pd.Series(root.class_proba)

        child = root.children[row_data[root.split_attribute]]
        return ID3._predict_proba_row(row_data, child)

    def predict_proba(self, X_test):
        '''
        Return DataFrame of prediction probabilities with the same index 
        as ``X_test``.
        '''
        return X_test.apply(lambda row_data: ID3._predict_proba_row(row_data,
                                                                    self.root),
                            axis=1)

    def _idxmax(nums):
        '''
        Determine the first index in nums that contains
        the maximum value.
        '''
        m = float("-inf")
        best = 0
        for i, x in enumerate(nums):
            if x > m:
                best = i
                m = x
        return best

    def predict(self, X_test):
        '''
        Return DataFrame of predictions with the same index 
        as ``X_test``.
        '''
        return ID3.predict_proba(self, X_test).apply(ID3._idxmax,
                                                     axis=1)

import graphlab as gl
import json

loans = gl.SFrame('lending-club-data.gl')
loans['safe_loans'] = loans['bad_loans'].apply(lambda bad: 1 if bad == 0 else -1)

features = ['grade',  # grade of the loan
            'term',  # the term of the loan
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'emp_length',  # number of years of employment
            ]
target = 'safe_loans'  # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
print('%s loans in total after balance' % len(loans_data))

loans_data = risky_loans.append(safe_loans)

categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

train_data, test_data = loans_data.random_split(.8, seed=1)


def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    # Count the number of 1's (safe loans)
    positives = sum(labels_in_node == 1)
    # Count the number of -1's (risky loans)
    negatives = sum(labels_in_node == -1)
    # Return the number of mistakes that the majority classifier makes.
    return min(positives, negatives)

# Test case 1
example_labels = gl.SArray([-1, -1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 1 failed... try again!'

# Test case 2
example_labels = gl.SArray([-1, -1, 1, 1, 1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'

# Test case 3
example_labels = gl.SArray([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'


def best_splitting_feature(data, features, target):
    best_feature = None  # Keep track of the best feature
    best_error = 10  # Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]

        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]

        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this!
        # (It was called intermediate_node_num_mistakes)
        left_mistakes = intermediate_node_num_mistakes(left_split[target])

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])

        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far,
        # store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature
            best_error = error

    return best_feature  # Return the best feature we found


def create_leaf(target_values):
    # Create a leaf node
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}

    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    leaf['prediction'] = 1 if num_ones > num_minus_ones else -1
    return leaf


def decision_tree_create(data, features, target, current_depth=0, max_depth=10):
    remaining_features = features[:]  # Make a copy of the features.

    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached."
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)

    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if len(remaining_features) == 0:
        print "Stopping condition 2 reached."
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)

        # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, remaining_features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(
        left_split, remaining_features, target, current_depth + 1, max_depth)
    right_tree = decision_tree_create(
        right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}


def classify(tree, x, annotate=False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


print test_data[0]
all_features = ['grade.A',
                'grade.B',
                'grade.C',
                'grade.D',
                'grade.E',
                'grade.F',
                'grade.G',
                'term. 36 months',
                'term. 60 months',
                'home_ownership.MORTGAGE',
                'home_ownership.OTHER',
                'home_ownership.OWN',
                'home_ownership.RENT',
                'emp_length.1 year',
                'emp_length.10+ years',
                'emp_length.2 years',
                'emp_length.3 years',
                'emp_length.4 years',
                'emp_length.5 years',
                'emp_length.6 years',
                'emp_length.7 years',
                'emp_length.8 years',
                'emp_length.9 years',
                'emp_length.< 1 year',
                'emp_length.n/a']
# my_decision_tree = decision_tree_create(train_data, all_features, target, max_depth=6)
# json.dump(my_decision_tree, open('module-5-decision-tree-assignment-2.json', 'w'))
my_decision_tree = json.load(open('module-5-decision-tree-assignment-2.json'))
print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0])

print('===== QUIZ 1~3 =====')
classify(my_decision_tree, test_data[0], annotate=True)


def evaluate_classification_error(tree, data, target):
    evaluation = data.copy()
    evaluation['prediction'] = data.apply(lambda x: classify(tree, x))
    incorrect = evaluation[evaluation[target] != evaluation['prediction']]
    return float(len(incorrect)) / len(evaluation)


print('===== QUIZ 4 =====')
print('Error rate of my_decision_tree on test data is %s' %
      evaluate_classification_error(my_decision_tree, test_data, target))


def print_stump(tree, name='root'):
    split_name = tree['splitting_feature']  # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
          % (('leaf, label: ' + str(tree['left']['prediction'])
              if tree['left']['is_leaf'] else 'subtree'),
             ('leaf, label: ' + str(tree['right']['prediction'])
              if tree['right']['is_leaf'] else 'subtree'))


print('===== QUIZ 5 =====')
print_stump(my_decision_tree)

print('===== QUIZ 6 =====')
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])

print('===== QUIZ 7 =====')
print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree['right']['right'], my_decision_tree['right']['splitting_feature'])

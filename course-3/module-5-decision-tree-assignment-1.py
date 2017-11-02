import graphlab as gl

loans = gl.SFrame('lending-club-data.gl')
loans['safe_loans'] = loans['bad_loans'].apply(lambda bad: 1 if bad == 0 else -1)

features = ['grade',  # grade of the loan
            'sub_grade',  # sub-grade of the loan
            'short_emp',  # one year or less of employment
            'emp_length_num',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'dti',  # debt to income ratio
            'purpose',  # the purpose of the loan
            'term',  # the term of the loan
            'last_delinq_none',  # has borrower had a delinquincy
            'last_major_derog_none',  # has borrower had 90 day or worse rating
            'revol_util',  # percent of available credit being used
            'total_rec_late_fee',  # total late fees received to day
            ]

target = 'safe_loans'  # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == 1]
print('There are {0} safe loans out of {1} total, safe rate {2}'.format(
    len(safe_loans_raw), len(loans), float(len(safe_loans_raw)) / len(loans)
))
risky_loans_raw = loans[loans[target] == -1]
print('There are %s risky loans out of %s total, risky rate %s' %
      (len(risky_loans_raw), len(loans), float(len(risky_loans_raw)) / len(loans)))

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
print(len(loans_data))

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

train_data, validation_data = loans_data.random_split(.8, seed=1)

decision_tree_model = gl.decision_tree_classifier.create(
    dataset=train_data, target=target, max_depth=6,
    validation_set=None, verbose=False)

small_model = gl.decision_tree_classifier.create(
    dataset=train_data, target=target, max_depth=2,
    validation_set=None, verbose=False)

validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

print('===== QUIZ 1 =====')
sample_prediction = decision_tree_model.predict(dataset=sample_validation_data)
for i in range(len(sample_validation_data)):
    print('Prediction: %s; actual: %s' % (sample_prediction[i], sample_validation_data[i][target]))

print('===== QUIZ 2 =====')
sample_prediction = decision_tree_model.predict(
    dataset=sample_validation_data, output_type='probability')
for i in range(len(sample_validation_data)):
    print('Prediction: %s; actual: %s' % (sample_prediction[i], sample_validation_data[i][target]))

print('===== QUIZ 3 =====')
sample_prediction = small_model.predict(
    dataset=sample_validation_data, output_type='probability')
for i in range(len(sample_validation_data)):
    print('Prediction: %s; actual: %s' % (sample_prediction[i], sample_validation_data[i][target]))

print('===== QUIZ 4 =====')
print(sample_validation_data[1])
print('My prediction seems to be -1')
# small_model.show(view='Tree')
# raw_input('...')

def accuracy(model, dataset, target):
    predictions = model.predict(dataset)
    dataset_with_prediction = dataset.copy()
    dataset_with_prediction['prediction'] = predictions
    correct_predictions = dataset_with_prediction[
        dataset_with_prediction[target] == dataset_with_prediction['prediction']]
    return float(len(correct_predictions)) / len(dataset_with_prediction)


print('===== QUIZ 5 =====')
print('Accuracy of decision_tree_model on training set: %s' %
      accuracy(decision_tree_model, train_data, target))
print('Accuracy of decision_tree_model on validation set: %s' %
      accuracy(decision_tree_model, validation_data, target))

print('===== QUIZ 6 =====')
big_model = gl.decision_tree_classifier.create(
    dataset=train_data, target=target, max_depth=10,
    validation_set=None, verbose=False)
print('Accuracy of big_model on training set: %s' %
      accuracy(big_model, train_data, target))
print('Accuracy of big_model on validation set: %s' %
      accuracy(big_model, validation_data, target))

def cost(model, dataset, target):
    predictions = model.predict(dataset)
    dataset_with_prediction = dataset.copy()
    dataset_with_prediction['prediction'] = predictions
    false_predictions = dataset_with_prediction[
        dataset_with_prediction[target] != dataset_with_prediction['prediction']]
    false_negative = false_predictions[false_predictions['prediction'] == -1]
    false_positive = false_predictions[false_predictions['prediction'] == 1]
    return len(false_negative) * 10000 + len(false_positive) * 20000

print('===== QUIZ 7 =====')
print('Cost of decision_tree_model on validation data is $%s' %
      cost(decision_tree_model, validation_data, target))

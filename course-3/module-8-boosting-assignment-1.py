import graphlab

loans = graphlab.SFrame('lending-club-data.gl')
loans['safe_loans'] = loans['bad_loans'].apply(lambda bad: 1 if bad == 0 else -1)

target = 'safe_loans'
features = ['grade',  # grade of the loan (categorical)
            'sub_grade_num',  # sub-grade of the loan as a number from 0 to 1
            'short_emp',  # one year or less of employment
            'emp_length_num',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'dti',  # debt to income ratio
            'purpose',  # the purpose of the loan
            'payment_inc_ratio',  # ratio of the monthly payment to income
            'delinq_2yrs',  # number of delinquincies
            'delinq_2yrs_zero',  # no delinquincies in last 2 years
            'inq_last_6mths',  # number of creditor inquiries in last 6 months
            'last_delinq_none',  # has borrower had a delinquincy
            'last_major_derog_none',  # has borrower had 90 day or worse rating
            'open_acc',  # number of open credit accounts
            'pub_rec',  # number of derogatory public records
            'pub_rec_zero',  # no derogatory public records
            'revol_util',  # percent of available credit being used
            'total_rec_late_fee',  # total late fees received to day
            'int_rate',  # interest rate of the loan
            'total_rec_int',  # interest received to date
            'annual_inc',  # annual income of borrower
            'funded_amnt',  # amount committed to the loan
            'funded_amnt_inv',  # amount committed by investors for the loan
            'installment',  # monthly payment owed by the borrower
            ]

loans, loans_with_na = loans[[target] + features].dropna_split()

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows)

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed=1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

train_data, validation_data = loans_data.random_split(.8, seed=1)

model_5 = graphlab.boosted_trees_classifier.create(
    train_data, target=target, features=features, max_iterations=5,
    validation_set=None, verbose=False)

# Select all positive and negative examples.
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

# Select 2 examples from the validation set for positive & negative loans
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

# Append the 4 examples into a single dataset
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

print('===== QUIZ 1~2 =====')
for data_point in sample_validation_data:
    print('Actual: %s; Predicted: %s; Probability: %s' %
          (data_point[target], model_5.predict(data_point),
           model_5.predict(data_point, output_type='probability')))


def cost(model, dataset, target):
    predictions = model.predict(dataset)
    dataset_with_prediction = dataset.copy()
    dataset_with_prediction['prediction'] = predictions
    false_predictions = dataset_with_prediction[
        dataset_with_prediction[target] != dataset_with_prediction['prediction']]
    false_negative = false_predictions[false_predictions['prediction'] == -1]
    false_positive = false_predictions[false_predictions['prediction'] == 1]
    return len(false_negative) * 10000 + len(false_positive) * 20000


def accuracy(model, dataset, target):
    predictions = model.predict(dataset)
    dataset_with_prediction = dataset.copy()
    dataset_with_prediction['prediction'] = predictions
    correct_predictions = dataset_with_prediction[
        dataset_with_prediction[target] == dataset_with_prediction['prediction']]
    return float(len(correct_predictions)) / len(dataset_with_prediction)


def count_false_positives(model, dataset, target):
    dataset_with_prediction = dataset.copy()
    dataset_with_prediction['prediction'] = model.predict(dataset)
    positives = dataset_with_prediction[dataset_with_prediction['prediction'] == 1]
    false_positives = positives[positives[target] != 1]
    return len(false_positives)


print('===== QUIZ 3 =====')
print('False positives: %s' % count_false_positives(model_5, validation_data, target))

print('===== QUIZ 4 =====')
print('Cost of model_5 on validation_data: %s' %
      cost(model_5, validation_data, target))

print('===== QUIZ 5 =====')
validation_with_prediction = validation_data.copy()
validation_with_prediction['prediction'] = model_5.predict(validation_data, 'probability')
validation_with_prediction = validation_with_prediction.sort('prediction', ascending=False)
for i in range(5):
    data_point = validation_with_prediction[i]
    print('Probability: %s; Grade: %s' % (data_point['prediction'], data_point['grade']))

models = {}
for max_iterations in [10, 50, 100, 200, 500]:
    # models[max_iterations] = graphlab.boosted_trees_classifier.create(
    #     train_data, target=target, features=features, max_iterations=max_iterations,
    #     validation_set=None, verbose=False)
    # models[max_iterations].save('module-8-1-%s' % max_iterations)
    models[max_iterations] = graphlab.load_model('module-8-1-%s' % max_iterations)

print('===== QUIZ 6~7 =====')
for max_iterations in models:
    print('Evaluating model with max_iterations %s:' % max_iterations)
    print(models[max_iterations].evaluate(validation_data))

print('===== QUIZ 8~9 =====')
for max_iterations in models:
    print('Errors of model with max_iterations %s: \ntraining: %s; validation: %s' %
          (max_iterations,
           1 - accuracy(models[max_iterations], train_data, target),
           1 - accuracy(models[max_iterations], validation_data, target)))

from math import sqrt

import graphlab as gl
import numpy as np

sales = gl.SFrame('kc_house_data.gl')
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms'] ** 2
sales['floors'] = sales['floors'].astype(float)
sales['floors_square'] = sales['floors'] ** 2
print(sales[0])


def rss(model, dataset, target):
    predictions = model.predict(dataset)
    return ((dataset[target] - predictions) ** 2).sum()


print('===== QUIZ 1 =====')
all_features = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated']

model_all = gl.linear_regression.create(
    dataset=sales, target='price', features=all_features, l1_penalty=1e10, l2_penalty=0,
    validation_set=None, verbose=False)
model_all.get('coefficients').print_rows(num_rows=20)

print('===== QUIZ 2 =====')
(training_and_validation, testing) = sales.random_split(.9, seed=1)
(training, validation) = training_and_validation.random_split(0.5, seed=1)

for l1_penalty in np.logspace(1, 7, num=13):
    model = gl.linear_regression.create(
        dataset=training, target='price', features=all_features,
        l1_penalty=l1_penalty, l2_penalty=0,
        validation_set=None, verbose=False)
    predictions = model.predict(validation)
    rss_of_current_model = rss(model, validation, 'price')
    print('Validation RSS of model with L1 penalty [{0}] : {1}'.format(
        l1_penalty, rss_of_current_model))
print('Selected L1 penalty is 10')

print('===== QUIZ 3 =====')
selected_model = gl.linear_regression.create(
    dataset=training, target='price', features=all_features,
    l1_penalty=10.0, l2_penalty=0,
    validation_set=None, verbose=False)
selected_model.get('coefficients').print_rows(num_rows=20)
print('RSS with selected L1 penalty on test data is {0}'.format(
    rss(selected_model, testing, 'price')))

print('===== QUIZ 4 =====')
max_non_zeros = 7
for l1_penalty in np.logspace(8, 10, num=20):
    model = gl.linear_regression.create(
        dataset=training, target='price', features=all_features,
        l1_penalty=l1_penalty, l2_penalty=0,
        validation_set=None, verbose=False)
    non_zeros = model['coefficients']['value'].nnz()
    print('L1 penalty: {0}; Non-zero weights: {1}'.format(l1_penalty, non_zeros))
print('l1_penalty_min: 2976351441.63')
print('l1_penalty_max: 3792690190.73')

print('===== QUIZ 5 =====')
l1_penalty_min = 2976351441.63
l1_penalty_max = 3792690190.73
for l1_penalty in np.linspace(l1_penalty_min, l1_penalty_max, 20):
    model = gl.linear_regression.create(
        dataset=training, target='price', features=all_features,
        l1_penalty=l1_penalty, l2_penalty=0,
        validation_set=None, verbose=False)
    non_zeros = model['coefficients']['value'].nnz()
    print('L1 penalty: {0}; Non-zero weights: {1}; RSS: {2}'.format(
        l1_penalty, non_zeros, rss(model, validation, 'price')))
print('Answer: L1 penalty: 3448968612.16; Non-zero weights: 7; RSS: 1.04693748875e+15')

print('===== QUIZ 6 =====')
model = gl.linear_regression.create(
    dataset=training, target='price', features=all_features,
    l1_penalty=3448968612.16, l2_penalty=0,
    validation_set=None, verbose=False)
model.get('coefficients').print_rows(num_rows=20)

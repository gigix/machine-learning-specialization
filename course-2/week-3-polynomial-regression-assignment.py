import graphlab as gl

house_data = gl.SFrame('kc_house_data.gl').sort(['sqft_living', 'price'])
print(house_data)


def polynomial(dataset, feature, max_degree):
    result = gl.SFrame()
    for degree in range(1, max_degree + 1):
        result['{0}_power_{1}'.format(feature, degree)] = dataset[feature] ** degree
    return result


# polynomial_dataset = polynomial(house_data, 'sqft_living', 3)
# print(polynomial_dataset)

# poly_datasets = []
# for max_degree in range(0, 3):
#     poly_data = polynomial(house_data, feature='sqft_living', max_degree=max_degree + 1)
#     poly_data['price'] = house_data['price']
#     poly_datasets.append(poly_data)
#
# poly_data_15 = polynomial(house_data, feature='sqft_living', max_degree=15)
# poly_data_15['price'] = house_data['price']
#
# model_1 = gl.linear_regression.create(
#     poly_data_15, target='price', features=['sqft_living_power_1'],
#     validation_set=None, verbose=False)
# model_2 = gl.linear_regression.create(
#     poly_data_15, target='price', features=['sqft_living_power_1', 'sqft_living_power_2'],
#     validation_set=None, verbose=False)
# model_3 = gl.linear_regression.create(
#     poly_data_15, target='price',
#     features=['sqft_living_power_1', 'sqft_living_power_2', 'sqft_living_power_3'],
#     validation_set=None, verbose=False)
#
# model_15 = gl.linear_regression.create(
#     poly_data_15, target='price',
#     features=[
#         'sqft_living_power_1', 'sqft_living_power_2', 'sqft_living_power_3',
#         'sqft_living_power_4', 'sqft_living_power_5', 'sqft_living_power_6',
#         'sqft_living_power_7', 'sqft_living_power_8', 'sqft_living_power_9',
#         'sqft_living_power_10', 'sqft_living_power_11', 'sqft_living_power_12',
#         'sqft_living_power_13', 'sqft_living_power_14', 'sqft_living_power_15'
#     ],
#     validation_set=None, verbose=False)

# import matplotlib.pyplot as plt
#
# plt.plot(
#     poly_data_15['sqft_living_power_1'], poly_data_15['price'], '.',
#     poly_data_15['sqft_living_power_1'], model_1.predict(poly_data_15), '-',
#     poly_data_15['sqft_living_power_1'], model_2.predict(poly_data_15), '-',
#     poly_data_15['sqft_living_power_1'], model_3.predict(poly_data_15), '-',
#     poly_data_15['sqft_living_power_1'], model_15.predict(poly_data_15), '-'
# )
# plt.show()

def power_features(base_feature, max_degree):
    result = []
    for degree in range(1, max_degree + 1):
        result.append('{0}_power_{1}'.format(base_feature, degree))
    return result


print('===== QUIZ 1 & 2 =====')
first_half, second_half = house_data.random_split(.5, seed=0)
set_1, set_2 = first_half.random_split(.5, seed=0)
set_3, set_4 = second_half.random_split(.5, seed=0)
for dataset in [set_1, set_2, set_3, set_4]:
    poly_data_15 = polynomial(dataset, feature='sqft_living', max_degree=15)
    poly_data_15['price'] = dataset['price']
    model_15 = gl.linear_regression.create(
        poly_data_15, target='price', features=power_features('sqft_living', 15),
        validation_set=None, verbose=False
    )
    model_15.get('coefficients').print_rows(num_rows=16)

print('===== QUIZ 3 =====')
training_and_validation_set, test_set = house_data.random_split(.9, seed=1)
training_set, validation_set = training_and_validation_set.random_split(.5, seed=1)
models = []
for i in range(0, 15):
    max_degree = i + 1
    poly_data_train = polynomial(training_set, feature='sqft_living', max_degree=15)
    poly_data_train['price'] = training_set['price']
    model = gl.linear_regression.create(
        dataset=poly_data_train, target='price', features=power_features('sqft_living', max_degree),
        validation_set=None, verbose=False
    )
    models.append(model)
    poly_data_validation = polynomial(validation_set, 'sqft_living', 15)
    predictions = model.predict(poly_data_validation)
    rss = ((predictions - validation_set['price']) ** 2).sum()
    print('RSS of power {0} model on validation set is: {1}'.format(max_degree, rss))

print('===== QUIZ 4 =====')
poly_data_test = polynomial(test_set, 'sqft_living', 15)
for i in range(0, 15):
    selected_model = models[i]
    predictions = selected_model.predict(poly_data_test)
    rss = ((test_set['price'] - predictions) ** 2).sum()
    print('RSS of power {0} model on test set is: {1}'.format(i + 1, rss))

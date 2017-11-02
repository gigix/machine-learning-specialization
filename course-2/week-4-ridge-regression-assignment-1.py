import graphlab as gl


def polynomial(dataset, feature, max_degree):
    result = gl.SFrame()
    for degree in range(1, max_degree + 1):
        result['{0}_power_{1}'.format(feature, degree)] = dataset[feature] ** degree
    return result


def power_features(base_feature, max_degree):
    result = []
    for degree in range(1, max_degree + 1):
        result.append('{0}_power_{1}'.format(base_feature, degree))
    return result


house_data = gl.SFrame('kc_house_data.gl').sort(['sqft_living', 'price'])
poly_15_data = polynomial(dataset=house_data, feature='sqft_living', max_degree=15)
poly_15_data['price'] = house_data['price']

plain_model = gl.linear_regression.create(dataset=poly_15_data, target='price',
                                          features=power_features('sqft_living', 15),
                                          l1_penalty=0, l2_penalty=0,
                                          validation_set=None, verbose=False)
plain_model.get('coefficients').print_rows(num_rows=16)

(semi_split1, semi_split2) = house_data.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

print('===== QUIZ 1 =====')
ridge_model = gl.linear_regression.create(dataset=poly_15_data, target='price',
                                          features=power_features('sqft_living', 15),
                                          l1_penalty=0, l2_penalty=1.5e-5,
                                          validation_set=None, verbose=False)
ridge_model.get('coefficients').print_rows(num_rows=16)

print('===== QUIZ 2 & 3 =====')
for dataset in [set_1, set_2, set_3, set_4]:
    poly_data_15 = polynomial(dataset, feature='sqft_living', max_degree=15)
    poly_data_15['price'] = dataset['price']
    model_15 = gl.linear_regression.create(
        poly_data_15, target='price',
        l1_penalty=0, l2_penalty=1e-5, validation_set=None, verbose=False
    )
    print('Coefficient of power_1: {0}'.format(model_15.get('coefficients')[1]))
    # import matplotlib.pyplot as plt
    # plt.plot(
    #     poly_data_15['sqft_living_power_1'], poly_data_15['price'], '.',
    #     poly_data_15['sqft_living_power_1'], model_15.predict(poly_data_15), '.'
    # )
    # plt.show()

print('==== QUIZ 4 & 5 ====')
for dataset in [set_1, set_2, set_3, set_4]:
    poly_data_15 = polynomial(dataset, feature='sqft_living', max_degree=15)
    poly_data_15['price'] = dataset['price']
    model_15 = gl.linear_regression.create(
        poly_data_15, target='price',
        l1_penalty=0, l2_penalty=1e5, validation_set=None, verbose=False
    )
    print('Coefficient of power_1: {0}'.format(model_15.get('coefficients')[1]))
    # import matplotlib.pyplot as plt
    # plt.plot(
    #     poly_data_15['sqft_living_power_1'], poly_data_15['price'], '.',
    #     poly_data_15['sqft_living_power_1'], model_15.predict(poly_data_15), '.'
    # )
    # plt.show()


def k_fold_cross_validation(k, l2_penalty, data, target):
    n = len(data)
    rssTotal = 0.0
    for i in xrange(k):
        # Validation subset of kth iteration
        start = (n * i) / k
        end = (n * (i + 1)) / k - 1
        validation = data[start:end + 1]
        # Train data is the rest appended
        first = data[0:start]
        last = data[end + 1:n]
        train = first.append(last)
        # Fit model and calculate rss
        model = gl.linear_regression.create(dataset=train, target=target, l2_penalty=l2_penalty,
                                            validation_set=None, verbose=False)
        prediction = model.predict(validation)
        rssTotal += ((prediction - validation['price']) ** 2).sum()
    return rssTotal / k


print('===== QUIZ 6 =====')
train_data, test_data = house_data.random_split(.9, seed=1)
train_data_shuffled = gl.toolkits.cross_validation.shuffle(train_data, random_seed=1)
poly_15_data = polynomial(dataset=train_data_shuffled, feature='sqft_living', max_degree=15)
features_list = poly_15_data.column_names()
poly_15_data['price'] = train_data_shuffled['price']

# import numpy as np
# penalties = np.logspace(3, 9, num=13)
# for penalty in penalties:
#     validation_result = k_fold_cross_validation(
#         k=10, l2_penalty=penalty, data=poly_15_data, target='price')
#     print('Mean rss for penalty {0} is {1}'.format(penalty, validation_result))
print('lowest rss happens with L2 penalty 1000')

print('===== QUIZ 7 =====')
model = gl.linear_regression.create(dataset=poly_15_data, target='price',
                                    l2_penalty=1000, validation_set=None, verbose=False)
poly_test_data = polynomial(dataset=test_data, feature='sqft_living', max_degree=15)
predictions = model.predict(dataset=poly_test_data)
rss = ((test_data['price'] - predictions) ** 2).sum()
print('RSS with L2 penalty 1000 on test data: {0}'.format(rss))

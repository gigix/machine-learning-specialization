import graphlab as gl
import numpy as np

sales = gl.SFrame('kc_house_data_small.gl')
(train_and_validation, test) = sales.random_split(.8, seed=1)
(train, validation) = train_and_validation.random_split(.8, seed=1)


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1  # add constant variable
    features = ['constant'] + features  # create list of features
    features_sframe = data_sframe[features]  # create sframe with all features
    feature_matrix = features_sframe.to_numpy()  # convert to np matrix

    output_sarray = data_sframe[output]  # select the output variable
    output_array = output_sarray.to_numpy()  # convert to np array
    return feature_matrix, output_array


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    return feature_matrix / norms, norms


feature_list = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train)
features_test /= norms
features_valid /= norms


def euclidean_distance(reference, query):
    try:
        reference.shape[1]  # test if there are multiple references
        axis = 1
    except IndexError:
        axis = 0
    return np.sqrt(((reference - query) ** 2).sum(axis=axis))


print('===== QUIZ 1 =====')
first_test_data = features_test[0]
distance = euclidean_distance(first_test_data, features_train[9])
print('Distance between the 2 observations is {0}'.format(distance))

print('===== QUIZ 2 =====')
multi_train_data = features_train[0:10]
distances = euclidean_distance(multi_train_data, first_test_data)
print('Distances between first ten train data to first test data:')
print(distances)


def nn(reference, query):
    return knn(reference, query, 1)


def knn(reference, query, k):
    return euclidean_distance(reference, query).argsort()[:k]


print('===== QUIZ 3&4 =====')
query = features_test[2]
nearest_neighbor_index = nn(features_train, query)
print('Nearest neighbor is #{0}, whose value is {1}'.format(
    nearest_neighbor_index, output_train[nearest_neighbor_index]))

print('===== QUIZ 5&6 =====')
k_nearest_neighbors_indices = knn(features_train, query, 4)
print('Nearest 4 neighbors are at indices {0}'.format(k_nearest_neighbors_indices))
prediction = output_train[k_nearest_neighbors_indices].mean()
print('House value prediction is {0}'.format(prediction))


def knn_prediction(reference, k, output, query):
    return output[knn(reference, query, k)].mean()


def knn_multi_prediction(reference, k, output, queries):
    return map(lambda q: knn_prediction(reference, k, output, q), queries)


print('===== QUIZ 7 =====')
for i in range(10):
    print('Predicted value of house #{0} is ${1}'.format(
        i, knn_prediction(features_train, 10, output_train, features_test[i])))


def rss(actuals, predictions):
    return ((actuals - predictions) ** 2).sum()


print('===== QUIZ 8 =====')
# for k in range(1, 16):
#     predictions = knn_multi_prediction(features_train, k, output_train, features_valid)
#     print('RSS when k={0} is {1}'.format(k, rss(output_valid, predictions)))
print('Best K: RSS when k=8 is 6.73616787355e+13')
predictions = knn_multi_prediction(features_train, 8, output_train, features_test)
print('RSS on test data is {0}'.format(rss(output_test, predictions)))

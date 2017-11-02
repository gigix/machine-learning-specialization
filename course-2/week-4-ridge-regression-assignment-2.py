import graphlab as gl
import numpy as np

house_data = gl.SFrame('kc_house_data.gl')


def to_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1  # add constant variable
    features = ['constant'] + features  # create list of features
    features_sframe = data_sframe[features]  # create sframe with all features
    feature_matrix = features_sframe.to_numpy()  # convert to np matrix

    output_sarray = data_sframe[output]  # select the output variable
    output_array = output_sarray.to_numpy()  # convert to np array
    return (feature_matrix, output_array)


def predict_output(feature_matrix, weights):
    return np.dot(feature_matrix, weights)


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    derivative = 2 * np.dot(errors, feature)
    regulation = 0 if feature_is_constant else 2 * l2_penalty * weight
    return derivative + regulation


(example_features, example_output) = to_numpy_data(house_data, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output  # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:, 1], my_weights[1], 1, False)
print np.sum(errors * example_features[:, 1]) * 2 + 20.
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:, 0], my_weights[0], 1, True)
print np.sum(errors) * 2.


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size,
                                      l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)  # make sure it's a numpy array
    # while not reached maximum number of iterations:
    # compute the predictions using your predict_output() function

    # compute the errors as predictions - output
    for i in xrange(max_iterations):
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output

        for i in xrange(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # (Remember: when i=0, you are computing the derivative of the constant!)

            derivative = feature_derivative_ridge(
                errors, feature_matrix[:, i], weights[i], l2_penalty, i == 0)
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size * derivative
    return weights


train_data, test_data = house_data.random_split(.8, seed=0)
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = to_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = to_numpy_data(test_data, simple_features, my_output)

# step_size = 1e-12
# max_iterations = 1000
initial_weights = np.array([0., 0.])
simple_weights_0_penalty = ridge_regression_gradient_descent(
    feature_matrix=simple_feature_matrix, output=output, initial_weights=initial_weights,
    step_size=1e-12, l2_penalty=0, max_iterations=1000)
# step_size = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(
    feature_matrix=simple_feature_matrix, output=output, initial_weights=initial_weights,
    step_size=1e-12, l2_penalty=1e11, max_iterations=1000)

print('===== QUIZ 1 & 2 =====')
print("Coefficients with no regularization: {0}".format(simple_weights_0_penalty))
print("Coefficients with high regularization: {0}".format(simple_weights_high_penalty))

# import matplotlib.pyplot as plt
# plt.plot(simple_feature_matrix, output, 'k.',
#          simple_feature_matrix, predict_output(simple_feature_matrix, simple_weights_0_penalty),
#          'b-',
#          simple_feature_matrix, predict_output(simple_feature_matrix, simple_weights_high_penalty),
#          'r-')
# plt.show()

def rss(feature_matrix, weights, output):
    predictions = predict_output(feature_matrix, weights)
    return ((output - predictions) ** 2).sum()


print('===== QUIZ 3 =====')
print('RSS of initial (all zero) model: {0}'.format(
    rss(simple_test_feature_matrix, initial_weights, test_output)))
print('RSS of no regularization model: {0}'.format(
    rss(simple_test_feature_matrix, simple_weights_0_penalty, test_output)))
print('RSS of high regularization model: {0}'.format(
    rss(simple_test_feature_matrix, simple_weights_high_penalty, test_output)))

print('===== QUIZ 4 =====')
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = to_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = to_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0., 0., 0.])
step_size = 1e-12
max_iterations = 1000

multiple_weights_0_penalty = ridge_regression_gradient_descent(
    feature_matrix, output, initial_weights, step_size, 0, max_iterations)
multiple_weights_high_penalty = ridge_regression_gradient_descent(
    feature_matrix, output, initial_weights, step_size, 1e11, max_iterations)
print("Coefficients with no regularization: {0}".format(multiple_weights_0_penalty))
print("Coefficients with high regularization: {0}".format(multiple_weights_high_penalty))

print('===== QUIZ 5 =====')
print('RSS of initial (all zero) model: {0}'.format(
    rss(test_feature_matrix, initial_weights, test_output)))
print('RSS of no regularization model: {0}'.format(
    rss(test_feature_matrix, multiple_weights_0_penalty, test_output)))
print('RSS of high regularization model: {0}'.format(
    rss(test_feature_matrix, multiple_weights_high_penalty, test_output)))

print('===== QUIZ 6 =====')
sample_house = test_feature_matrix[0]
print('About the house: {0} -> {1}'.format(sample_house, test_output[0]))
print('Price prediction with no regularization model: {0}'.format(
    predict_output(sample_house, multiple_weights_0_penalty)
))
print('Price prediction with high regularization model: {0}'.format(
    predict_output(sample_house, multiple_weights_high_penalty)
))

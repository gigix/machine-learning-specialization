import graphlab as gl
import numpy as np

house_data = gl.SFrame('kc_house_data.gl')


def to_numpy_data(sframe_data, features, target):
    sframe_data['constant'] = 1
    feature_matrix = sframe_data.select_columns(['constant'] + features)
    target_vector = sframe_data.select_column(target)
    return feature_matrix.to_numpy(), target_vector.to_numpy()


def predict_outcome(feature_matrix, weights):
    return np.dot(feature_matrix, weights)


def feature_derivative(errors, feature):
    return 2 * np.dot(errors, feature)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        # compute the errors as predictions - output:
        predictions = predict_outcome(feature_matrix, weights)
        errors = predictions - output

        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])

            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative ** 2

            # update the weight based on step size and derivative:
            weights[i] = weights[i] - step_size * derivative

        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return (weights)


train_data, test_data = house_data.random_split(.8, seed=0)

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = to_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(
    simple_feature_matrix, output, initial_weights, step_size, tolerance)
print('===== QUIZ 1 =====')
print('Coefficients of simple model: {0}'.format(simple_weights))

model_2_features = ['sqft_living', 'sqft_living15']
(model_2_feature_matrix, output) = to_numpy_data(train_data, model_2_features, my_output)
model_2_initial_weights = np.array([-100000., 1., 1.])
model_2_step_size = 4e-12
model_2_tolerance = 1e9

model_2_weights = regression_gradient_descent(model_2_feature_matrix, output,
                                              model_2_initial_weights, model_2_step_size,
                                              model_2_tolerance)
print('Coefficients of model 2: {0}'.format(model_2_weights))


def do_prediction(dataset, features, target, model):
    feature_matrix, target_array = to_numpy_data(dataset, features, target)
    return predict_outcome(feature_matrix, model), target_array


first_house = test_data[0:1]
simple_prediction, _ = do_prediction(first_house,
                                     features=['sqft_living'], target='price', model=simple_weights)
model_2_prediction, _ = do_prediction(first_house,
                                      ['sqft_living', 'sqft_living15'], 'price', model_2_weights)
print('===== QUIZ 2 =====')
print('Price of first test house based on simple model: {0}'.format(simple_prediction))
print('Price of first test house based on model 2: {0}'.format(model_2_prediction))
print('Price of first house is: {0}'.format(first_house['price']))


def rss(dataset, features, target, model):
    predictions, target_array = do_prediction(dataset, features, target, model)
    return np.sum((target_array - predictions) ** 2)


print('===== QUIZ 3 =====')
rss_of_simple_model = rss(test_data, ['sqft_living'], 'price', simple_weights)
print('RSS of simple model on test data is: {0}'.format(rss_of_simple_model))

rss_of_model_2 = rss(test_data, ['sqft_living', 'sqft_living15'], 'price', model_2_weights)
print('RSS of model 2 on test data is: {0}'.format(rss_of_model_2))

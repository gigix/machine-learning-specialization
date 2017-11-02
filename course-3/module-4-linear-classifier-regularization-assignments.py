import json
import string

import graphlab as gl
import numpy as np

products = gl.SFrame('amazon_baby_subset.gl/')
important_words = [str(s) for s in json.load(open('important_words.json'))]


def remove_punctuation(text):
    return text.translate(None, string.punctuation)

# Remove punctuation.
products['review_clean'] = products['review'].apply(remove_punctuation)

# Split out the words into individual columns
for word in important_words:
    products[word] = products['review_clean'].apply(lambda r: r.split().count(word))

train_data, validation_data = products.random_split(.8, seed=2)


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.to_numpy()
    label_sarray = dataframe[label]
    label_array = label_sarray.to_numpy()
    return feature_matrix, label_array


feature_matrix_train, sentiment_train = get_numpy_data(
    train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(
    validation_data, important_words, 'sentiment')


def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant):
    # Compute the dot product of errors and feature
    derivative = np.sum(feature * errors)
    # add L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant:
        derivative -= 2 * l2_penalty * coefficient
    return derivative


def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment == +1)
    scores = np.dot(feature_matrix, coefficients)
    return np.sum((indicator - 1) * scores - np.log(1. + np.exp(-scores))) - \
           l2_penalty * np.sum(coefficients[1:] ** 2)


def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients
    score = np.dot(feature_matrix, coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1 / (1 + np.exp(-score))
    # return predictions
    return predictions


def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients,
                                step_size, l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients)  # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment == +1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)):  # loop over each coefficient
            is_intercept = (j == 0)
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative_with_L2(
                errors, feature_matrix[:, j], coefficients[j], l2_penalty, is_intercept)

            # add the step size times the derivative to the current coefficient
            coefficients[j] += step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                  (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients


models = []
for l2_penalty in [0, 4, 10, 1e2, 1e3, 1e5]:
    model = logistic_regression_with_L2(
        feature_matrix=feature_matrix_train, sentiment=sentiment_train,
        initial_coefficients=np.zeros(194), step_size=5e-6, l2_penalty=l2_penalty, max_iter=501)
    models.append(model)


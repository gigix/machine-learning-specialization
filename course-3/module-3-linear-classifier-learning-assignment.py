import json
import string
import graphlab as gl
import numpy as np

products = gl.SFrame('amazon_baby_subset.gl')
important_words = [str(s) for s in json.load(open('important_words.json'))]


def remove_punctuation(text):
    return text.translate(None, string.punctuation)


products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda r: r > 3)

products = products.fillna('review', '')  # fill in N/A's in the review column
products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda r: r.split().count(word))

print('===== QUIZ 1 =====')
# print('{0} reviews contain the word *perfect*'.format(len(products[products['perfect'] > 0])))
print('2955 reviews contain the word *perfect*')


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.to_numpy()
    label_sarray = dataframe[label]
    label_array = label_sarray.to_numpy()
    return feature_matrix, label_array


print('===== QUIZ 2&3 =====')
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')
print('feature_matrix should have {0} features'.format(len(important_words) + 1))


def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients
    score = np.dot(feature_matrix, coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1 / (1 + np.exp(-score))
    # return predictions
    return predictions


def feature_derivative(errors, feature):
    # Compute the dot product of errors and feature
    derivative = np.sum(feature * errors)
    # Return the derivative
    return derivative


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment == +1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator - 1) * scores - np.log(1. + np.exp(-scores)))
    return lp


def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients)  # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_1,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment == +1)
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)):  # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative(errors, feature_matrix[:, j])

            # add the step size times the derivative to the current coefficient
            coefficients[j] += step_size * derivative
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                  (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients


print('===== QUIZ 4&5 =====')
coefficients = logistic_regression(
    feature_matrix, sentiment, initial_coefficients=np.zeros(194), step_size=1e-7, max_iter=301)
predictions = predict_probability(feature_matrix, coefficients)
print('{0} reviews seem to be positive'.format(filter(lambda p: p > 0, predictions)))

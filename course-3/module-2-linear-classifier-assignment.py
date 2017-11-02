import graphlab as gl
import string

products = gl.SFrame('amazon_baby.gl')


def remove_punctuation(text):
    return text.translate(None, string.punctuation)


products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda r: r > 3)

products = products.fillna('review', '')  # fill in N/A's in the review column
products['review_clean'] = products['review'].apply(remove_punctuation)
products['word_count'] = gl.text_analytics.count_words(products['review_clean'])

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                     'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
                     'work', 'product', 'money', 'would', 'return']
products['word_count_subset'] = \
    products['word_count'].dict_trim_by_keys(significant_words, exclude=False)

train_data, test_data = products.random_split(.8, seed=1)

# sentiment_model = gl.logistic_classifier.create(
#     dataset=train_data, target='sentiment', features=['word_count'], validation_set=None)
# sentiment_model.save('module-2-model-1')
sentiment_model = gl.load_model('module-2-model-1')

print('===== QUIZ 1 =====')
sentiment_coefficients = sentiment_model.get('coefficients')
print('There are {0} non-negative coefficients'.format(
    len(sentiment_coefficients[sentiment_coefficients['value'] >= 0])))

print('===== QUIZ 2 =====')
sample_test_data = test_data[10:13]
sample_predictions = sentiment_model.predict(sample_test_data, output_type='probability')
print(sample_predictions)

print('===== QUIZ 3&4 =====')
predict_result = test_data.copy()
predict_result['prediction'] = sentiment_model.predict(test_data, output_type='probability')
predict_result = predict_result.sort('prediction')
print('Following are most positive reviews:')
predict_result[-20:].print_rows(num_rows=20, num_columns=3)
print('Following are most negative reviews:')
predict_result[:20].print_rows(num_rows=20, num_columns=3)


def accuracy(model, dataset, actual_output):
    comparison = gl.SFrame(data={'actual': actual_output, 'prediction': model.predict(dataset)})
    accurate_predictions = comparison[comparison['prediction'] == comparison['actual']]
    return float(len(accurate_predictions)) / float(len(comparison))


print('===== QUIZ 5&6 =====')
print('Accuracy: {0}'.format(accuracy(sentiment_model, test_data, test_data['sentiment'])))

# simple_model = gl.logistic_classifier.create(
#     dataset=train_data, target='sentiment', features=['word_count_subset'], validation_set=None)
# simple_model.save('module-2-model-2')
simple_model = gl.load_model('module-2-model-2')

print('===== QUIZ 7&8 =====')
simple_coefficients = simple_model.get('coefficients')
positive_simple_coefficients = simple_coefficients[simple_coefficients['value'] >= 0]
print('There are {0} non-negative coefficients'.format(len(positive_simple_coefficients)))
for coefficient in positive_simple_coefficients:
    print('{0}: value in previous model is {1}'.format(
        coefficient['index'],
        sentiment_coefficients[sentiment_coefficients['index'] == coefficient['index']]['value']
    ))

print('===== QUIZ 9&10 =====')
print('Accuracy of sentiment_model on training data: {0}'.format(
    accuracy(sentiment_model, train_data, train_data['sentiment'])))
print('Accuracy of simple_model on training data: {0}'.format(
    accuracy(simple_model, train_data, train_data['sentiment'])))
print('Accuracy of sentiment_model on test data: {0}'.format(
    accuracy(sentiment_model, test_data, test_data['sentiment'])))
print('Accuracy of simple_model on test data: {0}'.format(
    accuracy(simple_model, test_data, test_data['sentiment'])))

print('===== QUIZ 11 & 12 =====')
positive_test_data = test_data[test_data['sentiment'] == 1]
print('Total test data: {0}'.format(len(test_data)))
print('Positive test data: {0}'.format(len(positive_test_data)))
print('Majority rate: {0}'.format(float(len(positive_test_data)) / float(len(test_data))))

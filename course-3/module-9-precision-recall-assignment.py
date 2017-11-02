import string

import graphlab
import numpy as np

products = graphlab.SFrame('amazon_baby.gl/')


def remove_punctuation(text):
    return text.translate(None, string.punctuation)

# Remove punctuation.
review_clean = products['review'].apply(remove_punctuation)

# Count words
products['word_count'] = graphlab.text_analytics.count_words(review_clean)

# Drop neutral sentiment reviews.
products = products[products['rating'] != 3]

# Positive sentiment to +1 and negative sentiment to -1
target = 'sentiment'
products[target] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

train_data, test_data = products.random_split(.8, seed=1)

# model = graphlab.logistic_classifier.create(train_data, target=target,
#                                             features=['word_count'], validation_set=None)
# model.save('module-9-model-1')
model = graphlab.load_model('module-9-model-1')

print('===== QUIZ 1 =====')
accuracy = model.evaluate(test_data, metric='accuracy')['accuracy']
print "Test Accuracy: %s" % accuracy

baseline = len(test_data[test_data[target] == 1]) / float(len(test_data))
print "Baseline accuracy (majority class classifier): %s" % baseline

print('===== QUIZ 2 =====')
confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']
print confusion_matrix


def cost(model, dataset):
    confusion_matrix = model.evaluate(dataset, metric='confusion_matrix')['confusion_matrix']
    errors = confusion_matrix[
        confusion_matrix['target_label'] != confusion_matrix['predicted_label']]
    false_positive = errors[errors['predicted_label'] == 1]['count']
    false_negative = errors[errors['predicted_label'] == -1]['count']
    return false_positive * 100 + false_negative


print('===== QUIZ 3 =====')
print('Cost of the model on test data: %s' % cost(model, test_data))

print('===== QUIZ 4~5 =====')
precision = model.evaluate(test_data, metric='precision')['precision']
print "Precision on test data: %s; false positive rate: %s" % (precision, 1 - precision)

print('===== QUIZ 6~7 =====')
recall = model.evaluate(test_data, metric='recall')['recall']
print "Recall on test data: %s" % recall


def apply_threshold(probabilities, threshold):
    return probabilities.apply(lambda p: +1 if p >= threshold else -1)


probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

print('===== QUIZ 8~9 =====')
print "Number of positive predicted reviews (threshold = 0.5): %s" % \
      len(predictions_with_default_threshold[predictions_with_default_threshold == 1])
print "Number of positive predicted reviews (threshold = 0.9): %s" % \
      len(predictions_with_high_threshold[predictions_with_high_threshold == 1])

print('===== QUIZ 10 =====')
threshold_values = np.linspace(0.5, 1, num=100)
probabilities = model.predict(test_data, output_type='probability')
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    precision = graphlab.evaluation.precision(test_data[target], predictions)
    recall = graphlab.evaluation.recall(test_data[target], predictions)
    print('Threshold: %s - precision: %s; recall: %s' % (threshold, precision, recall))

print('===== QUIZ 11 =====')
predictions = apply_threshold(probabilities, 0.98)
print('----- confusion matrix when threshold is 0.98 -----')
confusion_matrix = graphlab.evaluation.confusion_matrix(test_data[target], predictions)
print(confusion_matrix)

print('===== QUIZ 12~13 =====')
baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
probabilities = model.predict(baby_reviews, output_type='probability')
threshold_values = np.linspace(0.5, 1, num=100)
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    precision = graphlab.evaluation.precision(baby_reviews[target], predictions)
    recall = graphlab.evaluation.recall(baby_reviews[target], predictions)
    print('Threshold: %s - precision: %s; recall: %s' % (threshold, precision, recall))


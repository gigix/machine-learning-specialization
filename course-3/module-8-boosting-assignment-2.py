import graphlab

loans = graphlab.SFrame('lending-club-data.gl/')

features = ['grade',  # grade of the loan
            'term',  # the term of the loan
            'home_ownership',  # home ownership status: own, mortgage or rent
            'emp_length',  # number of years of employment
            ]
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans.remove_column('bad_loans')
target = 'safe_loans'
loans = loans[features + [target]]

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw))
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)
loans_data = risky_loans_raw.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

train_data, test_data = loans_data.random_split(0.8, seed=1)


def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    return (total_weight_positive, -1) \
        if total_weight_positive <= total_weight_negative \
        else (total_weight_negative, +1)


example_labels = graphlab.SArray([-1, -1, 1, 1, 1])
example_data_weights = graphlab.SArray([1., 2., .5, 1., 1.])
if intermediate_node_weighted_mistakes(example_labels, example_data_weights) == (2.5, -1):
    print 'Test passed!'
else:
    print 'Test failed... try again!'


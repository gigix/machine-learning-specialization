import graphlab as gl
import numpy as np

house_data = gl.SFrame('kc_house_data.gl')

house_data['bedrooms_squared'] = house_data['bedrooms'] ** 2
house_data['bed_bath_rooms'] = house_data['bedrooms'] * house_data['bathrooms']
house_data['log_sqft_living'] = np.log(house_data['sqft_living'])
house_data['lat_plus_long'] = house_data['lat'] + house_data['long']

print(house_data)
print(house_data[0])

train_data, test_data = house_data.random_split(.8, seed=0)

print('===== QUIZ 1 =====')
for new_feature in ['bedrooms_squared', 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']:
    print('Mean value of [{0}] is : {1}'.format(new_feature, test_data[new_feature].mean()))

feature_group_1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
feature_group_2 = feature_group_1 + ['bed_bath_rooms']
feature_group_3 = feature_group_2 + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

models = []
print('===== QUIZ 2 and 3 =====')
for feature_group in [feature_group_1, feature_group_2, feature_group_3]:
    model = gl.linear_regression.create(
        train_data, target='price', features=feature_group, verbose=False)
    models.append(model)
    print('Coefficients of model {0}'.format(feature_group))
    print(model.get('coefficients'))


def show_rss(model, data):
    evaluation = model.evaluate(data)
    rss = evaluation['rmse'] ** 2 * len(data)
    print('... and RSS is {0}'.format(rss))


print('===== QUIZ 4 =====')
for i in [0, 1, 2]:
    print('Evaluating of model {0} on train data'.format(i))
    show_rss(models[i], train_data)

print('===== QUIZ 5 =====')
for i in [0, 1, 2]:
    print('Evaluating of model {0} on test data'.format(i))
    show_rss(models[i], test_data)

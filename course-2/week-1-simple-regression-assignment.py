import graphlab as gl

sales = gl.SFrame('kc_house_data.gl')
print(sales)

train_data, test_data = sales.random_split(.8, seed=0)

simple_model = gl.linear_regression.create(train_data, target='price', features=['sqft_living'])
print(simple_model.get('coefficients'))


def simple_linear_regression(x, y):
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xx = (x ** 2).sum()
    sum_xy = (x * y).sum()
    w1 = (sum_xy - sum_y * sum_x / n) / (sum_xx - sum_x * sum_x / n)  # slope
    w0 = sum_y / n - w1 * sum_x / n  # intercept
    return w0, w1


def predict(x, w0, w1):
    return x * w1 + w0


def rss(x, y, w0, w1):
    y_hat = predict(x, w0, w1)
    return ((y_hat - y) ** 2).sum()


def reverse_regression_predictions(y, w0, w1):
    return (y - w0) / w1


w0_area, w1_area = simple_linear_regression(train_data['sqft_living'], train_data['price'])
print('Closed form coefficients are ({0}, {1})'.format(w0_area, w1_area))

print('===== QUIZ 1 =====')
print('Predicted house price is {0}'.format(predict(2650, w0_area, w1_area)))

print('===== QUIZ 2 =====')
print('RSS on training data is {0}'.format(
    rss(train_data['sqft_living'], train_data['price'], w0_area, w1_area)))

print('===== QUIZ 3 =====')
sqft_living = reverse_regression_predictions(800000, w0_area, w1_area)
print('Estimated living area of the house is {0}'.format(sqft_living))
print('(double check - {0})'.format(predict(sqft_living, w0_area, w1_area)))

print('===== QUIZ 4 =====')
rss_area = rss(test_data['sqft_living'], test_data['price'], w0_area, w1_area)
w0_room, w1_room = simple_linear_regression(train_data['bedrooms'], train_data['price'])
rss_room = rss(test_data['bedrooms'], test_data['price'], w0_room, w1_room)
print('RSS of areas is {0}\nRSS of rooms is {1}'.format(rss_area, rss_room))

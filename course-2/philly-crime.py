import graphlab as gl

sales = gl.SFrame.read_csv('Philadelphia_Crime_Rate_noNA.csv')
sales['CrimeRateSqr'] = sales['CrimeRate'] * sales['CrimeRate']
print(sales)

sales_without_downtown = sales[sales['MilesPhila'] > 0]

crime_model = gl.linear_regression.create(
    sales, target='HousePrice', features=['CrimeRate'])
crime_model_without_downtown = gl.linear_regression.create(
    sales_without_downtown, target='HousePrice', features=['CrimeRate'])
crime_sqr_model_without_downtown = gl.linear_regression.create(
    sales_without_downtown, target='HousePrice', features=['CrimeRate', 'CrimeRateSqr'])

# import matplotlib.pyplot as plt
# plt.plot(
#     sales_without_downtown['CrimeRate'], sales_without_downtown['HousePrice'], '.',
#     sales_without_downtown['CrimeRate'], crime_model.predict(sales_without_downtown), '.',
#     sales_without_downtown['CrimeRate'], crime_model_without_downtown.predict(sales_without_downtown), '.',
#     sales_without_downtown['CrimeRate'], crime_sqr_model_without_downtown.predict(sales_without_downtown), '.'
# )
# plt.show()

print(crime_model.get('coefficients'))
print(crime_model_without_downtown.get('coefficients'))
print(crime_sqr_model_without_downtown.get('coefficients'))

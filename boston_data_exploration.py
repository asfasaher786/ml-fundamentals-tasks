>>> # Step 1: Import Libraries
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import mean_squared_error, r2_score
>>>
>>> # Step 2: Load the Dataset (no headers, space-separated)
>>> file_path = r"C:\Users\PMYLS\OneDrive\Desktop\houseprediction.csv"
>>> df = pd.read_csv(file_path, header=None, delim_whitespace=True)
<python-input-9>:1: FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in afuture version. Use ``sep='\s+'`` instead
>>>
>>> # Step 3: Assign Column Names
>>> df.columns = [
...     "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
...         "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
...         ]
>>>
>>> # Step 4: Select One Feature ('RM') and Target ('MEDV')
>>> X = df[['RM']]
>>> y = df['MEDV']
>>>
>>> # Step 5: Split the Data into Train and Test Sets
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Step 6: Train the Linear Regression Model
>>> model = LinearRegression()
>>> model.fit(X_train, y_train)
LinearRegression()
>>>
>>> # Step 7: Make Predictions
>>> y_pred = model.predict(X_test)
>>>
>>> # Step 8: Evaluate the Model
>>> r2 = r2_score(y_test, y_pred)
>>> mse = mean_squared_error(y_test, y_pred)
>>> print(f"Coefficient: {model.coef_[0]}")
Coefficient: 9.348301406497722
>>> print(f"Intercept: {model.intercept_}")
Intercept: -36.24631889813792
>>> print(f"R² Score: {r2}")
R² Score: 0.3707569232254778
>>> print(f"Mean Squared Error: {mse}")
Mean Squared Error: 46.144775347317264
>>>
>>> # Step 9: Plotting
>>> plt.scatter(X_test, y_test, color='blue', label='Actual')
<matplotlib.collections.PathCollection object at 0x00000247DEC44EC0>
>>> plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
[<matplotlib.lines.Line2D object at 0x00000247DEC34A50>]
>>> plt.xlabel("Average Number of Rooms (RM)")
Text(0.5, 0, 'Average Number of Rooms (RM)')
>>> plt.ylabel("House Price (MEDV)")
Text(0, 0.5, 'House Price (MEDV)')
>>> plt.title("Simple Linear Regression: RM vs MEDV")
Text(0.5, 1.0, 'Simple Linear Regression: RM vs MEDV')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x00000247C3637CB0>
>>> plt.show()
>>>

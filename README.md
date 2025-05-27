>>> import pandas as pd
>>>
>>> df = pd.read_csv(r"C:\Users\PMYLS\OneDrive\Desktop\iris.csv")
>>> print(df.head())
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
>>> print(df.describe())
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
>>> print(df['species'].value_counts())
species
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64
>>> print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
>>> import matplotlib.pyplot as plt
Traceback (most recent call last):
  File "<python-input-7>", line 1, in <module>
    import matplotlib.pyplot as plt
ModuleNotFoundError: No module named 'matplotlib'
>>> import seaborn as sns
Traceback (most recent call last):
  File "<python-input-8>", line 1, in <module>
    import seaborn as sns
ModuleNotFoundError: No module named 'seaborn'
>>>
>>> sns.pairplot(df, hue='species')
Traceback (most recent call last):
  File "<python-input-10>", line 1, in <module>
    sns.pairplot(df, hue='species')
    ^^^
NameError: name 'sns' is not defined
>>> plt.show()
Traceback (most recent call last):
  File "<python-input-11>", line 1, in <module>
    plt.show()
    ^^^
NameError: name 'plt' is not defined
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns
>>>
>>> sns.pairplot(df, hue='species')
<seaborn.axisgrid.PairGrid object at 0x000001F0DC0A6120>
>>> plt.show()
>>>
>>>
>>> print(df.describe())
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
>>> print(df['species'].value_counts())
species
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64
>>> print(df.isnull().sum())
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
species         0
dtype: int64
>>> print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
>>> import matplotlib.pyplot as plt
>>>
>>> df.hist(figsize=(10,8))
array([[<Axes: title={'center': 'sepal_length'}>,
        <Axes: title={'center': 'sepal_width'}>],
       [<Axes: title={'center': 'petal_length'}>,
        <Axes: title={'center': 'petal_width'}>]], dtype=object)
>>> plt.show()
>>> import seaborn as sns
>>>
>>> sns.boxplot(x='species', y='sepal_length', data=df)
<Axes: xlabel='species', ylabel='sepal_length'>
>>> plt.show()
>>> corr = df.corr()
Traceback (most recent call last):
  File "<python-input-31>", line 1, in <module>
    corr = df.corr()
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\frame.py", line 11049, in corr
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\frame.py", line 1993, in to_numpy
    result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\internals\managers.py", line 1694, in as_array
    arr = self._interleave(dtype=dtype, na_value=na_value)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\internals\managers.py", line 1753, in _interleave
    result[rl.indexer] = arr
    ~~~~~~^^^^^^^^^^^^
ValueError: could not convert string to float: 'setosa'
>>> print(corr)
Traceback (most recent call last):
  File "<python-input-32>", line 1, in <module>
    print(corr)
          ^^^^
NameError: name 'corr' is not defined
>>>
>>> sns.heatmap(corr, annot=True)
Traceback (most recent call last):
  File "<python-input-34>", line 1, in <module>
    sns.heatmap(corr, annot=True)
                ^^^^
NameError: name 'corr' is not defined
>>> plt.show()
>>> # Step 1: Select only numeric columns (exclude 'species')
>>> numeric_df = df.select_dtypes(include=['float64', 'int64'])
>>>
>>> # Step 2: Now compute correlation
>>> corr = numeric_df.corr()
>>>
>>> # Step 3: Show correlation matrix
>>> print(corr)
              sepal_length  sepal_width  petal_length  petal_width
sepal_length      1.000000    -0.109369      0.871754     0.817954
sepal_width      -0.109369     1.000000     -0.420516    -0.356544
petal_length      0.871754    -0.420516      1.000000     0.962757
petal_width       0.817954    -0.356544      0.962757     1.000000
>>>
>>> # Step 4: Heatmap
>>> import seaborn as sns
>>> import matplotlib.pyplot as plt
>>>
>>> sns.heatmap(corr, annot=True, cmap='coolwarm')
<Axes: >
>>> plt.title('Correlation Heatmap')
Text(0.5, 1.0, 'Correlation Heatmap')
>>> plt.show()
>>> print("Shape of dataset:", df.shape)
Shape of dataset: (150, 5)
>>> print("\nColumn names:", df.columns.tolist())

Column names: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
>>> print("\nStatistical Summary:")

Statistical Summary:
>>> print(df.describe())
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
>>> print("\nClass Distribution:")

Class Distribution:
>>> print(df['species'].value_counts())
species
setosa        50
versicolor    50
virginica     50
Name: count, dtype: int64
>>> import seaborn as sns
>>> import matplotlib.pyplot as plt
>>>
>>> sns.countplot(x='species', data=df)
<Axes: xlabel='species', ylabel='count'>
>>> plt.title('Distribution of Iris Species')
Text(0.5, 1.0, 'Distribution of Iris Species')
>>> plt.xlabel('Species')
Text(0.5, 0, 'Species')
>>> plt.ylabel('Count')
Text(0, 0.5, 'Count')
>>> plt.show()
>>> from sklearn.preprocessing import LabelEncoder
>>>
>>> le = LabelEncoder()
>>> df['species'] = le.fit_transform(df['species'])  # setosa=0, versicolor=1, virginica=2
>>> print(df.head())
   sepal_length  sepal_width  petal_length  petal_width  species
0           5.1          3.5           1.4          0.2        0
1           4.9          3.0           1.4          0.2        0
2           4.7          3.2           1.3          0.2        0
3           4.6          3.1           1.5          0.2        0
4           5.0          3.6           1.4          0.2        0
>>> X = df.drop('species', axis=1)
>>> y = df['species']
>>>
>>> from sklearn.model_selection import train_test_split
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> from sklearn.tree import DecisionTreeClassifier
>>>
>>> model = DecisionTreeClassifier()
>>> model.fit(X_train, y_train)
DecisionTreeClassifier()
>>> y_pred = model.predict(X_test)
>>> from sklearn.metrics import accuracy_score
>>>
>>> accuracy = accuracy_score(y_test, y_pred)
>>> print("Accuracy:", accuracy)
Accuracy: 1.0
>>> from sklearn.tree import plot_tree
>>> plt.figure(figsize=(12,8))
<Figure size 1200x800 with 0 Axes>
>>> plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
Traceback (most recent call last):
  File "<python-input-88>", line 1, in <module>
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 205, in plot_tree
    return exporter.export(decision_tree, ax=ax)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 652, in export
    my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 628, in _make_tree
    name = self.node_to_str(et, node_id, criterion=criterion)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 394, in node_to_str
    node_string += class_name
TypeError: can only concatenate str (not "numpy.int64") to str
>>> plt.title("Decision Tree Visualization")
Text(0.5, 1.0, 'Decision Tree Visualization')
>>> plt.show()
>>> print(model)print(model)print(model)print(model)
  File "<python-input-91>", line 1
    print(model)print(model)print(model)print(model)
                ^^^^^
SyntaxError: invalid syntax
>>> from sklearn.tree import plot_tree
>>> import matplotlib.pyplot as plt
>>>
>>> plt.figure(figsize=(24, 14))  # Make it even larger for better visibility
<Figure size 2400x1400 with 0 Axes>
>>> plot_tree(
...     model,
...         feature_names=X.columns,
...             class_names=model.classes_,
...                 filled=True,
...                     rounded=True,
...                         fontsize=12
...                         )
Traceback (most recent call last):
  File "<python-input-96>", line 1, in <module>
    plot_tree(
    ~~~~~~~~~^
        model,
        ^^^^^^
    ...<4 lines>...
                            fontsize=12
                            ^^^^^^^^^^^
                            )
                            ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 205, in plot_tree
    return exporter.export(decision_tree, ax=ax)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 652, in export
    my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 628, in _make_tree
    name = self.node_to_str(et, node_id, criterion=criterion)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 394, in node_to_str
    node_string += class_name
TypeError: can only concatenate str (not "numpy.int64") to str
>>> plt.title("Decision Tree Visualization")
Text(0.5, 1.0, 'Decision Tree Visualization')
>>> plt.tight_layout()
>>> plt.show()
>>> plt.figure(figsize=(24, 14))
<Figure size 2400x1400 with 0 Axes>
>>> plot_tree(
...     model,
...         feature_names=X.columns,
...             class_names=model.classes_,
...                 filled=True,
...                     rounded=True,
...                         fontsize=12
...                         )
Traceback (most recent call last):
  File "<python-input-101>", line 1, in <module>
    plot_tree(
    ~~~~~~~~~^
        model,
        ^^^^^^
    ...<4 lines>...
                            fontsize=12
                            ^^^^^^^^^^^
                            )
                            ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 205, in plot_tree
    return exporter.export(decision_tree, ax=ax)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 652, in export
    my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 628, in _make_tree
    name = self.node_to_str(et, node_id, criterion=criterion)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\tree\_export.py", line 394, in node_to_str
    node_string += class_name
TypeError: can only concatenate str (not "numpy.int64") to str
>>> plt.title("Decision Tree Visualization")
Text(0.5, 1.0, 'Decision Tree Visualization')
>>> plt.tight_layout()
>>> plt.savefig("iris_decision_tree.png")  # Saves to working directory
>>> print("Plot saved as 'iris_decision_tree.png'")
Plot saved as 'iris_decision_tree.png'
>>> # Convert target to string
>>> y = df['species'].astype(str)
>>> from sklearn.tree import DecisionTreeClassifier
>>>
>>> model = DecisionTreeClassifier()
>>> model.fit(X, y)
DecisionTreeClassifier()
>>> plt.figure(figsize=(24, 14))
<Figure size 2400x1400 with 0 Axes>
>>> plot_tree(
...     model,
...         feature_names=X.columns,
...             class_names=model.classes_,  # these are now strings
...                 filled=True,
...                     rounded=True,
...                         fontsize=12
...                         )
[Text(0.5, 0.9166666666666666, 'petal_length <= 2.45\ngini = 0.667\nsamples = 150\nvalue = [50, 50, 50]\nclass = 0'), Text(0.4230769230769231, 0.75, 'gini = 0.0\nsamples = 50\nvalue = [50, 0, 0]\nclass = 0'), Text(0.46153846153846156, 0.8333333333333333, 'True  '),Text(0.5769230769230769, 0.75, 'petal_width <= 1.75\ngini = 0.5\nsamples = 100\nvalue = [0, 50, 50]\nclass = 1'), Text(0.5384615384615384, 0.8333333333333333, '  False'), Text(0.3076923076923077, 0.5833333333333334, 'petal_length <= 4.95\ngini = 0.168\nsamples = 54\nvalue = [0, 49, 5]\nclass = 1'), Text(0.15384615384615385, 0.4166666666666667, 'petal_width <= 1.65\ngini = 0.041\nsamples = 48\nvalue = [0, 47, 1]\nclass = 1'), Text(0.07692307692307693, 0.25, 'gini = 0.0\nsamples = 47\nvalue = [0, 47, 0]\nclass = 1'), Text(0.23076923076923078, 0.25, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 1]\nclass = 2'), Text(0.46153846153846156, 0.4166666666666667, 'petal_width <= 1.55\ngini = 0.444\nsamples = 6\nvalue = [0, 2, 4]\nclass = 2'), Text(0.38461538461538464, 0.25, 'gini = 0.0\nsamples = 3\nvalue = [0, 0, 3]\nclass = 2'), Text(0.5384615384615384, 0.25, 'petal_length <= 5.45\ngini = 0.444\nsamples = 3\nvalue = [0, 2, 1]\nclass = 1'), Text(0.46153846153846156, 0.08333333333333333, 'gini = 0.0\nsamples = 2\nvalue = [0, 2, 0]\nclass = 1'), Text(0.6153846153846154, 0.08333333333333333, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 1]\nclass = 2'), Text(0.8461538461538461, 0.5833333333333334, 'petal_length <= 4.85\ngini = 0.043\nsamples = 46\nvalue = [0, 1, 45]\nclass = 2'), Text(0.7692307692307693, 0.4166666666666667, 'sepal_length <= 5.95\ngini = 0.444\nsamples = 3\nvalue = [0, 1, 2]\nclass = 2'), Text(0.6923076923076923, 0.25, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0]\nclass = 1'), Text(0.8461538461538461, 0.25, 'gini = 0.0\nsamples = 2\nvalue = [0, 0, 2]\nclass = 2'), Text(0.9230769230769231, 0.4166666666666667, 'gini = 0.0\nsamples = 43\nvalue = [0, 0, 43]\nclass = 2')]
>>> plt.title("Decision Tree Visualization")
Text(0.5, 1.0, 'Decision Tree Visualization')
>>> plt.tight_layout()
>>> plt.show()
>>> y_pred = model.predict(X_test)
>>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>>>
>>> cm = confusion_matrix(y_test, y_pred)
Traceback (most recent call last):
  File "<python-input-120>", line 1, in <module>
    cm = confusion_matrix(y_test, y_pred)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 345, in confusion_matrix
    labels = unique_labels(y_true, y_pred)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\multiclass.py", line 117, in unique_labels
    raise ValueError("Mix of label input types (string and number)")
ValueError: Mix of label input types (string and number)
>>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
Traceback (most recent call last):
  File "<python-input-121>", line 1, in <module>
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                                                   ^^
NameError: name 'cm' is not defined
>>> disp.plot()
Traceback (most recent call last):
  File "<python-input-122>", line 1, in <module>
    disp.plot()
    ^^^^
NameError: name 'disp' is not defined
>>> plt.title("Confusion Matrix")
Text(0.5, 1.0, 'Confusion Matrix')
>>> plt.show()
>>> from sklearn.preprocessing import LabelEncoder
>>>
>>> # If you already used LabelEncoder earlier, re-use it:
>>> # Example:
>>> le = LabelEncoder()
>>> le.fit(y)  # 'y' is your original label column before splitting
LabelEncoder()
>>>
>>> # Convert numeric predictions back to string labels
>>> y_pred_labels = le.inverse_transform(y_pred)
Traceback (most recent call last):
  File "<python-input-133>", line 1, in <module>
    y_pred_labels = le.inverse_transform(y_pred)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\preprocessing\_label.py", line 162, in inverse_transform
    raise ValueError("y contains previously unseen labels: %s" % str(diff))
ValueError: y contains previously unseen labels: ['0' '1' '2']
>>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>>>
>>> cm = confusion_matrix(y_test, y_pred_labels)
Traceback (most recent call last):
  File "<python-input-136>", line 1, in <module>
    cm = confusion_matrix(y_test, y_pred_labels)
                                  ^^^^^^^^^^^^^
NameError: name 'y_pred_labels' is not defined
>>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
Traceback (most recent call last):
  File "<python-input-137>", line 1, in <module>
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                                                   ^^
NameError: name 'cm' is not defined
>>> disp.plot()
Traceback (most recent call last):
  File "<python-input-138>", line 1, in <module>
    disp.plot()
    ^^^^
NameError: name 'disp' is not defined
>>> plt.title("Confusion Matrix")
Text(0.5, 1.0, 'Confusion Matrix')
>>> plt.show()
>>> from sklearn.preprocessing import LabelEncoder
>>> from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
>>> import matplotlib.pyplot as plt
>>>
>>> # Assuming y_test is original string labels, and y_pred is numeric predictions from the model
>>>
>>> le = LabelEncoder()
>>> le.fit(y)  # y is your original label data before train-test split
LabelEncoder()
>>>
>>> y_pred_labels = le.inverse_transform(y_pred)
Traceback (most recent call last):
  File "<python-input-150>", line 1, in <module>
    y_pred_labels = le.inverse_transform(y_pred)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\preprocessing\_label.py", line 162, in inverse_transform
    raise ValueError("y contains previously unseen labels: %s" % str(diff))
ValueError: y contains previously unseen labels: ['0' '1' '2']
>>>
>>> cm = confusion_matrix(y_test, y_pred_labels)
Traceback (most recent call last):
  File "<python-input-152>", line 1, in <module>
    cm = confusion_matrix(y_test, y_pred_labels)
                                  ^^^^^^^^^^^^^
NameError: name 'y_pred_labels' is not defined
>>>
>>> plt.figure(figsize=(8,6))
<Figure size 800x600 with 0 Axes>
>>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
Traceback (most recent call last):
  File "<python-input-155>", line 1, in <module>
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                                                   ^^
NameError: name 'cm' is not defined
>>> disp.plot(cmap=plt.cm.Blues)
Traceback (most recent call last):
  File "<python-input-156>", line 1, in <module>
    disp.plot(cmap=plt.cm.Blues)
    ^^^^
NameError: name 'disp' is not defined
>>> plt.title("Confusion Matrix")
Text(0.5, 1.0, 'Confusion Matrix')
>>> plt.show()
>>> # Assuming y_pred is numpy array or list of string digits
>>>
>>> # Convert y_pred to integers
>>> y_pred_int = y_pred.astype(int)  # or list(map(int, y_pred))
>>>
>>> # Inverse transform numeric predictions to original string labels
>>> y_pred_labels = le.inverse_transform(y_pred_int)
>>>
>>> # Compute confusion matrix
>>> cm = confusion_matrix(y_test, y_pred_labels)
Traceback (most recent call last):
  File "<python-input-168>", line 1, in <module>
    cm = confusion_matrix(y_test, y_pred_labels)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 345, in confusion_matrix
    labels = unique_labels(y_true, y_pred)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\multiclass.py", line 117, in unique_labels
    raise ValueError("Mix of label input types (string and number)")
ValueError: Mix of label input types (string and number)
>>>
>>> # Plot confusion matrix
>>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
Traceback (most recent call last):
  File "<python-input-171>", line 1, in <module>
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                                                   ^^
NameError: name 'cm' is not defined
>>> plt.figure(figsize=(8,6))
<Figure size 800x600 with 0 Axes>
>>> disp.plot(cmap=plt.cm.Blues)
Traceback (most recent call last):
  File "<python-input-173>", line 1, in <module>
    disp.plot(cmap=plt.cm.Blues)
    ^^^^
NameError: name 'disp' is not defined
>>> plt.title("Confusion Matrix")
Text(0.5, 1.0, 'Confusion Matrix')
>>> plt.show()
>>> print(type(y_test[0]), y_test[:5])        # What type and sample of your test labels?
Traceback (most recent call last):
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\indexes\base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 2606, in pandas._libs.hashtable.Int64HashTable.get_item
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 2630, in pandas._libs.hashtable.Int64HashTable.get_item
KeyError: 0

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<python-input-176>", line 1, in <module>
    print(type(y_test[0]), y_test[:5])        # What type and sample of your test labels?
               ~~~~~~^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\series.py", line 1121, in __getitem__
    return self._get_value(key)
           ~~~~~~~~~~~~~~~^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\series.py", line 1237, in _get_value
    loc = self.index.get_loc(label)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 0
>>> print(type(y_pred_labels[0]), y_pred_labels[:5])  # What about predicted labels?
<class 'str'> ['1' '0' '2' '1' '1']
>>> print(type(y_test.iloc[0]), y_test.iloc[:5].values)
<class 'numpy.int64'> [1 0 2 1 1]
>>> print(type(y_pred_labels[0]), y_pred_labels[:5])
<class 'str'> ['1' '0' '2' '1' '1']
>>> # y_test: numeric
>>> # y_pred_labels: strings of digits like '0', '1', '2'
>>>
>>> # Convert predicted labels to int:
>>> y_pred_labels_int = y_pred_labels.astype(int)
>>>
>>> # Compute confusion matrix:
>>> cm = confusion_matrix(y_test, y_pred_labels_int)
>>>
>>> # Plot
>>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
>>> disp.plot(cmap=plt.cm.Blues)
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x000001F0C2ACACF0>
>>> plt.title("Confusion Matrix")
Text(0.5, 1.0, 'Confusion Matrix')
>>> plt.show()
>>> print("Confusion Matrix:\n", cm)
Confusion Matrix:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
>>> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
>>>
>>> accuracy = accuracy_score(y_test, y_pred_labels)
>>> precision = precision_score(y_test, y_pred_labels, average='weighted')
Traceback (most recent call last):
  File "<python-input-198>", line 1, in <module>
    precision = precision_score(y_test, y_pred_labels, average='weighted')
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 2247, in precision_score
    p, _, _, _ = precision_recall_fscore_support(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        y_true,
        ^^^^^^^
    ...<6 lines>...
        zero_division=zero_division,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1830, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1599, in _check_set_wise_labels
    present_labels = _tolist(unique_labels(y_true, y_pred))
                             ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\multiclass.py", line 117, in unique_labels
    raise ValueError("Mix of label input types (string and number)")
ValueError: Mix of label input types (string and number)
>>> recall = recall_score(y_test, y_pred_labels, average='weighted')
Traceback (most recent call last):
  File "<python-input-199>", line 1, in <module>
    recall = recall_score(y_test, y_pred_labels, average='weighted')
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 2429, in recall_score
    _, r, _, _ = precision_recall_fscore_support(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        y_true,
        ^^^^^^^
    ...<6 lines>...
        zero_division=zero_division,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1830, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1599, in _check_set_wise_labels
    present_labels = _tolist(unique_labels(y_true, y_pred))
                             ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\multiclass.py", line 117, in unique_labels
    raise ValueError("Mix of label input types (string and number)")
ValueError: Mix of label input types (string and number)
>>> f1 = f1_score(y_test, y_pred_labels, average='weighted')
Traceback (most recent call last):
  File "<python-input-200>", line 1, in <module>
    f1 = f1_score(y_test, y_pred_labels, average='weighted')
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 216, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1324, in f1_score
    return fbeta_score(
        y_true,
    ...<6 lines>...
        zero_division=zero_division,
    )
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1517, in fbeta_score
    _, _, f, _ = precision_recall_fscore_support(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        y_true,
        ^^^^^^^
    ...<7 lines>...
        zero_division=zero_division,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\_param_validation.py", line 189, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1830, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\metrics\_classification.py", line 1599, in _check_set_wise_labels
    present_labels = _tolist(unique_labels(y_true, y_pred))
                             ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\sklearn\utils\multiclass.py", line 117, in unique_labels
    raise ValueError("Mix of label input types (string and number)")
ValueError: Mix of label input types (string and number)
>>>
>>> print(f"Accuracy: {accuracy:.4f}")
Accuracy: 0.0000
>>> print(f"Precision (weighted): {precision:.4f}")
Traceback (most recent call last):
  File "<python-input-203>", line 1, in <module>
    print(f"Precision (weighted): {precision:.4f}")
                                   ^^^^^^^^^
NameError: name 'precision' is not defined
>>> print(f"Recall (weighted): {recall:.4f}")
Traceback (most recent call last):
  File "<python-input-204>", line 1, in <module>
    print(f"Recall (weighted): {recall:.4f}")
                                ^^^^^^
NameError: name 'recall' is not defined
>>> print(f"F1 Score (weighted): {f1:.4f}")
Traceback (most recent call last):
  File "<python-input-205>", line 1, in <module>
    print(f"F1 Score (weighted): {f1:.4f}")
                                  ^^
NameError: name 'f1' is not defined
>>> # Convert y_pred_labels (string) to integers
>>> y_pred_int = y_pred_labels.astype(int)
>>>
>>> # Now compute metrics with y_test and y_pred_int (both integer type)
>>> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
>>>
>>> accuracy = accuracy_score(y_test, y_pred_int)
>>> precision = precision_score(y_test, y_pred_int, average='weighted')
>>> recall = recall_score(y_test, y_pred_int, average='weighted')
>>> f1 = f1_score(y_test, y_pred_int, average='weighted')
>>>
>>> print(f"Accuracy: {accuracy:.4f}")
Accuracy: 1.0000
>>> print(f"Precision (weighted): {precision:.4f}")
Precision (weighted): 1.0000
>>> print(f"Recall (weighted): {recall:.4f}")
Recall (weighted): 1.0000
>>> print(f"F1 Score (weighted): {f1:.4f}")
F1 Score (weighted): 1.0000
>>> import matplotlib.pyplot as plt
>>> from sklearn.metrics import ConfusionMatrixDisplay
>>>
>>> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
>>> disp.plot(cmap=plt.cm.Blues)
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x000001F0EC68C690>
>>> plt.title("Confusion Matrix")
Text(0.5, 1.0, 'Confusion Matrix')
>>> plt.show()
>>> from sklearn.metrics import roc_curve, auc
>>> import numpy as np
>>>
>>> # Assuming y_test and y_pred_proba (probabilities) are available
>>> fpr, tpr, thresholds = roc_curve(y_test.astype(int), y_pred_proba[:,1])
Traceback (most recent call last):
  File "<python-input-232>", line 1, in <module>
    fpr, tpr, thresholds = roc_curve(y_test.astype(int), y_pred_proba[:,1])
                                                         ^^^^^^^^^^^^
NameError: name 'y_pred_proba' is not defined
>>> roc_auc = auc(fpr, tpr)
Traceback (most recent call last):
  File "<python-input-233>", line 1, in <module>
    roc_auc = auc(fpr, tpr)
                  ^^^
NameError: name 'fpr' is not defined
>>>
>>> plt.figure()
<Figure size 640x480 with 0 Axes>
>>> plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
Traceback (most recent call last):
  File "<python-input-236>", line 1, in <module>
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
             ^^^
NameError: name 'fpr' is not defined
>>> plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
[<matplotlib.lines.Line2D object at 0x000001F0EC743D90>]
>>> plt.xlim([0.0, 1.0])
(0.0, 1.0)
>>> plt.ylim([0.0, 1.05])
(0.0, 1.05)
>>> plt.xlabel('False Positive Rate')
Text(0.5, 0, 'False Positive Rate')
>>> plt.ylabel('True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
>>> plt.title('Receiver Operating Characteristic')
Text(0.5, 1.0, 'Receiver Operating Characteristic')
>>> plt.legend(loc="lower right")
<python-input-243>:1: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
<matplotlib.legend.Legend object at 0x000001F0EC743ED0>
>>> plt.show()
>>> importances = model.feature_importances_
>>> features = X.columns  # your feature names
>>>
>>> # Plot
>>> plt.figure(figsize=(10,6))
<Figure size 1000x600 with 0 Axes>
>>> plt.barh(features, importances)
<BarContainer object of 4 artists>
>>> plt.xlabel("Feature Importance")
Text(0.5, 0, 'Feature Importance')
>>> plt.title("Feature Importance in Model")
Text(0.5, 1.0, 'Feature Importance in Model')
>>> plt.show()
>>> from sklearn.metrics import classification_report
>>>
>>> # If y_test is numeric, make sure y_pred_labels is in the same format (strings or ints)
>>> # Since your y_test is numeric (ints), convert y_pred_labels to ints as well:
>>> y_pred_int_labels = le.transform(y_pred_labels)  # transform string labels back to integers
>>>
>>> report = classification_report(y_test, y_pred_int_labels, target_names=le.classes_)
>>> print(report)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.ensemble import RandomForestClassifier  # example model
>>> model = RandomForestClassifier(random_state=42)
>>> param_grid = {
...     'n_estimators': [50, 100, 200],        # Number of trees in the forest
...         'max_depth': [None, 10, 20, 30],      # Maximum depth of the tree
...             'min_samples_split': [2, 5, 10],      # Minimum samples required to split an internal node
...             }
>>> grid_search = GridSearchCV(
...     estimator=model,
...         param_grid=param_grid,
...             cv=5,                # 5-fold cross-validation
...                 scoring='accuracy',  # Metric to optimize (you can change to 'f1', 'precision' etc.)
...                     n_jobs=-1,           # Use all CPU cores
...                         verbose=2            # Show progress
...                         )
>>> grid_search.fit(X_train, y_train)
Fitting 5 folds for each of 36 candidates, totalling 180 fits
[CV] END max_depth=None, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END max_depth=None, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END max_depth=None, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END max_depth=None, min_samples_split=5, n_estimators=50; total time=   0.0s
[CV] END max_depth=None, min_samples_split=5, n_estimators=50; total time=   0.2s
[CV] END max_depth=None, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=None, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=None, min_samples_split=5, n_estimators=50; total time=   0.2s
[CV] END max_depth=None, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=5, n_estimators=100; total time=   0.1s
[CV] END max_depth=None, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=None, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=None, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=None, min_samples_split=10, n_estimators=50; total time=   0.0s
[CV] END max_depth=None, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=None, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=5, n_estimators=100; total time=   0.4s
[CV] END max_depth=None, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=10, n_estimators=50; total time=   0.2s
[CV] END max_depth=None, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=5, n_estimators=200; total time=   0.5s
[CV] END max_depth=None, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=None, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=5, n_estimators=200; total time=   0.7s
[CV] END max_depth=None, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=None, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=None, min_samples_split=10, n_estimators=100; total time=   0.4s
[CV] END .max_depth=10, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=10, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=None, min_samples_split=10, n_estimators=200; total time=   0.5s
[CV] END .max_depth=10, min_samples_split=2, n_estimators=50; total time=   0.2s
[CV] END .max_depth=10, min_samples_split=2, n_estimators=50; total time=   0.2s
[CV] END max_depth=None, min_samples_split=10, n_estimators=200; total time=   0.5s
[CV] END max_depth=10, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=None, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=None, min_samples_split=10, n_estimators=200; total time=   0.5s
[CV] END max_depth=None, min_samples_split=10, n_estimators=200; total time=   0.5s
[CV] END .max_depth=10, min_samples_split=2, n_estimators=50; total time=   0.2s
[CV] END max_depth=10, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END .max_depth=10, min_samples_split=5, n_estimators=50; total time=   0.0s
[CV] END max_depth=10, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END .max_depth=10, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=10, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=2, n_estimators=100; total time=   0.4s
[CV] END .max_depth=10, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=10, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=10, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=10, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=10, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=10, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=10, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=5, n_estimators=200; total time=   0.7s
[CV] END max_depth=10, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END max_depth=10, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=10, n_estimators=100; total time=   0.4s
[CV] END .max_depth=20, min_samples_split=2, n_estimators=50; total time=   0.2s
[CV] END .max_depth=20, min_samples_split=2, n_estimators=50; total time=   0.2s
[CV] END .max_depth=20, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=20, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=20, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=20, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=10, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=10, min_samples_split=10, n_estimators=200; total time=   0.5s
[CV] END max_depth=10, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END .max_depth=20, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=20, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=10, min_samples_split=10, n_estimators=200; total time=   0.7s
[CV] END .max_depth=20, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=20, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=20, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=20, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=20, min_samples_split=2, n_estimators=200; total time=   0.5s
[CV] END max_depth=20, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=20, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=20, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=20, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=20, min_samples_split=5, n_estimators=100; total time=   0.4s
[CV] END max_depth=20, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=20, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=20, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=20, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=20, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=5, n_estimators=200; total time=   0.5s
[CV] END max_depth=20, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=5, n_estimators=200; total time=   0.7s
[CV] END max_depth=20, min_samples_split=5, n_estimators=200; total time=   0.5s
[CV] END max_depth=20, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END .max_depth=30, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=2, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=20, min_samples_split=10, n_estimators=200; total time=   0.5s
[CV] END max_depth=20, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=30, min_samples_split=2, n_estimators=100; total time=   0.2s
[CV] END max_depth=30, min_samples_split=2, n_estimators=100; total time=   0.3s
[CV] END max_depth=20, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=20, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=2, n_estimators=100; total time=   0.4s
[CV] END .max_depth=30, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=5, n_estimators=50; total time=   0.2s
[CV] END .max_depth=30, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END .max_depth=30, min_samples_split=5, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=5, n_estimators=100; total time=   0.2s
[CV] END max_depth=30, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=30, min_samples_split=2, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=5, n_estimators=100; total time=   0.3s
[CV] END max_depth=30, min_samples_split=5, n_estimators=100; total time=   0.4s
[CV] END max_depth=30, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=5, n_estimators=100; total time=   0.4s
[CV] END max_depth=30, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=2, n_estimators=200; total time=   0.8s
[CV] END max_depth=30, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=10, n_estimators=50; total time=   0.1s
[CV] END max_depth=30, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END max_depth=30, min_samples_split=5, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=10, n_estimators=100; total time=   0.2s
[CV] END max_depth=30, min_samples_split=5, n_estimators=200; total time=   0.5s
[CV] END max_depth=30, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END max_depth=30, min_samples_split=10, n_estimators=100; total time=   0.4s
[CV] END max_depth=30, min_samples_split=5, n_estimators=200; total time=   0.7s
[CV] END max_depth=30, min_samples_split=10, n_estimators=100; total time=   0.3s
[CV] END max_depth=30, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=10, n_estimators=200; total time=   0.6s
[CV] END max_depth=30, min_samples_split=10, n_estimators=200; total time=   0.6s
GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,
             param_grid={'max_depth': [None, 10, 20, 30],
                         'min_samples_split': [2, 5, 10],
                         'n_estimators': [50, 100, 200]},
             scoring='accuracy', verbose=2)
>>> print("Best params:", grid_search.best_params_)
Best params: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
>>> print("Best cross-validated accuracy:", grid_search.best_score_)
Best cross-validated accuracy: 0.95
>>> best_model = grid_search.best_estimator_
>>>
>>> # Predict on test set
>>> y_pred = best_model.predict(X_test)
>>>
>>> # Now evaluate with metrics like accuracy, precision, recall, f1
>>> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
>>>
>>> print("Test Accuracy:", accuracy_score(y_test, y_pred))
Test Accuracy: 1.0
>>> print("Test Precision:", precision_score(y_test, y_pred, average='weighted'))
Test Precision: 1.0
>>> print("Test Recall:", recall_score(y_test, y_pred, average='weighted'))
Test Recall: 1.0
>>> print("Test F1 Score:", f1_score(y_test, y_pred, average='weighted'))
Test F1 Score: 1.0
>>> import pandas as pd
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.preprocessing import StandardScaler
>>>
>>> # Example: load dataset (replace with your dataset)
>>> data = pd.read_csv('your_dataset.csv')
Traceback (most recent call last):
  File "<python-input-287>", line 1, in <module>
    data = pd.read_csv('your_dataset.csv')
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ~~~~~~~~~~^
        f,
        ^^
    ...<6 lines>...
        storage_options=self.options.get("storage_options", None),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
        handle,
    ...<3 lines>...
        newline="",
    )
FileNotFoundError: [Errno 2] No such file or directory: 'your_dataset.csv'
>>>
>>> # Separate features and target label
>>> X = data.drop('target_column', axis=1)  # replace 'target_column' with actual label column name
Traceback (most recent call last):
  File "<python-input-290>", line 1, in <module>
    X = data.drop('target_column', axis=1)  # replace 'target_column' with actual label column name
        ^^^^
NameError: name 'data' is not defined
>>> y = data['target_column']
Traceback (most recent call last):
  File "<python-input-291>", line 1, in <module>
    y = data['target_column']
        ^^^^
NameError: name 'data' is not defined
>>>
>>> # Split into train/test sets (e.g., 80% train, 20% test)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Normalize features (important for KNN)
>>> scaler = StandardScaler()
>>> X_train = scaler.fit_transform(X_train)
>>> X_test = scaler.transform(X_test)
>>> import pandas as pd
>>> from sklearn.model_selection import train_test_split, GridSearchCV
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
>>>
>>> data = pd.read_csv('iris.csv')
Traceback (most recent call last):
  File "<python-input-306>", line 1, in <module>
    data = pd.read_csv('iris.csv')
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ~~~~~~~~~~^
        f,
        ^^
    ...<6 lines>...
        storage_options=self.options.get("storage_options", None),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\PMYLS\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
        handle,
    ...<3 lines>...
        newline="",
    )
FileNotFoundError: [Errno 2] No such file or directory: 'iris.csv'
>>> import pandas as pd
>>> from sklearn.model_selection import train_test_split, GridSearchCV
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
>>>
>>> # Load dataset
>>> df = pd.read_csv(r"C:\Users\PMYLS\OneDrive\Desktop\iris.csv")
>>>
>>> # Prepare features (X) and target (y)
>>> X = df.drop('species', axis=1)  # all columns except 'species'
>>> y = df['species']               # target column
>>>
>>> # Split into train and test (80% train, 20% test)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Normalize features for KNN (important)
>>> scaler = StandardScaler()
>>> X_train = scaler.fit_transform(X_train)
>>> X_test = scaler.transform(X_test)
>>>
>>> # Initialize KNN classifier
>>> knn = KNeighborsClassifier()
>>>
>>> # Define hyperparameter grid for GridSearchCV
>>> param_grid = {
...     'n_neighbors': [3, 5, 7, 9],       # try different neighbors
...         'weights': ['uniform', 'distance'],# different weighting schemes
...             'metric': ['euclidean', 'manhattan']  # distance metrics
...             }
>>>
>>> # Setup GridSearchCV with 5-fold cross-validation
>>> grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
>>>
>>> # Fit on training data
>>> grid_search.fit(X_train, y_train)
GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
             param_grid={'metric': ['euclidean', 'manhattan'],
                         'n_neighbors': [3, 5, 7, 9],
                         'weights': ['uniform', 'distance']},
             scoring='accuracy')
>>>
>>> # Best parameters from grid search
>>> print("Best parameters:", grid_search.best_params_)
Best parameters: {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'distance'}
>>>
>>> # Best cross-validated accuracy score
>>> print("Best cross-validated accuracy:", grid_search.best_score_)
Best cross-validated accuracy: 0.9583333333333334
>>>
>>> # Best model after grid search
>>> best_model = grid_search.best_estimator_
>>>
>>> # Predict on test data
>>> y_pred = best_model.predict(X_test)
>>>
>>> # Evaluate performance on test set
>>> print("Test Accuracy:", accuracy_score(y_test, y_pred))
Test Accuracy: 1.0
>>> print("Test Precision:", precision_score(y_test, y_pred, average='weighted'))
Test Precision: 1.0
>>> print("Test Recall:", recall_score(y_test, y_pred, average='weighted'))
Test Recall: 1.0
>>> print("Test F1 Score:", f1_score(y_test, y_pred, average='weighted'))
Test F1 Score: 1.0
>>>

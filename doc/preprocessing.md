# Preprocessing

## Spliting

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## Methods

To remove columns/rows with null values in a data frame, use the df.dropna() method

```
\# remove rows with NaN values

df.dropna()

\# remove columns with NaN values

df.dropna(axis=1)

\# all the values must be NaN to drop row

df.dropna(how='all')

\# at least 4 NaN

df.dropna(thresh=4)

\# only certain columns

df.dropna(subset=['x1', 'x3'])
```

To determine how many null values we have in each column

```
df.isnull().sum()
```

Instead of deleting columns/rows, we can set new values based of different strategies: mean, median or most_frequent

```
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, stategy='mean')
imp.fit(X_train)

imputed_data = imp.transform(X_test)
```

### Categorical data

#### Ordinal

Order matters. For example, t-shirt size.

Manually map values.

```
size_mapping = { 'XL': 3, 'L': 2, 'M': 1 }

df['tshirt_size'] = df['tshirt_size'].map(size_mapping)
```

For the labels, since it doesn't matter the value, we can use LabelEncoder

```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['label'].values)

\#and we can alwys revert
le.inverse_transform(y)
```

#### Nominal 

No order. For example, car brand.

We need to careful, cos assigning numeric values sets an order, that shouldn't exist in the first place. 

```

```

## Scalers

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X) # min value = 0, max value = 1, per column

### Question

If I scale the dataset, do I also have to use same scaler when predicting?
from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
lung_cancer = fetch_ucirepo(id=62) 
  
# data (as pandas dataframes) 
X = lung_cancer.data.features 
y = lung_cancer.data.targets 

# uzupełnienie brakujących wartości medianią 
X["Attribute4"] = X["Attribute4"].fillna(X["Attribute4"].median())
X["Attribute38"] = X["Attribute38"].fillna(X["Attribute38"].median())

# Połączenie danych
df = pd.concat([X, y], axis=1)

df.to_csv('lung_cancer_dataset_filled.csv', index=False)

print("Dane zapisane")


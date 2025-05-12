from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
lung_cancer = fetch_ucirepo(id=62) 
  
# data (as pandas dataframes) 
X = lung_cancer.data.features 
y = lung_cancer.data.targets 
  
# metadata 
print(lung_cancer.metadata) 
  
# variable information 
print(lung_cancer.variables) 


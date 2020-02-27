import pandas as pd
import numpy as np
from mlp import MLP
def main():
    # import dataset
    iris = pd.read_csv("Iris.csv")
    
    # Encoding data teks di kolom Species
    species = sorted(list(set(iris['Species'].values)))
    for i in range(len(species)): iris.loc[iris['Species'] == species[i], 'Species'] = i
    
    # training data
    iris_train = iris.drop('Species', 1).drop('Id', 1)

    # target
    iris_target = iris['Species']
    # print(len(set(iris_target)))
    # print(iris_train.values.tolist())
    mlp = MLP((7,5),5)
    print(mlp.fit(iris_train, iris_target))

if __name__ == "__main__":
    main()
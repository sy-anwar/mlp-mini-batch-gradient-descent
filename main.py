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
    mlp = MLP((200,),1)
    mlp.fit(iris_train, iris_target, threshold=0.01, learning_rate=0.005)

    # iris_goal = iris_target[:15].values.tolist() + iris_target[50:65].values.tolist() + iris_target[100:115].values.tolist()
    iris_test = iris_train[:15].values.tolist() + iris_train[50:65].values.tolist() + iris_train[100:115].values.tolist()
    # print(iris_goal)
    # print(iris_train.values)
    print(mlp.predict(iris_train.values.tolist()))
    print(mlp.score(iris_train.values.tolist(), iris_target.values.tolist()))
    # print(mlp.weights)

if __name__ == "__main__":
    main()
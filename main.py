import pandas as pd

def main():
    # import dataset
    iris = pd.read_csv("Iris.csv")
    
    # Encoding data teks di kolom Species
    species = sorted(list(set(iris['Species'].values)))
    for i in range(len(species)): iris.loc[iris['Species'] == species[i]] = i
    
    print(iris)

if __name__ == "__main__":
    main()
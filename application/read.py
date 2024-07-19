import pandas as pd

LOCALPATHTODATA = "./nd0821-c3-starter-code/starter/data/census.csv"

if __name__ == "__main__":
    print("reading data")
    df = pd.read_csv(LOCALPATHTODATA)
    print(df.head())
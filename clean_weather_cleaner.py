import pandas as pd

df = pd.read_csv("weatherHistory.csv")
temps = df["Temperature (C)"]
temps = temps.dropna()
temps.to_csv("cleaned_temps.txt", index=False, header=False)

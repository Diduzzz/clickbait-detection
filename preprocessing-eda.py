import re, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv("data/clickbait_data.csv")
# print(df.head)

##
print("Dimensioni:", df.shape)
print(df.dtypes, "\n")
print("Valori mancanti:\n", df.isna().sum(), "\n")
print("Titoli vuoti:", (df.headline.str.strip() == "").sum())
print("Titoli duplicati:", df.headline.duplicated().sum())
dup = df[df.headline.duplicated(keep=False)]

## -- DATA VISUALIZATION AND ANALYSIS

# visualizzazione classi se sono bilanciate o meno

print("\nDistribuzione Classi:")
print(df["clickbait"].value_counts())

print("\nDistribuzione Classi (%):")
print(df["clickbait"].value_counts(normalize=True)*100)     #CLASSI BILANCIATE OK!

# visualizzazione classi con grafico
counts = df["clickbait"].value_counts().sort_index()
etichette = [f"{n:,} ({n/len(df)*100:.1f}%)" for n in counts]

ax = counts.plot(
    kind="bar",
    title="Distribuzione delle classi",
    xlabel="Classe (0 = Non-Clickbait, 1 = Clickbait)",
    ylabel="Frequenza",
    color=["lightblue", "salmon"],
    rot=0,
)

ax.bar_label(ax.containers[0], labels=etichette, padding=3)
ax.margins(y=0.15)
plt.show()
import re, json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy import stats
from wordcloud import STOPWORDS, WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

df = pd.read_csv("data/clickbait_data.csv")
# print(df.head)

##      DATA CLEANING
print("Dimensioni:", df.shape)
print(df.dtypes, "\n")
# verifica dedei dati mancanti
print("Valori mancanti:\n", df.isna().sum(), "\n")      # NO MISSING DATA
print("Titoli vuoti:", (df.headline.str.strip() == "").sum())
print("Titoli duplicati:", df.headline.duplicated().sum())
df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True) #in caso dovessero esserci duplicati con questo vanno eliminati
dup = df[df.headline.duplicated(keep=False)]

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
    xlabel="Classe (0 = Non Clickbait, 1 = Clickbait)",
    ylabel="Frequenza",
    color=["lightblue", "salmon"],
    rot=0,
)

# ax.bar_label(ax.containers[0], labels=etichette, padding=3)
# ax.margins(y=0.15)
# plt.savefig("/Users/diduz/Desktop/TESI/clickbait-detection-tesi/results/classi-bilanciate.pdf", dpi=300, bbox_inches="tight")
# plt.show()

##      FEATURE ENGINEERING 
#creazione delle feature stilistiche
## 2. FEATURE ENGINEERING (STILISTICHE E STRUTTURALI)
s = df["headline"].astype(str)
s_lower = s.str.lower()

df["char_len"] = s.str.len()
df["word_count"] = s.str.split().str.len()
df["avg_word_len"] = df["char_len"] / df["word_count"].replace(0, 1)
# In letteratura (Chakraborty et al., 2016), i titoli clickbait tendono a essere più lunghi e discorsivi 
#rispetto ai titoli giornalistici tradizionali, pur impiegando un vocabolario più semplice e parole più brevi.

df["upper_word_ratio"] = s.apply(
    lambda t: sum(1 for w in t.split() if w[0].isupper())
    / max(len(t.split()), 1)
)
df["exclam_count"] = s.str.count(r"!")  
df["quest_count"] = s.str.count(r"\?")
df["has_quotes"] = s.str.contains(r'["\']').astype(int)
# Questi indicatori misurano l'enfasi grafica e la punteggiatura emotiva tipica dei portali virali (Biyani et al., 2016)

df["starts_with_digit"] = s.str.match(r"^\s*\d").astype(int)
df["has_digit"] = s.str.contains(r"\d").astype(int)
# i formati ppiù diffusi del clickbait sono i listicle ovvero articoli che vengono strutturati in elenchi numerati esempio: "17 reason why...", "20 pitcure that..."

# Le parole 5W1H sono le sei parole interrogative fondamentali della lingua inglese 
# che aprono una domanda o introducono una spiegazione
#Nel contesto del clickbait (come evidenziato nel paper di Bronakowski et al. citando Biyani et al.), i titoli clickbait iniziano molto più frequentemente con una di queste parole rispetto ai titoli giornalistici tradizionali — ad esempio "Why you should never...", "What happens when...", "How this man..." — proprio per attivare il curiosity gap, cioè promettere una risposta o un retroscena che spinge il lettore a cliccare per scoprirlo.
df["starts_with_5w1h"] = s_lower.str.match(
    r"^\s*(what|why|when|who|which|how)\b"
).astype(int)
df["has_pronoun"] = s_lower.str.contains(
    r"\b(you|your|yours|we|our|i|my|she|he|they)\b"
).astype(int)
df["has_demonstrative"] = s_lower.str.contains(
    r"\b(this|that|these|those)\b"
).astype(int)
df["has_contraction"] = s_lower.str.contains(
    r"\b\w+['’](t|s|re|ve|ll|d|m)\b"
).astype(int)
df["curiosity_words"] = s_lower.str.count(
    r"\b(reason|reasons|why|just|this|one|secret|secrets|things|actually|ever|never)\b"
)
# traduce in variabili numeriche il meccanismo psicologico e pragmatico del curiosity
# il titolo fornisce un'informazione incompleta che spinge il lettore a cliccare per colmare il vuoto informativo.

# Elenco ordinato delle 13 feature create senza upper_word_ratio poiché quasi tutti i titoli sono Title Case ovvero con la lettera iniziale maiuscola.
feature_cols = [
    "char_len",
    "word_count",
    "avg_word_len",
    "exclam_count",
    "quest_count",
    "has_quotes",
    "starts_with_digit",
    "has_digit",
    "starts_with_5w1h",
    "has_pronoun",
    "has_demonstrative",
    "has_contraction",
    "curiosity_words",
]

## MATRICE DI CORRELAZIONE
corr_cols = feature_cols + ["clickbait"]
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Matrice di correlazione tra feature stilistiche e target")
plt.tight_layout()
# plt.show()

## --- ANALISI ESPLORATIVA DEL TESTO (WORD CLOUD E FREQUENZE)

# si mantengono pronomi e parole 5W1H rilevanti per il curiosity gap
custom_stopwords = set(STOPWORDS) - {"you", "your", "this", "these", "why", "how", "what", "who", "when"}

testo_cb = " ".join(df[df["clickbait"] == 1]["headline"])
testo_nocb = " ".join(df[df["clickbait"] == 0]["headline"])

# WORD CLOUD CLICKBAIT
wc_cb = WordCloud(width=800, height=450, background_color="white", stopwords=custom_stopwords, max_words=80, random_state=42).generate(testo_cb)
plt.figure(figsize=(8, 4.5))
plt.imshow(wc_cb, interpolation="bilinear")
plt.title("Parole più frequenti Clickbait")
plt.axis("off")
plt.tight_layout()
plt.show()

# WORD CLOUD NON CLICKBAIT
wc_nocb = WordCloud(width=800, height=450, background_color="white", stopwords=custom_stopwords, max_words=80, random_state=42).generate(testo_nocb)
plt.figure(figsize=(8, 4.5))
plt.imshow(wc_nocb, interpolation="bilinear")
plt.title("Parole più frequenti dei Non Clickbait")
plt.axis("off")
plt.tight_layout()
plt.show()

# parole più frequenti nei titoli in ciascuna classe (in percentuale)
for classe, nome in [(1, "CLICKBAIT"), (0, "NON CLICKBAIT")]:
  parole = df[df["clickbait"] == classe]["headline"].str.lower().str.findall(r"\b[a-z]{2,}\b").explode()    #estrae da ogni titolo una lista di parole formate da almeno 2 lettere ed elimina punteggiatura, numeri. explode -> trasforma le liste di parola in un unica colonna, una parola per riga
  parole = parole[~parole.isin(custom_stopwords)] # filtra tutte le parole presenti nelle custom_stopword, quindi esclude tutte le parole che sono presenti nel custom_stopword
  print(f"\n Top 10 parole {nome}")
  print((parole.value_counts().head(10) / len(df[df["clickbait"] == classe]) * 100).round(2).astype(str)) # conta quante volte le parole appaiono e prende solo le prime 10 e le divide per il numero totale di titoli

## SPLIT TRAIN E TEST (80/20)
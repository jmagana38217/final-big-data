import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter
import re
import gender_guesser.detector as gender

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_es = set(stopwords.words('spanish'))

detector = gender.Detector()

plt.style.use('seaborn-v0_8-darkgrid')
sns.set(rc={'figure.figsize': (12, 6)})

df = pd.read_csv("mundial_tweets.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Fecha"] = df["Date"].dt.date
df["Hora"] = df["Date"].dt.time
df = df.drop(columns=["Date"])

# === 2. Palabras más utilizadas ===
plt.figure()
all_words = ' '.join(df['Tweet'].dropna()).lower()
all_words = re.findall(r'\b\w+\b', all_words)
filtered_words = [w for w in all_words if w not in stopwords_es and len(w) > 3]
word_freq = Counter(filtered_words).most_common(20)
word_df = pd.DataFrame(word_freq, columns=["Palabra", "Frecuencia"])
ax = word_df.plot(kind="barh", x="Palabra", y="Frecuencia", legend=False)
plt.title("Palabras más comunes en tweets")
plt.xlabel("Frecuencia")
plt.gca().invert_yaxis()
for i, v in enumerate(word_df["Frecuencia"]):
    ax.text(v + 1, i, str(v), va='center')
plt.tight_layout()

# === 3. Hashtags más utilizados ===
plt.figure()
hashtags = df['Hashtags'].dropna().astype(str).str.lower().str.split(', ')
hashtags_flat = [ht for sublist in hashtags for ht in sublist if ht]
hashtag_freq = Counter(hashtags_flat).most_common(20)
hashtag_df = pd.DataFrame(hashtag_freq, columns=["Hashtag", "Frecuencia"])
ax = hashtag_df.plot(kind="barh", x="Hashtag", y="Frecuencia", legend=False)
plt.title("Hashtags más utilizados")
plt.xlabel("Frecuencia")
plt.gca().invert_yaxis()
for i, v in enumerate(hashtag_df["Frecuencia"]):
    ax.text(v + 1, i, str(v), va='center')
plt.tight_layout()

# === 4. Tweets por día ===
plt.figure()
tweets_por_dia = df.groupby("Fecha").size()
tweets_por_dia.plot(kind="line", marker="o")
plt.title("Publicaciones por día")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de tweets")
plt.xticks(rotation=45)
plt.tight_layout()

# === 5. ¿Qué sexo publica más? ===
df["Primer_Nombre"] = df["Name"].apply(lambda x: x.split()[0] if pd.notnull(x) else None)
df["Sexo"] = df["Primer_Nombre"].apply(lambda x: detector.get_gender(x) if pd.notnull(x) else "unknown")
mapa_sexo = {
    "male": "Hombre", "mostly_male": "Hombre",
    "female": "Mujer", "mostly_female": "Mujer",
    "andy": "Ambiguo", "unknown": "Desconocido"
}
df["Sexo"] = df["Sexo"].map(mapa_sexo)
sexo_counts = df['Sexo'].value_counts()

plt.figure()
sexo_counts.plot(kind="pie", autopct=lambda p: f'{p:.1f}%\n({int(p*sexo_counts.sum()/100)})',
                 startangle=90, title="¿Qué sexo publica más?")
plt.ylabel("")
plt.tight_layout()

print("\nTotales por sexo:")
print(sexo_counts)
print("\n⚠️ Advertencia: El análisis de sexo se basa en el primer nombre y puede tener un margen de error considerable.")

# === 6. Posible spam ===
df['Es_spam'] = (df['Followers'] < 20) & (df['Friends'] < 20)
print("Usuarios potencialmente spam:", df['Es_spam'].sum())

# === 7. Tweets por hora y región ===
plt.figure()
df['Hora_int'] = pd.to_datetime(df['Hora'], format='%H:%M:%S', errors='coerce').dt.hour
tweets_por_hora_region = df.groupby([df['Place'].fillna("Sin región"), 'Hora_int']).size().unstack().fillna(0)
tweets_por_hora_region.T.plot()
plt.title("Tweets por hora y región")
plt.xlabel("Hora del día")
plt.ylabel("Cantidad de tweets")
plt.tight_layout()

# === 8. Plataforma más usada ===
plt.figure()
plataformas = df['Source'].dropna().str.lower()
plataformas = plataformas.replace({
    r'.*iphone.*': 'iPhone',
    r'.*android.*': 'Android',
    r'.*web.*': 'Web',
    r'.*ipad.*': 'iPad'
}, regex=True)
plataformas = plataformas.where(plataformas.isin(['iPhone', 'Android', 'Web', 'iPad']), 'Otro')
df['Plataforma'] = plataformas

plataformas_counts = plataformas.value_counts()
ax = plataformas_counts.plot(kind='bar')
plt.title("Plataforma desde la cual se tuiteó más")
plt.xlabel("Plataforma")
plt.ylabel("Cantidad de tweets")
for i, v in enumerate(plataformas_counts):
    ax.text(i, v + 1, str(v), ha='center')
plt.tight_layout()

# === 9. Sexo vs plataforma ===
plt.figure()
sexo_plataforma = df.pivot_table(index='Plataforma', columns='Sexo', values='Tweet', aggfunc='count').fillna(0)
sexo_plataforma.plot(kind='bar', stacked=True)
plt.title("Sexo que más tuiteó por plataforma")
plt.xlabel("Plataforma")
plt.ylabel("Tweets")
plt.tight_layout()

# === 10. Palabras promedio por tweet ===
df['Palabras_por_tweet'] = df['Tweet'].dropna().apply(lambda x: len(str(x).split()))
promedio_palabras = df['Palabras_por_tweet'].mean()
print(f"\nPromedio de palabras por tweet: {promedio_palabras:.2f}")

# === 11. Tweets más largos y más cortos ===
df['Longitud'] = df['Tweet'].dropna().apply(len)
max_tweet = df.loc[df['Longitud'].idxmax()]
min_tweet = df.loc[df['Longitud'].idxmin()]
print("\nTweet más largo:")
print(max_tweet['Tweet'])
print("\nTweet más corto:")
print(min_tweet['Tweet'])

# === 12. Likes promedio por sexo, región, plataforma ===
likes_sexo = df.groupby('Sexo')['Likes'].mean()
likes_region = df.groupby(df['Place'].fillna('Sin región'))['Likes'].mean().sort_values(ascending=False)
likes_plataforma = df.groupby('Plataforma')['Likes'].mean()

plt.figure()
ax = likes_sexo.plot(kind='bar', title="Likes promedio por sexo")
plt.ylabel("Promedio de likes")
for i, v in enumerate(likes_sexo):
    ax.text(i, v + 0.5, f'{v:.1f}', ha='center')
plt.tight_layout()

plt.figure()
ax = likes_plataforma.plot(kind='bar', title="Likes promedio por plataforma")
plt.ylabel("Promedio de likes")
for i, v in enumerate(likes_plataforma):
    ax.text(i, v + 0.5, f'{v:.1f}', ha='center')
plt.tight_layout()

print("\nLikes promedio por región (Top 10):")
print(likes_region.head(10))

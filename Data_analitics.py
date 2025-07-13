import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter
import re

# Descargar stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_es = set(stopwords.words('spanish'))

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(rc={'figure.figsize': (12, 6)})

# ========================================
# 1. Cargar y preparar datos
# ========================================
df = pd.read_csv("mundial_tweets.csv")

# Procesar fechas y horas
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Fecha"] = df["Date"].dt.date
df["Hora"] = df["Date"].dt.time
df = df.drop(columns=["Date"])

# ========================================
# 2. Palabras más utilizadas en los tweets
# ========================================
all_words = ' '.join(df['Tweet'].dropna()).lower()
all_words = re.findall(r'\b\w+\b', all_words)
filtered_words = [w for w in all_words if w not in stopwords_es and len(w) > 3]
word_freq = Counter(filtered_words).most_common(20)
pd.DataFrame(word_freq, columns=["Palabra", "Frecuencia"]).plot(kind="barh", x="Palabra", y="Frecuencia", legend=False)
plt.title("Palabras más comunes en tweets")
plt.xlabel("Frecuencia")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ========================================
# 3. Hashtags más utilizados
# ========================================
hashtags = df['Hashtags'].dropna().astype(str).str.lower().str.split(', ')
hashtags_flat = [ht for sublist in hashtags for ht in sublist if ht]
hashtag_freq = Counter(hashtags_flat).most_common(20)
pd.DataFrame(hashtag_freq, columns=["Hashtag", "Frecuencia"]).plot(kind="barh", x="Hashtag", y="Frecuencia", legend=False)
plt.title("Hashtags más utilizados")
plt.xlabel("Frecuencia")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ========================================
# 4. Número de publicaciones por día
# ========================================
tweets_por_dia = df.groupby("Fecha").size()
tweets_por_dia.plot(kind="line", marker="o")
plt.title("Publicaciones por día")
plt.xlabel("Fecha")
plt.ylabel("Cantidad de tweets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========================================
# 5. ¿Qué sexo publica más?
# ========================================
df['Sexo'] = df['Name'].str.extract(r'\b(M|F)\b', expand=False)  # suponiendo columna "Name" incluye eso (adaptar si aplica)
sexo_counts = df['Sexo'].value_counts()
sexo_counts.plot(kind="pie", autopct='%1.1f%%', startangle=90, title="¿Qué sexo publica más?")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ========================================
# 6. Posible spam: pocos followers y amigos
# ========================================
df['Es_spam'] = (df['Followers'] < 20) & (df['Friends'] < 20)
print("Usuarios potencialmente spam:", df['Es_spam'].sum())

# ========================================
# 7. Tweets por hora y región
# ========================================
df['Hora_int'] = pd.to_datetime(df['Hora'], format='%H:%M:%S', errors='coerce').dt.hour
tweets_por_hora_region = df.groupby([df['Place'].fillna("Sin región"), 'Hora_int']).size().unstack().fillna(0)
tweets_por_hora_region.T.plot()
plt.title("Tweets por hora y región")
plt.xlabel("Hora del día")
plt.ylabel("Cantidad de tweets")
plt.tight_layout()
plt.show()

# ========================================
# 8. Plataforma más usada
# ========================================
plataformas = df['Source'].dropna().str.lower()
plataformas = plataformas.replace({
    r'.*iphone.*': 'iPhone',
    r'.*android.*': 'Android',
    r'.*web.*': 'Web',
    r'.*ipad.*': 'iPad'
}, regex=True)
plataformas = plataformas.where(plataformas.isin(['iPhone', 'Android', 'Web', 'iPad']), 'Otro')
plataformas.value_counts().plot(kind='bar')
plt.title("Plataforma desde la cual se tuiteó más")
plt.xlabel("Plataforma")
plt.ylabel("Cantidad de tweets")
plt.tight_layout()
plt.show()

# ========================================
# 9. Sexo vs plataforma
# ========================================
df['Plataforma'] = plataformas
sexo_plataforma = df.pivot_table(index='Plataforma', columns='Sexo', values='Tweet', aggfunc='count').fillna(0)
sexo_plataforma.plot(kind='bar', stacked=True)
plt.title("Sexo que más tuiteó por plataforma")
plt.xlabel("Plataforma")
plt.ylabel("Tweets")
plt.tight_layout()
plt.show()

# ========================================
# 10. Palabras promedio por tweet
# ========================================
df['Palabras_por_tweet'] = df['Tweet'].dropna().apply(lambda x: len(str(x).split()))
promedio_palabras = df['Palabras_por_tweet'].mean()
print(f"Promedio de palabras por tweet: {promedio_palabras:.2f}")

# ========================================
# 11. Tweets más largos y más cortos
# ========================================
df['Longitud'] = df['Tweet'].dropna().apply(len)
max_tweet = df.loc[df['Longitud'].idxmax()]
min_tweet = df.loc[df['Longitud'].idxmin()]
print("\nTweet más largo:")
print(max_tweet['Tweet'])
print("\nTweet más corto:")
print(min_tweet['Tweet'])

# ========================================
# 12. Likes promedio por sexo, región, plataforma
# ========================================
likes_sexo = df.groupby('Sexo')['Likes'].mean()
likes_region = df.groupby(df['Place'].fillna('Sin región'))['Likes'].mean().sort_values(ascending=False)
likes_plataforma = df.groupby('Plataforma')['Likes'].mean()

print("\nLikes promedio por sexo:")
print(likes_sexo)

print("\nLikes promedio por región:")
print(likes_region.head(10))  # Mostrar las 10 principales

print("\nLikes promedio por plataforma:")
print(likes_plataforma)

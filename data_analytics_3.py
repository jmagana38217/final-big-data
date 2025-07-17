import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter
import re
import gender_guesser.detector as gender

# Cargar stopwords
try:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_es = set(stopwords.words('spanish'))
except Exception as e:
    print(f"Error al cargar stopwords: {e}")
    stopwords_es = set()

# Detector de género
detector = gender.Detector()

# Estilo visual
try:
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set(rc={'figure.figsize': (12, 6)})
except Exception as e:
    print(f"Error al aplicar estilos visuales: {e}")

# === 1. Cargar datos ===
try:
    df = pd.read_csv("mundial_tweets.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Fecha"] = df["Date"].dt.date
    df["Hora"] = df["Date"].dt.time
    df = df.drop(columns=["Date"])
except Exception as e:
    print(f"Error al cargar o procesar datos: {e}")
    df = pd.DataFrame()

if df.empty:
    print("El DataFrame está vacío. Revisa el archivo fuente.")
else:
    # === 2. Palabras más comunes ===
    try:
        all_words = ' '.join(df['Tweet'].dropna()).lower()
        words = re.findall(r'\b\w+\b', all_words)
        filtered = [w for w in words if w not in stopwords_es and len(w) > 3]
        freq = Counter(filtered).most_common(20)
        df_words = pd.DataFrame(freq, columns=["Palabra", "Frecuencia"])
        ax = df_words.plot(kind="barh", x="Palabra", y="Frecuencia", legend=False)
        for i, v in enumerate(df_words['Frecuencia']):
            plt.text(v, i, f'{v} ({v/df_words["Frecuencia"].sum()*100:.1f}%)', va='center')
        plt.title("Palabras más comunes en tweets")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al analizar palabras: {e}")

    # === 3. Hashtags más usados ===
    try:
        hashtags = df['Hashtags'].dropna().astype(str).str.lower().str.split(', ')
        hashtags_flat = [ht for sub in hashtags for ht in sub if ht]
        freq = Counter(hashtags_flat).most_common(20)
        df_ht = pd.DataFrame(freq, columns=["Hashtag", "Frecuencia"])
        df_ht.plot(kind="barh", x="Hashtag", y="Frecuencia", legend=False)
        for i, v in enumerate(df_ht['Frecuencia']):
            plt.text(v, i, f'{v} ({v/df_ht["Frecuencia"].sum()*100:.1f}%)', va='center')
        plt.title("Hashtags más utilizados")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al analizar hashtags: {e}")

    # === 4. Tweets por día ===
    try:
        df.groupby("Fecha").size().plot(marker='o')
        plt.title("Publicaciones por día")
        plt.xlabel("Fecha")
        plt.ylabel("Cantidad de tweets")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al graficar publicaciones por día: {e}")

    # === 5. Sexo que publica más ===
    try:
        df["Primer_Nombre"] = df["Name"].apply(lambda x: x.split()[0] if pd.notnull(x) else None)
        df["Sexo"] = df["Primer_Nombre"].apply(lambda x: detector.get_gender(x) if x else "unknown")
        mapa = {
            "male": "Hombre", "female": "Mujer",
            "mostly_male": "Hombre", "mostly_female": "Mujer",
            "andy": "Ambiguo", "unknown": "Desconocido"
        }
        df["Sexo"] = df["Sexo"].map(mapa)
        sexo_counts = df["Sexo"].value_counts()
        sexo_counts.plot(kind="pie", autopct='%1.1f%%', startangle=90, ylabel="")
        plt.title("¿Qué sexo publica más?")
        plt.tight_layout()
        plt.show()
        print("\n⚠️ Advertencia: El análisis de sexo puede no ser 100% preciso.")
    except Exception as e:
        print(f"Error al estimar sexo: {e}")

    # === 6. Posible spam ===
    try:
        df['Es_spam'] = (df['Followers'] < 20) & (df['Friends'] < 20)
        print("Usuarios potencialmente spam:", df['Es_spam'].sum())
    except Exception as e:
        print(f"Error al identificar spam: {e}")

    # === 7. Tweets por hora y región ===
    try:
        df['Hora_int'] = pd.to_datetime(df['Hora'], format='%H:%M:%S', errors='coerce').dt.hour
        por_region = df.groupby([df['Place'].fillna("Sin región"), 'Hora_int']).size().unstack().fillna(0)
        por_region.T.plot()
        plt.title("Tweets por hora y región")
        plt.xlabel("Hora")
        plt.ylabel("Cantidad de tweets")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al graficar tweets por hora y región: {e}")

    # === 8. Plataforma más usada ===
    try:
        plataformas = df['Source'].dropna().str.lower()
        plataformas = plataformas.replace({
            r'.*iphone.*': 'iPhone', r'.*android.*': 'Android',
            r'.*web.*': 'Web', r'.*ipad.*': 'iPad'
        }, regex=True)
        plataformas = plataformas.where(plataformas.isin(['iPhone', 'Android', 'Web', 'iPad']), 'Otro')
        df['Plataforma'] = plataformas
        plataformas.value_counts().plot(kind='bar')
        plt.title("Plataforma desde la cual se tuiteó más")
        for i, v in enumerate(plataformas.value_counts()):
            plt.text(i, v, f'{v} ({v/plataformas.value_counts().sum()*100:.1f}%)', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al analizar plataformas: {e}")

    # === 9. Sexo vs plataforma ===
    try:
        tabla = df.pivot_table(index='Plataforma', columns='Sexo', values='Tweet', aggfunc='count').fillna(0)
        tabla.plot(kind='bar', stacked=True)
        plt.title("Sexo que más tuiteó por plataforma")
        plt.xlabel("Plataforma")
        plt.ylabel("Tweets")
        for p in tabla.plot(kind='bar', stacked=True).patches:
            height = p.get_height()
            if height > 0:
                plt.text(p.get_x() + p.get_width()/2, p.get_y() + height/2,
                         f'{height:.0f} ({height/tabla.sum().sum()*100:.1f}%)',
                         ha='center', va='center')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al graficar sexo vs plataforma: {e}")

    # === 10. Palabras promedio por tweet ===
    try:
        df['Palabras_por_tweet'] = df['Tweet'].dropna().apply(lambda x: len(str(x).split()))
        print(f"Promedio de palabras por tweet: {df['Palabras_por_tweet'].mean():.2f}")
    except Exception as e:
        print(f"Error al calcular promedio de palabras: {e}")

    # === 11. Tweets más largos y cortos ===
    try:
        df['Longitud'] = df['Tweet'].dropna().apply(len)
        max_tweet = df.loc[df['Longitud'].idxmax()]
        min_tweet = df.loc[df['Longitud'].idxmin()]
        print("\nTweet más largo:\n", max_tweet['Tweet'])
        print("\nTweet más corto:\n", min_tweet['Tweet'])
    except Exception as e:
        print(f"Error al obtener tweets más largos/cortos: {e}")

    # === 12. Likes por sexo, región, plataforma ===
    try:
        df['Place'] = df['Place'].fillna('Sin región')
        likes_sexo = df.groupby('Sexo')['Likes'].mean()
        likes_region = df.groupby('Place')['Likes'].mean().sort_values(ascending=False)
        likes_plataforma = df.groupby('Plataforma')['Likes'].mean()

        likes_sexo.plot(kind='bar', title="Likes promedio por sexo")
        plt.ylabel("Promedio de likes")
        plt.tight_layout()
        plt.show()

        likes_plataforma.plot(kind='bar', title="Likes promedio por plataforma")
        plt.ylabel("Promedio de likes")
        plt.tight_layout()
        plt.show()

        print("\nLikes promedio por región:")
        print(likes_region.head(10))
    except Exception as e:
        print(f"Error al calcular likes promedio: {e}")

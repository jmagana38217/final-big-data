import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(rc={'figure.figsize': (12, 6)})

def cargar_y_procesar_datos(ruta_archivo="mundial_tweets.csv"):
    """Carga un archivo CSV y extrae fecha y hora."""
    try:
        df = pd.read_csv(ruta_archivo)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Fecha"] = df["Date"].dt.date
        df["Hora"] = df["Date"].dt.time
        df = df.drop(columns=["Date"])
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{ruta_archivo}'.")
    except Exception as e:
        print(f"Ocurrió un error al cargar los datos: {e}")
    return None

def analizar_hashtags(df):
    """Analiza los hashtags más utilizados y los visualiza."""
    try:
        hashtags = df['Hashtags'].dropna().astype(str).str.lower().str.split(', ')
        hashtags_flat = [ht for lista in hashtags for ht in lista if ht]

        frecuencia = Counter(hashtags_flat).most_common(20)
        df_hashtags = pd.DataFrame(frecuencia, columns=["Hashtag", "Frecuencia"])
        df_hashtags["Porcentaje"] = df_hashtags["Frecuencia"] / df_hashtags["Frecuencia"].sum() * 100

        # Visualización
        plt.close()
        df_hashtags.plot(kind="barh", x="Hashtag", y="Frecuencia", legend=False, color="orchid")
        for i, (freq, pct) in enumerate(zip(df_hashtags["Frecuencia"], df_hashtags["Porcentaje"])):
            plt.text(freq + 1, i, f'{freq} ({pct:.1f}%)', va='center')
        plt.title("Hashtags más utilizados")
        plt.xlabel("Frecuencia")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        # Mostrar en consola
        print("\nAnálisis de hashtags más utilizados:")
        print(df_hashtags)

    except KeyError:
        print("Error: La columna 'Hashtags' no se encuentra en el DataFrame.")
    except Exception as e:
        print(f"Ocurrió un error durante el análisis de hashtags: {e}")

if __name__ == "__main__":
    datos = cargar_y_procesar_datos()

    if datos is not None:
        analizar_hashtags(datos)

import pandas as pd             # Biblioteca fundamental para la manipulación y análisis de datos tabulares (DataFrames).
import matplotlib.pyplot as plt # Módulo para la creación de gráficos estáticos, interactivos y animados en Python.
import seaborn as sns           # Biblioteca para la visualización de datos basada en Matplotlib, que ofrece una interfaz de alto nivel para crear gráficos estadísticos atractivos e informativos.
import gender_guesser.detector as gender # Herramienta para inferir el género a partir del nombre de pila.

# Inicialización del detector de género, cargando los modelos de datos necesarios.
d = gender.Detector()

# Configuración del estilo visual global para los gráficos generados por Matplotlib y Seaborn.
plt.style.use('seaborn-v0_8-darkgrid')

# --- Funciones Auxiliares para Preparación de Datos ---

def f1(n: str, dt: gender.Detector) -> str:
    """
    Infiere el sexo de un individuo basándose en su nombre de pila utilizando la librería gender_guesser.
    Clasifica los resultados en categorías más amplias y en español.

    Args:
        n (str): El nombre de pila del usuario.
        dt (gender.Detector): Una instancia del detector de género.

    Returns:
        str: Categoría de sexo inferida ('Hombre', 'Mujer', 'Desconocido', 'Ambiguo').
    """
    if pd.notnull(n):
        pred = dt.get_gender(n)
        # Mapeo de las categorías de género en inglés a español, agrupando variantes.
        m = {
            "male": "Hombre", "mostly_male": "Hombre",
            "female": "Mujer", "mostly_female": "Mujer",
            "unknown": "Desconocido", "andy": "Ambiguo"
        }
        return m.get(pred, "Desconocido")
    return "Desconocido"

def f2(col: pd.Series) -> pd.Series:
    """
    Normaliza y agrupa los nombres de las plataformas de origen de los tweets (ej., "Twitter for iPhone" a "iPhone").
    Utiliza expresiones regulares para capturar patrones.

    Args:
        col (pd.Series): Serie de Pandas que contiene los nombres de las fuentes de los tweets.

    Returns:
        pd.Series: Serie con los nombres de las plataformas simplificados y agrupados.
    """
    # Elimina valores nulos y convierte a minúsculas para una normalización consistente.
    clean_names = col.dropna().str.lower()
    # Reemplaza patrones de texto por nombres de plataforma simplificados usando regex.
    unified_platforms = clean_names.replace({
        r'.*iphone.*': 'iPhone', r'.*android.*': 'Android',
        r'.*web.*': 'Web', r'.*ipad.*': 'iPad'
    }, regex=True)

    # Clasifica las plataformas que no coinciden con las principales categorías como 'Otro'.
    return unified_platforms.where(
        unified_platforms.isin(['iPhone', 'Android', 'Web', 'iPad']), 'Otro'
    )

# --- Función Principal de Carga y Preprocesamiento de Datos ---

def cargar_y_preparar_datos(ruta_archivo: str = "FIFA.csv") -> pd.DataFrame:
    """
    Carga un archivo CSV de tweets, realiza la limpieza inicial de datos
    y enriquece el DataFrame con columnas derivadas (Fecha, Hora, Sexo estimado, Plataforma simplificada).

    Args:
        ruta_archivo (str): Ruta al archivo CSV que contiene los datos de los tweets.

    Returns:
        pd.DataFrame: DataFrame de Pandas con los datos de tweets procesados.
    """
    try:
        df = pd.read_csv(ruta_archivo)
        print(f"✔️ Datos cargados exitosamente desde '{ruta_archivo}'.")

        # Conversión de la columna 'Date' a tipo datetime, manejando errores con 'coerce'.
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # Extracción de componentes de fecha y hora en nuevas columnas.
        df["Fecha"] = df["Date"].dt.date
        df["Hora"] = df["Date"].dt.time
        # Eliminación de la columna 'Date' original.
        df = df.drop(columns=["Date"])
        print("✔️ Fechas y horas organizadas.")

        # Extracción del primer nombre para la inferencia de género.
        df["Primer_Nombre"] = df["Name"].apply(
            lambda x: x.split()[0] if pd.notna(x) else None
        )

        # Aplicación de la función auxiliar para estimar el sexo de cada usuario.
        df["Sexo"] = df["Primer_Nombre"].apply(
            lambda x: f1(x, d)
        )
        print("✔️ Sexo estimado para cada usuario.")

        # Aplicación de la función auxiliar para simplificar los nombres de las plataformas de origen.
        df['Plataforma'] = f2(
            df['Source']
        )
        print("✔️ Plataformas de tuiteo también organizadas.")

        return df

    except FileNotFoundError:
        print(f"❌ ¡ERROR! No encontré el archivo '{ruta_archivo}'.")
        print("Asegúrate de que está en la misma carpeta que este programa.")
        return pd.DataFrame()

# ========================================
# Módulo de Análisis 1: Distribución por Plataforma
# ========================================
def analisis_plataforma(datos: pd.DataFrame):
    """
    Calcula y visualiza la frecuencia de uso de cada plataforma de tuiteo.

    Args:
        datos (pd.DataFrame): DataFrame que contiene la columna 'Plataforma'.
    """
    print("\n--- Analizando la distribución de tweets por plataforma ---")

    # Cuenta las ocurrencias únicas de cada valor en la columna 'Plataforma'.
    conteo = datos['Plataforma'].value_counts()
    print(f"✔️ Conteo de tweets por plataforma realizado. Se identificaron {len(conteo)} tipos de plataformas principales.")

    # Creación de la figura y los ejes para el gráfico.
    plt.figure(figsize=(10, 6))
    # Generación del gráfico de barras.
    ax = conteo.plot(
        kind='bar',
        color='mediumseagreen'
    )
    plt.title("Distribución de Tweets por Plataforma de Origen")
    plt.xlabel("Tipo de Plataforma")
    plt.ylabel("Cantidad de Tweets")

    # Anotación de los valores numéricos sobre cada barra.
    for i, val in enumerate(conteo):
        ax.text(i, val + 1, str(val), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print("\n--- Resultados Detallados: Frecuencia de Plataformas ---")
    print(conteo)

# ========================================
# Módulo de Análisis 2: Distribución de Sexo por Plataforma
# ========================================
def analisis_sexo_plataforma(datos: pd.DataFrame):
    """
    Examina y visualiza la distribución de tweets por sexo estimado en cada plataforma.

    Args:
        datos (pd.DataFrame): DataFrame con las columnas 'Plataforma' y 'Sexo'.
    """
    print("\n--- Analizando la distribución de sexo por plataforma de tuiteo ---")

    # Creación de una tabla pivote para contar tweets, con 'Plataforma' como índice y 'Sexo' como columnas.
    pivote = datos.pivot_table(
        index='Plataforma',
        columns='Sexo',
        values='Tweet',
        aggfunc='count'
    ).fillna(0)
    print("✔️ Conteo de tweets por sexo y plataforma completado.")

    plt.figure(figsize=(12, 7))
    # Generación de un gráfico de barras apiladas.
    pivote.plot(
        kind='bar',
        stacked=True,
        colormap='viridis',
        ax=plt.gca()
    )
    plt.title("Distribución de Tweets por Sexo Estimado y Plataforma")
    plt.xlabel("Plataforma de Tuiteo")
    plt.ylabel("Cantidad de Tweets")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sexo Estimado')
    plt.tight_layout()
    plt.show()

    print("\n--- Resultados Detallados: Conteo de Sexo por Plataforma ---")
    print(pivote)

# ========================================
# Módulo de Análisis 3: Promedio de Palabras por Tweet
# ========================================
def analisis_promedio_palabras(datos: pd.DataFrame):
    """
    Calcula y muestra el promedio de palabras por tweet en el dataset.

    Args:
        datos (pd.DataFrame): DataFrame que contiene la columna 'Tweet'.
    """
    print("\n--- Calculando el promedio de palabras por tweet ---")

    # Calcula la longitud de cada tweet (en número de palabras) y la almacena.
    datos['Conteo_Palabras'] = datos['Tweet'].dropna().apply(
        lambda x: len(str(x).split())
    )

    # Calcula la media de la columna 'Conteo_Palabras'.
    prom = datos['Conteo_Palabras'].mean()
    print(f"✔️ Promedio de palabras por tweet calculado: {prom:.2f}")

    plt.figure(figsize=(8, 5))
    # Gráfico de barras simple para visualizar el promedio.
    plt.bar(["Promedio de Palabras"], [prom], color="skyblue")
    plt.ylabel("Número Promedio de Palabras")
    plt.title("Promedio de Palabras por Tweet")
    plt.ylim(0, prom * 1.2)
    # Anota el valor promedio sobre la barra.
    plt.text(0, prom + 0.5, f"{prom:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    print("\n--- Resultado Final: Promedio de Palabras ---")
    print(f"En promedio, cada tweet tiene **{prom:.2f}** palabras.")

# ========================================
# Módulo de Análisis 4: Tweets Más Largos y Más Cortos
# ========================================
def analisis_tweets_extremos(datos: pd.DataFrame):
    """
    Identifica y muestra los tweets con la mayor y menor cantidad de caracteres.

    Args:
        datos (pd.DataFrame): DataFrame que contiene la columna 'Tweet'.
    """
    print("\n--- Identificando los tweets más largos y más cortos ---")

    # Calcula la longitud de cada tweet (en número de caracteres) y la almacena.
    datos['Longitud_Tweet'] = datos['Tweet'].dropna().astype(str).apply(len)

    # Localiza la fila correspondiente al tweet con la longitud máxima.
    t_largo = datos.loc[datos['Longitud_Tweet'].idxmax()]
    # Localiza la fila correspondiente al tweet con la longitud mínima.
    t_corto = datos.loc[datos['Longitud_Tweet'].idxmin()]
    print("✔️ Tweets más largo y más corto identificados.")

    print("\n--- Tweet Más Largo ---")
    print(f"Contenido: '{t_largo['Tweet']}'")
    print(f"Caracteres: {t_largo['Longitud_Tweet']}")

    print("\n--- Tweet Más Corto ---")
    print(f"Contenido: '{t_corto['Tweet']}'")
    print(f"Caracteres: {t_corto['Longitud_Tweet']}")

    plt.figure(figsize=(10, 7))

    # Añade el texto del tweet más largo al gráfico.
    plt.text(0.1, 0.7,
             f"**Tweet Más Largo ({t_largo['Longitud_Tweet']} caracteres):**\n"
             f"{t_largo['Tweet']}",
             fontsize=11, wrap=True,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="k", lw=0.5, alpha=0.7))

    # Añade el texto del tweet más corto al gráfico.
    plt.text(0.1, 0.2,
             f"**Tweet Más Corto ({t_corto['Longitud_Tweet']} caracteres):**\n"
             f"{t_corto['Tweet']}",
             fontsize=11, wrap=True,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", ec="k", lw=0.5, alpha=0.7))

    plt.title("Tweets con Longitud Extrema", fontsize=14, color="navy")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ========================================
# Módulo de Análisis 5: Promedio de 'Likes' por Categoría
# ========================================
def analisis_promedio_likes(datos: pd.DataFrame):
    """
    Calcula y visualiza el promedio de "Likes" por sexo estimado, las 10 regiones principales,
    y la plataforma de origen del tweet.

    Args:
        datos (pd.DataFrame): DataFrame con las columnas 'Likes', 'Sexo', 'Place', y 'Plataforma'.
    """
    print("\n--- Analizando el promedio de 'Likes' por diversas categorías ---")

    # Calcula el promedio de 'Likes' agrupado por la columna 'Sexo'.
    likes_sexo = datos.groupby('Sexo')['Likes'].mean().dropna()
    print("✔️ Promedio de 'Likes' por sexo calculado.")

    # Calcula el promedio de 'Likes' agrupado por 'Place', maneja nulos y toma el top 10.
    likes_region_top10 = datos.groupby(datos['Place'].fillna('Sin Región'))['Likes'].mean() \
                                                         .sort_values(ascending=False) \
                                                         .head(10) \
                                                         .dropna()
    print("✔️ Promedio de 'Likes' por región (Top 10) calculado.")

    # Calcula el promedio de 'Likes' agrupado por la columna 'Plataforma'.
    likes_plataforma = datos.groupby('Plataforma')['Likes'].mean().dropna()
    print("✔️ Promedio de 'Likes' por plataforma calculado.")

    # --- Visualizaciones de Promedio de Likes ---

    plt.figure(figsize=(10, 6))
    ax_sexo = likes_sexo.plot(
        kind='bar',
        title="Promedio de 'Likes' por Sexo (Estimado) del Autor",
        color=['skyblue', 'salmon', 'lightgray', 'lightgreen']
    )
    plt.ylabel("Promedio de 'Likes'")
    plt.xlabel("Sexo del Autor (Estimado)")
    plt.xticks(rotation=0)
    # Anotación de valores sobre las barras.
    for i, val in enumerate(likes_sexo):
        ax_sexo.text(i, val + 0.5, f'{val:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    ax_plataforma = likes_plataforma.plot(
        kind='bar',
        title="Promedio de 'Likes' por Plataforma Usada",
        color='gold'
    )
    plt.ylabel("Promedio de 'Likes'")
    plt.xlabel("Plataforma Usada")
    plt.xticks(rotation=45, ha='right')
    # Anotación de valores sobre las barras.
    for i, val in enumerate(likes_plataforma):
        ax_plataforma.text(i, val + 0.5, f'{val:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    print("\n--- Resultados Detallados: Promedio de 'Likes' para las 10 Regiones más populares ---")
    print(likes_region_top10)

# --- Punto de Entrada Principal del Script ---

if __name__ == "__main__":
    # La ejecución comienza aquí.
    # Se carga y preprocesa el dataset una única vez.
    datos_procesados = cargar_y_preparar_datos(ruta_archivo="FIFA.csv")

    # Verifica si la carga de datos fue exitosa antes de proceder.
    if datos_procesados.empty:
        print("El DataFrame está vacío. No se pueden realizar análisis.")
    else:
        # Ejecución secuencial de cada módulo de análisis.
        # Se pasa una copia del DataFrame.
        print("\n===== INICIANDO ANÁLISIS DE DISTRIBUCIÓN POR PLATAFORMA =====")
        analisis_plataforma(datos_procesados.copy())

        print("\n===== INICIANDO ANÁLISIS DE DISTRIBUCIÓN DE SEXO POR PLATAFORMA =====")
        analisis_sexo_plataforma(datos_procesados.copy())

        print("\n===== INICIANDO ANÁLISIS DE PROMEDIO DE PALABRAS POR TWEET =====")
        analisis_promedio_palabras(datos_procesados.copy())

        print("\n===== INICIANDO ANÁLISIS DE TWEETS CON LONGITUD EXTREMA =====")
        analisis_tweets_extremos(datos_procesados.copy())

        print("\n===== INICIANDO ANÁLISIS DE PROMEDIO DE 'LIKES' =====")
        analisis_promedio_likes(datos_procesados.copy())

    print("\n¡Todos los análisis han sido completados! 🎉")
# ========================================
# Script 1: ¿Desde Qué Plataforma se Tuiteó Más?
# ========================================

# --- Preparativos (Las Herramientas que Necesitamos) ---
import pandas as pd     # Para manejar los datos en tablas (como Excel).
import matplotlib.pyplot as plt # Para dibujar gráficos.
import seaborn as sns   # Para que los gráficos se vean más bonitos.

# Configuramos el estilo de nuestros gráficos.
plt.style.use('seaborn-v0_8-darkgrid') # Un estilo de cuadrícula oscura.
sns.set(rc={'figure.figsize': (10, 6)}) # Tamaño de la ventana del gráfico.

# --- Paso 1: Cargar los Datos (Abrir el Cuaderno de Tweets) ---
def cargar_datos_de_tweets(ruta_archivo="FIFA.csv"):
    """
    Esta parte es como abrir el archivo donde guardamos todos los tweets.
    Lee los tweets, organiza las fechas y se asegura de que todo esté listo para usar.
    """
    try:
        # Leemos el archivo CSV y lo convertimos en una tabla.
        tabla_de_tweets = pd.read_csv(ruta_archivo)
        print(f"✔️ Datos cargados exitosamente desde '{ruta_archivo}'.")

        # Convertimos la columna 'Date' a un formato de fecha real para que sea útil.
        tabla_de_tweets["Date"] = pd.to_datetime(tabla_de_tweets["Date"], errors="coerce")
        # Creamos una columna 'Fecha' solo con el día, mes, año.
        tabla_de_tweets["Fecha"] = tabla_de_tweets["Date"].dt.date
        # Creamos una columna 'Hora' solo con la hora.
        tabla_de_tweets["Hora"] = tabla_de_tweets["Date"].dt.time
        # Borramos la columna 'Date' original porque ya la separamos.
        tabla_de_tweets = tabla_de_tweets.drop(columns=["Date"])

        return tabla_de_tweets

    except FileNotFoundError:
        print(f"❌ ¡ERROR! No encontré el archivo '{ruta_archivo}'.")
        print("Asegúrate de que está en la misma carpeta que este programa.")
        return pd.DataFrame() # Devolvemos una tabla vacía si hay un error.

# --- Paso 2: Analizar las Plataformas (Clasificar los Dispositivos) ---
def analizar_plataforma_mas_usada(datos_de_tweets):
    """
    Esta es la parte principal de este script:
    1. Revisa de dónde vino cada tweet (iPhone, Android, la página web, etc.).
    2. Cuenta cuántos tweets vinieron de cada tipo de lugar.
    3. Muestra los resultados en un gráfico fácil de entender.
    """
    print("\n--- ¡Vamos a ver qué dispositivos usó la gente para tuitear! ---")

    # Tomamos la columna 'Source' (la fuente del tweet).
    # Quitamos los que no tienen información y ponemos todo en minúsculas.
    nombres_de_plataformas_originales = datos_de_tweets['Source'].dropna().str.lower()

    # Simplificamos los nombres para que sean fáciles de entender:
    # Si dice 'Twitter for iPhone', lo llamamos solo 'iPhone'. Hacemos lo mismo para Android, Web y iPad.
    plataformas_simplificadas = nombres_de_plataformas_originales.replace({
        r'.*iphone.*': 'iPhone',
        r'.*android.*': 'Android',
        r'.*web.*': 'Web',
        r'.*ipad.*': 'iPad'
    }, regex=True) # El 'regex=True' es para buscar patrones en el texto.

    # Si una plataforma no es ninguna de las anteriores, la llamamos 'Otro' para agrupar las menos comunes.
    datos_de_tweets['Plataforma'] = plataformas_simplificadas.where(
        plataformas_simplificadas.isin(['iPhone', 'Android', 'Web', 'iPad']),
        'Otro'
    )

    # Contamos cuántas veces aparece cada plataforma en nuestra lista de tweets.
    conteo_de_tweets_por_plataforma = datos_de_tweets['Plataforma'].value_counts()
    print(f"✔️ ¡Contamos los tweets por plataforma! Encontramos {len(conteo_de_tweets_por_plataforma)} tipos de plataformas principales.")

    # --- Paso 3: Mostrar los Resultados (El Gráfico y la Lista) ---
    plt.figure() # Creamos el espacio para nuestro gráfico.
    # Dibujamos un gráfico de barras.
    eje_del_grafico = conteo_de_tweets_por_plataforma.plot(
        kind='bar', # Queremos un gráfico de barras verticales.
        color='mediumseagreen' # Un color verde bonito para las barras.
    )
    plt.title("¿Desde Qué Plataforma Se Tuiteó Más Sobre el Mundial?") # El título grande del gráfico.
    plt.xlabel("Tipo de Plataforma") # La etiqueta para la parte de abajo del gráfico.
    plt.ylabel("Cantidad de Tweets") # La etiqueta para la parte de la izquierda del gráfico.

    # Ponemos el número exacto de tweets encima de cada barra.
    for i, valor_barra in enumerate(conteo_de_tweets_por_plataforma):
        eje_del_grafico.text(i, valor_barra + 1, str(valor_barra), ha='center', va='bottom')

    plt.tight_layout() # Asegura que el gráfico se vea ordenado y todo quepa bien.
    plt.show() # ¡Mostramos la ventana con el gráfico!

    print("\n--- Resultados Detallados ---")
    print("Aquí está la lista de plataformas y cuántos tweets se hicieron desde cada una:")
    print(conteo_de_tweets_por_plataforma)

# --- ¡Aquí empieza a correr el programa! ---
if __name__ == "__main__":
    nuestros_tweets_cargados = cargar_datos_de_tweets() # Cargamos nuestros tweets.

    # Si no se pudo cargar nada, le decimos al usuario y terminamos.
    if nuestros_tweets_cargados.empty:
        print("No hay datos para analizar. ¡Asegúrate de tener el archivo CSV!")
    else:
        # Si todo está bien, pasamos nuestros tweets a la función que los analiza.
        analizar_plataforma_mas_usada(nuestros_tweets_cargados)
    print("\n¡Análisis de Plataformas Completado!")
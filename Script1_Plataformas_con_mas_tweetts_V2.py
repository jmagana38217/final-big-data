import pandas as pd     
import matplotlib.pyplot as plt 
import seaborn as sns   


plt.style.use('seaborn-v0_8-darkgrid') 
sns.set(rc={'figure.figsize': (10, 6)}) 


def cargar_datos_de_tweets(nombre_archivo_csv="FIFA.csv"):
    """
    Esta parte es como abrir el archivo donde guardamos todos los tweets.
    Lee los tweets, organiza las fechas y se asegura de que todo esté listo para usar.
    """
    try:
        
        tabla_completa_de_tweets = pd.read_csv(nombre_archivo_csv)
        print(f"✔️ Datos cargados exitosamente desde '{nombre_archivo_csv}'.")

        
        tabla_completa_de_tweets["Date"] = pd.to_datetime(tabla_completa_de_tweets["Date"], errors="coerce")
        
        tabla_completa_de_tweets["Fecha"] = tabla_completa_de_tweets["Date"].dt.date
        
        tabla_completa_de_tweets["Hora"] = tabla_completa_de_tweets["Date"].dt.time
        
        tabla_completa_de_tweets = tabla_completa_de_tweets.drop(columns=["Date"])

        return tabla_completa_de_tweets

    except FileNotFoundError:
        print(f"❌ ¡ERROR! No encontré el archivo '{nombre_archivo_csv}'.")
        print("Asegúrate de que está en la misma carpeta que este programa.")
        return pd.DataFrame() 


def analizar_plataforma_mas_usada(datos_de_todos_los_tweets):
    """
    Esta es la parte principal de este script:
    1. Revisa de dónde vino cada tweet (iPhone, Android, la página web, etc.).
    2. Cuenta cuántos tweets vinieron de cada tipo de lugar.
    3. Muestra los resultados en un gráfico fácil de entender.
    """
    print("\n--- ¡Vamos a ver qué dispositivos usó la gente para tuitear! ---")


    fuentes_originales_de_tweets = datos_de_todos_los_tweets['Source'].dropna().str.lower()

   
    fuentes_de_plataformas_simplificadas = fuentes_originales_de_tweets.replace({
        r'.*iphone.*': 'iPhone',
        r'.*android.*': 'Android',
        r'.*web.*': 'Web',
        r'.*ipad.*': 'iPad'
    }, regex=True) 


    datos_de_todos_los_tweets['Plataforma'] = fuentes_de_plataformas_simplificadas.where(
        fuentes_de_plataformas_simplificadas.isin(['iPhone', 'Android', 'Web', 'iPad']),
        'Otro'
    )

    
    conteo_total_de_tweets_por_plataforma = datos_de_todos_los_tweets['Plataforma'].value_counts()
    print(f"✔️ ¡Contamos los tweets por plataforma! Encontramos {len(conteo_total_de_tweets_por_plataforma)} tipos de plataformas principales.")

    
    plt.figure() 
    
    objeto_eje_del_grafico = conteo_total_de_tweets_por_plataforma.plot(
        kind='bar', 
        color='mediumseagreen' 
    )
    plt.title("¿Desde Qué Plataforma Se Tuiteó Más Sobre el Mundial?")
    plt.xlabel("Tipo de Plataforma")
    plt.ylabel("Cantidad de Tweets")

    
    for indice_barra, valor_cantidad_tweets in enumerate(conteo_total_de_tweets_por_plataforma):
        objeto_eje_del_grafico.text(indice_barra, valor_cantidad_tweets + 1, str(valor_cantidad_tweets), ha='center', va='bottom')

    plt.tight_layout() 
    plt.show() 

    print("\n--- Resultados Detallados ---")
    print("Aquí está la lista de plataformas y cuántos tweets se hicieron desde cada una:")
    print(conteo_total_de_tweets_por_plataforma)


if __name__ == "__main__":
    
    tabla_de_datos_de_tweets = cargar_datos_de_tweets() 

    
    if tabla_de_datos_de_tweets.empty:
        print("No hay datos para analizar. ¡Asegúrate de tener el archivo CSV!")
    else:
        
        analizar_plataforma_mas_usada(tabla_de_datos_de_tweets)
    print("\n¡Análisis de Plataformas Completado! 🎉")
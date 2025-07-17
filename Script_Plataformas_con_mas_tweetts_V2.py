import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gender_guesser.detector as gender

detector_de_genero = gender.Detector()

plt.style.use('seaborn-v0_8-darkgrid')

def adivinar_sexo_del_usuario(nombre_a_revisar, herramienta_detector):
    if pd.notnull(nombre_a_revisar):
        genero_predicho = herramienta_detector.get_gender(nombre_a_revisar)
        mapa_espanol_sexo = {
            "male": "Hombre", "mostly_male": "Hombre",
            "female": "Mujer", "mostly_female": "Mujer",
            "unknown": "Desconocido", "andy": "Ambiguo"
        }
        return mapa_espanol_sexo.get(genero_predicho, "Desconocido")
    return "Desconocido"

def simplificar_nombre_de_plataforma(columna_fuente_original):
    nombres_de_plataformas_limpios = columna_fuente_original.dropna().str.lower()
    plataformas_unificadas = nombres_de_plataformas_limpios.replace({
        r'.*iphone.*': 'iPhone', r'.*android.*': 'Android',
        r'.*web.*': 'Web', r'.*ipad.*': 'iPad'
    }, regex=True)
    return plataformas_unificadas.where(
        plataformas_unificadas.isin(['iPhone', 'Android', 'Web', 'iPad']), 'Otro'
    )

def cargar_y_preparar_todos_los_datos(nombre_archivo_csv="FIFA.csv"):
    try:
        tabla_maestra_de_tweets = pd.read_csv(nombre_archivo_csv)
        print(f"✔️ Datos cargados exitosamente desde '{nombre_archivo_csv}'.")

        tabla_maestra_de_tweets["Date"] = pd.to_datetime(tabla_maestra_de_tweets["Date"], errors="coerce")
        tabla_maestra_de_tweets["Fecha"] = tabla_maestra_de_tweets["Date"].dt.date
        tabla_maestra_de_tweets["Hora"] = tabla_maestra_de_tweets["Date"].dt.time
        tabla_maestra_de_tweets = tabla_maestra_de_tweets.drop(columns=["Date"])
        print("✔️ Fechas y horas organizadas.")

        tabla_maestra_de_tweets["Primer_Nombre"] = tabla_maestra_de_tweets["Name"].apply(
            lambda nombre: nombre.split()[0] if pd.notna(nombre) else None
        )

        tabla_maestra_de_tweets["Sexo"] = tabla_maestra_de_tweets["Primer_Nombre"].apply(
            lambda nombre: adivinar_sexo_del_usuario(nombre, detector_de_genero)
        )
        print("✔️ Sexo estimado para cada usuario.")

        tabla_maestra_de_tweets['Plataforma'] = simplificar_nombre_de_plataforma(
            tabla_maestra_de_tweets['Source']
        )
        print("✔️ Plataformas de tuiteo también organizadas.")

        return tabla_maestra_de_tweets

    except FileNotFoundError:
        print(f"❌ ¡ERROR! No encontré el archivo '{nombre_archivo_csv}'.")
        print("Asegúrate de que está en la misma carpeta que este programa.")
        return pd.DataFrame()

def analizar_plataforma_mas_usada(datos_para_analisis):
    print("\n--- ¡Vamos a ver qué dispositivos usó la gente para tuitear! ---")

    conteo_total_de_tweets_por_plataforma = datos_para_analisis['Plataforma'].value_counts()
    print(f"✔️ ¡Contamos los tweets por plataforma! Encontramos {len(conteo_total_de_tweets_por_plataforma)} tipos de plataformas principales.")

    plt.figure(figsize=(10, 6))
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

    print("\n--- Resultados Detallados - Plataformas ---")
    print(conteo_total_de_tweets_por_plataforma)

def analizar_sexo_y_plataforma(datos_para_analisis):
    print("\n--- ¡Vamos a ver si los hombres y mujeres usan diferentes plataformas para tuitear! ---")

    conteo_sexo_por_plataforma = datos_para_analisis.pivot_table(
        index='Plataforma',
        columns='Sexo',
        values='Tweet',
        aggfunc='count'
    ).fillna(0)
    print("✔️ Conteo de tweets por sexo y plataforma listo.")

    plt.figure(figsize=(12, 7))
    conteo_sexo_por_plataforma.plot(
        kind='bar',
        stacked=True,
        colormap='viridis',
        ax=plt.gca()
    )
    plt.title("¿Qué Sexo (Estimado) Tuiteó Más por Plataforma?")
    plt.xlabel("Plataforma de Tuiteo")
    plt.ylabel("Cantidad de Tweets")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sexo Estimado')
    plt.tight_layout()
    plt.show()

    print("\n--- Resultados Detallados - Sexo por Plataforma ---")
    print(conteo_sexo_por_plataforma)

def calcular_promedio_palabras_por_tweet(datos_para_analisis):
    print("\n--- ¡Vamos a ver cuántas palabras tiene un tweet en promedio! ---")

    datos_para_analisis['Conteo_de_Palabras'] = datos_para_analisis['Tweet'].dropna().apply(
        lambda texto_completo: len(str(texto_completo).split())
    )

    promedio_final_de_palabras = datos_para_analisis['Conteo_de_Palabras'].mean()
    print(f"✔️ ¡Promedio de palabras por tweet calculado!")

    plt.figure(figsize=(8, 5))
    plt.bar(["Promedio de Palabras"], [promedio_final_de_palabras], color="skyblue")
    plt.ylabel("Número Promedio de Palabras")
    plt.title("Promedio de Palabras por Tweet en los Tweets del Mundial")
    plt.ylim(0, promedio_final_de_palabras * 1.2)
    plt.text(0, promedio_final_de_palabras + 0.5, f"{promedio_final_de_palabras:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    print("\n--- Resultado Final - Promedio de Palabras ---")
    print(f"En promedio, cada tweet del Mundial tiene **{promedio_final_de_palabras:.2f}** palabras.")

def encontrar_tweets_extremos(datos_para_analisis):
    print("\n--- ¡Vamos a buscar el tweet más largo y el más corto! ---")

    datos_para_analisis['Largo_del_Tweet'] = datos_para_analisis['Tweet'].dropna().astype(str).apply(len)

    el_tweet_mas_largo = datos_para_analisis.loc[datos_para_analisis['Largo_del_Tweet'].idxmax()]
    el_tweet_mas_corto = datos_para_analisis.loc[datos_para_analisis['Largo_del_Tweet'].idxmin()]
    print("✔️ ¡Tweets más largo y más corto identificados!")

    print("\n--- El Tweet Más Largo ---")
    print(f"Contenido: '{el_tweet_mas_largo['Tweet']}'")
    print(f"Tiene {el_tweet_mas_largo['Largo_del_Tweet']} caracteres.")

    print("\n--- El Tweet Más Corto ---")
    print(f"Contenido: '{el_tweet_mas_corto['Tweet']}'")
    print(f"Tiene {el_tweet_mas_corto['Largo_del_Tweet']} caracteres.")

    plt.figure(figsize=(10, 7))

    plt.text(0.1, 0.7,
             f"**Tweet Más Largo ({el_tweet_mas_largo['Largo_del_Tweet']} caracteres):**\n"
             f"{el_tweet_mas_largo['Tweet']}",
             fontsize=11, wrap=True,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="k", lw=0.5, alpha=0.7))

    plt.text(0.1, 0.2,
             f"**Tweet Más Corto ({el_tweet_mas_corto['Largo_del_Tweet']} caracteres):**\n"
             f"{el_tweet_mas_corto['Tweet']}",
             fontsize=11, wrap=True,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", ec="k", lw=0.5, alpha=0.7))

    plt.title("Los Tweets Más Largos y Más Cortos del Mundial", fontsize=14, color="navy")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analizar_likes_promedio(datos_para_analisis):
    print("\n--- ¡Vamos a ver qué tweets reciben más 'Me Gusta' en promedio! ---")

    likes_promedio_por_sexo = datos_para_analisis.groupby('Sexo')['Likes'].mean().dropna()
    print("✔️ Promedio de 'Likes' por sexo calculado.")

    likes_promedio_por_region_top_10 = datos_para_analisis.groupby(datos_para_analisis['Place'].fillna('Sin región'))['Likes'].mean() \
                                                         .sort_values(ascending=False) \
                                                         .head(10) \
                                                         .dropna()
    print("✔️ Promedio de 'Likes' por región (Top 10) calculado.")

    likes_promedio_por_plataforma = datos_para_analisis.groupby('Plataforma')['Likes'].mean().dropna()
    print("✔️ Promedio de 'Likes' por plataforma calculado.")

    plt.figure(figsize=(10, 6))
    eje_grafico_sexo = likes_promedio_por_sexo.plot(
        kind='bar',
        title="Promedio de 'Likes' por Sexo (Estimado) del Autor",
        color=['skyblue', 'salmon', 'lightgray', 'lightgreen']
    )
    plt.ylabel("Promedio de 'Likes'")
    plt.xlabel("Sexo del Autor (Estimado)")
    plt.xticks(rotation=0)
    for i, valor_barra in enumerate(likes_promedio_por_sexo):
        eje_grafico_sexo.text(i, valor_barra + 0.5, f'{valor_barra:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    eje_grafico_plataforma = likes_promedio_por_plataforma.plot(
        kind='bar',
        title="Promedio de 'Likes' por Plataforma Usada",
        color='gold'
    )
    plt.ylabel("Promedio de 'Likes'")
    plt.xlabel("Plataforma Usada")
    plt.xticks(rotation=45, ha='right')
    for i, valor_barra in enumerate(likes_promedio_por_plataforma):
        eje_grafico_plataforma.text(i, valor_barra + 0.5, f'{valor_barra:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

    print("\n--- Resultados: Promedio de 'Likes' para las 10 Regiones más populares ---")
    print(likes_promedio_por_region_top_10)

if __name__ == "__main__":
    datos_completos_de_tweets = cargar_y_preparar_todos_los_datos(nombre_archivo_csv="FIFA.csv")

    if datos_completos_de_tweets.empty:
        print("No hay datos para analizar. ¡Asegúrate de tener el archivo CSV 'FIFA.csv' en la misma carpeta!")
    else:
        print("\n===== INICIANDO ANÁLISIS DE PLATAFORMAS =====")
        analizar_plataforma_mas_usada(datos_completos_de_tweets.copy())

        print("\n===== INICIANDO ANÁLISIS DE SEXO POR PLATAFORMA =====")
        analizar_sexo_y_plataforma(datos_completos_de_tweets.copy())

        print("\n===== INICIANDO ANÁLISIS DE PROMEDIO DE PALABRAS =====")
        calcular_promedio_palabras_por_tweet(datos_completos_de_tweets.copy())

        print("\n===== INICIANDO ANÁLISIS DE TWEETS EXTREMOS =====")
        encontrar_tweets_extremos(datos_completos_de_tweets.copy())

        print("\n===== INICIANDO ANÁLISIS DE LIKES PROMEDIO =====")
        analizar_likes_promedio(datos_completos_de_tweets.copy())

    print("\nLos análisis fueron completados! ")
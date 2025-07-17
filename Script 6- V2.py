import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(rc={'figure.figsize': (6, 4)})

def cargar_y_limpiar_datos(ruta_archivo="mundial_tweets.csv"):
    """Carga el archivo CSV y maneja errores si no se encuentra o está mal formateado."""
    try:
        datos = pd.read_csv(ruta_archivo)
        return datos
    except FileNotFoundError:
        print(f"❌ Error: El archivo '{ruta_archivo}' no fue encontrado.")
    except pd.errors.EmptyDataError:
        print("❌ Error: El archivo está vacío.")
    except pd.errors.ParserError:
        print("❌ Error: El archivo no se pudo analizar correctamente.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")
    return None

if __name__ == "__main__":
    datos = cargar_y_limpiar_datos()

    if datos is not None:
        # Identificación de posible spam
        datos['Es_spam'] = (datos['Followers'] < 20) & (datos['Friends'] < 20)
        conteo_spam = datos['Es_spam'].value_counts()
        total_usuarios = len(datos)

        # Crear DataFrame resumen
        resumen_spam = pd.DataFrame({
            'Categoría': ['No Spam', 'Potencial Spam'],
            'Cantidad': [conteo_spam.get(False, 0), conteo_spam.get(True, 0)],
            'Porcentaje': [f"{conteo_spam.get(False, 0) / total_usuarios * 100:.1f}%",
                           f"{conteo_spam.get(True, 0) / total_usuarios * 100:.1f}%"]
        })

        # Visualización
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')  # Ocultar ejes
        tabla = ax.table(
            cellText=resumen_spam[['Categoría', 'Cantidad', 'Porcentaje']].values,
            colLabels=['Categoría', 'Cantidad', 'Porcentaje'],
            cellLoc='center',
            loc='upper center'
        )
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        plt.title("Análisis de Usuarios Potencialmente Spam")

        # Agregar leyenda
        leyenda = "Análisis de potencial spam:\nUsuarios con menos de 20 seguidores Y menos de 20 amigos se consideran potencialmente spam."
        plt.figtext(0.5, 0.01, leyenda, ha="center", fontsize=9,
                    bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

        # Mostrar resultados en consola
        print("\n📊 Análisis de posible spam:")
        print(resumen_spam)
        print("\n📌 Definición de potencial spam: Usuarios con menos de 20 seguidores Y menos de 20 amigos.")

import pandas as pd

# Cargar datos originales
print("Cargando datos originales...")
df = pd.read_csv("mundial_tweets.csv")

# Eliminar columna 'lang' porque todos los valores son iguales
print("Eliminando columna 'lang'...")
df = df.drop(columns=["lang"])

# Convertir columna 'Date' a datetime, y separar en 'Fecha' y 'Hora'
print("Separando fecha y hora...")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Fecha"] = df["Date"].dt.date
df["Hora"] = df["Date"].dt.time
df = df.drop(columns=["Date"])

# Mostrar resumen de datos
print("\n Vista previa de los datos limpios (primeros 10 registros):")
print(df.head(10))

print("\n Información general del DataFrame:")
print(df.info())

print("\n Conteo de valores nulos por columna:")
print(df.isnull().sum())

# Guardar archivo limpio
print("\n Guardando archivo limpio como 'mundial_tweets_limpio.csv'...")
# df.to_csv("mundial_tweets_limpio.csv", index=False)

print("\n Proceso de limpieza completado.")

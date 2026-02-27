import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. Cargar Datos
print("Cargando dataset California Housing...")
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# 2. Estadísticas Básicas
print("\n--- Estadísticas Descriptivas ---")
print(df.describe())

# 3. Visualización de Correlación
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación - California Housing")
plt.savefig("correlacion.png")
print("\n[INFO] Matriz de correlación guardada como 'correlacion.png'")

# 4. Distribución del Target
plt.figure(figsize=(8, 6))
sns.histplot(df['MedHouseVal'], kde=True, color='blue')
plt.title("Distribución del Valor Medio de la Vivienda")
plt.xlabel("MedHouseVal ($100,000s)")
plt.savefig("distribucion_target.png")
print("[INFO] Distribución del target guardada como 'distribucion_target.png'")

# 5. Visualización de las variables más correlacionadas con el target
# MedInc parece ser la más importante habitualmente
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'], alpha=0.5)
plt.title("Relación entre Ingreso Medio y Valor de la Vivienda")
plt.savefig("medinc_vs_target.png")
print("[INFO] Gráfico MedInc vs Target guardado como 'medinc_vs_target.png'")

print("\nEDA completado.")

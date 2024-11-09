import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("winequality.csv")
df = df.sample(n=50, random_state=42)
X = df.drop(columns=['quality'], errors='ignore')

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

explained_variance = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['quality'], cmap='viridis', edgecolor='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Wine Quality Dataset (2D)")
plt.colorbar(label="Wine Quality")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
components = np.arange(1, len(explained_variance) + 1)
plt.plot(components, explained_variance * 100, marker='o', linestyle='--', color='b')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("Scree Plot")
plt.show()

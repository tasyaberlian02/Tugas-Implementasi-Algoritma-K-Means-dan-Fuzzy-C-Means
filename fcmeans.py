import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def initialize_membership_matrix(n_samples, n_clusters):
    return np.random.rand(n_samples, n_clusters)

def update_centroids(data, membership_matrix):
    centroids = np.dot(data.T, membership_matrix**2) / np.sum(membership_matrix**2, axis=0)
    return centroids

def update_membership_matrix(data, centroids, m):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    return 1 / distances**(2/(m-1))

def fuzzy_cmeans(data, n_clusters, max_iters=100, m=2):
    n_samples, n_features = data.shape

    # Initialize membership matrix randomly
    membership_matrix = initialize_membership_matrix(n_samples, n_clusters)

    for _ in range(max_iters):
        # Update centroids
        centroids = update_centroids(data, membership_matrix)

        # Update membership matrix
        new_membership_matrix = update_membership_matrix(data, centroids, m)

        # Check for convergence
        if np.allclose(new_membership_matrix, membership_matrix):
            break

        membership_matrix = new_membership_matrix

    return centroids, membership_matrix

# Membaca data dari file CSV
file_path = r'C:\Users\62812\Downloads\cereal.csv'
df = pd.read_csv(file_path)

# Memilih kolom yang akan digunakan (name, calories, protein, carbo)
selected_columns = ['calories', 'protein', 'carbo']
data = df[selected_columns].values

# Meminta pengguna memasukkan jumlah cluster
n_clusters = int(input("Masukkan jumlah cluster yang diinginkan: "))

# Menjalankan algoritma Fuzzy C-Means
centroids, membership_matrix = fuzzy_cmeans(data, n_clusters)

# Menentukan keanggotaan setiap data point pada cluster
cluster_membership = np.argmax(membership_matrix, axis=1)
df['Cluster'] = cluster_membership + 1  # Penomoran cluster dimulai dari 1

# Menampilkan hasil
print("\nData Sereal setelah di-Cluster:")
print(df[['name', 'calories', 'protein', 'carbo', 'Cluster']])

# Visualisasi hasil cluster
plt.scatter(df['calories'], df['protein'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.title(f'Fuzzy C-Means Clustering of Cereal (Clusters: {n_clusters})')
plt.show()

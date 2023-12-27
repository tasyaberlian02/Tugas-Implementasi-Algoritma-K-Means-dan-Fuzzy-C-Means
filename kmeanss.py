import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Membaca data dari file CSV
file_path = r'C:\Users\62812\Downloads\cereal.csv'
df = pd.read_csv(file_path)

# Memilih kolom yang akan digunakan (name, calories, protein, carbo)
selected_columns = ['calories', 'protein', 'carbo']
X = df[selected_columns]

# Pengguna memasukan jumlah cluster
n_clusters =  int(input("Masukan Jumlah Cluster : "))

# Membuat model K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Menampilkan hasil
print("\nData Sereal setelah di-Cluster:")
print(df[['name', 'calories', 'protein', 'carbo', 'Cluster']])

# Visualisasi hasil cluster
plt.scatter(df['calories'], df['protein'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.title('K-Means Clustering of Cereal based on Calories and Protein')
plt.show()

import numpy as np
from sklearn.decomposition import PCA   
import matplotlib.pyplot as plt

#leer el numpy arrange
flat_faces_np = np.loadtxt("C:\\DMA-Grupo3\\faces_numpy.csv", delimiter=",")

n_rows=16
n_cols = 4
# def pca_caras(n_componentes, dataset, n_rows=16, n_cols = 4):
pca = PCA(n_components= 60)
pca.fit(flat_faces_np)
U = pca.components_
#U_mod = U[3:]
#print(len(U_mod))
Z = pca.transform(flat_faces_np)
X_recover = np.dot(Z,U)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.6*n_cols, 2*n_rows))

# Remover los ejes para todo
for ax in axs.flatten():
    ax.set_axis_off()

# Mostrar la cara promedio
_ = axs.flatten()[0].imshow(pca.mean_.reshape(30, 30), cmap='gray')
_ = axs.flatten()[0].set_title('Mean')

# Hacer gráficas de las auto-caras
for i, ax in zip(range(60), axs.flatten()[1:]):
    eigenface = pca.components_[i]
    _ = ax.imshow(eigenface.reshape(30,30), cmap = plt.cm.gray)
    _ = ax.set_title('PC{}'.format(i+1))
    
# pca_caras(n_componentes = 60, dataset = flat_faces_np)
from utils import *
from PIL import Image
from scipy.ndimage.filters import gaussian_filter


# Images à utiliser
save_path = "save/"
path_1 = "pictures/joconde.jpg"  # image à colorier
path_2 = "pictures/picasso.jpg"  # image palette de couleurs


# Paramètres
q = 60  # nombre de points qu'on veut tracer dans le graphique en 3d
N = 15  # nombre d'axes aléatoires sur lesquels on projette
gamma = 5e-2  # facteur de pénalisation des grandes variations
res = 2.5 * 10e4  # résolution
alpha = gamma * res  # pénalisation des grandes variations
# epsilon = 0.1
smooth = True  # régularisation post-transport
sigma = 0.5  # facteur de régularisation


# Téléversement des images
img_to_colorize = Image.open(path_1)
img_to_pick_colors = Image.open(path_2)


# Prise des dimensions et calcul des dimensions finales en fonction de la résolution
n, m, p = np.array(img_to_colorize).shape
ratio = n / m
w = np.sqrt(res / ratio)
l = ratio * w
w, l = int(w), int(l)
print("w, l : "+str((w, l)))


# Redimensionnement des images
img_1 = img_to_colorize.resize((w, l))
img_2 = img_to_pick_colors.resize((w, l))


# Transformation en tableau numpy
X = np.array(img_1)
X = X[:, :, :3]
X_c = X.copy().reshape((w * l, 3)) / 255
Y = np.array(img_2)


# Transformation des images en vecteurs
X = X[:, :, :3].reshape((w * l, 3)) / 255
Y = Y[:, :, :3].reshape((w * l, 3)) / 255


# Calcul du transport optimal
i_X, i_Y = transport3d_bis(X, Y, N, alpha=alpha, w=w, l=l)
X[i_X] = Y[i_Y]


# Redimensionnement de l'image finale
X = X.reshape((l, w, 3))
X_f = X.copy()


# Régularisaiton
if smooth:
    for k in range(3):
        X_f[:, :, k] = gaussian_filter(X_f[:, :, k], sigma=sigma)


# On trace sur une première figure les images à colorier et la palette de couleur
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(img_to_colorize.resize((w, l)))
ax2.imshow(img_to_pick_colors)
ax1.set_title("Image à colorier")
ax2.set_title("Palette de couleur")


# On trace sur une autre figure l'image coloriée
plt.figure()
plt.imshow(X_f)


# Sauvegarde de l'image coloriée
X_s = np.round(X_f * 255, 0)
X_s = np.array(X_s, dtype=np.uint8)
# print(X_s)
img = Image.fromarray(X_s, 'RGB')
img.save(save_path + path_1[9:len(path_1)-4] + '_modified' + str('.png'))


# Tracé des points dans l'espace et des segments reliant les pixels
show3d(X_c, Y, i_X, i_Y, q)


# Affichage
plt.show()


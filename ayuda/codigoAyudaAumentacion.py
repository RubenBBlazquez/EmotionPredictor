# Horizontal Flip - Damos la vuelta a las imágenes entorno al eje y
import numpy as np

keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 1))

# dado que estamos volteando horizontalmente, los valores de la coordenada y serían los mismos
# Solo cambiarían los valores de la coordenada x, todo lo que tenemos que hacer es restar nuestros valores iniciales de la coordenada x del ancho de la imagen (96)
for i in range(len(columns)):
  if i%2 == 0:
    keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x) )
	


#rotar imagen
from scipy import ndimage, misc

img_45 = ndimage.rotate(image, -45, reshape=False)
plt.imshow(img_45 , cmap=plt.cm.gray)

def rotar_puntos(points, angle):
    # Cambia los puntos en el plano de manera que la rotación está justo en el origen
    # nuestra imagen es de 96*96 ,así que restamos 48
    points = points-48

    # matriz de rotación
	# R = [ [cos(t), -sin(t)],[sin(t),cos(t)]
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # rotar los puntos
    for i in range(0,len(points),2):
        xy = np.array([points[i],points[i+1]])
        xy_rot = R@xy
        points[i],points[i+1]= xy_rot

    # volver al origen del centro de rotación
    points = points+48
    return points
	
	
# Aumentar aleatoriamente el brillo de las imágenes
# Multiplicamos los valores de los píxeles por valores aleatorios entre 1,5 y 2 para aumentar el brillo de la imagen
# Recortamos el valor entre 0 y 255
import random

keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x:np.clip(random.uniform(1.5, 2)* x, 0.0, 255.0))
augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape



# Decrementar aleatoriamente el brillo de las imágenes
#se multiplica el valor del pixel por números aleatorios entre 0 and 1
#clip entre 0 y 255

df_copy = copy.copy(keyfacial_df)
df_copy['Image'] = keyfacial_df['Image'].apply(lambda x:np.clip(random.uniform(0, 1)* x,0.0, 255.0))
augmented_df = np.concatenate((augmented_df,keyfacial_df))
augmented_df.shape

#borroso
img = image.copy()
noise = np.random.randint(low=0, high=255, size=img.shape)
factor = 0.25
plt.imshow(img+(noise*factor), cmap='gray');


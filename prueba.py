#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import gc

gc.collect()


def visualize_images_without_points(n_images: int, dataset, shape):
    """Visualize n_images images from the dataset."""
    images = dataset.sample(n_images)
    images = images.apply(lambda x: x.values.reshape(shape), axis=1).values

    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    del images

    plt.show()


def visualize_images_with_points(n_images: int, dataset, shape, points):
    """Visualize n_images images from the dataset."""
    images = dataset.sample(n_images)
    points = points.loc[images.index]
    images = images.apply(lambda x: x.values.reshape(shape), axis=1).values

    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.plot(points.values[i][0::2], points.values[i][1::2], 'ro', markersize=2)

        ax.axis('off')

    del images
    del points

    plt.show()


def split_images_points(df):
    images = df['Image'].apply(lambda x: pd.Series(x.split(' ')))
    images = images.astype(np.float32)
    points = df.iloc[:, :-1]

    return images, points


def split_images_pixels(df):
    images = df[' pixels'].apply(lambda x: pd.Series(x.split(' ')))
    images = images.astype(np.float32)

    return images


# In[3]:


facial_face_image_shape = (48, 48)
facial_face_points_shape = (96, 96)

original_facial_face_points = pd.read_csv('datasets/data.csv')
original_facial_face_images = pd.read_csv('datasets/icml_face_data.csv')

original_facial_face_points

# In[4]:


facial_emotions_images = split_images_pixels(original_facial_face_images)

visualize_images_without_points(5, dataset=facial_emotions_images, shape=facial_face_image_shape)

del facial_emotions_images

# In[5]:


facial_emotions_points_images, points = split_images_points(original_facial_face_points)

visualize_images_with_points(5, dataset=facial_emotions_points_images, shape=facial_face_points_shape, points=points)

del facial_emotions_points_images
del points


# In[6]:


def rotar_puntos(points, angle):
    # Cambia los puntos en el plano de manera que la rotación está justo en el origen
    # nuestra imagen es de 96*96 ,así que restamos 48
    points = points - 48

    # matriz de rotación
    # R = [ [cos(t), -sin(t)],[sin(t),cos(t)]
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # rotar los puntos
    for i in range(0, len(points), 2):
        xy = np.array([points[i], points[i + 1]])
        xy_rot = R @ xy
        points[i], points[i + 1] = xy_rot

    # volver al origen del centro de rotación
    points = points + 48
    return points


def get_data_augmentation_flip(df):
    df_flip = df.copy()
    columns = df.columns[:-1]

    images = pd.Series(df_flip['Image'])
    # Horizontal Flip - Damos la vuelta a las imágenes entorno al eje y
    images = images.apply(lambda x: np.flip(x).tolist())

    # dado que estamos volteando horizontalmente, los valores de la coordenada y serían los mismos
    # Solo cambiarían los valores de la coordenada x, todo lo que tenemos que hacer es restar nuestros valores iniciales de la coordenada x del ancho de la imagen (96)
    for i in range(len(columns)):
        if i % 2 == 0:
            df[columns[i]] = df[columns[i]].apply(lambda x: 96. - float(x))

    df_flip['Image'] = ' '.join(images.values.tolist())

    df_flip[columns] = rotar_puntos(df_flip.iloc[:, :-1].values, 180)

    return df_flip


facial_face_points_flipped = get_data_augmentation_flip(original_facial_face_points)
facial_face_points_flipped

# In[7]:


from scipy import ndimage, misc


def get_data_augmentation_rotate(df_to_rotate, angles):
    def rotate_images(element, rotation_angles: list):
        rotation_angle = np.random.choice(rotation_angles, 1)[0]

        image = element['Image']

        rotated_image = np.array(image.split(' ')).astype(np.float32).reshape(facial_face_points_shape)
        rotated_image = np.array(ndimage.rotate(rotated_image, -rotation_angle, reshape=False)).astype(np.float32)

        rotated_points = rotar_puntos(element[:-1].values, rotation_angle)
        element.loc[element[:-1].index.values] = rotated_points
        element['Image'] = ' '.join(rotated_image.reshape(-1).astype('str').tolist())

        return element

    new_df_rotations = df_to_rotate.apply(rotate_images, rotation_angles=angles, axis=1)
    df_rotations = pd.concat([df_to_rotate, new_df_rotations])

    print(f'Rotated {len(df_rotations)} images')

    return df_rotations


angles_to_rotate = [-120, -80, -50, -10, 10, 15, 25, 50, 80, 90, 120, 150, 165]

facial_face_points_rotated1 = get_data_augmentation_rotate(original_facial_face_points, angles_to_rotate)
facial_face_points_rotated1 = get_data_augmentation_rotate(facial_face_points_rotated1, angles_to_rotate)
facial_face_points_rotated1 = get_data_augmentation_rotate(facial_face_points_rotated1, angles_to_rotate)

# In[8]:


facial_face_points_rotated2 = get_data_augmentation_rotate(original_facial_face_points, angles_to_rotate)
facial_face_points_rotated2 = get_data_augmentation_rotate(facial_face_points_rotated2, angles_to_rotate)
facial_face_points_rotated2 = get_data_augmentation_rotate(facial_face_points_rotated2, angles_to_rotate)

# In[9]:


facial_face_points_rotated3 = get_data_augmentation_rotate(original_facial_face_points, angles_to_rotate)
facial_face_points_rotated3 = get_data_augmentation_rotate(facial_face_points_rotated3, angles_to_rotate)
facial_face_points_rotated3 = get_data_augmentation_rotate(facial_face_points_rotated3, angles_to_rotate)

# In[15]:


facial_face_points_rotated = pd.concat(
    [facial_face_points_rotated1, facial_face_points_rotated2, facial_face_points_rotated3])
gc.collect()


# In[16]:


def set_random_brightness(dataframe_to_brightness):
    def change_image_brightness(element):
        brightness = np.random.uniform(0, 2)

        image = np.array(element['Image'].split(' ')).astype(np.float32).reshape(facial_face_points_shape)

        element['Image'] = ' '.join(np.array(image * brightness).reshape(-1).astype(str).tolist())

        return element

    new_dataframe = dataframe_to_brightness.apply(change_image_brightness, axis=1)

    return pd.concat([dataframe_to_brightness, new_dataframe])


facial_face_points_with_brightness = set_random_brightness(original_facial_face_points)
facial_face_points_with_brightness = set_random_brightness(facial_face_points_with_brightness)

facial_face_points_with_brightness

# In[12]:


gc.collect()

# In[13]:


augmented_facial_face_points = pd.DataFrame()
augmented_facial_face_points = pd.concat([augmented_facial_face_points, original_facial_face_points], ignore_index=True,
                                         axis=0)
del original_facial_face_points

augmented_facial_face_points = pd.concat([augmented_facial_face_points, facial_face_points_flipped], ignore_index=True,
                                         axis=0)
del facial_face_points_flipped

augmented_facial_face_points = pd.concat([augmented_facial_face_points, facial_face_points_rotated], ignore_index=True,
                                         axis=0)
del facial_face_points_rotated

augmented_facial_face_points = pd.concat([augmented_facial_face_points, facial_face_points_with_brightness],
                                         ignore_index=True, axis=0)
del facial_face_points_with_brightness

print(f'Augmented dataset has {len(augmented_facial_face_points)} images')

# In[14]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

augmented_facial_face_points['Image'] = augmented_facial_face_points['Image'].apply(
    lambda x: scaler.fit_transform(np.array(x).reshape(-1, 1)).reshape(-1).astype(np.float32))

augmented_facial_face_points

# In[ ]:


# suffle pandas dataframe
augmented_facial_face_points = augmented_facial_face_points.sample(frac=1).reset_index(drop=True)
augmented_facial_face_points

# In[ ]:


import json

print(augmented_facial_face_points['Image'].tolist())
augmented_facial_face_points['Image'] = augmented_facial_face_points['Image'].apply(lambda x: json.dumps(x.tolist()))
augmented_facial_face_points.to_csv('datasets/augmented_data.csv', sep=':', index=False)

# In[ ]:


from ast import literal_eval

augmented_data = pd.read_csv('datasets/augmented_data.csv', converters={'Image': lambda x: print(x)}, sep=':')
augmented_data

# In[ ]:


dataframe_shape = normalized_augmented_facial_face_points.shape
dataframe_range = range(dataframe_shape[0] - 1000, dataframe_shape[0])
print(dataframe_range)
images = normalized_augmented_facial_face_points.loc[dataframe_range, 'Image'].apply(lambda x: pd.Series(x))

visualize_images_with_points(5, dataset=images, shape=facial_face_points_shape,
                             points=normalized_augmented_facial_face_points.iloc[dataframe_range, :-1])

# In[ ]:

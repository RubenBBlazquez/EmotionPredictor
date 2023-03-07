#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time

import numpy as np
import pandas as pd
import gc

from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler

gc.collect()


def clean_original_dataset(dataset_path: str):
    """Clean the original dataset."""
    df = pd.read_csv(dataset_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    columns = df.columns[:-1]

    df[columns] = df[columns].astype('float32')

    for index in range(0, 30):
        df.drop(df[df[columns[index]] > 96].index, inplace=True)

    def normalize_images(image):
        if image.between(0, 1).all():
            return ' '.join(image.astype(str).tolist())

        return ' '.join((image / 255).astype(str).tolist())

    images_split = pd.DataFrame(
        df['Image'].str.split(
            ' ', expand=True).to_numpy().astype('float16')
    )

    df['Image'] = images_split.apply(normalize_images, axis=1)

    df.to_csv('datasets/data.csv', index=False)

def visualize_images_with_points(n_images: int, dataset, shape, points):
    """Visualize n_images images from the dataset."""
    images = dataset.sample(n_images)
    points = points.loc[images.index]
    images = images.apply(
        lambda x: x.str.split(' ', expand=True).astype(np.float32).to_numpy().reshape(shape), axis=1
    ).values

    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.plot(points.values[i][0::2], points.values[i][1::2], 'ro', markersize=2)

        ax.axis('off')

    plt.show()


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
        next_i = i + 1

        xy = np.array([points[i], points[next_i]])
        xy_rot = R @ xy

        points[i], points[next_i] = xy_rot

    # volver al origen del centro de rotación
    points = points + 48
    return points


def get_data_augmentation_flip(df):
    df_flip = df.copy()
    columns = df.columns[:-1]

    images_flip = df_flip['Image'].apply(
        lambda x: ' '.join(np.flip(np.array(x.split(' ')).reshape(96, 96), axis=1).reshape(-1).astype(str).tolist())
    )

    # dado que estamos volteando horizontalmente, los valores de la coordenada y serían los mismos
    # Solo cambiarían los valores de la coordenada x, todo lo que tenemos que hacer es restar nuestros valores iniciales de la coordenada x del ancho de la imagen (96)
    for i in range(len(columns)):
        if i % 2 == 0:
            df_flip[columns[i]] = df_flip[columns[i]].apply(lambda x: 96. - float(x))

    df_flip['Image'] = images_flip

    return df_flip


def manipulate_images(facial_face_images, images_shape, save_original) -> pd.DataFrame:
    facial_face_points_flipped = get_data_augmentation_flip(facial_face_images)

    # print(f'facial_face_points_flipped dataset has {len(facial_face_points_flipped)} images')

    def get_data_augmentation_rotate(df_to_rotate, angles):
        def rotate_images(element, rotation_angles: list):
            rotation_angle = np.random.choice(rotation_angles, 1)[0]

            image = np.array(element['Image'].split(' ')).astype(np.float32)
            rotated_image = image.reshape(images_shape)
            rotated_image = np.array(ndimage.rotate(rotated_image, -rotation_angle, reshape=False))

            rotated_points = rotar_puntos(element[:-1].values, rotation_angle)
            element.loc[element[:-1].index.values] = rotated_points
            element['Image'] = ' '.join(rotated_image.reshape(-1).astype(str).tolist())

            return element

        new_df_rotations = df_to_rotate.apply(rotate_images, rotation_angles=angles, axis=1)

        return new_df_rotations

    angles_to_rotate = [-15, -25, -50, -80, -10, 10, 15, 25, 50, 80]

    facial_face_points_rotated = get_data_augmentation_rotate(facial_face_images, angles_to_rotate)

    # print(f'facial_face_points_rotated dataset has {len(facial_face_points_rotated)} images')

    def set_random_brightness(dataframe_to_brightness):
        def change_image_brightness(element):
            brightness = np.random.uniform(0, 2)

            image = np.array(element['Image'].split(' ')).astype(np.float32).reshape(images_shape)

            element['Image'] = ' '.join(np.array(image * brightness).reshape(-1).astype(str).tolist())

            return element

        new_dataframe = dataframe_to_brightness.apply(change_image_brightness, axis=1)

        return pd.concat([dataframe_to_brightness, new_dataframe])

    facial_face_points_with_brightness = set_random_brightness(facial_face_images)

    # print(f'facial_face_points_with_brightness dataset has {len(facial_face_points_with_brightness)} images')

    augmented_facial_face_points = [
        facial_face_points_flipped,
        facial_face_points_rotated,
        facial_face_points_with_brightness
    ]

    if save_original:
        augmented_facial_face_points.append(facial_face_images)

    augmented_facial_face_points = pd.concat(augmented_facial_face_points, ignore_index=True, axis=0)
    # print(f'augmented_facial_face_points dataset has {len(augmented_facial_face_points)} images')

    return augmented_facial_face_points


def data_augmentation(images_shape, dataset_path, augmented_file_path, is_feather=False, save_original=True,
                      chunk_size=10000, n_elements_to_generate=50000):
    if is_feather:
        facial_face_images = pd.read_feather(dataset_path)
    else:
        facial_face_images = pd.read_csv(dataset_path)

    print(f'original dataset has {chunk_size} chunks with {len(facial_face_images)} images in total')

    facial_face_images = np.array_split(facial_face_images, chunk_size)

    all_manipulated_images = pd.DataFrame()

    for facial_face_images_chunk in facial_face_images:
        if len(all_manipulated_images) >= n_elements_to_generate:
            break

        augmented_facial_face_points = manipulate_images(facial_face_images_chunk, images_shape, save_original)
        all_manipulated_images = pd.concat([all_manipulated_images, augmented_facial_face_points]).reset_index(
            drop=True)

        print(f'all_manipulated_images dataset has {len(all_manipulated_images)} images')

    print(f'Suffelling dataset.....')
    all_manipulated_images = all_manipulated_images.sample(frac=1).reset_index(drop=True)

    print(f'Saving dataset.....')
    all_manipulated_images.to_feather(f'{augmented_file_path}.feather')


if __name__ == '__main__':
    clean_original_dataset('datasets/data.csv')

    print(f'cleaned_original_dataset dataset has {len(pd.read_csv("datasets/data.csv"))} images')

    time_start = time.time()
    data_augmentation((96, 96), 'datasets/data.csv', 'datasets/augmented_data', is_feather=False, chunk_size=300)
    print(time.time() - time_start)
    time_start = time.time()
    print()

    data_augmentation((96, 96), 'datasets/augmented_data.feather', 'datasets/augmented_data', is_feather=True,
                      save_original=False, chunk_size=600, n_elements_to_generate=50000)
    print(time.time() - time_start)
    time_start = time.time()
    print()




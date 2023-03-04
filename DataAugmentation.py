#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

import numpy as np
import pandas as pd
import gc
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler

gc.collect()

if __name__ == '__main__':
    facial_face_image_shape = (48, 48)
    facial_face_points_shape = (96, 96)

    original_facial_face_points = pd.read_csv('datasets/data.csv')
    original_facial_face_images = pd.read_csv('datasets/icml_face_data.csv')

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

        images_to_flip = pd.Series(df_flip['Image'])
        images_flip = images_to_flip.apply(lambda x: ' '.join(np.flip(np.array(x.split(' ')).astype(np.float32)).astype(str).tolist()))

        # dado que estamos volteando horizontalmente, los valores de la coordenada y serían los mismos
        # Solo cambiarían los valores de la coordenada x, todo lo que tenemos que hacer es restar nuestros valores iniciales de la coordenada x del ancho de la imagen (96)
        for i in range(len(columns)):
            if i % 2 == 0:
                df[columns[i]] = df[columns[i]].apply(lambda x: 96. - float(x))

        df_flip['Image'] = pd.Series(images_flip.values.tolist())

        df_flip[columns] = rotar_puntos(df_flip.iloc[:, :-1].values, 180)

        return df_flip


    facial_face_points_flipped = get_data_augmentation_flip(original_facial_face_points)

    points = facial_face_points_flipped.iloc[:, :-1]
    images = facial_face_points_flipped['Image'].apply(lambda x: pd.Series(x))


    def get_data_augmentation_rotate(df_to_rotate, angles):
        def rotate_images(element, rotation_angles: list):
            rotation_angle = np.random.choice(rotation_angles, 1)[0]

            image = np.array(element['Image'].split(' ')).astype(np.float32)
            rotated_image = image.reshape(facial_face_points_shape)
            rotated_image = np.array(ndimage.rotate(rotated_image, -rotation_angle, reshape=False))

            rotated_points = rotar_puntos(element[:-1].values, rotation_angle)
            element.loc[element[:-1].index.values] = rotated_points
            element['Image'] = ' '.join(rotated_image.reshape(-1).astype(str).tolist())

            return element

        new_df_rotations = df_to_rotate.apply(rotate_images, rotation_angles=angles, axis=1)
        df_rotations = pd.concat([df_to_rotate, new_df_rotations])

        print(f'Rotated {len(df_rotations)} images')

        return df_rotations


    angles_to_rotate = [-120, -80, -50, -10, 10, 15, 25, 50, 80, 90, 120, 150, 165]

    facial_face_points_rotated = get_data_augmentation_rotate(original_facial_face_points, angles_to_rotate)
    facial_face_points_rotated = get_data_augmentation_rotate(facial_face_points_rotated, angles_to_rotate)
    facial_face_points_rotated = get_data_augmentation_rotate(facial_face_points_rotated, angles_to_rotate)
    print(f'facial_face_points_rotated dataset has {len(facial_face_points_rotated)} images')


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
    print(f'facial_face_points_with_brightness dataset has {len(facial_face_points_with_brightness)} images')

    augmented_facial_face_points = [
        original_facial_face_points,
        facial_face_points_flipped,
        facial_face_points_rotated,
        facial_face_points_with_brightness
    ]
    augmented_facial_face_points = pd.concat(augmented_facial_face_points, ignore_index=True, axis=0)
    print(f'augmented_facial_face_points dataset has {len(augmented_facial_face_points)} images')

    scaler = MinMaxScaler()
    normalized_augmented_facial_face_points = augmented_facial_face_points.copy()

    normalized_augmented_facial_face_points['Image'] = normalized_augmented_facial_face_points['Image'].apply(
        lambda x: ' '.join(
            scaler.fit_transform(np.array(x.split(' ')).reshape(-1, 1).astype(np.float32)).reshape(-1).astype(str)
        )
    )

    normalized_augmented_facial_face_points = normalized_augmented_facial_face_points.\
        sample(frac=1)\
        .reset_index(drop=True)
    normalized_augmented_facial_face_points.to_csv('datasets/augmented_data.csv', index=False)
    print(f'normalized_augmented_facial_face_points dataset has {len(normalized_augmented_facial_face_points)} images')

    augmented_data = pd.read_csv('datasets/augmented_data.csv',
                                 converters={'Image': lambda x: np.array(x.split(' ')).astype(np.float32)})
    print(f'Augmented dataset has {len(augmented_data)} images')

    dataframe_shape = normalized_augmented_facial_face_points.shape
    dataframe_range = range(dataframe_shape[0] - 1000, dataframe_shape[0])
    images = normalized_augmented_facial_face_points.loc[dataframe_range, 'Image'].apply(lambda x: pd.Series(x))
    print(f'final Augmented dataset has {len(images)} images')

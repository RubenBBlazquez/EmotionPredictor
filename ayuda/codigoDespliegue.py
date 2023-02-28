import json
import tensorflow.keras.backend as K

def deploy(directory, model):
  MODEL_DIR = directory
  version = 1 

  # Juntamos el directorio del temp model con la versión elegida
  # El resultado será = '\tmp\version number'
  export_path = os.path.join(MODEL_DIR, str(version))
  print('export_path = {}\n'.format(export_path))

  # Guardemos el modelo con saved_model.save
  # Si el directorio existe, debemos borrarlo con '!rm' 
  # rm elimina cada fichero especificado usando la consola de comandos. 

  if os.path.isdir(export_path):
    print('\nmodelo ya guardado, limpiando...\n')
    !rm -r {export_path}

  tf.saved_model.save(model, export_path)

  os.environ["MODEL_DIR"] = MODEL_DIR



directorio='C:/modelos/tensorflow_serving/serving/tensorflow_serving/servables/tensorflow/modelos/facialKeyPoints'
deploy(directorio, model_1_facialKeyPoints)

#si usas docker, despliega el contenedor con 
#set-Variable -Name "MODELO" -Value "C:/modelos/tensorflow_serving/serving/tensorflow_serving/servables/tensorflow/modelos/facialKeyPoints"
#docker run --name=TensorServerFacialKeyPoints -t --rm -p 8501:8501  -v "$MODELO/:/models/facialKeyPoints" -e MODEL_NAME=facialKeyPoints tensorflow/serving

directorio='C:/modelos/tensorflow_serving/serving/tensorflow_serving/servables/tensorflow/modelos/facialExpressions'
deploy(directorio, model_2_emotion)

#si usas docker, despliega el contenedor con 
#set-Variable -Name "MODELO" -Value "C:/modelos/tensorflow_serving/serving/tensorflow_serving/servables/tensorflow/modelos/facialExpressions"
#docker run --name=TensorServerFacialExpressions -t --rm -p 8502:8501  -v "$MODELO/:/models/facialExpressions" -e MODEL_NAME=facialExpressions tensorflow/serving

import json
import requests

# Función para hacer predicciones con el modelo publicado
def response(data):
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://localhost:8501/v1/models/facialKeyPoints/versions/1:predict', data=data, headers=headers, verify = False)
  df_predict = json.loads(json_response.text)['predictions']
  json_response = requests.post('http://localhost:8502/v1/models/facialExpressions/versions/1:predict', data=data, headers=headers, verify = False)
  df_emotion = np.argmax(json.loads(json_response.text)['predictions'], axis = 1)
  
  # Redimensión de (856,) a (856,1)
  df_emotion = np.expand_dims(df_emotion, axis = 1)

  # Convertir las predicciones en un dataframe
  df_predict= pd.DataFrame(df_predict, columns = columns)

  # Añadimos la emoción al dataframe de predicciones 
  df_predict['emotion'] = df_emotion

  return df_predict


# HACER PETICIONES AL MODELO CON TENSORFLOW SERVING
# Vamos a crear un objeto JSON y hacer 10 predicciones
data = json.dumps({"signature_name": "serving_default", "instances": X_test[300:310].tolist()})
print(data)

# Hacer una predicción 
df_predict = response(data)


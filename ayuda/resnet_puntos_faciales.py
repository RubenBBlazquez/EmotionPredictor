#BLOQUE RES (1 bloque convolucional + 2 identidad)
def res_block(X, filter, stage):
	# Bloque Convolucional
	X_copy = X

	f1 , f2, f3 = filter

	# Camino Principal
	X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = MaxPool2D((2,2))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
	X = Activation('relu')(X) 

	X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
	X = Activation('relu')(X) 

	X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


	# Camino Corto
	X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
	X_copy = MaxPool2D((2,2))(X_copy)
	X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

	# Añadir
	X = Add()([X,X_copy])
	X = Activation('relu')(X)

	# Bloque de Identidad 1
	X_copy = X


	# Camino Principal
	X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
	X = Activation('relu')(X) 

	X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
	X = Activation('relu')(X) 

	X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

	# Añadir
	X = Add()([X,X_copy])
	X = Activation('relu')(X)

	# Bloque de Identidad 2
	X_copy = X


	# Camino Principal
	X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
	X = Activation('relu')(X) 

	X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
	X = Activation('relu')(X) 

	X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
	X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

	# Añadir
	X = Add()([X,X_copy])
	X = Activation('relu')(X)

	return X
  
  

#DEFINICIÓN DE LA RES NET
input_shape = (96, 96, 1)

# Tamaño del tensor de entrada
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3,3))(X_input)

# 1 - Fase
X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2 - Fase
X = res_block(X, filter= [64,64,256], stage= 2)

# 3 - Fase
X = res_block(X, filter= [128,128,512], stage= 3)

# 4 - Fase
#X = res_block(X, filter= [256,256,1024], stage= 4)

# Average Pooling
X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

# Capa Final
X = Flatten()(X)
X = Dense(4096, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation = 'relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation = 'relu')(X)


model_1_facialKeyPoints = keras.Model( inputs= X_input, outputs = X)
model_1_facialKeyPoints.summary()

adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
model_1_facialKeyPoints.compile(loss = "mean_squared_error", optimizer = adam , metrics = ['accuracy'])

checkpointer = ModelCheckpoint(filepath = "FacialKeyPoints_weights.hdf5", verbose = 1, save_best_only = True)
history = model_1_facialKeyPoints.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_split = 0.05, callbacks=[checkpointer])
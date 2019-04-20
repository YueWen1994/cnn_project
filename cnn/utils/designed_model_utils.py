def designedCNN_Model():
    data = helper.prepDataforCNN(numChannel = 3, feat_norm = True)
    X_train = data["trainX"]
    X_val  = data["valdX"]
    y_train = data["trainY"]
    y_val  = data["valdY"]

    _,row, col,channel = X_train.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 75
    batch_size = 64

    optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    # optim = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config = config)

    input = keras.Input(shape=(row,col,channel), name='customModel')
    M = Conv2D(16,(3,3),activation='relu',padding='same',name = 'conv_16_1')(input)
    M = Conv2D(16,(3, 3), activation ='relu', padding='same',name = 'conv_16_2')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)

    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_32_01')(M)
    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_32_02')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)
    M = Dropout(0.5)(M)

    M = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv2_48_01')(M)
    M = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv2_48_02')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)

    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_64_1')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same', name = 'conv2_64_2')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_64_3')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)

    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_1')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_2')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_3')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)

    M = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_5')(M)
    M = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_6')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)
    M = Dropout(0.5)(M)

    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_1')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_2')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_3')(M)
                # kernel_regularizer=regularizers.l2(0.01),
                # activity_regularizer=regularizers.l1(0.01))(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)

    M = Conv2D(512, (5, 5), activation='relu', padding='same',name = 'conv2_512_1')(M)
    M = Conv2D(512, (5, 5), activation='relu', padding='same',name = 'conv2_512_2')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides= 1)(M)
    M = Dropout(0.25)(M)
    # M = keras.layers.BatchNormalization(axis=-1)(M)

    Mout = Flatten()(M)
    Mout = Dense(2048, activation='relu', name = 'FC1_2048')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC1_1024')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC2_1024')(Mout)
    # Mout = Dropout(0.5)(Mout)

    numd_SM = Dense(digLen,    activation='softmax',name = 'num')(Mout)
    dig1_SM = Dense(numDigits, activation='softmax',name = 'dig1')(Mout)
    dig2_SM = Dense(numDigits, activation='softmax',name = 'dig2')(Mout)
    dig3_SM = Dense(numDigits, activation='softmax',name = 'dig3')(Mout)
    dig4_SM = Dense(numDigits, activation='softmax',name = 'dig4')(Mout)
    numB_SM = Dense(2,         activation='softmax',name = 'nC')(Mout)
    out = [numd_SM, dig1_SM ,dig2_SM, dig3_SM, dig4_SM, numB_SM]

    svhnModel = keras.Model(inputs = input, outputs = out)

    lr_metric = get_lr_metric(optim)
    svhnModel.compile(loss = 'sparse_categorical_crossentropy', #ceLoss ,
                      optimizer= optim,
                      metrics=  ['accuracy']) #[])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                  factor = 0.1,
                                                  verbose = 1,
                                                  patience= 2,
                                                  cooldown= 1,
                                                  min_lr = 0.00001)
    svhnModel.summary()

    callback = []
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/designedBGRClassifier.hdf5',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)
    tb = keras.callbacks.TensorBoard(log_dir = 'logs',
                                      write_graph = True,
                                      batch_size = batch_size,
                                      write_images = True)
    es = keras.callbacks.EarlyStopping(monitor= 'loss',  #'dig1_loss',
                                       min_delta=0.000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(tb)
    callback.append(es)
    callback.append(checkpointer)
    callback.append(reduce_lr)


    # svhnModel.fit_generator(
    #                   datagen.flow(ctrain, ctrlab, batch_size=batch_size),
    #                   batch_size = batch_size,
    #                   epochs=epochs,
    #                   verbose=1,
    #                   shuffle = True,
    #                   validation_data=(cvald, cvlab),
    #                   callbacks= callback)


    # fits the model on batches with real-time data augmentation:
    # svhnModel.fit_generator(datagen.flow(ctrain, ctrlab, batch_size=batch_size),
    #                         steps_per_epoch=len(ctrain) / batch_size,
    #                         epochs=epochs,
    #                         verbose=1,
    #                         validation_data = (cvald, cvlab),
    #                         callbacks= callback)
    #
    designHist = svhnModel.fit(x = X_train,
                              y = y_train,
                              batch_size = batch_size,
                              epochs = epochs,
                              verbose=1,
                              shuffle = True,
                              validation_data = (X_val, y_val),
                              callbacks= callback)

    print(designHist.history.keys())
    modName = 'customDesign'
    print(designHist.history.keys())
    createSaveMetricsPlot(designHist,modName,data,svhnModel)
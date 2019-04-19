import keras
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras import optimizers
from utils.model_evaluation_utils import *
from utils.preprocess_utils import *


def scratchVGG16_Model():
    SHAPE = (64, 64)
    CHANNEL = 3

    data = generate_data_set_for_training(shape=SHAPE, channel=CHANNEL)
    # Extract Train, val from data
    X_train = data['X_train']
    X_val = data['X_val']
    # Format as the input of VGG
    y_train = list(data['y_train'].T)
    y_val = list(data['y_val'].T)

    _,row, col,channel = X_train.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 50
    batch_size = 64

    vgg16Model = VGG16(include_top = False,
                       weights = None)
    vgg16Model.summary()
    ptInput = keras.Input(shape = (row,col,channel), name  = 'vgg16Scratch')
    vgg16 = vgg16Model(ptInput)

    vgg16 = Flatten()(vgg16)
    vgg16 = Dense(512, activation='relu')(vgg16)
    vgg16 = Dense(512, activation='relu')(vgg16)
    # vgg16 = Dense(1000, activation='relu')(vgg16)
    vgg16 = Dropout(0.5)(vgg16)

    numd_SM = Dense(digLen,    activation='softmax',name = 'num')(vgg16)
    dig1_SM = Dense(numDigits, activation='softmax',name = 'dig1')(vgg16)
    dig2_SM = Dense(numDigits, activation='softmax',name = 'dig2')(vgg16)
    dig3_SM = Dense(numDigits, activation='softmax',name = 'dig3')(vgg16)
    dig4_SM = Dense(numDigits, activation='softmax',name = 'dig4')(vgg16)
    numB_SM = Dense(2,         activation='softmax',name = 'nC')(vgg16)
    out = [numd_SM, dig1_SM ,dig2_SM, dig3_SM, dig4_SM, numB_SM]

    vgg16 = keras.Model(inputs = ptInput, outputs = out)

    callback = []
    optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/vgg16.classifier.hdf5',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss',
                                                  factor = 0.1,
                                                  verbose = 1,
                                                  patience= 3,
                                                  cooldown= 0,
                                                  min_lr = 0.000001)
    # tb = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, write_images=True)
    es = keras.callbacks.EarlyStopping(monitor= 'val_loss',
                                       min_delta=0.00000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(es)
    callback.append(checkpointer)
    callback.append(reduce_lr)
    vgg16.summary()

    vgg16.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer= optim,
                  metrics=  ['accuracy'])

    vgg16History = vgg16.fit(x = X_train,
                             y = y_train,
                             batch_size = batch_size,
                             epochs=epochs,
                             verbose=1,
                             shuffle = True,
                             validation_data = (X_val, y_val),
                             callbacks = callback)

    print(vgg16History.history.keys())
    modName = 'vgg16_Scratch'
    print(vgg16History.history.keys())
    evaluate_model_performance(vgg16History, modName, data, vgg16)
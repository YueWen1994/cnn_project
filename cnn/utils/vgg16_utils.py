import keras
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras import optimizers
from utils.model_evaluation_utils import *
from utils.preprocess_utils import *


def run_pretrained_vgg16():
    SHAPE = (64, 64)
    CHANNEL = 3
    # 0 - 9 plus 10 as input to keras
    # Generate Data
    data = generate_data_set_for_training(shape=SHAPE, channel=CHANNEL)
    # Extract Train, val from data
    X_train = data['X_train']
    X_val = data['X_val']
    # Format as the input of VGG
    y_train = list(data['y_train'].T)
    y_val = list(data['y_val'].T)
    n, row_num, col_num, channel = X_train.shape
    assert channel == CHANNEL
    # Train params
    epochs = 50
    batch_size = 64
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = None
    decay = 0.0
    amsgrad = True
    # defind optimizer

    # build model and compile
    vgg_pretrain_model = create_pretrain_vgg16(row_num, col_num, channel)

    optim = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay, amsgrad=amsgrad)
    vgg_pretrain_model.compile(loss='sparse_categorical_crossentropy',
                               optimizer=optim,
                               metrics=['accuracy'])  # [])

    # add callback
    model_checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/VGGPreTrained.classifier.hdf5',
                                                         monitor='loss',
                                                         save_best_only=True,
                                                         verbose=2)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                  factor=0.1,
                                                  verbose=1,
                                                  patience=4,
                                                  cooldown=1,
                                                  min_lr=0.0001)
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=0.000001,
                                               patience=5,
                                               verbose=1,
                                               mode='auto')
    callback = [early_stop, model_checkpointer, reduce_lr]

    # Fit model
    model_fit_result = vgg_pretrain_model.fit(x=X_train,
                                              y=y_train,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              verbose=1,
                                              shuffle=True,
                                              validation_data=(X_val, y_val),
                                              callbacks=callback)

    model_name = 'pretrain_vgg16'
    # list all data in history
    evaluate_model_performance(model_fit_result, model_name, data, vgg_pretrain_model)


def create_pretrain_vgg16(row_num, col_num, channel):
    NUM_DIGITS = 11
    LEN_DIGITS = 5
    # load pretrain and define input
    vgg16_model = VGG16(include_top=False, weights='imagenet')
    input_shape = (row_num, col_num, channel)
    pretrain_input = keras.Input(shape=input_shape, name='inputVGGPreTrain')
    pretrain_vgg16 = vgg16_model(pretrain_input)

    model_output = Flatten(name='flatten')(pretrain_vgg16)
    model_output = Dense(1024, activation='relu', name='FC1_4096')(model_output)
    model_output = Dense(1024, activation='relu', name='FC1_512')(model_output)

    number_of_digits = Dense(LEN_DIGITS, activation='softmax', name='num')(model_output)
    dig1 = Dense(NUM_DIGITS, activation='softmax', name='dig1')(model_output)
    dig2 = Dense(NUM_DIGITS, activation='softmax', name='dig2')(model_output)
    dig3 = Dense(NUM_DIGITS, activation='softmax', name='dig3')(model_output)
    dig4 = Dense(NUM_DIGITS, activation='softmax', name='dig4')(model_output)
    is_num = Dense(2, activation='softmax', name='nC')(model_output)
    output = [number_of_digits, dig1, dig2, dig3, dig4, is_num]

    # build model and compile
    vgg_pretrain_model = keras.Model(inputs=pretrain_input, outputs=output)
    return vgg_pretrain_model


def load_trained_model(weights_path):
   model = create_pretrain_vgg16()
   model.load_weights(weights_path)
   return model

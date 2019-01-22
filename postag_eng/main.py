from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence

from postag_eng.parser.EnglishNerParser import EnglishNerParser
from postag_eng.vectorizer.Vectorizer import Vectorizer
from keras.utils import np_utils
from postag_eng.recurrentneuralnetwork.RecurrentNeuralNetwork import RecurrentNeuralNetwork

import numpy as np

if __name__ == '__main__':

    print('Reading training data')
    documents = EnglishNerParser().read_file('F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/eng_train.txt')


    print('Create features')
    vectorizer = Vectorizer(word_embedding_path='F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/glove.6B.50d.txt')
    word, shape = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    print('Loaded {} data samples'.format(len(word)))

    print('Split training/validation')
    max_length = 20
    # --------------- Features_word ----------------
    # 2. Padd sequences
    word = sequence.pad_sequences(word, maxlen=max_length)
    # --------------- Features_shape ----------------
    # 2. Padd sequences
    shape = sequence.pad_sequences(shape, maxlen=max_length)


    # --------------- Labels -------------------
    # 1. Convert to one-hot vectors
    labels = [np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels]
    # 2. (only for sequence tagging) Pad sequences
    labels = sequence.pad_sequences(labels, maxlen=max_length)

    to = int(len(word)*0.8)

    x_train = [np.asarray(word[:to]), np.asarray(shape[:to])]
    y_train = np.array(labels[:to])
    x_validation = [np.asarray(word[to+1:]), np.asarray(shape[to+1:])]
    y_validation = np.array(labels[to+1:])

    print(x_train[1].shape)
    print(x_validation[1].shape)

    print('Building network...')
    model = RecurrentNeuralNetwork.build_sequence(word_embeddings=vectorizer.word_embeddings,
                                          input_shape={'shape': (len(vectorizer.shape2index), 46)},
                                          out_shape=len(vectorizer.labels),
                                          units=100, dropout_rate=0.5)

    print('Train...')
    trained_model_name = 'ner_weights.h5'

    # Callback that stops training based on the loss fuction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Callback that saves the best model across epochs
    saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    model.fit(x_train, y_train,
              validation_data=(x_validation, y_validation),
              batch_size=32,  epochs=50, callbacks=[saveBestModel, early_stopping])

    # # Load the best weights in the model
    # model.load_weights(trained_model_name)
    #
    # # Save the complete model
    # model.save('rnn.h5')






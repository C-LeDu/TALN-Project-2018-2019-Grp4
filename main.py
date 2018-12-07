from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence

from amazon.parser.EnglishNerParser import EnglishNerParser
from amazon.vectorizer.Vectorizer import Vectorizer

if __name__ == '__main__' :

    print('Reading training data')
    documents = EnglishNerParser().read_file('F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/eng_train.txt')


    print('Create features')
    vectorizer = Vectorizer(word_embedding_path='F:/Utilisateur/Documents/ESGI/Cours/Traitement Automatique du Langage Naturel/glove.6B.50d.txt')
    word, shape = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    print('Loaded {} data samples'.format(len(word)))


    from keras.utils import np_utils

    print('Split training/validation')
    max_length = 60
    # --------------- Features_word ----------------
    # 2. Padd sequences
    word = sequence.pad_sequences(word, maxlen=max_length)
    # --------------- Features_pos ----------------
    # 2. Padd sequences
    # pos = sequence.pad_sequences(pos, maxlen=max_length)
    # --------------- Features_shape ----------------
    # 2. Padd sequences
    shape = sequence.pad_sequences(shape, maxlen=max_length)


    # --------------- Labels -------------------
    # 1. Convert to one-hot vectors
    labels = [np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels]
    # 2. (only for sequence tagging) Pad sequences
    labels = sequence.pad_sequences(labels, maxlen=max_length)

    x_train, x_validation = [], []
    y_train, y_validation = [], []

    for i in range(len(word)):
        if i < len(word)*0.8 :
            x_train.append([word[i], shape[i]])
            y_train.append(labels[i])
        else :
            x_validation.append([word[i], shape[i]])
            y_validation.append(labels[i])


    from amazon.recurrentneuralnetwork.RecurrentNeuralNetwork import RecurrentNeuralNetwork

    print('Building network...')
    model = RecurrentNeuralNetwork.build_sequence(word_embeddings=vectorizer.word_embeddings,
                                          input_shape={'word': (len(vectorizer.pos2index), 10),
                                                       'shape': (len(vectorizer.shape2index), 2)},
                                          out_shape=len(vectorizer.labels),
                                          units=100, dropout_rate=0.5)
    # or
    RecurrentNeuralNetwork.build_classification(word_embeddings=vectorizer.word_embeddings,
                                                input_shape={'word': (len(vectorizer.pos2index), 10),
                                                             'shape': (len(vectorizer.shape2index), 2),
                                                             'max_length': max_length},
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
              batch_size=32,  epochs=10, callbacks=[saveBestModel, early_stopping])

    # Load the best weights in the model
    model.load_weights(trained_model_name)

    # Save the complete model
    model.save('rnn.h5')

    #
    # print('Reading training data')
    # documents = EnglishNerParser.read('/Path/to/testingdata')
    #
    # print('Create features')
    # vectorizer = Vectorizer(word_embedding_path='/Path/to/embeddings file')
    # features = vectorizer.encode_features(documents)
    # labels = vectorizer.encode_annotations(documents)
    # print('Loaded {} data samples'.format(len(features)))
    #
    #
    # model = RecurrentNeuralNetwork.load('/Path/to/modelfile')
    #
    # y_predictied = []
    # # Loop over features
    #     # Predict labels and convert from probabilities to classes
    #     # model.predict(features, batch_size=1, verbose=0)
    #     # RecurrentNeuralNetwork.probas_to_classes()
    #


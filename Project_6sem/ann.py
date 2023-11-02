def ANN(x_train, y_train, X_test, Y_test, epoc=100):
    import keras as k
    from keras.models import Sequential
    from keras.layers import Activation, Flatten
    from keras.layers.core import Dense
    from keras.optimizers import Adam
    from keras.metrics import categorical_crossentropy
    import numpy as np

    feature_count = 1
    for i in x_train[0].shape:
        feature_count *= i

    model = Sequential()
    if len(x_train[0].shape) > 1:
        print("Flattning....")
        model.add(Flatten(input_shape=x_train[0].shape))

    output = int(y_train[y_train.argmax(axis=0)]) + 1
    print("range: ", output, " y : ", y_train)

    model.add(Dense(feature_count, activation="relu"))
    model.add(Dense(int(feature_count / 10), activation="relu", use_bias=True))
    model.add(Dense(output, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # print(model.summary())

    history = model.fit(
        x=x_train, y=y_train, epochs=epoc, shuffle="True", validation_split=0.2
    )

    result = model.evaluate(X_test, Y_test, batch_size=10)

    print("test Loss : ", result[0], " , Test Accuracy : ", result[1])

    # preditions = model.predict(X_test)

    return model

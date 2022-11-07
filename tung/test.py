import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model


class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__(model=self.init_model())
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, loss="mean_squared_error", metrics=["accuracy"])

    def init_model(self):
        # x = Input(shape=(image_dimensions['height'],
        #                  image_dimensions['width'],
        #                  image_dimensions['channels']))
        x = Input(shape=(256, 256, 3))
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)


if __name__ == "__main__":
    image_dimensions = {
        "height": 256,
        "width": 256,
        "channels": 3
    }
    print(image_dimensions)
    meso = Meso4()
    meso.load("./weights/Meso4_DF")

    data_generator = ImageDataGenerator(rescale=1. / 255)
    generator = data_generator.flow_from_directory(
        "./images1/",
        target_size=(256, 256),
        batch_size=1,
        class_mode="binary"
    )
    print(generator.class_indices)

    X, y = generator.next()

    # Evaluating prediction
    print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
    print(f"Actual label: {int(y[0])}")
    print(f"\nCorrect prediction: {round(meso.predict(X)[0][0]) == y[0]}")

    # Showing image
    plt.imshow(np.squeeze(X))
    # Creating separate lists for correctly classified and misclassified images
    correct_real = []
    correct_real_pred = []

    correct_deepfake = []
    correct_deepfake_pred = []

    misclassified_real = []
    misclassified_real_pred = []

    misclassified_deepfake = []
    misclassified_deepfake_pred = []

    # Generating predictions on validation set, storing in separate lists
    for i in range(len(generator.labels)):

        # Loading next picture, generating prediction
        X, y = generator.next()
        pred = meso.predict(X)[0][0]

        # Sorting into proper category
        if round(pred) == y[0] and y[0] == 1:
            correct_real.append(X)
            correct_real_pred.append(pred)
        elif round(pred) == y[0] and y[0] == 0:
            correct_deepfake.append(X)
            correct_deepfake_pred.append(pred)
        elif y[0] == 1:
            misclassified_real.append(X)
            misclassified_real_pred.append(pred)
        else:
            misclassified_deepfake.append(X)
            misclassified_deepfake_pred.append(pred)

            # Printing status update
        if i % 1000 == 0:
            pass
            # print(i, ' predictions completed.')

        if i == len(generator.labels) - 1:
            print("All", len(generator.labels), "predictions completed")

    print(len(correct_real_pred), len(correct_deepfake_pred), len(misclassified_real_pred), len(misclassified_deepfake_pred))


    def plotter(images, preds):
        fig = plt.figure(figsize=(16, 9))
        subset = np.random.randint(0, len(images) - 1, 12)
        for i, j in enumerate(subset):
            fig.add_subplot(3, 4, i + 1)
            plt.imshow(np.squeeze(images[j]))
            plt.xlabel(f"Model confidence: \n{preds[j]:.4f}")
            plt.tight_layout()
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        plt.show()


    plotter(correct_real, correct_real_pred)
    plotter(misclassified_real, misclassified_real_pred)
    plotter(correct_deepfake, correct_deepfake_pred)
    plotter(misclassified_deepfake, misclassified_deepfake_pred)

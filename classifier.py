import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.datasets import mnist
# import keras.preprocessing.image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Reshape the data for CNN input
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

def create_model():
    # Model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def train_model(model, x_training, y_training, x_validation, y_validation):

    # Data augmentation
    datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
    datagen.fit(x_training)

    # Learning rate scheduler
    reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

    # Training parameters
    batch_size = 64
    epochs = 50

    # Train the model
    history = model.fit(
        datagen.flow(x_training, y_training, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        verbose=1,
        # steps_per_epoch=x_train.shape[0] // batch_size,
        callbacks=[reduce_lr]
    )
    return model, history

def model_evaluation(model, x, y, run_num):

    # Predicting on the test set
    pred_digits_test = np.argmax(model.predict(x), axis=1)
    
    # Training history plot
    plt.figure(figsize=(13, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    # save figure
    plt.savefig(f'run_{run_num}_accuracy.png')

    # Evaluation metrics
    print(classification_report(y.argmax(axis=1), pred_digits_test))

    # Accuracy score for test data
    accuracy = accuracy_score(y_test.argmax(axis=1), pred_digits_test)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    # Save accuracy score
    with open(f'run_{run_num}_accuracy.txt', 'w') as f:
        f.write('Test Accuracy: ' + str(accuracy))

        # add train and validation accuracy
        f.write('\n')
        f.write('Train Accuracy: ' + str(history.history['accuracy'][-1]))
        f.write('\n')        
        f.write('Validation Accuracy: ' + str(history.history['val_accuracy'][-1]))
        
    # Confusion matrix
    cm = confusion_matrix(y_test.argmax(axis=1), pred_digits_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # save figure
    plt.savefig(f'run_{run_num}_confusion_matrix.png')



if __name__ == "__main__":
    # Set number of runs
    number_of_runs = 3
    # Create model
    model = create_model()
    # Allow for multiple runs
    for i in range(number_of_runs):

        print('Number of runs: ', i+1)
        
        # Create a validation dataset used to tune hyperparameters
        X_train, x_val, Y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        print(x_train.shape)

        # Train model
        trained_model, history = train_model(model, X_train, Y_train, x_val, y_val)

        # Evaluate model
        model_evaluation(trained_model, x_test, y_test, i)

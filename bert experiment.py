import ktrain
from ktrain import text as txt

import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Setup GPU
# ------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6172)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.001, patience=20, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)

# ------------------------------------------------------
# Parameters
# ------------------------------------------------------
# NN s general parameters
batch_size = 6
epochs = 3
# ------------------------------------------------------
# Load the data
# ------------------------------------------------------
# (x_train, y_train), (x_test, y_test), preproc = txt.texts_from_folder('data', maxlen=500,
#                                                                       preprocess_mode='bert',
#                                                                       train_test_names=['train', 'test'],
#                                                                       classes=['1', '0'])

(x_train, y_train), (x_test, y_test), preproc = txt.texts_from_csv('data/mid_cleaned_data.csv', maxlen=30,
                                                                   preprocess_mode='bert',
                                                                   text_column='clean_text',
                                                                   label_columns='polarity')
# ------------------------------------------------------
# Load BERT model
# ------------------------------------------------------

model = txt.text_classifier('bert', (x_train, y_train), preproc=preproc)

# wrap model and data in ktrain.Learner object
learner = ktrain.get_learner(model,
                             train_data=(x_train, y_train),
                             val_data=(x_test, y_test),
                             batch_size=batch_size)

# find good learning rate
learner.lr_find(max_epochs=3)             # briefly simulate training to find good learning rate
learner.lr_plot()             # visually identify best learning rate

# ------------------------------------------------------
# Train the model
# ------------------------------------------------------
# train using 1cycle learning rate schedule for 3 epochs
history = learner.fit_onecycle(2e-5, epochs=epochs)

# ------------------------------------------------------
# Evaluate the model performance
# ------------------------------------------------------
scores = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy : ", scores[1])


loss = history.history['loss']
loss_val = history.history['val_loss']
accuracy = history.history['accuracy']
accuracy_val = history.history['val_accuracy']
epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, loss_val, 'b--', label='validation loss')
plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, accuracy_val, 'r--', label='validation accuracy')

plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.show()

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Enable GPU usage if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and enabled.")
    except RuntimeError as e:
        print("Error enabling GPU:", e)
else:
    print("GPU is not available, using CPU.")

# Define directories
data_dir = 'data'
train_dataset_dir = os.path.join(data_dir, 'train')
test_dataset_dir = os.path.join(data_dir, 'test')

# Ensure data directories exist
if not os.path.exists(train_dataset_dir) or not os.path.exists(test_dataset_dir):
    raise FileNotFoundError("Train or test dataset directory not found.")

# Define model parameters
width = 224
height = 224
input_shape = (width, height, 3)
num_classes = 11  # Number of classes in your dataset

# Function to preprocess images using OpenCV
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.keras.applications.xception.preprocess_input(image)
    return image

# Custom data generator using OpenCV
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, num_classes, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = np.array([preprocess_image(path) for path in batch_paths])
        labels = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)

        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            np.random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)

# Load image paths and labels
def load_dataset(directory):
    image_paths = []
    labels = []
    class_names = os.listdir(directory)
    class_indices = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image_paths.append(image_path)
                labels.append(class_indices[class_name])

    return image_paths, labels, class_names

train_image_paths, train_labels, class_names = load_dataset(train_dataset_dir)
test_image_paths, test_labels, _ = load_dataset(test_dataset_dir)

train_size = int(len(train_image_paths) * 0.8)
validation_image_paths = train_image_paths[train_size:]
validation_labels = train_labels[train_size:]
train_image_paths = train_image_paths[:train_size]
train_labels = train_labels[:train_size]

# Create data generators
batch_size = 32
train_generator = CustomDataGenerator(train_image_paths, train_labels, batch_size, num_classes)
validation_generator = CustomDataGenerator(validation_image_paths, validation_labels, batch_size, num_classes, shuffle=False)
test_generator = CustomDataGenerator(test_image_paths, test_labels, batch_size, num_classes, shuffle=False)

# Model architecture
model = tf.keras.Sequential([
    tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# Plotting training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(os.path.join(model_dir, 'mushroom_model.h5'))

# Evaluating the model on the test dataset
evaluation = model.evaluate(test_generator)
print("Test Accuracy:", evaluation[1])

# Generating classification report
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

try:
    print('Classification Report:')
    print(classification_report(test_generator.labels, y_pred, target_names=class_names))
except ValueError as e:
    print("Error generating classification report:", str(e))
    print("Check that the number of predictions matches the number of true labels, and that all predicted labels are valid class indices.")

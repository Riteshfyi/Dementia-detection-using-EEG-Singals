import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

DATA_DIR = Path('/kaggle/input/bipolar-t3-f3-dataset/T3-F3_Electrode_dataset')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 35
N_FOLDS = 10
AUTOTUNE = tf.data.AUTOTUNE

image_paths = sorted(list(DATA_DIR.glob("/*.png")))
class_names = sorted({p.parent.name for p in image_paths})
class_to_index = {name: idx for idx, name in enumerate(class_names)}
labels = np.array([class_to_index[p.parent.name] for p in image_paths])

image_paths = np.array([str(p) for p in image_paths])
image_paths, labels = shuffle(image_paths, labels, random_state=42)

print("Found classes:", class_names)
print(f"Total samples: {len(image_paths)}")

def process_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=len(class_names))
    return image, label

def build_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_image, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

def build_alexnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
    print(f"\n===== Fold {fold}/{N_FOLDS} =====")

    train_paths = image_paths[train_idx]
    val_paths = image_paths[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(train_labels),
                                         y=train_labels)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    train_ds = build_dataset(train_paths, train_labels, training=True)
    val_ds = build_dataset(val_paths, val_labels)

    train_ds.shuffle(1000)

    model = build_alexnet((*IMG_SIZE, 3), len(class_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    from tensorflow.keras.callbacks import ReduceLROnPlateau

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Epoch start — sample train paths & labels:")
    for p, l in zip(train_paths[:3], train_labels[:3]):
        print(os.path.basename(p), class_names[l])
    print("…and sample val paths & labels:")
    for p, l in zip(val_paths[:3], val_labels[:3]):
        print(os.path.basename(p), class_names[l])

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS,
                        callbacks=[early_stop, lr_scheduler],
                        verbose=1)

    loss, acc = model.evaluate(val_ds)
    print(f"Fold {fold} Validation Accuracy: {acc*100:.2f}%")
    fold_accuracies.append(acc)

    y_true, y_pred = [], []
    for images, labels_batch in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(tf.argmax(labels_batch, axis=1).numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    model.save(f"fp2alexnet_fold_{fold}.h5")
    print(f"Model saved: alexnet_fold_{fold}.h5")

print(f"\n>>> Mean Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

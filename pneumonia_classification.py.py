import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom
from keras.optimizers import Adam
from keras.applications import VGG16
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

batch_size = 12
img_width = 128
img_height = 128
img_channels = 3
epochs = 5 

train_dir = 'C:\\Users\\olide\\Music\\Computer-Vision\\chest_xray\\train'
test_dir = 'C:\\Users\\olide\\Music\\Computer-Vision\\chest_xray\\test'

with tf.device('/gpu:0'):
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=False) 

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print('Class Names', class_names)
    
    # Calculate class weights to handle dataset imbalance
    class_counts = {i: 0 for i in range(num_classes)}
    total_samples = 0
    
    for images, labels in train_ds.unbatch():
        class_counts[labels.numpy()] += 1
        total_samples += 1
        
    class_weights = {i: total_samples / (num_classes * count) for i, count in class_counts.items()}
    print('Computed Class Weights', class_weights)

    # Data Augmentation layer
    data_augmentation = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    # Transfer Learning using pre-trained VGG16 replacing basic CNN
    base_model = VGG16(input_shape=(img_height, img_width, img_channels), include_top=False, weights='imagenet')
    base_model.trainable = False 
    
    model = keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1.0/255),
        base_model,
        GlobalAveragePooling2D(), 
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia_improved.keras", save_freq='epoch', save_best_only=True)

    print("Beginning training...")
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        validation_data=val_ds,
        callbacks=[save_callback],
        class_weight=class_weights, 
        epochs=epochs)

    # Evaluation
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy', score[1])

    # Plot Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    # Generate Classification metrics (Precision, Recall, F1)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Implementing tf-explain GradCam logic
    try:
        from tf_explain.core.grad_cam import GradCAM
        explainer = GradCAM()
        
        test_batch = test_ds.take(1)
        for images, labels in test_batch:
            img_to_explain = images[0].numpy()
            target_class = labels[0].numpy()
            
            # VGG16 ultimate conv layer is block5_conv3
            grid = explainer.explain(validation_data=([img_to_explain], None), model=model, class_index=target_class, layer_name="block5_conv3")
            plt.imshow(grid)
            plt.title(f"GradCAM vision for target class {class_names[target_class]}")
            plt.axis("off")
            plt.show()
            break
            
    except ImportError:
        print("tf-explain not installed. To see GradCam, pip install tf-explain and run again.")
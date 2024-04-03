import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Ruta del modelo discriminatorio previamente entrenado
model_path = 'discriminator_model.h5'

# Cargar el modelo discriminatorio
discriminator = load_model(model_path)

# Parámetros de entrenamiento
batch_size = 32
epochs = 5
learning_rate = 0.0001

# Optimizador y función de pérdida
optimizer = Adam(learning_rate)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)

# Función para cargar y preprocesar una imagen
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Función para etiquetar imágenes como falsas (0) o verdaderas (1)
def label_images(images):
    labels = []
    for image in images:
        print("Es esta imagen una GAN (0) o una imagen real (1)?")
        label = int(input("Ingrese 0 o 1: "))
        labels.append(label)
    return labels

# Función principal
def main():
    while True:
        # Solicitar al usuario la ruta de la imagen
        image_path = input("Ingrese la ruta de la imagen (o 'salir' para finalizar): ")
        
        # Salir del programa si se ingresa 'salir'
        if image_path.lower() == 'salir':
            break
        
        # Verificar si la ruta de la imagen es válida
        if not os.path.exists(image_path):
            print("Error: La ruta de la imagen no es válida.")
            continue
        
        # Preprocesar la imagen
        image = preprocess_image(image_path)
        
        # Etiquetar la imagen
        label = label_images([image])[0]
        
        # Entrenar el modelo discriminatorio con la imagen etiquetada
        with tf.GradientTape() as tape:
            logits = discriminator(image, training=True)
            loss = loss_fn([label], logits)
        gradients = tape.gradient(loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
        
        print("Imagen etiquetada y modelo discriminatorio entrenado con éxito.")

# Ejecutar la función principal
if __name__ == "__main__":
    main()

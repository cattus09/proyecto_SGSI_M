import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Cargar el modelo discriminatorio previamente entrenado
discriminator_path = 'discriminator_model.h5'
discriminator = keras.models.load_model(discriminator_path)

# Función para cargar y procesar la imagen seleccionada por el usuario
def load_and_process_image(file_path):
    image = keras.preprocessing.image.load_img(file_path, target_size=(28, 28), color_mode='grayscale')
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = (image_array.astype(np.float32) - 127.5) / 127.5  # Normalizar
    image_array = image_array.reshape(1, 784)
    return image_array 

# Función para clasificar la imagen como real o generada por GANs
def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_array = load_and_process_image(file_path)
        prediction = discriminator.predict(image_array)
        print (prediction)
        if prediction < 0.5:
            result_label.config(text="Real Image")
        else:
            result_label.config(text="GAN Generated Image")
        # Mostrar la imagen seleccionada en la interfaz
        image = Image.open(file_path)
        image.thumbnail((250, 250))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

# Crear la ventana principal
root = tk.Tk()
root.title("Image Classifier")

# Crear etiquetas para mostrar la imagen y el resultado
image_label = tk.Label(root)
image_label.pack(pady=10)
result_label = tk.Label(root, font=('Helvetica', 18))
result_label.pack(pady=10)

# Botón para cargar la imagen y clasificarla
classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack(pady=10)

# Ejecutar el bucle principal de la aplicación
root.mainloop()
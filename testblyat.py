import os
import numpy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_FILEPATH = './saves/model-test.h5'

# print("TensorFlow version:", tf.__version__)


def get_digit_image_values(image_filepath):
    values = numpy.zeros(shape=(28, 28), dtype=float)

    try:
        with Image.open(image_filepath) as im:
            grayscale_img = im.resize((28, 28), resample=Image.Resampling.NEAREST).convert('L')
            bytes_arr = numpy.asarray(grayscale_img, dtype=numpy.float32) / 255.0

            print(bytes_arr.shape)

            # plt.figure()
            # plt.imshow(bytes_arr, cmap='gray', vmin=0, vmax=1)
            # plt.show()

            numpy.copyto(dst=values, src=bytes_arr)

            print(values.shape)
    finally:
        return values


def create_model():
  model = tf.keras.models.Sequential(          #Создание последовательной модели (???)
      [
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation="relu"),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10),
      ]
  )

  predictions = model(x_train[:1]).numpy()
  # print(x_train[0].dtype)

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  loss_fn(y_train[:1], predictions).numpy()

  model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=2)

  model.evaluate(x_test,  y_test, verbose=2)

  return model



mnist = tf.keras.datasets.mnist # импортируем датасет с рукописными сука числами

(x_train, y_train), (x_test, y_test) = mnist.load_data() #здесь мы назначаем датасету 2 координатных плоскости для обучения и тестирования.
x_train, x_test = x_train / 255.0, x_test / 255.0 #Здесь мы преобразуем полученные значения в число с плавающей точкой (от 255 к 0-1)

print(x_train.shape)

model = None

if os.path.isfile(MODEL_FILEPATH):
  model = tf.keras.models.load_model(MODEL_FILEPATH)
else:
  model = create_model()
  model.save(MODEL_FILEPATH)


# probability_model = tf.keras.Sequential([
#   model,
#   tf.keras.layers.Softmax()
# ])

# probability_model(x_test[:5])

print(model.input_shape)

values = get_digit_image_values('./numbers/1inv.png')
flattened_values = numpy.ravel(values)
# print(flattened_values)
# print(flattened_values.shape)
# n = tf.Tensor(values, shape=(28, 28), dtype=tf.float32)
# print(n)
# values_tensor = tf.reshape(tensor, shape=())

# # img = (numpy.expand_dims(values_tensor,0))

# probability_model = tf.keras.Sequential([model, 
#                                          tf.keras.layers.Softmax()])

# # predictions = model.predict(values_tensor.numpy())
predictions = model.predict(flattened_values)
print(predictions)
import nbformat as nbf

# 새로운 노트북 생성
nb = nbf.v4.new_notebook()

# Python 코드 파일 읽기
python_code = """
# Load libraries
import sys

# 코랩의 경우 깃허브 저장소로부터 utils.py를 다운로드 합니다.
if 'google.colab' in sys.modules:
    !wget https://raw.githubusercontent.com/rickiepark/Generative_Deep_Learning_2nd_Edition/main/notebooks/utils.py
    !mkdir -p notebooks
    !mv utils.py notebooks

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, datasets, callbacks
import tensorflow.keras.backend as K

from notebooks.utils import display

# Set hyperparameters
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 100
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 3

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    """
이미지를 정규화하고 크기를 변경합니다.
"""
imgs = imgs.astype("float32") / 255.0
imgs = np.pad(imgs, ((0, 0), (2, 2), (2, 2)), constant_values=0.0)
imgs = np.expand_dims(imgs, -1)
return imgs

x_train = preprocess(x_train)
x_test = preprocess(x_test)

# Display training set images
display(x_train)

# Define encoder
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = layers.Conv2D(32, (3, 3), strides=2, activation="relu", padding="same")(encoder_input)
x = layers.Conv2D(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(128, (3, 3), strides=2, activation="relu", padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]  # 디코더에 필요합니다!

x = layers.Flatten()(x)
encoder_output = layers.Dense(EMBEDDING_DIM, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output)
encoder.summary()

# Define decoder
decoder_input = layers.Input(shape=(EMBEDDING_DIM,), name="decoder_input")
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
decoder_output = layers
.Conv2D(CHANNELS, (3, 3), strides=1, activation=“sigmoid”, padding=“same”, name=“decoder_output”)(x)

decoder = models.Model(decoder_input, decoder_output)
decoder.summary()

Define autoencoder

autoencoder = models.Model(encoder_input, decoder(encoder_output))
autoencoder.summary()

Compile autoencoder

autoencoder.compile(optimizer=“adam”, loss=“binary_crossentropy”)

Create callbacks for model checkpoints and TensorBoard

model_checkpoint_callback = callbacks.ModelCheckpoint(
filepath=”./checkpoint”,
save_weights_only=False,
save_freq=“epoch”,
monitor=“loss”,
mode=“min”,
save_best_only=True,
verbose=0,
)
tensorboard_callback = callbacks.TensorBoard(log_dir=”./logs”)

Train the autoencoder

autoencoder.fit(
x_train,
x_train,
epochs=EPOCHS,
batch_size=BATCH_SIZE,
shuffle=True,
validation_data=(x_test, x_test),
callbacks=[model_checkpoint_callback, tensorboard_callback],
)

Save the final model

autoencoder.save(”./models/autoencoder”)
encoder.save(”./models/encoder”)
decoder.save(”./models/decoder”)

Select sample images for prediction

n_to_predict = 5000
example_images = x_test[:n_to_predict]
example_labels = y_test[:n_to_predict]

Encode the sample images

embeddings = encoder.predict(example_images)

Predict using the autoencoder

predictions = autoencoder.predict(example_images)

print(“실제 의류 아이템”)
display(example_images)
print(“재구성 이미지”)
display(predictions)

Visualize embeddings in 2D space

figsize = 8
plt.figure(figsize=(figsize, figsize))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=“black”, alpha=0.5, s=3)
plt.show()

Color embeddings by labels

figsize = 12
grid_size = 15
plt.figure(figsize=(figsize, figsize))
plt.scatter(
embeddings[:, 0],
embeddings[:, 1],
cmap=“rainbow”,
c=example_labels,
alpha=0.8,
s=300,
)
plt.colorbar()

x = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), grid_size)
y = np.linspace(max(embeddings[:, 1]), min(embeddings[:, 1]), grid_size)
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

reconstructions = decoder.predict(grid)
plt.show()

fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for I in range(grid_size**2):
ax = fig.add_subplot(grid_size, grid_size, I + 1)
ax.axis(“off”)
ax.imshow(reconstructions[I, :, :], cmap=“Greys”)
plt.show()
“””

설명과 함께 셀 추가

cells = []

Introduction cell

intro_text = “””\

오토인코더를 사용한 Fashion MNIST 데이터셋 처리 및 시각화

이 노트북에서는 오토인코더를 사용하여 Fashion MNIST 데이터셋을 처리하고 시각화하는 방법을 설명합니다. 각 단계마다 자세한 설명을 포함하고 있습니다.
“””
cells.append(nbf.v4.new_markdown_cell(intro_text))

각 코드 블록에 대한 설명과 코드 셀 추가

code_blocks = python_code.split(”\n\n”)
for block in code_blocks:
if block.startswith(”#”):
# 설명 텍스트를 추출하여 Markdown 셀로 추가
explanation = block.replace(”#”, “”).strip()
cells.append(nbf.v4.new_markdown_cell(f”### {explanation}\n{explanation}에 대한 자세한 설명을 추가합니다.”))
else:
# 코드 블록을 코드 셀로 추가
cells.append(nbf.v4.new_code_cell(block))

노트북에 셀 추가

nb[‘cells’] = cells

노트북 파일로 저장

with open(‘Autoencoder_Fashion_MNIST_Detailed.ipynb’, ‘w’) as f:
nbf.write(nb, f)
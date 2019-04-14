# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"colab": {}, "colab_type": "code", "id": "dRdv0ml29PSa"}
from keras.layers import Dense, Activation, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from numpy import argmax, array_equal
import matplotlib.pyplot as plt
from keras.models import Model
from random import randint
import pandas as pd
import numpy as np

# + {"colab_type": "text", "id": "KCbgwQaA0ZRv", "cell_type": "markdown"}
# ### データセットの前処理
# - データセットを読み込み、訓練データと予測用のデータに分割
# - 正規化

# + {"colab": {}, "colab_type": "code", "id": "nA9_qW1_9WOo"}
train = pd.read_csv('./data/fashion-mnist_train.csv')
train_x = train[list(train.columns)[1:]].values
train_y = train['label'].values

train_x = train_x / 255
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

train_x = train_x.reshape(-1, 784)
val_x = val_x.reshape(-1, 784)


# + {"colab_type": "text", "id": "DOICGWUx1LQo", "cell_type": "markdown"}
# ### オートエンコーダーのアーキテクチャを作成
# - エンコードは2000, 1200, 500の3つの層で構成されている

# + {"colab": {}, "colab_type": "code", "id": "OXSfU2iH9ZbZ"}
input_layer = Input(shape=(784,))

encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

latent_view = Dense(10, activation='sigmoid')(encode_layer3)

decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)
output_layer = Dense(784)(decode_layer3)

model = Model(input_layer, output_layer)
# -

# ### モデルのアーキテクチャを可視化

# +
import sys

sys.path.append('../../convnet-drawer')

# +
from keras.models import Sequential
from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense

drawer_model = Model(input_shape=(28, 28, 3))

drawer_model.add(Flatten())
drawer_model.add(Dense(1500))
drawer_model.add(Dense(1000))
drawer_model.add(Dense(500))
drawer_model.add(Dense(10))
drawer_model.add(Dense(500))
drawer_model.add(Dense(1000))
drawer_model.add(Dense(1500))
drawer_model.add(Dense(784))

drawer_model.save_fig('alexnet.svg')

# + {"colab_type": "text", "id": "mvFdHRtk2GGY", "cell_type": "markdown"}
# ### モデルの概要を表示

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 442}, "colab_type": "code", "id": "cr3WlpfZBd7a", "outputId": "75a74782-6a78-4ed2-bbb1-34dc3d770b83"}
model.summary()

# + {"colab_type": "text", "id": "t9yTfVZY2O01", "cell_type": "markdown"}
# ### モデルのコンパイル
# - EarlyStoppingの設定

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 731}, "colab_type": "code", "id": "Nv1BnA-39Z63", "outputId": "f9f6583b-e3ab-49da-f7e0-2a76d77488a5"}
model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
history = model.fit(train_x, train_x, epochs=20, batch_size=2048, validation_data=(val_x, val_x), callbacks=[early_stopping])
# -

# ### 学習結果の可視化

# +
import matplotlib.pyplot as plt
# %matplotlib inline

# Accuracy
print(history.history)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# -

preds = model.predict(val_x)

# + {"colab_type": "text", "id": "_zLwSOGC3OO6", "cell_type": "markdown"}
# ### 入力画像を表示

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 923}, "colab_type": "code", "id": "05dZ5nRyC1TO", "outputId": "a6998a6a-c28d-46ab-9d56-4ccf2cd89458"}
from PIL import Image
f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5):
  ax[i].imshow(val_x[i].reshape(28, 28))
  
plt.show()

# + {"colab_type": "text", "id": "mC1Em-XT5Czu", "cell_type": "markdown"}
# ### 出力画像を表示

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 838}, "colab_type": "code", "id": "jRXMDooXDRku", "outputId": "8430089e-69a0-44d8-abe9-145a1700e389"}
f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5):
  ax[i].imshow(preds[i].reshape(28, 28))
  
plt.show()

# + {"colab_type": "text", "id": "z3Lf3MiK508h", "cell_type": "markdown"}
# ### 画像のノイズ除去

# + {"colab_type": "text", "id": "MdFfo8095b9z", "cell_type": "markdown"}
#

# + {"colab": {}, "colab_type": "code", "id": "G2V02lFtEXqV"}
train_x = train[list(train.columns)[1:]].values
train_x, val_x = train_test_split(train_x, test_size=0.2)

## normalize and reshape
train_x = train_x/255.
val_x = val_x/255.

# + {"colab": {}, "colab_type": "code", "id": "yom_kL8VwUw0"}
train_x = train_x.reshape(-1, 28, 28, 1)
val_x = val_x.reshape(-1, 28, 28, 1)

# + {"colab_type": "text", "id": "8BR7leP76D4g", "cell_type": "markdown"}
# ### ノイズを付与して、画像認識の精度を上げる
# - https://qiita.com/bohemian916/items/9630661cd5292240f8c7

# + {"colab": {}, "colab_type": "code", "id": "aQZ-668PwUzM"}
noise = augmenters.SaltAndPepper(0.1)
seq_object = augmenters.Sequential([noise])

train_x_n = seq_object.augment_images(train_x * 255) / 255
val_x_n = seq_object.augment_images(val_x * 255) / 255

# + {"colab_type": "text", "id": "44444jTc6bGi", "cell_type": "markdown"}
# ### 加工前の画像を表示

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 838}, "colab_type": "code", "id": "KGuNwobroAW4", "outputId": "07dc019e-c36a-411c-ca3b-10d071eb4b50"}
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5,10):
  ax[i-5].imshow(train_x[i].reshape(28,28))
  
plt.show()

# + {"colab_type": "text", "id": "VP9a2jpY6f4O", "cell_type": "markdown"}
# ### ノイズ付与後の画像を表示

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 838}, "colab_type": "code", "id": "uBDH7meYojhm", "outputId": "1118f7ac-a8a7-4962-ae6f-3b0003142261"}
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5,10):
  ax[i-5].imshow(train_x_n[i].reshape(28,28))
  
plt.show()

# + {"colab_type": "text", "id": "9SoVatND6kdx", "cell_type": "markdown"}
# ### プーリング層を追加してモデル構築

# + {"colab": {}, "colab_type": "code", "id": "UkX3hoNaFJLt"}
input_layer = Input(shape=(28, 28, 1))

encoded_layer1 = Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
encoded_layer1 = MaxPool2D((2,2), padding='same')(encoded_layer1)

encoded_layer2 = Conv2D(32, (3,3), activation='relu', padding='same')(encoded_layer1)
encoded_layer2 = MaxPool2D((2,2), padding='same')(encoded_layer2)

encoded_layer3 = Conv2D(16, (3,3), activation='relu', padding='same')(encoded_layer2)
latent_view = MaxPool2D((2,2), padding='same')(encoded_layer3)

decoded_layer1 = Conv2D(16, (3,3), activation='relu', padding='same')(latent_view)
decoded_layer1 = UpSampling2D((2,2))(decoded_layer1)

decoded_layer2 = Conv2D(32, (3,3), activation='relu', padding='same')(decoded_layer1)
decoded_layer2 = UpSampling2D((2,2))(decoded_layer2)

decoded_layer3 = Conv2D(64, (3,3), activation='relu')(decoded_layer2)
decoded_layer3 = UpSampling2D((2,2))(decoded_layer3)

output_layer = Conv2D(1, (3,3), padding='same')(decoded_layer3)

model_2 = Model(input_layer, output_layer)
model_2.compile(optimizer='adam', loss='mse')


# + {"colab_type": "text", "id": "fHH62KrI7Um2", "cell_type": "markdown"}
# ### モデルの概要を表示

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 629}, "colab_type": "code", "id": "Kob51f3Lpjrk", "outputId": "9daf8bfa-5e2b-4930-b6ea-d5b5ce478503"}
model_2.summary()

# + {"colab_type": "text", "id": "utPawuva7e6X", "cell_type": "markdown"}
# ### EarlyStoppingの設定

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 374}, "colab_type": "code", "id": "qgfNMei5ftH1", "outputId": "eb8986b1-c026-463a-981f-bb33848b62a3"}
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=5, mode='auto')
history = model_2.fit(train_x_n, train_x, epochs=10, batch_size=2048, validation_data=(val_x_n, val_x), callbacks=[early_stopping])

# + {"colab": {"base_uri": "https://localhost:8080/", "height": 838}, "colab_type": "code", "id": "_OZPx2kcinZi", "outputId": "a54ee176-3daa-4a27-b185-b8c393697458"}
preds = model_2.predict(val_x_n[:10])
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5,10):
    ax[i-5].imshow(preds[i].reshape(28, 28))
plt.show()


# + {"colab_type": "text", "id": "7o963kGV78SK", "cell_type": "markdown"}
# ### オートエンコーダを用いたシーケンス間予測

# + {"colab": {}, "colab_type": "code", "id": "zlMOCK-EqVhm"}
def dataset_preparation(n_in, n_out, n_unique, n_samples):
  X1, X2, y = [], [], []
  for _ in range(n_samples):
    inp_seq = [randint(1, n_unique-1) for _ in range(n_in)]
    
    target_seq = list(reversed(target))
    send_seq = [0] + target_seq[:-1]
    
    X1.append(to_categorical([inp_seq], num_classes=n_unique))
    X2.append(to_categorical([seed_seq], num_classes=n_unique))
    y.append(to_categorical([target_seq], num_classes=n_unique))
    
    X1 = np.squeeze(np.array(X1), axis=1)
    X2 = np.squeeze(np.array(X2), axis=1)
    y = np.squeeze(np.array(y), axis=1)
    
    return X1, X2, y
  
  samples = 100000
  features = 51
  inp_size = 6
  out_size = 3
  
  inputs, seeds, outputs = dataset_preparation(inp_size, out_size, features, samples)
  
  print("Shapes: ", inputs.shape, seeds.shape, outputs.shape)
  print("Here is first categorically encoded input seqquence looks like: ",)
  inputs[0][0]

# + {"colab": {}, "colab_type": "code", "id": "f-hp0rkK0URf"}


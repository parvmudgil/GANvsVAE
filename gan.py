import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

def loadingData():
    (x_train, y_train), (x_test, y_test) = mnist.loadingData()
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    # convert shape of x_train from (60000, 28, 28) to (60000, 784)
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)
(X_train, y_train,X_test, y_test)=loadingData()
print(X_train.shape)

def adamOptimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def discriminatorInit():
    dscmtr=Sequential()
    dscmtr.add(Dense(units=1024,input_dim=784))
    dscmtr.add(LeakyReLU(0.2))
    dscmtr.add(Dropout(0.3))


    dscmtr.add(Dense(units=512))
    dscmtr.add(LeakyReLU(0.2))
    dscmtr.add(Dropout(0.3))

    dscmtr.add(Dense(units=256))
    dscmtr.add(LeakyReLU(0.2))

    dscmtr.add(Dense(units=1, activation='sigmoid'))

    dscmtr.compile(loss='binary_crossentropy', optimizer=adamOptimizer())
    return dscmtr
d =discriminatorInit()
d.summary()

def generatorInit():
    gntr=Sequential()
    gntr.add(Dense(units=256,input_dim=100))
    gntr.add(LeakyReLU(0.2))

    gntr.add(Dense(units=512))
    gntr.add(LeakyReLU(0.2))

    gntr.add(Dense(units=1024))
    gntr.add(LeakyReLU(0.2))

    gntr.add(Dense(units=784, activation='tanh'))

    gntr.compile(loss='binary_crossentropy', optimizer=adamOptimizer())
    return gntr
g=generatorInit()
g.summary()



def ganInit(dscmtr, gntr):
    dscmtr.trainable=False
    gan_input = Input(shape=(100,))
    x = gntr(gan_input)
    gan_output= dscmtr(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan
gan = ganInit(d,g)
gan.summary()

def imagePlots(epoch, gntr, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    prodImages = gntr.predict(noise)
    prodImages = prodImages.reshape(100,28,28)
    plt.figure(figsize=figsize)
    for i in range(prodImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(prodImages[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)

def mainTrain(epochs=1, batch_size=128):

    #Loading the data
    (X_train, y_train, X_test, y_test) = loadingData()
    batch_count = X_train.shape[0] / batch_size

    # Creating GAN
    gntr= generatorInit()
    dscmtr= discriminatorInit()
    gan = ganInit(dscmtr, gntr)

    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 100])

            # Generate fake MNIST images from noised input
            prodImages = gntr.predict(noise)

            # Get a random set of  real images
            imgBatch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]

            #Construct different batches of  real and fake data
            X= np.concatenate([imgBatch, prodImages])

            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9

            #Pre train discriminator on  fake and real data  before starting the gan.
            dscmtr.trainable=True
            dscmtr.train_on_batch(X, y_dis)

            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of dscmtr should be fixed.
            #We can enforce that by setting the trainable flag
            dscmtr.trainable=False

            #training  the GAN by alternating the training of the Discriminator
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:

            imagePlots(e, gntr)
mainTrain(400,128)
# import packages we need

from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, AveragePooling2D, Activation
from keras.layers import BatchNormalization, Dropout
from keras import backend


class Basic_CNN:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        
    def create_empty_model(self):
        model = Sequential()
        self.model = model
    
    def add_layer(self,use_batch = True, use_drop = True):
        self.model.add(Conv2D(32, (3,3),strides=2))
        self.model.add(DepthwiseConv2D((3,3), strides=1))

        self.model.add(AveragePooling2D((7,7)))
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, activation="softmax"))
        
    def train(self,optimizer,loss='sparse_categorical_crossentropy',val_split = 0.2,
             epochs = 2, batch_size = 256):
        
        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        self.model.fit(x=x_train, 
                       y=y_train,
                       validation_split = val_split,
                       batch_size=batch_size,
                       epochs=epochs)
        
    def summary(self):
        return self.model.summary()


class mobilenet:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = None
        
    def create_empty_model(self):
        model = Sequential()
        self.model = model
    
    def add_layer(self,use_batch = True, use_drop = True):
        self.model.add(Conv2D(32, (3,3),strides=2))
        self.model.add(DepthwiseConv2D((3,3), strides=1))
        
        self.model.add(Conv2D(64, (1,1),strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=2))
        
        self.model.add(Conv2D(128, (1,1),strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=1))  
        
        self.model.add(Conv2D(128, (1,1),strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=2))         
        
        self.model.add(Conv2D(256, (1,1),strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=1)) 
        
        self.model.add(Conv2D(256, (1,1),strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=1))
        
        self.model.add(Conv2D(512, (1,1),strides=1))
        for i in range(0,5):
            self.model.add(DepthwiseConv2D((3,3), strides=1))
            self.model.add(Conv2D(512, (1,1), strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=2))
        
        self.model.add(Conv2D(1024, (1,1),strides=1))
        self.model.add(DepthwiseConv2D((3,3), strides=2))
        self.model.add(Conv2D(1024, (1,1),strides=1))
        self.model.add(AveragePooling2D((7,7)))
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, activation="softmax"))
        
    def train(self,optimizer,loss='sparse_categorical_crossentropy',val_split = 0.2,
             epochs = 2, batch_size = 256):
        
        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        self.model.fit(x=x_train, 
                       y=y_train,
                       validation_split = val_split,
                       batch_size=batch_size,
                       epochs=epochs)
        
    def summary(self):
        return self.model.summary()
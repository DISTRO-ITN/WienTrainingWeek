import numpy as np
import os
from PIL import Image
import xml.etree.ElementTree

class DataGenerator(object):
    def __init__(self, inputPath, batch_size = 64, shuffle = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inputPath = inputPath # should be the POI dir
        self.filesNumber = -1
        self.height, self.width = (0,0)
        
        #get global informations
        self.list_IDs = [i[:-4] for i in  os.listdir(os.path.join(self.inputPath)) if i[-3:] == "png"]
        self.filesNumber = len(self.list_IDs)
        imgPatch = np.asarray(Image.open(os.path.join(self.inputPath, self.list_IDs[0] + '.png')), dtype=np.uint8)
        infoLight = self.loadInfo(os.path.join(self.inputPath, self.list_IDs[0] + '.xml'))
          
        # fill informations
        self.height = imgPatch.shape[0]
        self.width = imgPatch.shape[1]
        print(self.height, self.width)
        self.labelShape = len(infoLight)
    
    def generate(self):
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            self.__get_exploration_order()
            indexes = self.list_IDs

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [k for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
    
                # Generate data
                X, y = self.__data_generation(list_IDs_temp)
    
                yield X, y

    def __get_exploration_order(self):
        # Find exploration order
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, self.height, self.width, 3))
        y = np.empty((self.batch_size, self.labelShape), dtype = np.float32)

        # Generate data
        for i in range(0, self.filesNumber):
            X[i, :, :, :] = np.asarray(Image.open(os.path.join(self.inputPath, list_IDs_temp[i] + '.png')), dtype=np.uint8)            
            y[i] = self.loadInfo(os.path.join(self.inputPath, list_IDs_temp[i] +'.xml'))
                
            return X, y

    def loadInfo(self, pathname):
        root = xml.etree.ElementTree.parse(pathname).getroot()
        for child in root:
            if child.tag == "quaternion":
                data = child.text
        return np.array(eval(data))

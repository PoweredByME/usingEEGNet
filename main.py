from scipy.io import loadmat
import numpy as np
from EEGModels import EEGNet_SSVEP
import keras

def loadDataSet(subject_no):
    dataset = loadmat('DataSet/data/s' + str(subject_no) + '.mat')
    eegDataset = dataset['eeg']
    tragetFreqs = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
    trails = 15

    print(eegDataset[0,:,:,1].shape)
    
    dataX = []
    dataY = []
    for i in range(len(tragetFreqs)):
        for j in range(trails):
            eeg = eegDataset[i, :, 39:1114, j]
            eeg = np.pad(eeg, ((0,0),(0, 1280 - 1114 + 39)), 'constant')
            for k in range(4):
                dataX.append(np.reshape(eeg[:, k * 256: (k + 1) * 256], (1,8,256)))
                dataY.append(i);
    
    dataX = np.asarray(dataX)
    dataY = np.asarray(dataY)

    dataY = keras.utils.to_categorical(dataY);
    dataX = keras.utils.normalize(dataX)
    
    return (dataX, dataY)
            

def main():
    detector = EEGNet_SSVEP();
    detector.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc', 'mae', 'mse'])
    (dataX, dataY) = loadDataSet(1);

    dataX = dataX[0:1, :, :, :];
    dataY = dataY[110:120, :];
    
    
    print dataX;
    return;

    for i in range(10):
        y_predicted = detector.predict(x = dataX);
        print(str(i) + ". Before Fitting -> y = " + str(y_predicted));
        detector.fit(x = dataX, y = dataY, verbose = 1, epochs = 100, validation_split=0.0);
        y_predicted = detector.predict(x = dataX);
        print(str(i) + ". After Fitting -> y = " + str(y_predicted));
        print("#" * 20);        


if __name__ == "__main__":
    main()




'''
def main():
    detector = EEGNet_SSVEP()
    try:
        detector.load_weights("weights.h5")
        detector.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc', 'mae', 'mse'])
            
        for i in range(1,3):
            (dataX, dataY) = loadDataSet(i)
            detector.fit(x = dataX[0:1,:,:,:], y = dataY[0:1,:,:], verbose = 1, validation_split=0.1, epochs=5)
            print(dataY[0:10,:])
            x_hat = detector.predict(x = dataX[0:10, :, :, :])
            print(x_hat);
            h = detector.evaluate(verbose=1, x = dataX, y = dataY, )
                
            print(h)

        detector.save_weights("weights.h5")
    except:
        print("Error")
    finally:
        detector.save_weights("weights.h5")
'''

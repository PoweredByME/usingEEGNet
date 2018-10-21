from scipy.io import loadmat
import numpy as np
from EEGModels import EEGNet_SSVEP;

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
                targetVec = np.zeros((len(tragetFreqs)))
                targetVec[i] = 1.0
                dataY.append(targetVec)
    
    dataX = np.asarray(dataX)
    dataY = np.asarray(dataY)
    
    return (dataX, dataY)
            

def main():
    detector = EEGNet_SSVEP()
    try:
        detector.load_weights("weights.h5")
        detector.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc', 'mae', 'mse'])
            
        for i in range(1,3):
            (dataX, dataY) = loadDataSet(i)
            detector.fit(x = dataX, y = dataY, verbose = 1, validation_split=0.1, epochs=500, batch_size=64)
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


if __name__ == "__main__":
    main()
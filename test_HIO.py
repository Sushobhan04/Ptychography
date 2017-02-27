from keras.models import load_model
import h5py
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import sys
from skimage.measure import compare_psnr

def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]

def main():

    path_test = "/home/sushobhan/Documents/data/ptychography/"
    home = "/home/sushobhan/Documents/research/ptychography/"
    model_name = sys.argv[1]

    border_mode = 'valid'

    # file_name = 'lena_1.h5'
    # file_name = 'resChart.h5'
    file_name = 'valset_1.h5'

    file = h5py.File(path_test+'data/'+ file_name,'r')
    ks = file.keys()
    print ks

    data = file['data']
    label =file['label']

    # im.imsave('label.png',label[0,0,],cmap=plt.cm.gray)
    # im.imsave('data.png',data[0,24,],cmap=plt.cm.gray)

    model = load_model(path_test+'models/'+model_name+'.h5')
    y_output = np.array(model.predict(data[:,5:6,]))

    print np.max(data), np.max(label)

    print y_output.shape, np.max(y_output)
    print data.shape , label.shape

    im.imsave(path_test+'results/'+'label.png',label[0,0,],cmap=plt.cm.gray)
    im.imsave(path_test+'results/'+'output.png',y_output[0,0,],cmap=plt.cm.gray)

    fig = plt.figure(0)
    m,n = 2,2
    for i in range(0,1):
        # print i
        j,k = i//n, i%n
        # print j,k
        plt.subplot2grid((m,n), (j, k))
        plt.imshow(label[i,0,],cmap=plt.cm.gray)
        # print j+2, k
        plt.subplot2grid((m,n), (j+1, k))
        plt.imshow(y_output[i,0,],cmap=plt.cm.gray)

        print compare_psnr(label[i,0,],y_output[i,0,])

    plt.subplot_tool()
    plt.savefig(path_test+'results/'+model_name+'.png')



if __name__ == '__main__':
    main()
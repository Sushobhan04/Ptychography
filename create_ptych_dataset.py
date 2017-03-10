import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as im


def plot_arr(arr,name):
	im.imsave(name+'.png',arr,cmap=plt.cm.gray)

def patchify(img,size):
	H = img.shape[0]
	W = img.shape[1]

	batch = []

	for i in range((2*H)//size-1):
		for j in range((2*W)//size-1):
			x = i*size/2
			y = j*size/2
			batch.append(img[x:x+size,y:y+size])
	return np.array(batch)

def batch_fft(batch):
	batch_f = []
	for x in batch:
		batch_f.append(np.absolute(fft.fft2(x)))
	return batch_f

def create_dataset(source,destination):


def main():
	output_path = '/home/sushobhan/Documents/data/ptychography/datasets/'
	source = '/home/sushobhan/Documents/data/ptychography/data/'
	N = 64

	dataset_name = 'patch_'+str(N)



	

	plot_arr(arr,'test_img')
	plot_arr(trans,'test_fft',t='c')

if __name__=='__main__':
	main()

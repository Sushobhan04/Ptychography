import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as im
from PIL import Image
import os
import h5py


def plot_arr(arr,name):
	im.imsave(name+'.png',arr,cmap=plt.cm.gray)

def add_dataset(data,label,grp):
	grp.create_dataset("data",data=data)
	grp.create_dataset("label",data = label)

def tvt_split(data,label, f, split = (0.8,0.1,0.1)):
	l = data.shape[0]
	train_pt = int(split[0]*l)
	val_pt = int(split[1]*l)+train_pt

	# data_sp = {}
	train = f.create_group("train")
	val = f.create_group("val")
	test = f.create_group("test")

	add_dataset(data[0:train_pt,],label[0:train_pt,],train)
	add_dataset(data[train_pt:val_pt,],label[train_pt:val_pt,],val)
	add_dataset(data[val_pt:,],label[val_pt:,],test)

	# f['train'] = {'data':data[0:train_pt,], 'label':label[0:train_pt,]}
	# f['val'] = {'data':data[train_pt:val_pt,], 'label':label[train_pt:val_pt,]}
	# f['test'] = {'data':data[val_pt:,], 'label':label[val_pt:,]}

	return f


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
		batch_f.append(fft.fftshift(fft.fft2(x)))
	return np.array(batch_f)

def filter_data(data,factor,thresh = (-1.0,1.0)):
	d = data/factor
	d = np.where(d<thresh[0],0.0, d)
	d = np.where(d>thresh[1],0.0, d)

	return d


def create_dataset(N,source,destination,dataset_name,factor):
	dataset = []
	labelset = []

	for filename in os.listdir(source):
		img = np.asarray(Image.open(source+filename).convert('L'))/255.0
		patches = patchify(img,N)
		patches_fft = batch_fft(patches)

		dataset.extend(np.absolute(patches_fft))
		labelset.extend(np.array([patches_fft.real,patches_fft.imag]).transpose(1,0,2,3))

	dataset = np.expand_dims(np.array(dataset),axis=1)
	labelset =np.array(labelset)

	thresh = (-10.0,10.0)

	dataset = filter_data(dataset, factor,thresh)
	labelset = filter_data(labelset, factor,thresh)

	print dataset.shape
	print labelset.shape


	f = h5py.File(destination+dataset_name+'.h5','w')
	f = tvt_split(dataset, labelset, f)
	# print f.keys()
	# k = f['train']['data'][()]
	# print k
	f.close()


	print "dataset created"

	plot_arr(labelset[0,0],'test_real_img')
	plot_arr(labelset[0,1],'test_img_img')
	plot_arr(dataset[0,0],'test_fft')
	# print labelset[0], dataset[0]
	print np.max(dataset), np.min(np.absolute(dataset)),np.min(dataset), np.mean(dataset), np.median(dataset)
	print np.max(labelset), np.min(np.absolute(labelset)),np.min(labelset), np.mean(labelset),np.median(labelset)
	# print len(dataset)

def main():
	output_path = '/home/sushobhan/Documents/data/ptychography/datasets/'
	source = '/home/sushobhan/Documents/data/ptychography/data/Set91/'
	N = 64
	factor = 1000.0

	dataset_name = 'fft_'+str(N)

	create_dataset(N,source,output_path,dataset_name, factor)



	

	# plot_arr(arr,'test_img')
	# plot_arr(trans,'test_fft',t='c')

if __name__=='__main__':
	main()

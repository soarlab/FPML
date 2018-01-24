import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
from matplotlib.pyplot import draw



matplotlib.rcParams.update({'font.size': 17})

def splitDotValue(column):
	values=column.split(":")
	return values[1]
	
def getAccuracy(val):
	acc=0
	try:
		acc=float(splitDotValue(val))/100.0
	except ValueError:
		acc=0
	return acc
	
data=[]

fileNames=["/home/roki/GIT/FPML/src/final/fourclass/MPFR/part3SVMTest.txt",
		   "/home/roki/GIT/FPML/src/final/fourclass/FLEX/part3SVMTest.txt"]
maxi=0
mini=1.0
for fileName in fileNames:
	with open(fileName) as f:
		data=f.readlines()
	for line in data:
		columns=line.split("$")
		if getAccuracy(columns[6])!=0:
			maxi=max(maxi,getAccuracy(columns[6]))
			mini=min(mini,getAccuracy(columns[6]))

for fileName in fileNames:
	with open(fileName) as f:
		data=f.readlines()
	
	labelsComputation=[]
	labelsReading=[]
	labelsTest=[]
	
	for line in data:
		columns=line.split("$")
		if "exc" not in line:
			if columns[1]!="" and columns[1] not in labelsReading: 
				labelsReading.append(columns[1])
			if columns[2]!="" and columns[2] not in labelsComputation: 
				labelsComputation.append(columns[2])
			if columns[3]!="" and columns[3] not in labelsTest: 
				labelsTest.append(columns[3])
	
	matrix=numpy.zeros((len(labelsReading),len(labelsComputation),len(labelsTest)))
	
	for line in data:
		columns=line.split("$")
		if getAccuracy(columns[6])!=0:
			matrix[labelsReading.index(columns[1])][labelsComputation.index(columns[2])][labelsTest.index(columns[3])]=getAccuracy(columns[6])

	labelHiddenX=range(0,len(labelsComputation),1)
	labelHiddenY=range(0,len(labelsReading),1)
	labelHiddenZ=range(0,len(labelsTest),1)
	
	for i,val in enumerate(labelsReading): 
		if i not in labelHiddenY:
			labelsReading[i]=""
	
	for i,val in enumerate(labelsComputation): 
		if i not in labelHiddenX:
			labelsComputation[i]=""
	
	for i,val in enumerate(labelsTest): 
		if i not in labelHiddenZ:
			labelsTest[i]=""
	
	indexX=range(0,len(labelsReading),1)
	indexY=range(0,len(labelsComputation),1)
	indexZ=range(0,len(labelsTest),1)
	
	X, Y = np.meshgrid(indexX, indexY)
	
	fig = plt.figure(num=None, dpi=80, facecolor='w', edgecolor='k')
	
	norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi, clip = False)
	
	plt.subplots_adjust(top = 0.98, bottom = 0.02, right = 0.92, left = 0, hspace = 0.5, wspace = 0.1)
	
	for i in range(0,len(labelsTest)):
		Z=matrix[indexX][indexY][i]
		Z[Z<0.5]=np.NaN
		ax = fig.add_subplot(3,4,i+1, projection='3d')
		ax.view_init(30,-20)
		ax.set_zlim(mini, maxi)
		ax.set_xticklabels(labelsReading)
		ax.set_yticklabels(labelsComputation)
		ax.set_title("Test Precision:"+str(labelsTest[i]))
		im=ax.plot_surface(X, Y, Z, cmap="cool", antialiased=True,norm=norm,alpha=1)
	
	cbar_ax = fig.add_axes([0.95, 0.05, 0.01, 0.9])
	cbar_ax.set_yticklabels(['< -1', '0', '> 1'])
	fig.colorbar(im,cax=cbar_ax)
	title=fileName.split("/")[-1].split(".")[0]
	
	#fileName.split("/")[-3]+"\n"+
	#if "AP" in fileName:
	#	title=title+"AP"
	#if "SVM" in fileName:
	#	title=title+"SVM"
	#if "P" in fileName:
	#	title=title+"P"

	plt.title(title),  
	draw()
plt.show()
	

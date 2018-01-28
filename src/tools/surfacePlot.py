import numpy as np
import numpy.random
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
from matplotlib.pyplot import draw



matplotlib.rcParams.update({'font.size': 10})

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

### Configuration ###
#Select from AP, SVM, P
algoritmh="AP"
folder="/home/roki/GIT/src/final/heart/"
#####################

### Note: the script can save the graphs in the folder: /folder/GRAPHS/*. To do it just remove comment from line 124 'fig.savefig'

for tt in ["Train","Test"]:
	for indFile in range(1,5): 
		fileNames=[folder+"FLEX/part"+str(indFile)+algoritmh+tt+".txt", 
				   folder+"MPFR/part"+str(indFile)+algoritmh+tt+".txt"]
		maxi=0
		mini=1.0
		for fileName in fileNames:
			with open(fileName) as f:
				data=f.readlines()
			for line in data:
				columns=line.split("$")
				if getAccuracy(columns[6])>0:
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
			
			matrix=numpy.zeros((len(labelsTest),len(labelsReading),len(labelsComputation)))
			
			for line in data:
				columns=line.split("$")
				if getAccuracy(columns[6])!=0:
					matrix[labelsTest.index(columns[3])][labelsReading.index(columns[1])][labelsComputation.index(columns[2])]=getAccuracy(columns[6])
			
			labelHiddenX=[0,len(labelsComputation)-1]
			labelHiddenY=[0,len(labelsReading)-1]
			
			for i,val in enumerate(labelsReading):
				if i not in labelHiddenX:
					labelsReading[i]=""
			
			for i,val in enumerate(labelsComputation): 
				if i not in labelHiddenY:
					labelsComputation[i]=""
			
			indexY=range(0,len(labelsReading),1)
			indexX=range(0,len(labelsComputation),1)
			indexZ=range(0,len(labelsTest),1)
			
			X, Y = np.meshgrid(indexX,indexY)
			fig = plt.figure(num=None, dpi=100, facecolor='w', edgecolor='k')
			fig.set_size_inches(18, 10)
			norm = matplotlib.colors.Normalize(vmin = mini, vmax = maxi, clip = False)
			plt.subplots_adjust(top = 0.98, bottom = 0.02, right = 0.92, left = 0, hspace = 0.1, wspace = 0.1)
		
			for i in range(0,len(labelsTest)):
				Z=matrix[i][:][:]
				Z[Z<0.5]=np.NaN #Used to remove accuracy values under 0.5
				ax = fig.add_subplot(3,4,i+1, projection='3d')
				ax.view_init(30,-20)
				ax.set_zlim(mini, maxi)
				ax.set_xticks(indexY)
				ax.set_xticklabels(labelsReading)
				ax.set_yticks(indexX)
				ax.set_yticklabels(labelsComputation)
				ax.set_title("Test Precision:"+str(labelsTest[i]))
				ax.set_xlabel("R")
				ax.xaxis.labelpad = -8
				ax.set_ylabel("C")
				ax.yaxis.labelpad = -8
				ax.tick_params(axis='both', which='major', pad=3)
				im=ax.plot_surface(Y, X, Z, cmap="cool", antialiased=True,norm=norm,alpha=1)
				
			cbar_ax = fig.add_axes([0.95, 0.05, 0.01, 0.9])
			cbar_ax.set_yticklabels(['< -1', '0', '> 1'])
			fig.colorbar(im,cax=cbar_ax)
			title=fileName.split("/")[-2]+fileName.split("/")[-1].split(".")[0]
			plt.title(title),
			fig.canvas.set_window_title(title)
			manager = plt.get_current_fig_manager()
			manager.resize(*manager.window.maxsize())
			draw()
			#fig.savefig(folder+"GRAPHS/"+algoritmh+"/"+title+".pdf") 
plt.show()

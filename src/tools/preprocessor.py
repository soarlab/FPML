import numpy
import sys
data=[]
filenamePath="../tmp/local/german/german"
nAttributes=0
with open(filenamePath+".txt") as f:
	data=f.readlines()
maximo=0

for attr in data[0].split():
	if ":" in attr:
		pair=attr.split(":")
		if (int(pair[0])>maximo):
			maximo=int(pair[0])
			
nAttributes=maximo
nColumns=nAttributes+2
matrix=numpy.zeros((len(data),nColumns))
matrix[:,nColumns-2]=1.0
minimo=sys.float_info.max
maximo=sys.float_info.min
for index,line in enumerate(data):
	splitted=line.split()
	if ("-" in splitted[0]) or ("2" in splitted[0]):
		matrix[index][nColumns-1]=-1.0
	else:
		matrix[index][nColumns-1]=1.0
	for value in line.split():
		if ":" in value:
			attr=value.split(":")
			matrix[index,int(attr[0])-1]=attr[1]
			if float(attr[1])>maximo:
				maximo=float(attr[1])
			if float(attr[1])<minimo:
				minimo=float(attr[1])

div=float(maximo)-float(minimo)
i=0
j=0

for i in range(0,len(matrix)):
	for j in range(0,len(matrix[0])-2):
		matrix[i,j]=float((matrix[i,j]-minimo)/(div))
			
tmp=int(len(matrix)/4)

numpy.savetxt(filenamePath+"Proc1.txt", matrix[0:tmp,:], delimiter=" ", fmt="%s") 
numpy.savetxt(filenamePath+"Proc2.txt", matrix[tmp:2*tmp,:], delimiter=" ", fmt="%s") 
numpy.savetxt(filenamePath+"Proc3.txt", matrix[2*tmp:3*tmp,:], delimiter=" ", fmt="%s") 
numpy.savetxt(filenamePath+"Proc4.txt", matrix[3*tmp:,:], delimiter=" ", fmt="%s") 

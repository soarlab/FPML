import os
import sys
import shutil

### Modify with correct path ###
pathDatasets="/home/roki/GIT/src/datasets/";
mpfrcpp="/home/roki/GIT/src/MPFR-ML/src/mpfr.cpp"
softfloat="/home/roki/softFloat/"
flexfloatcpp="/home/roki/GIT/src/FlexFloat-ML/flexfloat.cpp"
################################

mpfr="mpic++ -std=c++11 "+mpfrcpp+" -o mpfr -lmpfr -lgmp"
val=os.system(mpfr);
if (val!=0):
	print "ERROR COMPILER(0)"
	sys.exit()

#### Format configuration ###

#### Exponent Configuration
LexponentDataset=[10];
LexponentComputation=[10];
LexponentTest=[10];
############

#### Mantissa Configuration
LmantissaDataset=[2,3,4,5,6,7,8,9,10,23,52];
LmantissaComputation=[2,3,4,5,6,7,8,9,10,23,52];
LmantissaTest=[2,3,4,5,6,7,8,9,10,23,52];
###########################

for fileName in os.listdir(pathDatasets):
	print "Dataset: "+str(fileName)
	if (os.path.isdir(pathDatasets+fileName+"/")):
		path=pathDatasets+fileName+"/"
		if os.path.exists(path+"FLEX/"):
			shutil.rmtree(path+"FLEX/")
		if os.path.exists(path+"MPFR/"):
			shutil.rmtree(path+"MPFR/")
		if os.path.exists(path+"GRAPHS/"):
			shutil.rmtree(path+"GRAPHS/")	
		findConf="g++ -std=c++11 "+str(flexfloatcpp) +" "+softfloat+"build/Linux-386-GCC/softfloat.a -I "+softfloat+"source/include/ -o flexfloat" 
		#print findConf
		val=os.system(findConf);
		if (val==0):
			os.system("./flexfloat "+path);
		else:
			print "ERROR COMPILER(1)!!"
			sys.exit()
			
		with open(path+"conf.txt") as f:
			conf=f.readlines()
		
		elems=conf[0].split(",")
		epochSVM=elems[1].split("=")[1]
		learningRateSVM=elems[2].split("=")[1]
		CRegularizerSVM=elems[3].split("=")[1]

		elems=conf[1].split(",")
		epochP=elems[1].split("=")[1]
		learningRateP=elems[2].split("=")[1]
		
		elems=conf[2].split(",")
		epochAP=elems[1].split("=")[1]
		learningRateAP=elems[2].split("=")[1]
		
		os.makedirs(path+"FLEX/")
		os.makedirs(path+"MPFR/")
		os.makedirs(path+"GRAPHS/")
		os.makedirs(path+"GRAPHS/AP")
		os.makedirs(path+"GRAPHS/P")
		os.makedirs(path+"GRAPHS/SVM")
		for mantissaDataset in LmantissaDataset:
			for exponentDataset in LexponentDataset:
				for mantissaComputation in LmantissaComputation:
					for exponentComputation in LexponentComputation:
						for mantissaTest in LmantissaTest:
							for exponentTest in LexponentTest:
								flexFloat=("g++ -std=c++11 -Danalysis=true -DdatasetMantissa="+str(mantissaDataset)+" -DdatasetExponent="+str(exponentDataset)+
																		 " -DcomputationMantissa="+str(mantissaComputation)+" -DcomputationExponent="+str(exponentComputation)+
																		 " -DtestMantissa="+str(mantissaTest)+ " -DtestExponent="+str(exponentTest)+ 
																		 " "+flexfloatcpp+" "+str(softfloat)+"build/Linux-386-GCC/softfloat.a -I "+
																		 str(softfloat)+"source/include/ -o flexfloat")
								print flexFloat
								val=os.system(flexFloat);
								if (val==0):
									analysis="./flexfloat "+path+" "+ learningRateSVM+" "+CRegularizerSVM+" "+epochSVM+ " " + learningRateP +" "+ epochP+ " "+ learningRateAP+ " " +epochAP
									print analysis
									os.system(analysis)
								else:
									print "ERROR COMPILER(2)!!"
									sys.exit()
								mpfrAnalysis=("./mpfr "+path+" "+ learningRateSVM+" "+CRegularizerSVM+" "+epochSVM+ " " + learningRateP +" "+ epochP+ " "+ learningRateAP+ " " +epochAP+
								" "+str(mantissaDataset)+" "+str(exponentDataset)+" "+str(mantissaComputation)+" "+str(exponentComputation)+" "+str(mantissaTest)+" "+str(exponentTest))
								print mpfrAnalysis
								os.system(mpfrAnalysis)

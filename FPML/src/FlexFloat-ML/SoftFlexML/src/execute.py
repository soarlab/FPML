import os

path="/home/roki/workspace/FlexFloatML/fourclass/";

for fileName in os.listdir(path):
    if ("Train" in fileName) or ("Test" in fileName):
        os.remove(path+fileName)

LexponentDataset=[10];
LexponentComputation=[10];
LexponentTest=[10];

LmantissaDataset=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,52];
LmantissaComputation=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,52];
LmantissaTest=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,52];

for mantissaDataset in LmantissaDataset:
	for exponentDataset in LexponentDataset:
		for mantissaComputation in LmantissaComputation:
			for exponentComputation in LexponentComputation:
				for mantissaTest in LmantissaTest:
					for exponentTest in LexponentTest:
						exe=("g++ -std=c++11 -Danalysis=true -DdatasetMantissa="+str(mantissaDataset)+" -DdatasetExponent="+str(exponentDataset)+" -DcomputationMantissa="+str(mantissaComputation)+
						" -DcomputationExponent="+str(exponentComputation)+" -DtestMantissa="+str(mantissaTest)+
						" -DtestExponent="+str(exponentTest)+ " flexfloat.cpp /home/roki/softFloat/build/Linux-386-GCC/softfloat.a -I /home/roki/softFloat/source/include/"+
						" -o \"example_flexfloat\"");
						print exe
						val=os.system(exe);
						if (val==0):
							os.system("./example_flexfloat");
						else:
							print "ERROR COMPILER!!"

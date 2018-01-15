/*
 * Find types and Operations in source/include/flexfloat.hpp
 *
 * To compile:
 * g++ -std=c++11 example_flexfloat.cpp -I ../source/include/ ../build/Linux-386-GCC/softfloat.a -o example_flexfloat_cpp
 *
 * 
 * -Lpath .... -lmpfr -> look in path for libmpfr.a or libmpfr.so ...
 * 
 * libsoftfloat.a -> -LLinux-386 -lsoftfloat
 * g++ -std=c++11 ../src/example_flexfloat.cpp -L/home/roki/softFloat/build/Linux-386-GCC -lsoftfloat -I/home/roki/softFloat/source/include/ -O0 -o example_flexfloat
 *
 */
#include "flexfloat.hpp"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <fstream>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <utility> 

#ifndef datasetMantissa
	#define datasetMantissa 100
#endif
#ifndef datasetExponent
	#define datasetExponent 10
#endif
#ifndef computationMantissa
	#define computationMantissa 100
#endif
#ifndef computationExponent
	#define computationExponent 10
#endif
#ifndef testMantissa
	#define testMantissa 100
#endif
#ifndef testExponent
	#define testExponent 10
#endif
#ifndef analysis
	#define analysis false
#endif

using namespace std;

void printMatrix(vector<vector<flexfloat<datasetExponent,datasetMantissa>>> &matrix){
	cout<<matrix.size()<<endl;
	cout<<matrix[0].size()<<endl;
	for (int j=0;j<matrix.size();j++){
		for (int i=0;i<matrix[0].size();i++){
			cout<<flexfloat_as_double<<matrix[j][i]<<" ";
		}
		cout<<endl;
	}
}

void printArray(vector<flexfloat<datasetExponent,datasetMantissa>> &vect){
	for (int i=0;i<vect.size();i++)
		cout<<flexfloat_as_double<<vect[i]<<" ";
	cout<<endl;
}

vector<vector<flexfloat<datasetExponent,datasetMantissa>>> readData(vector<string> &data){
	vector<vector<flexfloat<datasetExponent,datasetMantissa>>> dataset={};
	for(int i=0;i<data.size();i++){
		vector<string> strs={};
		boost::split(strs,data[i],boost::is_any_of("\t "));
		vector<flexfloat< datasetExponent,datasetMantissa>> tmp={};
		for(int j=0;j<strs.size()-1;j++){
			tmp.push_back(atof(strs[j].c_str()));
		}
		dataset.push_back(tmp);
	}
	return dataset;
}

vector<string> parseDataset(string nameFile){
	ifstream file(nameFile);
	string line;
	vector<string> data={};
	while (getline(file, line)){
		if (!line.empty()){
			data.push_back(line);
		}
	}
	file.close();
	return data;
}

vector<flexfloat<datasetExponent,datasetMantissa>> readLabel(vector<string> &data){
	vector<flexfloat<datasetExponent,datasetMantissa>> label={};
	for(int i=0;i<data.size();i++){
		vector<string> strs={};
		boost::split(strs,data[i],boost::is_any_of("\t "));
		label.push_back(atof(strs.back().c_str()));
	}
	return label;
}

void shuffleData(vector<vector<flexfloat<datasetExponent,datasetMantissa>>> &matrix, vector<flexfloat<datasetExponent,datasetMantissa>> &label){
	srand (time(NULL));
	for (int i=0;i<label.size()/2.0;i++){
		int v1 = rand() % label.size();
		int v2 = rand() % label.size();
		swap(matrix[v1],matrix[v2]);
		swap(label[v1],label[v2]);
	}
}

double testDataset(vector<vector<flexfloat<datasetExponent, datasetMantissa>>> &matrix,vector<flexfloat<datasetExponent, datasetMantissa>> &label,vector<flexfloat<computationExponent, computationMantissa>> &weights){
	int mistakes=0;
	for(int i=0;i<matrix.size();i++){
		flexfloat<testExponent,testMantissa> dotProduct=0;
		for (int index=0;index<matrix[i].size();index++){
			dotProduct=dotProduct+((flexfloat<testExponent,testMantissa>)matrix[i][index])*((flexfloat<testExponent,testMantissa>)weights[index]);
		}
		dotProduct=(dotProduct)*(flexfloat<testExponent,testMantissa>)label[i];
		if (dotProduct<=0){
			mistakes++;
		}
		if (isnan(dotProduct.getValue())){
			return 0;
		}
	}
	return (1-((double)mistakes/label.size()))*100;
}

void SVM(vector<vector<flexfloat<datasetExponent, datasetMantissa>>> &matrix,vector<flexfloat<datasetExponent, datasetMantissa>> &label, vector<flexfloat<computationExponent, computationMantissa>> &weights, flexfloat<computationExponent, computationMantissa> &C, flexfloat<computationExponent, computationMantissa> &learningRate){
	flexfloat<computationExponent, computationMantissa> updateLearningRate={0};
	for (int n=0;n<matrix.size();n++){
		updateLearningRate=learningRate/(1.0+(learningRate*(n/C)));
		flexfloat<computationExponent, computationMantissa> dotProduct=0;
		for (int index=0;index<matrix[n].size();index++){
			dotProduct=dotProduct+(((flexfloat<computationExponent, computationMantissa>)matrix[n][index])*weights[index]);
		}
		dotProduct=(dotProduct)*(flexfloat<computationExponent, computationMantissa>)label[n];
		if (dotProduct<=1){
			for (int index=0;index<weights.size();index++){
				weights[index]=weights[index]*(1.0-updateLearningRate) + updateLearningRate*C*((flexfloat<computationExponent, computationMantissa>)label[n])*((flexfloat<computationExponent, computationMantissa>)matrix[n][index]);
			}
		}
		else{
			for (int index=0;index<weights.size();index++){
				weights[index]=weights[index]*(1.0-updateLearningRate);
			}
		}
	}
}

void Perceptron(vector<vector<flexfloat<datasetExponent, datasetMantissa>>> &matrix,vector<flexfloat<datasetExponent, datasetMantissa>> &label, vector<flexfloat<computationExponent, computationMantissa>> &weights, flexfloat<computationExponent, computationMantissa> &learningRate){
	int mistakes=0;
	for (int n=0;n<matrix.size();n++){
		flexfloat<computationExponent, computationMantissa> dotProduct=0;
		for (int index=0;index<matrix[n].size();index++){
			dotProduct=dotProduct+(((flexfloat<computationExponent, computationMantissa>)matrix[n][index])*weights[index]);
		}
		dotProduct=(dotProduct)*(flexfloat<computationExponent, computationMantissa>)label[n];
		if (dotProduct<=0){
			mistakes++;
			for (int index=0;index<weights.size();index++){
				weights[index]=weights[index]+learningRate*((flexfloat<computationExponent, computationMantissa>)label[n])*((flexfloat<computationExponent, computationMantissa>)matrix[n][index]);
			}
		}
	}
	//cout<<"P:"<<mistakes<<" "<<matrix.size()<<endl;
}

void AveragePerceptron(vector<vector<flexfloat<datasetExponent, datasetMantissa>>> &matrix,vector<flexfloat<computationExponent, computationMantissa>> &averageWeights,double &c,vector<flexfloat<datasetExponent, datasetMantissa>> &label, vector<flexfloat<computationExponent, computationMantissa>> &weights, flexfloat<computationExponent, computationMantissa> &learningRate){
	int mistakes=0;
	for (int n=0;n<matrix.size();n++){
		flexfloat<computationExponent, computationMantissa> dotProduct=0;
		for (int index=0;index<matrix[n].size();index++){
			dotProduct=dotProduct+(((flexfloat<computationExponent, computationMantissa>)matrix[n][index])*weights[index]);
		}
		dotProduct=(dotProduct)*(flexfloat<computationExponent, computationMantissa>)label[n];
		if (dotProduct<=0){
			mistakes++;
			for (int index=0;index<weights.size();index++){
				weights[index]=weights[index]+learningRate*((flexfloat<computationExponent, computationMantissa>)label[n])*((flexfloat<computationExponent, computationMantissa>)matrix[n][index]);
				averageWeights[index]=averageWeights[index]+learningRate*((flexfloat<computationExponent, computationMantissa>)label[n])*((flexfloat<computationExponent, computationMantissa>)c)*((flexfloat<computationExponent, computationMantissa>)matrix[n][index]);
			}
		}
		c=c+1;
	}
	//cout<<"AP:"<<mistakes<<" "<<matrix.size()<<endl;
}

void mapBestConfiguration(stringstream &myStream, map<string,vector<double>> &myMap, double accuracyTraining){
	string hashString;
	if (!analysis){
		hashString=myStream.str();
		if (myMap.find(hashString) == myMap.end()) {
			myMap[hashString]={accuracyTraining};
		}
		else{
			myMap[hashString].push_back(accuracyTraining);
		}
	}
}
void printBestConfiguration(string label, map<string,vector<double>> &myMap){
	double max=0;
	string best="";
	for(map<string, vector<double> >::const_iterator it = myMap.begin(); it != myMap.end(); ++it) {
		cout <<label<<" "<< it->first<< " ";
		double acc=0;
		for(auto val:it->second){	
			acc=acc+val;
			cout<< to_string(val)<<" ";
		}
		acc=acc/((it->second).size());
		if (acc>max){
			max=acc;
			best=it->first+ " Acc:" + to_string(acc);	
		}
		cout <<label<<" Mean: "+to_string(acc)+"\n";
	}
	cout << label<<" Best: " << best<<"\n";
}


int main(int argc, char* argv[]){
	ofstream trainingSVM;
	ofstream testSVM;
	
	ofstream trainingPerceptron;
	ofstream testPerceptron;
	
	ofstream trainingAverage;
	ofstream testAverage;
	
	vector<string> trainingData={};
	vector<string> testData={};
	string fileName="/home/roki/GIT/FPML/src/FlexFloat-ML/fourclass/fourclassProc";
		
	map<string,vector<double>> myMapTrainingSVM;
	map<string,vector<double>> myMapTestSVM;
	
	map<string,vector<double>> myMapTrainingP;
	map<string,vector<double>> myMapTestP;
	
	map<string,vector<double>> myMapTrainingAP;
	map<string,vector<double>> myMapTestAP;

	vector<flexfloat<computationExponent, computationMantissa>> learningRates;
	vector<flexfloat<computationExponent, computationMantissa>> CRegularizers;

	vector<int> epochsVect;
		
	if (!analysis){
		learningRates={10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};
		CRegularizers={10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};
		epochsVect={1,5,10,20,30,50};
	}
	
	for (int i=1;i<5;i++){
		trainingSVM.open (fileName+"SVMTrain"+to_string(i)+".txt",std::ofstream::app);
		testSVM.open (fileName+"SVMTest"+to_string(i)+".txt",std::ofstream::app);
		
		trainingPerceptron.open (fileName+"PTrain"+to_string(i)+".txt",std::ofstream::app);
		testPerceptron. open (fileName+"PTest"+to_string(i)+".txt",std::ofstream::app);
		
		trainingAverage.open (fileName+"APTrain"+to_string(i)+".txt",std::ofstream::app);
		testAverage.open (fileName+"APTest"+to_string(i)+".txt",std::ofstream::app);
		
		trainingData={};
		testData={};
		for (int j=1;j<5;j++){
			if (i!=j){
				cout<<"Train:"+fileName+to_string(j)+".txt"<<endl;
				vector<string> tmp=parseDataset(fileName+to_string(j)+".txt");
				for (int val=0;val<tmp.size();val++){
					trainingData.push_back(tmp[val]);
				}
			}
			else{
				cout<<"Test:"+fileName+to_string(j)+".txt"<<endl;
				vector<string> tmp=parseDataset(fileName+to_string(j)+".txt");
				for (int val=0;val<tmp.size();val++){
					testData.push_back(tmp[val]);
				}
			}
		}
		cout<<endl;
		vector<vector<flexfloat<datasetExponent, datasetMantissa>>> matrix=readData(trainingData);
		vector<flexfloat<datasetExponent, datasetMantissa>> label= readLabel(trainingData);
		vector<vector<flexfloat<datasetExponent, datasetMantissa>>> matrixTest=readData(testData);
		vector<flexfloat<datasetExponent, datasetMantissa>> labelTest= readLabel(testData);
		shuffleData(matrix,label);
		shuffleData(matrixTest,labelTest);
	
		if (analysis){
			learningRates={0.0001};
			CRegularizers={10};
			epochsVect={10};
		}
		for (flexfloat<computationExponent, computationMantissa> &learningRate:learningRates){
			for(flexfloat<computationExponent, computationMantissa> &C:CRegularizers){
				flexfloat<computationExponent, computationMantissa> updateLearningRate={0};
				vector<flexfloat<computationExponent, computationMantissa>> weights(matrix[0].size(),0);
				for(int epochs:epochsVect){
					for (int epoch=0;epoch<epochs;epoch++){
						SVM(matrix,label,weights,C,learningRate);
						shuffleData(matrix,label);
					}
					
					double accuracyTraining=testDataset(matrix,label,weights);
					
					stringstream trainingString;
					trainingString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"(N.R,N.R)$"<<"LR="<<flexfloat_as_double<<learningRate<<",C="<<flexfloat_as_double<<C<<"$SIZE$SVM:"<<to_string(accuracyTraining)<<"$";
					trainingSVM<<trainingString.str()<<endl;
					
					trainingString.str("");
					trainingString<<"Epochs="<<flexfloat_as_double<<epochs<<",LR="<<flexfloat_as_double<<learningRate<<",C="<<flexfloat_as_double<<C;
					
					mapBestConfiguration(trainingString,myMapTrainingSVM,accuracyTraining);
					
					double accuracyTest=testDataset(matrixTest,labelTest,weights);
					
					stringstream testString;
					testString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"("<<to_string(testMantissa)<<","<<to_string(testExponent)<<")$"<<"LR="<<flexfloat_as_double<<learningRate<<",C="<<flexfloat_as_double<<C<<"$SIZE$SVM:"<<to_string(accuracyTest)<<"$";
					testSVM<<testString.str()<<endl;
					
					mapBestConfiguration(trainingString,myMapTestSVM,accuracyTest);
				}
			}
		}
		
		if (analysis){
			learningRates={0.001};
			epochsVect={20};
		}
		for (flexfloat<computationExponent, computationMantissa> &learningRate:learningRates){
			for(int epochs:epochsVect){
				vector<flexfloat<computationExponent, computationMantissa>> weights(matrix[0].size(),0);
				for (int epoch=0;epoch<epochs;epoch++){
					Perceptron(matrix,label,weights,learningRate);
					shuffleData(matrix,label);
				}
			
				double accuracyTraining=testDataset(matrix,label,weights);
				
				stringstream trainingString;
				trainingString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"(N.R,N.R)$"<<"LR="<<flexfloat_as_double<<learningRate<<"$SIZE$P:"<<to_string(accuracyTraining)<<"$";
				trainingPerceptron<<trainingString.str()<<endl;
				
				trainingString.str("");
				trainingString<<"Epochs="<<flexfloat_as_double<<epochs<<",LR="<<flexfloat_as_double<<learningRate;
				
				mapBestConfiguration(trainingString,myMapTrainingP,accuracyTraining);
				
				double accuracyTest=testDataset(matrixTest,labelTest,weights);
				
				stringstream testString;
				testString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"("<<to_string(testMantissa)<<","<<to_string(testExponent)<<")$"<<"LR="<<flexfloat_as_double<<learningRate<<"$SIZE$P:"<<to_string(accuracyTest)<<"$";
				testPerceptron<<testString.str()<<endl;
				
				mapBestConfiguration(trainingString,myMapTestP,accuracyTest);
			}
		}
		
		if (analysis){
			learningRates={0.001};
			epochsVect={20};
		}
		for (flexfloat<computationExponent, computationMantissa> &learningRate:learningRates){		
			for(int epochs:epochsVect){
				vector<flexfloat<computationExponent, computationMantissa>> weights(matrix[0].size(),0);
				vector<flexfloat<computationExponent, computationMantissa>> averageWeights(matrix[0].size(),0);
				double c=1.0;
				for (int epoch=0;epoch<epochs;epoch++){
					AveragePerceptron(matrix,averageWeights,c,label,weights,learningRate);
					shuffleData(matrix,label);
				}
				
				c=1.0/c;
				for (int index=0;index<weights.size();index++){
					averageWeights[index]=weights[index]-(c*averageWeights[index]);
				}

				double accuracyTraining=testDataset(matrix,label,averageWeights);
				
				stringstream trainingString;
				trainingString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"(N.R,N.R)$"<<"LR="<<flexfloat_as_double<<learningRate<<"$SIZE$AP:"<<to_string(accuracyTraining)<<"$";
				trainingAverage<<trainingString.str()<<endl;
				
				trainingString.str("");
				trainingString<<"Epochs="<<flexfloat_as_double<<epochs<<",LR="<<flexfloat_as_double<<learningRate;
				
				mapBestConfiguration(trainingString,myMapTrainingAP,accuracyTraining);
				
				double accuracyTest=testDataset(matrixTest,labelTest,averageWeights);
				
				stringstream testString;
				testString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"("<<to_string(testMantissa)<<","<<to_string(testExponent)<<")$"<<"LR="<<flexfloat_as_double<<learningRate<<"$SIZE$AP:"<<to_string(accuracyTest)<<"$";
				testAverage<<testString.str()<<endl;
				
				mapBestConfiguration(trainingString,myMapTestAP,accuracyTest);
			}
		}
		trainingAverage.close();
		testAverage.close();
		
		trainingPerceptron.close();
		testPerceptron.close();
		
		trainingSVM.close();
		testSVM.close();
	}
	if (!analysis){
		printBestConfiguration("SVM Train",myMapTrainingSVM);
		cout<<"\n\n"<<endl;
		printBestConfiguration("SVM Test",myMapTestSVM);
		cout<<"\n\n"<<endl;
		printBestConfiguration("Perceptron Train",myMapTrainingP);
		cout<<"\n\n"<<endl;
		printBestConfiguration("Perceptron Test",myMapTestP);
		cout<<"\n\n"<<endl;
		printBestConfiguration("Average Train",myMapTrainingAP);
		cout<<"\n\n"<<endl;
		printBestConfiguration("Average Test",myMapTestAP);
	}
}

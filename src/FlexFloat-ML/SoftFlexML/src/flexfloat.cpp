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
#include <sys/stat.h>

//When parameters are not fixed by the analysis (using execute.py), it means the flexfloat.cpp is used for hyperparameter detection.
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
	//It means that the FP analysis is not performed, instead the flexfloat.cpp is used to detect the best configuration of hyperparameters.
	#define analysis false
#endif

using namespace std;

/***
 *It prints the matrix after converting each element to double representation.
 * */ 
void printMatrix(vector<vector<flexfloat<datasetExponent,datasetMantissa>>> &matrix){
	cout<<"Matrix Dim"<<endl;
	cout<<matrix.size()<<endl;
	cout<<matrix[0].size()<<endl;
	for (int j=0;j<matrix.size();j++){
		for (int i=0;i<matrix[0].size();i++){
			cout<<flexfloat_as_double<<matrix[j][i]<<" ";
		}
		cout<<endl;
	}
}

/***
 * It prints the array after casting each element to double precision;
 * */
void printArray(vector<flexfloat<datasetExponent,datasetMantissa>> &vect){
	for (int i=0;i<vect.size();i++)
		cout<<flexfloat_as_double<<vect[i]<<" ";
	cout<<endl;
}

/***
 * Converter of the dataset: it receives in input the file as it is (in form of lines). It returns the matrix where each element has <datasetExponent,datasetMantissa> precision;
 * */
vector<vector<flexfloat<datasetExponent,datasetMantissa>>> readData(vector<string> &data){
	vector<vector<flexfloat<datasetExponent,datasetMantissa>>> dataset={};
	for(int i=0;i<data.size();i++){
		vector<string> strs={};
		boost::split(strs,data[i],boost::is_any_of("\t "));
		vector<flexfloat<datasetExponent,datasetMantissa>> tmp={};
		for(int j=0;j<strs.size()-1;j++){
			tmp.push_back(stod(strs[j].c_str()));
		}
		dataset.push_back(tmp);
	}
	return dataset;
}

/***
 * Converter of the label: it receives in input the file as it is (in form of lines). It returns the label matrix where each element has <datasetExponent,datasetMantissa> precision;
 * */
vector<flexfloat<datasetExponent,datasetMantissa>> readLabel(vector<string> &data){
	vector<flexfloat<datasetExponent,datasetMantissa>> label={};
	for(int i=0;i<data.size();i++){
		vector<string> strs={};
		boost::split(strs,data[i],boost::is_any_of("\t "));
		label.push_back(stod(strs.back().c_str()));
	}
	return label;
}

/***
 * Parse the dataset from file.
 * */
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

/***
 * Shuffle data
 * */
void shuffleData(vector<vector<flexfloat<datasetExponent,datasetMantissa>>> &matrix, vector<flexfloat<datasetExponent,datasetMantissa>> &label){
	srand (time(NULL));
	for (int i=0;i<label.size()/2;i++){
		int v1 = rand() % label.size();
		int v2 = rand() % label.size();
		swap(matrix[v1],matrix[v2]);
		swap(label[v1],label[v2]);
	}
}

/***
 * Testing procedure of Perceptron, AVP and SVM. Before performing dot product each element of the 
 * dataset is casted to the precision used for testing. The same for weight vector and label.
 * */
double testDataset(vector<vector<flexfloat<datasetExponent, datasetMantissa>>> &matrix,vector<flexfloat<datasetExponent, datasetMantissa>> &label,vector<flexfloat<computationExponent, computationMantissa>> &weights){
	int mistakes=0;
	for(int i=0;i<matrix.size();i++){
		flexfloat<testExponent,testMantissa> dotProduct=0;
		for (int index=0;index<matrix[i].size();index++){
			dotProduct=dotProduct+((flexfloat<testExponent,testMantissa>)matrix[i][index])*((flexfloat<testExponent,testMantissa>)weights[index]);
		}
		dotProduct=(dotProduct)*((flexfloat<testExponent,testMantissa>)label[i]);
		if (dotProduct<=0){
			mistakes++;
		}
		if (isnan(dotProduct.getValue())){
			return 0;
		}
	}
	return (1-((double)mistakes/label.size()))*100;
}

/***
 * SVM training. Before the dot product is performed, each element of the matrix is mapped to the precision for computation.
 * Every computation performed in training is bounded by <computationExponent, computationMantissa>.
 * */
void SVM(vector<vector<flexfloat<datasetExponent, datasetMantissa>>> &matrix,vector<flexfloat<datasetExponent, datasetMantissa>> &label, vector<flexfloat<computationExponent, computationMantissa>> &weights, flexfloat<computationExponent, computationMantissa> &C, flexfloat<computationExponent, computationMantissa> &learningRate){
	flexfloat<computationExponent, computationMantissa> updateLearningRate={0};
	for (int n=0;n<matrix.size();n++){
		updateLearningRate=learningRate/(1.0+(learningRate*(((flexfloat<computationExponent, computationMantissa>)n)/C)));
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
/***
 * Perceptron training. Before the dot product is performed, each element of the matrix is mapped to the precision for computation.
 * Every computation performed in training is bounded by <computationExponent, computationMantissa>.
 * */
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
}
/***
 * Average Perceptron training. Before the dot product is performed, each element of the matrix is mapped to the precision for computation.
 * Every computation performed in training is bounded by <computationExponent, computationMantissa>.
 * */
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
}

/***
 * Utility used to detect the best hyperparameters configuration
 * */
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
/***
 * Utility used to print out the best hyperparameters configuration
 * */
void printBestConfiguration(ofstream &confFile,string label, map<string,vector<double>> &myMap){
	double max=0;
	string best="";
	for(map<string, vector<double> >::const_iterator it = myMap.begin(); it != myMap.end(); ++it) {
		//cout <<label<<" "<< it->first<< " ";
		double acc=0;
		for(auto val:it->second){	
			acc=acc+val;
			//cout<< to_string(val)<<" ";
		}
		acc=acc/((it->second).size());
		if (acc>max){
			max=acc;
			best=it->first+ ",Acc:" + to_string(acc);	
		}
		//cout <<label<<" Mean: "+to_string(acc)+"\n";
	}
	cout << label<<" Best: " << best<<"\n";
	confFile<<label<<","<<best<<"\n";
}

int main(int argc, char* argv[]){
	if (argc<2){
		cout<<"Input the path of the folder. The dataset has to be diveded in 4 parts with names: part1.txt, part2.txt, part3.txt, part4.txt"<<endl;
		return 0;
	}
	if ((analysis) && (argc<9)){
		cout<<"Parameters problem"<<endl;
		return 0;
	}
	ofstream trainingSVM;
	ofstream testSVM;
	
	ofstream trainingPerceptron;
	ofstream testPerceptron;
	
	ofstream trainingAverage;
	ofstream testAverage;
	
	vector<string> trainingData={};
	vector<string> testData={};
	string fileName=argv[1];
	
	map<string,vector<double>> myMapTrainingSVM;
	map<string,vector<double>> myMapTestSVM;
	
	map<string,vector<double>> myMapTrainingP;
	map<string,vector<double>> myMapTestP;
	
	map<string,vector<double>> myMapTrainingAP;
	map<string,vector<double>> myMapTestAP;
	
	vector<flexfloat<computationExponent, computationMantissa>> learningRatesPerceptron;
	vector<int> epochsVectPerceptron;
		
	vector<flexfloat<computationExponent, computationMantissa>> learningRatesAverage;
	vector<int> epochsVectAverage;

	vector<flexfloat<computationExponent, computationMantissa>> learningRatesSVM;
	vector<flexfloat<computationExponent, computationMantissa>> CRegularizersSVM;
	vector<int> epochsVectSVM;
	
	ofstream outConfFile;
	
	//When the analysis is not performed the program looks for the best hyperparametrs configuration.
	if (!analysis){
		outConfFile.open(fileName+"conf.txt");
		
		learningRatesSVM={10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};
		epochsVectSVM={5,10,20};
		CRegularizersSVM={10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};	
		
		learningRatesPerceptron={10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};
		epochsVectPerceptron={5,10,20};
		
		learningRatesAverage={10.0,1.0,0.1,0.01,0.001,0.0001,0.00001};
		epochsVectAverage={5,10,20};
		
	}
	if (analysis){
		learningRatesSVM={stod(argv[2])};
		CRegularizersSVM={stod(argv[3])};
		epochsVectSVM={stoi(argv[4])};
		
		learningRatesPerceptron={stod(argv[5])};
		epochsVectPerceptron={stoi(argv[6])};
		
		learningRatesAverage={stod(argv[7])};
		epochsVectAverage={stoi(argv[8])};
	}
	
	for (int i=1;i<5;i++){
		if (analysis){
			trainingSVM.open(fileName+"FLEX/part"+to_string(i)+"SVMTrain.txt",std::ofstream::app);
			testSVM.open(fileName+"FLEX/part"+to_string(i)+"SVMTest.txt",std::ofstream::app);
		
			trainingPerceptron.open(fileName+"FLEX/part"+to_string(i)+"PTrain.txt",std::ofstream::app);
			testPerceptron.open(fileName+"FLEX/part"+to_string(i)+"PTest.txt",std::ofstream::app);
		
			trainingAverage.open(fileName+"FLEX/part"+to_string(i)+"APTrain.txt",std::ofstream::app);
			testAverage.open(fileName+"FLEX/part"+to_string(i)+"APTest.txt",std::ofstream::app);
		}
		trainingData={};
		testData={};
		for (int j=1;j<5;j++){
			if (i!=j){
				//cout<<"Train:"+fileName+"part"+to_string(j)+".txt"<<endl;
				vector<string> tmp=parseDataset(fileName+"part"+to_string(j)+".txt");
				for (int val=0;val<tmp.size();val++){
					trainingData.push_back(tmp[val]);
				}
			}
			else{
				vector<string> tmp=parseDataset(fileName+"part"+to_string(j)+".txt");
				for (int val=0;val<tmp.size();val++){
					testData.push_back(tmp[val]);
				}
			}
		}
		//Parse the dataset
		vector<vector<flexfloat<datasetExponent, datasetMantissa>>> matrix=readData(trainingData);
		vector<flexfloat<datasetExponent, datasetMantissa>> label= readLabel(trainingData);
		vector<vector<flexfloat<datasetExponent, datasetMantissa>>> matrixTest=readData(testData);
		vector<flexfloat<datasetExponent, datasetMantissa>> labelTest= readLabel(testData);
		shuffleData(matrix,label);
		shuffleData(matrixTest,labelTest);
	
		//SVM
		for (flexfloat<computationExponent, computationMantissa> &learningRate:learningRatesSVM){
			for(flexfloat<computationExponent, computationMantissa> &C:CRegularizersSVM){
				flexfloat<computationExponent, computationMantissa> updateLearningRate={0};
				vector<flexfloat<computationExponent, computationMantissa>> weights(matrix[0].size(),0);
				for(int epochs:epochsVectSVM){
					for (int epoch=0;epoch<epochs;epoch++){
						SVM(matrix,label,weights,C,learningRate);
						shuffleData(matrix,label);
					}
					
					double accuracyTraining=testDataset(matrix,label,weights);
					
					stringstream trainingString;
					trainingString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"("<<to_string(testMantissa)<<","<<to_string(testExponent)<<")$"<<"LR="<<flexfloat_as_double<<learningRate<<",C="<<flexfloat_as_double<<C<<"$SIZE$SVM:"<<to_string(accuracyTraining)<<"$";
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
		
		//Perceptron
		for (flexfloat<computationExponent, computationMantissa> &learningRate:learningRatesPerceptron){
			for(int epochs:epochsVectPerceptron){
				vector<flexfloat<computationExponent, computationMantissa>> weights(matrix[0].size(),0);
				for (int epoch=0;epoch<epochs;epoch++){
					Perceptron(matrix,label,weights,learningRate);
					shuffleData(matrix,label);
				}
			
				double accuracyTraining=testDataset(matrix,label,weights);
				
				stringstream trainingString;
				trainingString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"("<<to_string(testMantissa)<<","<<to_string(testExponent)<<")$"<<"LR="<<flexfloat_as_double<<learningRate<<"$SIZE$P:"<<to_string(accuracyTraining)<<"$";
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
		
		//Average Perceptron
		for (flexfloat<computationExponent, computationMantissa> &learningRate:learningRatesAverage){		
			for(int epochs:epochsVectAverage){
				vector<flexfloat<computationExponent, computationMantissa>> weights(matrix[0].size(),0);
				vector<flexfloat<computationExponent, computationMantissa>> averageWeights(matrix[0].size(),0);
				double c=1.0;
				for (int epoch=0;epoch<epochs;epoch++){
					AveragePerceptron(matrix,averageWeights,c,label,weights,learningRate);
					shuffleData(matrix,label);
				}
				//cout<<c<<endl;
				//cout<<flexfloat_as_double<<((flexfloat<computationExponent, computationMantissa>)c)<<endl;
				c=1.0/c;
				for (int index=0;index<weights.size();index++){
					averageWeights[index]=weights[index]-(((flexfloat<computationExponent, computationMantissa>)c)*averageWeights[index]);
				}

				double accuracyTraining=testDataset(matrix,label,averageWeights);
				
				stringstream trainingString;
				trainingString<<"$("<<to_string(datasetMantissa)<<","<<to_string(datasetExponent)<<")$"<<"("<<to_string(computationMantissa)<<","<<to_string(computationExponent)<<")$"<<"("<<to_string(testMantissa)<<","<<to_string(testExponent)<<")$"<<"LR="<<flexfloat_as_double<<learningRate<<"$SIZE$AP:"<<to_string(accuracyTraining)<<"$";
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
		if (analysis){
			trainingAverage.close();
			testAverage.close();
		
			trainingPerceptron.close();
			testPerceptron.close();
		
			trainingSVM.close();
			testSVM.close();
		}
	}
	if (!analysis){
		//printBestConfiguration("SVM Train",myMapTrainingSVM);
		//cout<<"\n\n"<<endl;
		printBestConfiguration(outConfFile,"SVM",myMapTestSVM);
		//cout<<"\n\n"<<endl;
		//printBestConfiguration("Perceptron Train",myMapTrainingP);
		//cout<<"\n\n"<<endl;
		printBestConfiguration(outConfFile,"Perceptron",myMapTestP);
		//cout<<"\n\n"<<endl;
		//printBestConfiguration("Average Train",myMapTrainingAP);
		//cout<<"\n\n"<<endl;
		printBestConfiguration(outConfFile,"Average",myMapTestAP);
		outConfFile.close();
	}
}

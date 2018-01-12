#include "tools.h"

static vector<char> delimiters={'\t',' '};
static int numOfAttributes;
static int numElementsInRow;
static int precComputation;
static int ROUND;

namespace std
{
template<>
class default_delete<mpfr_t>
{
public:
  void operator()(mpfr_t *ptr)
  {
	for(int i = 0; i < numElementsInRow; i++){
		mpfr_clear(ptr[i]);
	}
	delete[] ptr;
  }
};
}

void printMPFRvar(mpfr_t &var){
	printf("Decimal: ");
	mpfr_out_str (stdout, 10, 0, var, (mpfr_rnd_t)ROUND);
	printf(" Binary: ");
	mpfr_out_str (stdout, 2, 0, var, (mpfr_rnd_t)ROUND);
	printf("\n");
	fflush(stdout);
}

string printExcValue(mpfr_t m){
	// example: for u=3.1416 we have s = "31416" and e = 1
	mpfr_exp_t  e;
	char* charsDec=mpfr_get_str(NULL, &e, 10, 0, m, (mpfr_rnd_t)ROUND);
	string sDec=charsDec;
	if (sDec.at(0)=='-')
		sDec.replace(0,1,"-0.");
	else
		sDec="0."+sDec;
	sDec=sDec+"E"+to_string(e);
	char* charsBin=mpfr_get_str(NULL, &e, 2, 0, m, (mpfr_rnd_t)ROUND);
	string sBin=charsDec;
	sBin=sBin+"E"+to_string(e);
	if (sBin.at(0)=='-')
		sBin.replace(0,1,"-0.");
	else
		sBin="0."+sBin;
	string temp="Dec.: "+sDec+", Bin.: "+sBin;
	mpfr_free_str(charsDec);
	mpfr_free_str(charsBin);
	return temp;
}

void printMPFRarray(unique_ptr<mpfr_t> &weights,int size){
	for (int i=0;i<size;i++){
		printMPFRvar(weights.get()[i]);
	}
	printf("\n");
}

void printWeightDecimal(mpfr_t* a, int size){
	for (int i=0;i<size;i++){
		mpfr_out_str (stdout, 10, 0, a[i], (mpfr_rnd_t)ROUND);
		printf(", ");
	}
	printf("\n");
	fflush(stdout);
}

int checkOverflow(){
	if (mpfr_underflow_p()){
		mpfr_clear_flags();
		return -1;
	}
	if (mpfr_overflow_p()){
		mpfr_clear_flags();
		return -2;
	}
	if (mpfr_erangeflag_p()){
		mpfr_clear_flags();
		return -3;
	}
	return 0;
}

string evaluateRes(int res){
	string s={(res==-1?"UNDERFLOW":(res==-2?"OVERFLOW":"ERANGE"))};
	return s;
}

void throwExcExponent(int n,string message){
	if (n!=0){
		string ret=evaluateRes(n);
		throw out_of_range(message+", Exception:"+ret+", Exp.Range:"+to_string(mpfr_get_emin())+", "+to_string(mpfr_get_emax()));//ERANGE: one of the operand is not a number;
	}
}

vector<string>* split(string s1){
	vector<string>* vs=new vector<string>();
	boost::trim(s1);
	boost::split(*vs, s1, boost::is_any_of("\t "),boost::token_compress_on);
	return vs;
}

int checkRangeVar(mpfr_t &var){
	return mpfr_check_range(var,0,(mpfr_rnd_t)ROUND);
}

int mulMPFR(mpfr_t &result, mpfr_t &first, mpfr_t &second){
	mpfr_mul(result,first,second,(mpfr_rnd_t)ROUND);
	return checkOverflow();
}
int addMPFR(mpfr_t &result,mpfr_t &first,mpfr_t &second){
	mpfr_add(result,first,second,(mpfr_rnd_t)ROUND);
	return checkOverflow();
}

int subMPFR(mpfr_t &result,mpfr_t &first,mpfr_t &second){
	mpfr_sub(result,first,second,(mpfr_rnd_t)ROUND);
	return checkOverflow();
}

int setMPFR(mpfr_t &destination,string tmp,int base){
	mpfr_set_str (destination,tmp.c_str(), base, (mpfr_rnd_t)ROUND);
	int ret=checkRangeVar(destination);
	int ret2=checkOverflow();
	return (ret!=0||ret2!=0);
}

int setMPFR(mpfr_t &destination,mpfr_t &source){
	mpfr_set(destination,source,(mpfr_rnd_t)ROUND);
	int ret=checkRangeVar(destination);
	int ret2=checkOverflow();
	return (ret!=0||ret2!=0);
}

int divMPFR(mpfr_t &result, mpfr_t &first, mpfr_t &second){
	mpfr_div (result, first, second, (mpfr_rnd_t)ROUND);
	return checkOverflow();
}

void myDotProduct(unique_ptr<mpfr_t> &weights,vector<mpfr_t> &x, mpfr_t &tot){
	bool go_on=true;
	#pragma omp parallel shared(go_on) num_threads(1)
	{
		mpfr_t tmp;
		mpfr_init2 (tmp, precComputation);
		mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);

		mpfr_t container;
		mpfr_init2 (container, precComputation);
		mpfr_set_d (container, 0, (mpfr_rnd_t)ROUND);

		int res1=0;
		int res2=0;
		int resTot=0;

		#pragma omp for nowait
		for (int i=0;i<numElementsInRow;i++){
			#pragma omp flush(go_on)
			if (go_on){
				mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);
				res1=mulMPFR(tmp, weights.get()[i], x[i]);
				if (res1!=0){
					go_on=false;
				}
				//printMPFRvar(tmp);
				res2=addMPFR(container,container,tmp);
				if (res2!=0){
					go_on=false;
				}
			}
		}
		if (go_on){
			#pragma omp critical
			{
				resTot=addMPFR(tot,tot,container);
				if (resTot!=0){
					go_on=false;
				}
			}
		}
		mpfr_clear(container);
		mpfr_clear(tmp);
	}
	if (!go_on){
		mpfr_clear_flags();
		throwExcExponent(1,"Dot Product Parallel Exception");
	}
}

void averageWeightsAdjust(unique_ptr<mpfr_t> &weightsBis,unique_ptr<mpfr_t> &averageWeights,int c){
	mpfr_t tmp;
	mpfr_init2 (tmp, precComputation);
	mpfr_set_si (tmp, c, (mpfr_rnd_t)ROUND);

	int res1=0;
	int res2=0;
	for (int i=0;i<numElementsInRow;i++){
		string save=printExcValue(averageWeights.get()[i]);
		res1=divMPFR(averageWeights.get()[i], averageWeights.get()[i], tmp);
		if (res1!=0){
			string stringTmp=printExcValue(tmp);
			throwExcExponent(res2, "(1)Exp. Exc. adjust weights normalization, value1: "+save+ ", value2: "+stringTmp);
		}
		res2=subMPFR(averageWeights.get()[i], weightsBis.get()[i], averageWeights.get()[i]);
		if (res2!=0){
			string stringTmp=printExcValue(weightsBis.get()[i]);
			throwExcExponent(res2, "(2)Exp. Exc. adjust weights normalization, value1: "+stringTmp+ ", value2: "+save);
		}
	}
	mpfr_clear(tmp);
}


void updateWeights(unique_ptr<mpfr_t> &weights, vector<mpfr_t> &rowDataset,mpfr_t &label,mpfr_t &learningRate){
	mpfr_t tmp;
	mpfr_init2 (tmp, precComputation);
	mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);

	int res1=0;
	int res2=0;
	int res3=0;

	string save1;
	string save2;
	for (int i=0;i<numElementsInRow;i++){
		res1=mulMPFR(tmp,rowDataset[i],label);
		if (res1!=0){
			string val1=printExcValue(rowDataset[i]);
			string val2=printExcValue(label);
			throwExcExponent(res1, "(1)Exp. Exc. updating weights, value1: "+val1+ ", value2: "+val2);
		}
		save1=printExcValue(tmp);
		res2=mulMPFR(tmp,tmp,learningRate);
		if (res2!=0){
			string val2=printExcValue(learningRate);
			throwExcExponent(res2, "(2)Exp. Exc. updating weights, value1: "+save1+ ", value2: "+val2);
		}
		save2=printExcValue(weights.get()[i]);
		res3=addMPFR(weights.get()[i], weights.get()[i], tmp);
		if (res3!=0){
			string val2=printExcValue(tmp);
			throwExcExponent(res3, "(3)Exp. Exc. updating weights, value1: "+save2+ ", value2: "+val2);
		}
		mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);
	}
	mpfr_clear(tmp);
}

void updateWeightsAverage(unique_ptr<mpfr_t> &weights, vector<mpfr_t> &rowDataset,mpfr_t &label,mpfr_t &learningRate,unique_ptr<mpfr_t> &averageWeights,long int &c){
	mpfr_t tmp;
	cout<<c<<endl;
	mpfr_init2 (tmp, precComputation);
	mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);

	mpfr_t iterationAverage;
	mpfr_init2 (iterationAverage, precComputation);
	mpfr_set_si (iterationAverage, c, (mpfr_rnd_t)ROUND);

	printMPFRvar(iterationAverage);

	int res1=0;
	int res2=0;
	int res3=0;
	int res4=0;
	int res5=0;

	string save1;
	string save2;
	string save3;

	//printWeightDecimal(averageWeights,numElementsInRow);
	for (int i=0;i<numElementsInRow;i++){
		//printWeightDecimal(averageWeights,numElementsInRow);
		res1=mulMPFR(tmp,rowDataset[i],label);
		if (res1!=0){
			string val1=printExcValue(rowDataset[i]);
			string val2=printExcValue(label);
			throwExcExponent(res1, "(1 AP)Exp. Exc. updating weights, value1: "+val1+ ", value2: "+val2);
		}
		save1=printExcValue(tmp);
		res2=mulMPFR(tmp,tmp,learningRate);
		if (res2!=0){
			string val2=printExcValue(learningRate);
			throwExcExponent(res2, "(2 AP)Exp. Exc. updating weights, value1: "+save1+ ", value2: "+val2);
		}
		save2=printExcValue(weights.get()[i]);
		res3=addMPFR(weights.get()[i], weights.get()[i], tmp);
		if (res3!=0){
			string val2=printExcValue(tmp);
			throwExcExponent(res3, "(3 AP)Exp. Exc. updating weights, value1: "+save2+ ", value2: "+val2);
		}
		///////////////
		//average
		///////////////
		//printMPFRvar(tmp);
		//printMPFRvar(iterationAverage);
		res4=mulMPFR(tmp,iterationAverage,tmp);
		if (res4!=0){
			throwExcExponent(res3, "(4 AP)Exp. Exc. updating average weights, value1: "+to_string(c)+ ", value2: "+save1);
		}
		//printMPFRvar(tmp);
		//printWeightDecimal(averageWeights, numElementsInRow);
		save3=printExcValue(averageWeights.get()[i]);
		res5=addMPFR(averageWeights.get()[i], averageWeights.get()[i], tmp);
		//printWeightDecimal(averageWeights, numElementsInRow);
		if (res5!=0){
			string val2=printExcValue(tmp);
			throwExcExponent(res3, "(5 AP)Exp. Exc. updating average weights, value1: "+save3+ ", value2: "+val2);
		}
		////////////////
		////////////////
		mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);
	}
	mpfr_clear(tmp);
	mpfr_clear(iterationAverage);
}

double computeAccuracy(double mistakes, int Nrows){
	if (mistakes<0)
		return -1.0;
	double accuracySummary=(1-(mistakes/(float)Nrows))*100;
	return accuracySummary;
}


int perceptronAverage(unique_ptr<mpfr_t> &weights,vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label, mpfr_t &learningRate,unique_ptr<mpfr_t> &averageWeights,long int &c){
	int mistakes=0;
	mpfr_t dotProduct;
	mpfr_init2 (dotProduct, precComputation);
	int res1=0;
	for (int i=0;i<dataset.size();i++){
		mpfr_set_d (dotProduct, 0, (mpfr_rnd_t)ROUND);
		myDotProduct(weights,dataset[i],dotProduct);
		res1=mulMPFR(dotProduct, label[i], dotProduct);
		if (res1!=0){
			string val1=printExcValue(label[i]);
			string val2=printExcValue(dotProduct);
			throwExcExponent(res1, "(AP)Exp. Exc. for the dot product, value1: "+val1+", value2: "+ val2);
		}
		if (mpfr_cmp_d (dotProduct, 0.0)<=0){
			updateWeightsAverage(weights,dataset[i],label[i],learningRate,averageWeights,c);
			mistakes++;
		}
		c++;
	}
	mpfr_clear(dotProduct);
	return mistakes;
}

int perceptron(unique_ptr<mpfr_t> &weights,vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label, mpfr_t &learningRate){
	int mistakes=0;
	mpfr_t dotProduct;
	mpfr_init2 (dotProduct, precComputation);
	int res1=0;
	for (int i=0;i<dataset.size();i++){
		mpfr_set_d (dotProduct, 0, (mpfr_rnd_t)ROUND);
		myDotProduct(weights,dataset[i],dotProduct);
		res1=mulMPFR(dotProduct, label[i], dotProduct);
		//summWeights(averageWeights,weights);
		if (res1!=0){
			string val1=printExcValue(label[i]);
			string val2=printExcValue(dotProduct);
			throwExcExponent(res1, "Exp. Exc. for the dot product, value1: "+val1+", value2: "+ val2);
		}
		if (mpfr_cmp_d (dotProduct, 0.0)<=0){
			updateWeights(weights,dataset[i],label[i],learningRate);
			mistakes++;
		}
	}
	mpfr_clear(dotProduct);
	return mistakes;
}

double testPerceptronAndSVM(unique_ptr<mpfr_t> &weights,vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label){
	int mistakes=0;
	mpfr_t dotProduct;
	mpfr_init2 (dotProduct, precComputation);
	int res1=0;
	for (int i=0;i<dataset.size();i++){
		mpfr_set_d (dotProduct, 0, (mpfr_rnd_t)ROUND);
		myDotProduct(weights,dataset[i],dotProduct);
		res1=mulMPFR(dotProduct, label[i], dotProduct);
		if (res1!=0){
			throwExcExponent(res1, "Exponent exception while computing test: ");
		}
		if (mpfr_cmp_d (dotProduct, 0.0)<=0){
			mistakes++;
		}
	}
	mpfr_clear(dotProduct);
	return computeAccuracy(mistakes,dataset.size());
}

void swapLine(vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label){
	srand (time(NULL));
	vector<mpfr_t> temp;
	for (int i=0; i<dataset.size()/2;i++){
		int v1 = rand() % dataset.size();
		int v2 = rand() % dataset.size();
		swap(dataset[v1],dataset[v2]);//dataset.back(); //[v1];
		swap(label[v1],label[v2]);
	}
}

void clearDatasetLabel(vector<vector<mpfr_t>> &dataset,vector<mpfr_t> &label){
	for(int i = 0; i < label.size(); i++){
		mpfr_clear(label[i]);
	}
	for(int i = 0; i < dataset.size(); i++){
		for(int j = 0; j <numElementsInRow; j++){
			mpfr_clear(dataset[i][j]);
		}
		dataset[i].clear();
	}
	dataset.clear();
	label.clear();
}

void setExponentPrecision(int exponentPrecision){
	//int infLimit=(2-pow(2.0,exponentPrecision-1))-significandPrecision+1+1;//one +1 for subnormal definition, one +1 for simulate IEEE754
	//int supLimit=pow(2.0,exponentPrecision-1); // always+1 to simulate IEEE754
	int infLimit=-pow(2.0,exponentPrecision-1);
	int supLimit=pow(2.0,exponentPrecision-1);
	mpfr_set_emin (infLimit);
	mpfr_set_emax (supLimit);
}

vector<string>* readData(string nameFile){
	ifstream file(nameFile);
	string line;
	vector<string>* data=new vector<string>();
	while (getline(file, line)){
		if (!line.empty())
			data->push_back(line);
	}
	vector<string>* firstRow=split(data->front());
	numElementsInRow=firstRow->size()-1;
	numOfAttributes=numElementsInRow-2;
	file.close();
	firstRow->clear();
	delete firstRow;
	return data;
}

void builtDataset(vector<vector<mpfr_t>> &dataset, vector<string> &data){
	vector<string>* tmp;
	double min=INT_MAX;
	double max=INT_MIN;

	int res1=0;
	for (int i=0;i<dataset.size();i++){
		tmp=split(data[i]);
		if (tmp->size()-1!=numElementsInRow){
			throw out_of_range("Error in the dataset, number of elements in line differs from line 1: (line)"+to_string(i));
		}
		dataset[i]=vector<mpfr_t>(numElementsInRow);
		for(int j=0;j<numElementsInRow;j++){
			mpfr_init2 (dataset[i][j],precComputation);
			res1=setMPFR(dataset[i][j], ((*tmp)[j]), 10);
			if (res1!=0){
				throwExcExponent(res1, "Exponent exception while reading dataset: "+to_string(res1)+", Row:"+to_string(i)+", value:"+((*tmp)[j]).c_str());
			}
			string val=(*tmp)[j];
			if(atof(val.c_str())<min && j<=numOfAttributes){
				min=atof(val.c_str());
			}
			if(atof(val.c_str())>max && j<=numOfAttributes){
				max=atof(val.c_str());
			}
		}
		tmp->clear();
		delete tmp;
	}
	//cout<<"Range Dataset. Min: "+to_string(min)+" - Max: "+to_string(max)<<endl;
}

void builtLabel(vector<mpfr_t> &label, vector<string> &data){
	vector<string>* tmp;
	int res1=0;
	for (int i=0;i<label.size();i++){
		tmp=split(data[i]);
		mpfr_init2 (label[i],precComputation);
		res1=setMPFR(label[i], tmp->back(), 10);
		if (res1!=0){
			throwExcExponent(res1, "Exponent exception while building label dataset: ");
		}
		tmp->clear();
		delete tmp;
	}
}

unique_ptr<mpfr_t> initWeights(){
	unique_ptr<mpfr_t> weights(new mpfr_t[numElementsInRow]);
	for (int i=0; i<numElementsInRow;i++){
		mpfr_init2 (weights.get()[i],precComputation);
		mpfr_set_d (weights.get()[i], 0, (mpfr_rnd_t)ROUND);
	}
	return weights;
}

void printToFile(ofstream &f, int count, ...)
{
    va_list ap;
    int j;
    string separator="$";
    string ret="";
    va_start(ap, count); /* Requires the last fixed parameter (to get the address) */
    for (j = 0; j < count; j++) {
        ret=ret+separator+va_arg(ap, string); /* Increments ap to the next argument. */
    }
    ret=ret+"\n";
    va_end(ap);
	f<<ret;
	f.flush();
}

string trainPerceptron(int epochs, int precExponentTrainingPerceptron, unique_ptr<mpfr_t> &weights, vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label, mpfr_t &learningRate){
	try{
		setExponentPrecision(precExponentTrainingPerceptron);
		for (int epoch=0;epoch<epochs;epoch++){
			perceptron(weights, dataset, label, learningRate);
			swapLine(dataset, label);
		}
	}
	catch(out_of_range& e){
		return e.what();
	}
	return "";
}

string trainAveragePerceptron(int epochs, int precExponentTrainingAverage, unique_ptr<mpfr_t> &weights,unique_ptr<mpfr_t> &averageWeights, vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label, mpfr_t &learningRate){
	try{
		long int countSurviveEpochs=1;
		setExponentPrecision(precExponentTrainingAverage);
		for (int epoch=0;epoch<epochs;epoch++){
			perceptronAverage(weights, dataset, label, learningRate, averageWeights, countSurviveEpochs);
			swapLine(dataset, label);
		}
		averageWeightsAdjust(weights,averageWeights,countSurviveEpochs);
	}
	catch(out_of_range& e){
		return e.what();
	}
	return "";
}

void updateWeightsSVMMistakes(unique_ptr<mpfr_t> &weights, vector<mpfr_t> &rowDataset,mpfr_t &label,mpfr_t &updateLR,mpfr_t &regularizer){
	//weight=(1-updateLR)*weight+updateLR*C*labelTrain[index]*x
	mpfr_t tmp;
	mpfr_init2 (tmp, precComputation);
	mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);

	int res1=0;

	string save1;

	mpfr_t one;
	mpfr_init2 (one, precComputation);
	res1=setMPFR(one, "1.0", 10);
	if (res1!=0){
		throwExcExponent(res1, "(SVM-Update-Wrong)Exp. Exc. for represent '1'");
	}

	for (int i=0;i<numElementsInRow;i++){

		res1=mulMPFR(tmp,rowDataset[i],label);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.1) Exp. Exc. updating weights, value1: "+printExcValue(rowDataset[i])+ ", value2: "+printExcValue(label));
		}

		save1=printExcValue(tmp);
		res1=mulMPFR(tmp,tmp,regularizer);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.2) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(regularizer));
		}

		save1=printExcValue(tmp);
		res1=mulMPFR(tmp, tmp,updateLR);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.3) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(updateLR));
		}

		save1=printExcValue(one);
		res1=subMPFR(one, one, updateLR);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.4) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(updateLR));
		}

		save1=printExcValue(one);
		res1=mulMPFR(one, one,weights.get()[i]);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.5) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(weights.get()[i]));
		}

		save1=printExcValue(tmp);
		res1=addMPFR(tmp, tmp,one);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.6) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(one));
		}

		res1=setMPFR(weights.get()[i],tmp);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.7) Exp. Exc. updating weights, value1: "+printExcValue(tmp));
		}

		res1=setMPFR(one, "1.0", 10);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Wrong.8) Exp. Exc. for reset '1'");
		}

		mpfr_set_d (tmp, 0, (mpfr_rnd_t)ROUND);
	}
	mpfr_clear(tmp);
}

void updateWeightsSVMCorrect(unique_ptr<mpfr_t> &weights,mpfr_t &updateLR){
	int res1=0;

	string save1;
	mpfr_t one;
	mpfr_init2 (one, precComputation);

	res1=setMPFR(one, "1.0", 10);
	if (res1!=0){
		throwExcExponent(res1, "(SVM-Update-Correct) Exp. Exc. for represent '1'");
	}

	for (int i=0;i<numElementsInRow;i++){

		save1=printExcValue(one);
		res1=subMPFR(one, one, updateLR);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Corr.1) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(updateLR));
		}

		save1=printExcValue(one);
		res1=mulMPFR(one, one,weights.get()[i]);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Corr.2) Exp. Exc. updating weights, value1: "+save1+ ", value2: "+printExcValue(weights.get()[i]));
		}

		res1=setMPFR(weights.get()[i],one);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Corr.3) Exp. Exc. updating weights, value1: "+printExcValue(one));
		}

		res1=setMPFR(one, "1.0", 10);
		if (res1!=0){
			throwExcExponent(res1, "(SVM-Update-Corr.4) Exp. Exc. for reset '1'");
		}
	}
}


int SVM(unique_ptr<mpfr_t> &weights,vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label, mpfr_t &learningRate, mpfr_t &regularizer){
	int mistakes=0;
	mpfr_t dotProduct;
	mpfr_init2 (dotProduct, precComputation);

	mpfr_t updateLR;
	mpfr_init2 (updateLR, precComputation);

	mpfr_t index;
	mpfr_init2 (index, precComputation);

	int res1=0;

	mpfr_t one;
	mpfr_init2(one,precComputation);
	res1=setMPFR(one, "1.0", 10);
	if (res1!=0){
		throwExcExponent(res1, "(SVM )Exp. Exc. for represent '1'");
	}

	for (int i=0;i<dataset.size();i++){

		res1=setMPFR(index, to_string(i), 10);
		if (res1!=0){
			throwExcExponent(res1, "(SVM )Exp. Exc. for the index, value: "+to_string(i));
		}

		res1=divMPFR(updateLR, index, regularizer);
		if (res1!=0){
			throwExcExponent(res1, "(SVM )Exp. Exc. for the index division, value1: "+printExcValue(index)+", value2:"+printExcValue(regularizer));
		}

		string save=printExcValue(updateLR);
		res1=mulMPFR(updateLR, updateLR, learningRate);
		if (res1!=0){
			throwExcExponent(res1, "(SVM )Exp. Exc. for the index multiplication, value1: "+save+", value2:"+printExcValue(learningRate));
		}

		save=printExcValue(updateLR);
		res1=addMPFR(updateLR,updateLR,one);
		if (res1!=0){
			throwExcExponent(res1, "(SVM )Exp. Exc. for summation with '1' multiplication, value1: "+save+", value2: "+printExcValue(one));
		}

		save=printExcValue(updateLR);
		res1=divMPFR(updateLR, learningRate, updateLR);
		if (res1!=0){
			throwExcExponent(res1, "(SVM )Exp. Exc. for division with learning Rate, value1: "+save+", value2: "+printExcValue(learningRate));
		}

		mpfr_set_d (dotProduct, 0, (mpfr_rnd_t)ROUND);
		myDotProduct(weights,dataset[i],dotProduct);

		string val1=printExcValue(dotProduct);
		res1=mulMPFR(dotProduct, label[i], dotProduct);

		if (res1!=0){
			throwExcExponent(res1, "(SVM)Exp. Exc. for the dot product, value1: "+printExcValue(label[i])+", value2: "+ val1);
		}

		if (mpfr_cmp_d (dotProduct, 1.0)<=0){
			updateWeightsSVMMistakes(weights,dataset[i],label[i],learningRate,regularizer);
			mistakes++;
		}
		else{
			updateWeightsSVMCorrect(weights, updateLR);
		}
	}
	mpfr_clear(dotProduct);
	return mistakes;
}

string trainSVM(int epochs, int precExponentTrainingSVM, unique_ptr<mpfr_t> &weights, vector<vector<mpfr_t>> &dataset, vector<mpfr_t> &label, mpfr_t &learningRate, mpfr_t &regularizer){
	try{
		setExponentPrecision(precExponentTrainingSVM);
		for(int epoch=0;epoch<epochs;epoch++){
			SVM(weights, dataset, label, learningRate, regularizer);
			swapLine(dataset, label);
		}
	}
	catch(out_of_range& e){
		return e.what();
	}
	return "";
}

//va parallelizzato
string cloneWeights(unique_ptr<mpfr_t> &destination,unique_ptr<mpfr_t> &source){
	int res1=0;
	for (int i=0; i<numElementsInRow;i++){
		res1=setMPFR(destination.get()[i], source.get()[i]);
		if (res1!=0){
			return "Exp. Exc.for cloning test weights";
		}
	}
	return "";
}


int main() {
	mpfr_clear_flags();
	vector<string> trainingData;

	string fileName="/home/roki/workspace/PerceptronMPFR/fourclass/fourclassS/fourclassProc";

	ofstream testPerceptron;
	ofstream testAveragePerceptron;
	ofstream testSVM;

	ofstream trainingPerceptron;
	ofstream trainingAveragePerceptron;
	ofstream trainingSVM;

	int mistakes;
	int mistakeCheck;
	int epochs;
	mpfr_t learningRate;

	unique_ptr<mpfr_t> weights;
	unique_ptr<mpfr_t> weightsForAverage;
	unique_ptr<mpfr_t> averageWeights;
	unique_ptr<mpfr_t> weightsSVM;

	vector<string> testData;
	vector<string>* tmp;
	vector<vector<mpfr_t>> dataset;
	vector<mpfr_t> label;
	vector<vector<mpfr_t>> datasetTest;
	vector<mpfr_t> labelTest;

	string separator="$";
	string settingsData="";
	string settingsComputationTraining="";
	string settingsComputationTest="";
	string settingsComputationPerceptronTraining="";
	string settingsComputationAverageTraining="";
	string settingsComputationSVM="";
	string exc="exc";
	string numericExc="-1";

	vector<string> stringLearningRate;
	vector<string> stringRegularizer;

	int precExponentData;
	int precExponentTrainingSVM;
	int precExponentTrainingPerceptron;
	int precExponentTrainingAverage;

	int precExponentTestSVM;
	int precExponentTestPerceptron;
	int precExponentTestAverage;

	int upperLimitMantissa;
	int lowerBoundMantissa;

	int startExponent=5;

	precExponentData=startExponent;
	precExponentTrainingSVM=startExponent;
	precExponentTrainingPerceptron=startExponent;
	precExponentTrainingAverage=startExponent;
	precExponentTestPerceptron=startExponent;
	precExponentTestAverage=startExponent;
	precExponentTestSVM=startExponent;


	upperLimitMantissa=10;
	lowerBoundMantissa=2;

	epochs=10;

	ROUND=MPFR_RNDZ;

	for (int i=1;i<5;i++){
		trainingPerceptron.open (fileName+"P-Train"+to_string(i)+".txt");
		trainingAveragePerceptron.open (fileName+"AP-Train"+to_string(i)+".txt");
		testPerceptron.open (fileName+"P-Test"+to_string(i)+".txt");
		testAveragePerceptron.open(fileName+"AP-Test"+to_string(i)+".txt");
		trainingSVM.open(fileName+"SVM-Train"+to_string(i)+".txt");
		testSVM.open(fileName+"SVM-Test"+to_string(i)+".txt");

		trainingData={};
		for (int j=1;j<5;j++){
			if (i!=j){
				cout<<fileName+to_string(j)+".txt"<<endl;
				vector<string>* tmp=readData(fileName+to_string(j)+".txt");
				for (int val=0;val<tmp->size();val++){
					trainingData.push_back((*tmp)[val]);
				}
				delete tmp;
			}
		}
		int precDataset=lowerBoundMantissa;
		while (precDataset<=upperLimitMantissa){
			setExponentPrecision(precExponentData);
			precComputation=precDataset;
			try{
				//START - DATASET 4 TRAIN
				settingsComputationTraining="(N.R,N.R)";
				settingsComputationTest="(N.R,N.R)";
				settingsData="("+to_string(precDataset)+","+to_string(precExponentData) +")";
				int numOfRows=trainingData.size();

				dataset=vector<vector<mpfr_t>>(numOfRows);
				label=vector<mpfr_t>(numOfRows);
				builtDataset(dataset,trainingData);
				builtLabel(label,trainingData);
				//END - DATASET 4 TEST

				//START - DATASET 4 TEST
				testData={};
				tmp=readData(fileName+to_string(i)+".txt");
				for (int val=0;val<tmp->size();val++){
					testData.push_back((*tmp)[val]);
				}
				numOfRows=testData.size();

				datasetTest=vector<vector<mpfr_t>>(numOfRows);
				labelTest=vector<mpfr_t>(numOfRows);

				builtDataset(datasetTest,testData);
				builtLabel(labelTest,testData);
				//END - DATASET 4 TEST

				int precisionTraining=lowerBoundMantissa;
				while (precisionTraining<=upperLimitMantissa){
					precComputation=precisionTraining;
					int l=0;

					vector<string> stringLearningRate={"0.001"};
					while(l<stringLearningRate.size()){
						//START-PERCEPTRON TRAINING
						setExponentPrecision(precExponentTrainingPerceptron);
						weights=initWeights();
						settingsComputationPerceptronTraining="("+to_string(precComputation)+","+to_string(precExponentTrainingPerceptron)+")";
						mpfr_init2 (learningRate, precComputation);
						int res1=setMPFR(learningRate, stringLearningRate[l], 10);
						if (res1!=0){
							printToFile(trainingPerceptron, 7,settingsData,settingsComputationPerceptronTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"ACC:"+exc,string("Exp. Exc.for learning rate: "+stringLearningRate[l]));
							precExponentTrainingPerceptron++;
							continue;
						}
						string message=trainPerceptron(epochs, precExponentTrainingPerceptron, weights, dataset, label, learningRate);
						if (message.size()!=0){
							printToFile(trainingPerceptron, 7,settingsData,settingsComputationPerceptronTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"P:"+exc,message);
							precExponentTrainingPerceptron++;
							continue;
						}
						//END-PERCEPTRON TRAINING
						//START-PERCEPTRON TEST
						double trainingAccuracy=0;
						setExponentPrecision(precExponentTrainingPerceptron);
						trainingAccuracy=testPerceptronAndSVM(weights, dataset, label);
						printToFile(trainingPerceptron, 7,settingsData,settingsComputationPerceptronTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"ACC:"+to_string(trainingAccuracy),string("N/A"));
						//END-PERCEPTRON TEST

						int precisionTest=lowerBoundMantissa;

						while (precisionTest<=upperLimitMantissa){
							precComputation=precisionTest;

							settingsComputationTest="("+to_string(precComputation)+","+to_string(precExponentTestPerceptron)+")";
							setExponentPrecision(precExponentTestPerceptron);

							unique_ptr<mpfr_t> backupWeights=initWeights();
							string res1=cloneWeights(backupWeights,weights);
							if (res1.size()!=0){
								printToFile(testPerceptron, 7,settingsData,settingsComputationPerceptronTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"P:"+exc,res1);
								precExponentTestPerceptron++;
								continue;
							}

							float accuracy=-1;
							try{
								accuracy=testPerceptronAndSVM(backupWeights, datasetTest, labelTest);
								printToFile(testPerceptron, 7,settingsData,settingsComputationPerceptronTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"P:"+to_string(accuracy),string("N/A"));
							}
							catch(out_of_range& e){
								printToFile(testPerceptron, 7,settingsData,settingsComputationPerceptronTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"P:"+exc,string(e.what()));
								precExponentTestPerceptron++;
								continue;
							}
							precisionTest++;
						}
						settingsComputationTest="(N.R,N.R)";
						l++;
					}

					precComputation=precisionTraining;
					l=0;
					stringLearningRate={"0.001"};
					while(l<stringLearningRate.size()){

						//START-AVERAGE PERCEPTRON
						setExponentPrecision(precExponentTrainingAverage);
						weightsForAverage=initWeights();
						averageWeights=initWeights();
						settingsComputationAverageTraining="("+to_string(precComputation)+","+to_string(precExponentTrainingAverage)+")";
						mpfr_init2 (learningRate, precComputation);
						int res1=setMPFR(learningRate, stringLearningRate[l], 10);
						if (res1!=0){
							printToFile(trainingAveragePerceptron, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"ACC:"+exc,string("Exp. Exc.for learning rate: "+stringLearningRate[l]));
							precExponentTrainingAverage++;
							continue;
						}
						string message="";
						message=trainAveragePerceptron(epochs, precExponentTrainingAverage, weightsForAverage, averageWeights, dataset, label, learningRate);
						if (message.size()!=0){
							printToFile(trainingAveragePerceptron, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"AP:"+exc,string(message));
							precExponentTrainingAverage++;
							continue;
						}
						//END-AVERAGE PERCETRON
						//START-PERCEPTRON AVERAGE TEST
						int trainingAccuracy=0;
						setExponentPrecision(precExponentTrainingAverage);
						trainingAccuracy=testPerceptronAndSVM(averageWeights, dataset, label);
						printToFile(trainingAveragePerceptron, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"ACC:"+to_string(trainingAccuracy),string("N/A"));

						//END-PERCEPTRON AVERAGE TEST
						int precisionTest=lowerBoundMantissa;
						while (precisionTest<=upperLimitMantissa){
							precComputation=precisionTest;

							settingsComputationTest="("+to_string(precComputation)+","+to_string(precExponentTestAverage)+")";
							setExponentPrecision(precExponentTestAverage);

							unique_ptr<mpfr_t> backupAverageWeights=initWeights();
							string res1=cloneWeights(backupAverageWeights, averageWeights);
							if (res1.size()!=0){
								printToFile(testAveragePerceptron, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"AP:"+exc,res1);
								precExponentTestAverage++;
								continue;
							}

							float accuracy=-1;
							try{
								accuracy=testPerceptronAndSVM(backupAverageWeights, datasetTest, labelTest);
								printToFile(testAveragePerceptron, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"AP:"+to_string(accuracy),string("N/A"));
							}
							catch(out_of_range& e){
								printToFile(testAveragePerceptron, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"AP:"+exc,string(e.what()));
								precExponentTestAverage++;
								continue;
							}
							precisionTest++;
						}
						settingsComputationTest="(N.R,N.R)";
						l++;
					}

					precComputation=precisionTraining;
					l=0;
					stringLearningRate={"0.001"};
					while(l<stringLearningRate.size()){
						//START-SVM TRAINING
						int c=0;
						stringRegularizer={"0.1"};
						while (c<stringRegularizer.size()){
							weightsSVM=initWeights();
							settingsComputationSVM="("+to_string(precComputation)+","+to_string(precExponentTrainingSVM)+")";
							setExponentPrecision(precExponentTrainingSVM);
							mpfr_t regularizer;
							mpfr_init2 (regularizer, precComputation);
							int res1=setMPFR(regularizer, stringRegularizer[c], 10);
							if (res1!=0){
								printToFile(trainingSVM, 7,settingsData,settingsComputationSVM,settingsComputationTest,"LR:"+stringLearningRate[l]+"-C:"+stringRegularizer[c],to_string(label.size()),"ACC:"+exc,string("Exp. Exc.for regularizer: "+stringRegularizer[c]));
								precExponentTrainingSVM++;
								continue;
							}
							string message="";
							message=trainSVM(epochs, precExponentTrainingSVM, weightsSVM, dataset, label, learningRate, regularizer);
							if (message.size()!=0){
								printToFile(trainingSVM, 7,settingsData,settingsComputationSVM,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"AP:"+exc,string(message));
								precExponentTrainingSVM++;
								continue;
							}


							//END-SVM TRAINING
							//START-SVM TEST
							float trainingAccuracy=0;
							setExponentPrecision(precExponentTrainingSVM);
							trainingAccuracy=testPerceptronAndSVM(weightsSVM, dataset, label);
							printToFile(trainingSVM, 7,settingsData,settingsComputationAverageTraining,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(label.size()),"ACC:"+to_string(trainingAccuracy),string("N/A"));
							//END-SVM TEST

							int precisionTest=lowerBoundMantissa;
							while (precisionTest<=upperLimitMantissa){
								precComputation=precisionTest;

								settingsComputationTest="("+to_string(precComputation)+","+to_string(precExponentTestSVM)+")";
								setExponentPrecision(precExponentTestSVM);

								unique_ptr<mpfr_t> backupWeightsSVM=initWeights();
								string res1=cloneWeights(backupWeightsSVM, weightsSVM);
								if (res1.size()!=0){
									printToFile(testSVM, 7,settingsData,settingsComputationSVM,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"SVM:"+exc,res1);
									precExponentTestSVM++;
									continue;
								}

								float accuracy=-1;
								try{
									accuracy=testPerceptronAndSVM(backupWeightsSVM, datasetTest, labelTest);
									printToFile(testSVM, 7,settingsData,settingsComputationSVM,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"SVM:"+to_string(accuracy),string("N/A"));
								}
								catch(out_of_range& e){
									printToFile(testSVM, 7,settingsData,settingsComputationSVM,settingsComputationTest,"LR:"+stringLearningRate[l],to_string(labelTest.size()),"SVM:"+exc,string(e.what()));
									precExponentTestSVM++;
									continue;
								}
								precisionTest++;
							}
							c++;
						}
						settingsComputationTest="(N.R,N.R)";
						l++;
					}
					precisionTraining++;
				}
				delete tmp;
				mpfr_clear(learningRate);

				clearDatasetLabel(datasetTest,labelTest);

				clearDatasetLabel(dataset, label);
				precDataset++;
			}
			catch(out_of_range& e){
				printToFile(testPerceptron, 7,settingsData,settingsComputationTraining,settingsComputationTest,"LR:"+exc,numericExc,"P:"+exc,string(e.what()));
				printToFile(trainingPerceptron, 7,settingsData,settingsComputationTraining,settingsComputationTest,"LR:"+exc,numericExc,"P:"+exc,string(e.what()));
				printToFile(testAveragePerceptron, 7,settingsData,settingsComputationTraining,settingsComputationTest,"LR:"+exc,numericExc,"AP:"+exc,string(e.what()));
				printToFile(trainingAveragePerceptron, 7,settingsData,settingsComputationTraining,settingsComputationTest,"LR:"+exc,numericExc,"AP:"+exc,string(e.what()));
				precExponentData=precExponentData+1;
			}
		}
		testPerceptron.close();
		testAveragePerceptron.close();
		trainingPerceptron.close();
		trainingAveragePerceptron.close();
		testSVM.close();
		trainingSVM.close();
	}
}

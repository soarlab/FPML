#include "tools.h"

void printMPFRvar2(mpfr_t var){
	printf("Decimal:\n");
	int tmpD=mpfr_out_str (stdout, 10, 0, var, (mpfr_rnd_t)MPFR_RNDZ);
	printf("\nBinary \n");
	int tmpB=mpfr_out_str (stdout, 2, 0, var, (mpfr_rnd_t)MPFR_RNDZ);
	printf("\n");
	cout <<"Value Of Errno:" <<errno<<endl;
	fflush(stdout);
}

void setdemoExponentPrecision(int exponentPrecision, int significandPrecision){
	int infLimit=(2-(int)pow(2.0,exponentPrecision-1))-significandPrecision+1+1;//one +1 for subnormal definition, one +1 for simulate IEEE754
	int supLimit=pow(2.0,exponentPrecision-1); // always+1 to simulate IEEE754
	mpfr_set_emin (infLimit);
	mpfr_set_emax (supLimit);
}

/*
 *
//typedef enum {
//MPFR_RNDN=0,  /* round to nearest, with ties to even */
// * MPFR_RNDZ,    /* round toward zero */
// * MPFR_RNDU,    /* round toward +Inf */
// * MPFR_RNDD,    /* round toward -Inf */
// * MPFR_RNDA,    /* round away from zero */
// * MPFR_RNDF,    /* faithful rounding (not implemented yet) */
// * MPFR_RNDNA=-1 /* round to nearest, with ties away from zero (mpfr_round) */
//}
// */

/*void roundModeAffect(){
	//(7, 0) (7, 0) 0.1 Exp. Exc. dot product (MUL), value1: Decimal: 0.9765E-3, Binary: 0.1000000E-9, value2: Decimal: 0.5798E-2, Binary: 0.1011111E-7, Exception:UNDERFLOW, Exp.Range:-16, 16

	setExponentPrecision(4);

	string s="-0.418205E-1";
	string t="0.418167E-1";
	int n;
	mpfr_t tmpS;
	mpfr_t tmpT;
	mpfr_t res;
	//0.10101011010010E-4
	//-0.10101011010011E-4

	mpfr_init2 (res, 11);
	n=setMPFR(res, "0", 10);
	mpfr_init2 (tmpS, 11);
	n=setMPFR(tmpS, s, 10);
	mpfr_init2 (tmpT, 11);
	n=setMPFR(tmpT, t, 10);
	cout<<*printExcValue(tmpS)<<endl;
	cout<<*printExcValue(tmpT)<<endl;
	n=addMPFR(res, tmpS, tmpT);
	cout<<*printExcValue(res)<<endl;
	cout<<n<<endl;

	mpfr_init2 (res, 12);
	n=setMPFR(res, "0", 10);
	mpfr_init2 (tmpS, 12);
	n=setMPFR(tmpS, s, 10);
	mpfr_init2 (tmpT, 12);
	n=setMPFR(tmpT, t, 10);
	cout<<*printExcValue(tmpS)<<endl;
	cout<<*printExcValue(tmpT)<<endl;
	n=addMPFR(res, tmpS, tmpT);
	cout<<*printExcValue(res)<<endl;
	cout<<n<<endl;

	mpfr_init2 (res, 13);
	n=setMPFR(res, "0", 10);
	mpfr_init2 (tmpS, 13);
	n=setMPFR(tmpS, s, 10);
	mpfr_init2 (tmpT, 13);
	n=setMPFR(tmpT, t, 10);
	cout<<*printExcValue(tmpS)<<endl;
	cout<<*printExcValue(tmpT)<<endl;
	n=addMPFR(res, tmpS, tmpT);
	cout<<*printExcValue(res)<<endl;
	cout<<n<<endl;

	mpfr_init2 (res, 14);
	n=setMPFR(res, "0", 10);
	mpfr_init2 (tmpS, 14);
	n=setMPFR(tmpS, s, 10);
	mpfr_init2 (tmpT, 14);
	n=setMPFR(tmpT, t, 10);
	cout<<*printExcValue(tmpS)<<endl;
	cout<<*printExcValue(tmpT)<<endl;
	n=addMPFR(res, tmpS, tmpT);
	cout<<*printExcValue(res)<<endl;
	cout<<n<<endl;

}*/

/*
 * tools.h
 *
 *  Created on: Oct 17, 2017
 *      Author: roki
 */

#include <stdio.h>
#include <iostream>
#include <mpfr.h>
#include <string>
#include <string.h>
#include <fstream>
#include <array>
#include <cstdlib>
#include <gmp.h>
#include <iterator>
#include <assert.h>
#include <limits.h>
#include <unistd.h>
#include <vector>
#include <list>
#include <algorithm>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <stdexcept>
#include <map>
#include <cerrno>
#include <cmath>
#include <stdarg.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

void printMPFRvar(mpfr_t &var);
void setExponentPrecision(int n);
int setMPFR(mpfr_t &dataset,string tmp,int base);
int mulMPFR(mpfr_t &result, mpfr_t &first, mpfr_t &second);
string printExcValue(mpfr_t m);
int addMPFR(mpfr_t &result,mpfr_t &first,mpfr_t &second);
int checkOverflow();

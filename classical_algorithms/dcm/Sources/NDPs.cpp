/* 
 * File:   NDPs.cpp
 * Author: c3156840
 * 
 * Created on 16 September 2014, 11:15 AM
 */

#include "NDPs.h"
#include "extern_parameters.h"

NDPs::NDPs() {
    Obj_Value = new double [N_Objectives];
    Variables_Values = new double [N_Variables];
    f_x_Value = Null;
}

NDPs::NDPs(const NDPs& orig) {
}

NDPs::~NDPs() {
    delete [] Obj_Value;
    delete [] Variables_Values;
}


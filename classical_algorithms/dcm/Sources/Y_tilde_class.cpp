/* 
 * File:   Y_tilde_class.cpp
 * Author: c3156840
 * 
 * Created on 15 September 2014, 5:13 PM
 */
#include "extern_parameters.h"
#include "Y_tilde_class.h"

Y_tilde_class::Y_tilde_class() {
    Number_NonInfty_Components = 0;
    point_y = new double[N_Objectives];
    for (int i = 0; i < N_Objectives; i++) {
        point_y[i] = Negative_Infty;
    }
}

void Y_tilde_class::Compute_Number_NonInfty_Components() {
    for (int i = 0; i < N_Objectives; i++) {
        if (point_y[i] > Negative_Infty + Hybrid_eps) {
            Number_NonInfty_Components++;
        }
    }
}

Y_tilde_class::~Y_tilde_class() {
    delete [] point_y;
}


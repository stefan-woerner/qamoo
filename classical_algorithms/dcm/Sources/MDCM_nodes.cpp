/* 
 * File:   MDCM_nodes.cpp
 * Author: c3156840
 * 
 * Created on 5 October 2014, 6:59 PM
 */

#include "MDCM_nodes.h"
#include "extern_parameters.h"

MDCM_nodes::MDCM_nodes() {
    y_point_is_Null=1;
    y_point = new double[N_Objectives];
    y_L_point = new double[N_Objectives];
    Local_x = new double [N_Variables];
    Local_z_i_x = new double[N_Objectives];
    for (int i = 0; i < N_Objectives; i++) {
        y_point[i] = Null;
        y_L_point[i] = Negative_Infty;
    }
}

MDCM_nodes::MDCM_nodes(const MDCM_nodes& orig) {
}

MDCM_nodes::~MDCM_nodes() {
    delete [] y_point;
    delete [] y_L_point;
    delete [] Local_x;
    delete [] Local_z_i_x;
    while (Y_tilde_list.size()) {
        delete Y_tilde_list.front();
        Y_tilde_list.erase(Y_tilde_list.begin());
    }
}


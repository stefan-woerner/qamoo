/* 
 * File:   DCM_nodes.cpp
 * Author: c3156840
 * 
 * Created on 15 September 2014, 5:08 PM
 */

#include "DCM_nodes.h"
#include "extern_parameters.h"
#include "Y_tilde_class.h"

DCM_nodes::DCM_nodes() {
    Number_NonInfty_Components_of_y_point = 0;
    Sum_BOX_LB_Components = 0;
    Sum_BOX_UB_Components = 0;
    y_point = new double[N_Objectives];
    y_L_point = new double[N_Objectives];
    Box_UB_Point = new double[N_Objectives];
    for (int i = 0; i < N_Objectives; i++) {
        y_point[i] = Null;
        y_L_point[i] = Negative_Infty;
        Box_UB_Point[i] = Max_z_i_x[i];
    }
}

void DCM_nodes::Compute_Number_NonInfty_Components_of_y_point() {
    if (y_point[0] > Null + Hybrid_eps) {
        for (int i = 0; i < N_Objectives; i++) {
            if (y_point[i] > Negative_Infty + Hybrid_eps) {
                Number_NonInfty_Components_of_y_point++;
            }
        }
    }
}

void DCM_nodes::Compute_Summations_of_BOX_LB_UB_Components() {
    for (int i = 0; i < N_Objectives; i++) {
        Sum_BOX_LB_Components += y_L_point[i];
        Sum_BOX_UB_Components += Box_UB_Point[i];
    }
}

void DCM_nodes::Computing_BOX_UB_Point() {
    Compute_Number_NonInfty_Components_of_y_point();
    for (int i = 0; i < N_Objectives; i++) {
        for (int j = 0; j < Y_tilde_list.size(); j++) {
            if (Y_tilde_list.at(j)->Number_NonInfty_Components == 1 && Y_tilde_list.at(j)->point_y[i] > Negative_Infty + Hybrid_eps) {
                Box_UB_Point[i] = fmin(Box_UB_Point[i], Y_tilde_list.at(j)->point_y[i]);
            }
        }
    }
    if (Number_NonInfty_Components_of_y_point == 1) {
        for (int i = 0; i < N_Objectives; i++) {
            if (y_point[i] > Negative_Infty + Hybrid_eps) {
                Box_UB_Point[i] = fmin(Box_UB_Point[i], y_point[i]);
            }
        }
    }
    Compute_Summations_of_BOX_LB_UB_Components();
}

DCM_nodes::~DCM_nodes() {
    delete [] y_point;
    delete [] y_L_point;
    delete [] Box_UB_Point;
    while (Y_tilde_list.size()) {
        delete Y_tilde_list.front();
        Y_tilde_list.erase(Y_tilde_list.begin());
    }
}


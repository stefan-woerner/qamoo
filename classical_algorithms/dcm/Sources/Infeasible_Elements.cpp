/* 
 * File:   Infeasible_Elements.cpp
 * Author: c3156840
 * 
 * Created on 3 October 2014, 4:00 PM
 */

#include "Infeasible_Elements.h"
#include "extern_parameters.h"

Infeasible_Elements::Infeasible_Elements() {
    Box_UB_Point = new double [N_Objectives];
}

Infeasible_Elements::Infeasible_Elements(const Infeasible_Elements& orig) {
}

Infeasible_Elements::~Infeasible_Elements() {
    delete [] Box_UB_Point;
    while (Y_tilde_list.size()) {
        delete Y_tilde_list.front();
        Y_tilde_list.erase(Y_tilde_list.begin());
    }
}


/* 
 * File:   DCM_nodes.h
 * Author: c3156840
 *
 * Created on 15 September 2014, 5:08 PM
 */
#include "Y_tilde_class.h"
#include <vector>

#ifndef DCM_NODES_H
#define	DCM_NODES_H

class DCM_nodes {
public:
    std::vector<Y_tilde_class*> Y_tilde_list;
    double* y_point;
    double* y_L_point;
    double* Box_UB_Point;
    int Number_NonInfty_Components_of_y_point;
    double Sum_BOX_UB_Components;
    double Sum_BOX_LB_Components;
    DCM_nodes();
    void Computing_BOX_UB_Point();
    virtual ~DCM_nodes();
private:
void Compute_Number_NonInfty_Components_of_y_point();
void Compute_Summations_of_BOX_LB_UB_Components();
};

#endif	/* DCM_NODES_H */


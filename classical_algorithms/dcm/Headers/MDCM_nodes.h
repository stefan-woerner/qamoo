/* 
 * File:   MDCM_nodes.h
 * Author: c3156840
 *
 * Created on 5 October 2014, 6:59 PM
 */
#include "Y_tilde_class.h"
#include <vector>

#ifndef MDCM_NODES_H
#define	MDCM_NODES_H

class MDCM_nodes {
public:
    std::vector<Y_tilde_class*> Y_tilde_list;
    double* y_point;
    bool y_point_is_Null;
    double* y_L_point;
    double* Local_x;
    double* Local_z_i_x;
    double Local_f_x;
    MDCM_nodes();
    MDCM_nodes(const MDCM_nodes& orig);
    virtual ~MDCM_nodes();
private:

};

#endif	/* MDCM_NODES_H */


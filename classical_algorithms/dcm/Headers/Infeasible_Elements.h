/* 
 * File:   Infeasible_Elements.h
 * Author: c3156840
 *
 * Created on 3 October 2014, 4:00 PM
 */
#include "Y_tilde_class.h"
#include <vector>

#ifndef INFEASIBLE_ELEMENTS_H
#define	INFEASIBLE_ELEMENTS_H

class Infeasible_Elements {
public:
    std::vector<Y_tilde_class*> Y_tilde_list;
    double* Box_UB_Point;
    double Sum_BOX_UB_Components;
    Infeasible_Elements();
    Infeasible_Elements(const Infeasible_Elements& orig);
    virtual ~Infeasible_Elements();
private:

};

#endif	/* INFEASIBLE_ELEMENTS_H */


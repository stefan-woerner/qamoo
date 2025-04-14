/* 
 * File:   Y_tilde_class.h
 * Author: c3156840
 *
 * Created on 15 September 2014, 5:13 PM
 */

#ifndef Y_TILDE_CLASS_H
#define	Y_TILDE_CLASS_H

class Y_tilde_class {
public:
    double* point_y;
    int Number_NonInfty_Components;
    Y_tilde_class();
    void Compute_Number_NonInfty_Components();
    virtual ~Y_tilde_class();
private:

};

#endif	/* Y_TILDE_CLASS_H */


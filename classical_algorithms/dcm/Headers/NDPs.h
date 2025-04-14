/* 
 * File:   NDPs.h
 * Author: c3156840
 *
 * Created on 16 September 2014, 11:15 AM
 */

#ifndef NDPS_H
#define	NDPS_H

class NDPs {
public:
    double* Obj_Value;
    double* Variables_Values;
    double f_x_Value;
    NDPs();
    NDPs(const NDPs& orig);
    virtual ~NDPs();
private:

};

#endif	/* NDPS_H */


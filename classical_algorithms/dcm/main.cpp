/* 
 * File:   main.cpp
 * Author: c3156840
 *
 * Created on 23 July 2013, 11:28 AM
 */

#include <cstdlib>
//all global variables are declared in parameters.h 
#include"parameters.h"
//all parameters can be modified in extern_parameters.h 
#include "extern_parameters.h"
//all CPLEX and global variables are defined (allocating memory) precisely Resetting_functions.h 
#include"Resetting_functions.h"
// The Disjunctive Constraint Algorithm functions
#include"DCM_Solver.h"
// The Modified Disjunctive Constraint Algorithm functions


using namespace std;

int main(int argc, char *argv[]) {
    LP_file_name = argv[1];
    N_Objectives = atoi(argv[2]);
    char *output = argv[3];
    eps_modifier = Abs_eps_modifier;
    generating_all_variables();
    DCM_Optimizer(output);

    return 0;
}


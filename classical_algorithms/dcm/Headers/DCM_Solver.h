/* 
 * File:   DCM_Solver.h
 * Author: c3156840
 *
 * Created on 15 September 2014, 5:04 PM
 */
#include "extern_parameters.h"
#ifndef DCM_SOLVER_H
#define	DCM_SOLVER_H
void Initialize_queue();
double Finding_epsilon_value(double value_copy);
void Search_For_Feasible_Solution();
bool FirstArg_is_Smaller_than_SecondArg(double* Point1, double* Point2);
void Searching_Among_Infeasible_Cases();
void Add_Sort_The_NDPs_list(double* Nondominated_Val, double* Efficient_Val);
void Add_Infeasible_Set_To_The_List();
void Adding_Disjunctive_Constraints();
void Generate_Binary_Variables();
void Find_DCM_NDP();
void Update_Prority_Queue(DCM_nodes* Element_Copy);
void Remove_Dominated_Points_of_Tilde_Y(DCM_nodes* Element_Copy, double* Ref_Point);
void Add_The_Internal_Element_To_The_Queue();
void Add_The_External_Element_To_The_Queue();
void DCM_Optimizer(char* output);
void Write_Results(char* output);
#endif	/* DCM_SOLVER_H */


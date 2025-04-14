/* 
 * File:   MDCM_Solver.h
 * Author: c3156840
 *
 * Created on 5 October 2014, 6:02 PM
 */
#include "extern_parameters.h"
#ifndef MDCM_SOLVER_H
#define	MDCM_SOLVER_H
void MDCM_Initialize_queue();
double MDCM_Finding_epsilon_value(double value_copy);
void MDCM_Add_Sort_The_NDPs_list(double* Nondominated_Val, double* Efficient_Val, double Value_f_x);
void MDCM_Search_For_Feasible_Solution();
void MDCM_LB_Generate_Binary_Variables(MDCM_nodes* element_copy_copy);
void MDCM_LB_Adding_Disjunctive_Constraints(MDCM_nodes* element_copy_copy);
void Find_MDCM_LB(MDCM_nodes* element_copy);
void Find_MDCM_NDP();
void Find_MDCM_Best_Sol();
bool Points_Are_Equal(double* Point1, double* Point2);
void Set_x_f_Equal_to_x_l_b();
void Set_Max_x_f_Equal_to_f_x_f();
void MDCM_Update_Prority_Queue(MDCM_nodes* element_Copy);
void MDCM_Remove_Dominated_Points_of_Tilde_Y(DCM_nodes* Element_Copy, double* Ref_Point);
void MDCM_Add_The_Internal_Element_To_The_Queue();
void MDCM_Add_The_External_Element_To_The_Queue();
void MDCM_Write_Results();
double Computing_Gap();
void MDCM_Optimizer();
#endif	/* MDCM_SOLVER_H */


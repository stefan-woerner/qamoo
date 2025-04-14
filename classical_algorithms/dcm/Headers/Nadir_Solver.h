/* 
 * File:   Nadir_Solver.h
 * Author: c3156840
 *
 * Created on 7 October 2014, 11:17 AM
 */
#include "extern_parameters.h"
#ifndef NADIR_SOLVER_H
#define	NADIR_SOLVER_H
void Nadir_MDCM_Initialize_queue(int index);
double Nadir_MDCM_Finding_epsilon_value(double value_copy);
void Nadir_MDCM_Add_Sort_The_NDPs_list(double* Nondominated_Val, double* Efficient_Val, double Value_f_x);
void Nadir_MDCM_Search_For_Feasible_Solution(int index);
void Nadir_MDCM_LB_Generate_Binary_Variables(MDCM_nodes* element_copy_copy);
void Nadir_MDCM_LB_Adding_Disjunctive_Constraints(MDCM_nodes* element_copy_copy);
void Nadir_Find_MDCM_LB(MDCM_nodes* element_copy, int index_copy);
void Nadir_Find_MDCM_NDP(int index);
void Nadir_Find_MDCM_Best_Sol(int index);
void Nadir_Set_Max_x_f_Equal_to_f_x_f();
void Nadir_MDCM_Update_Prority_Queue(MDCM_nodes* element_Copy);
void Nadir_MDCM_Remove_Dominated_Points_of_Tilde_Y(MDCM_nodes* Element_Copy, double* Ref_Point);
void Nadir_MDCM_Add_The_Internal_Element_To_The_Queue(int index);
void Nadir_MDCM_Add_The_External_Element_To_The_Queue(int index);
void Nadir_MDCM_Write_Results();
double Nadir_Computing_Gap(double Min_f_x_copy, double Max_f_x_copy);
void Nadir_Optimizer_Single_Component(int comp_copy);
void Nadir_Optimizer_All_Components();
#endif	/* NADIR_SOLVER_H */


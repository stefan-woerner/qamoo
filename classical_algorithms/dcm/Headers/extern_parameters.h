/* 
 * File:   extern_parameters.h
 * Author: c3156840
 *
 * Created on 23 July 2013, 11:39 AM
 */

#ifndef EXTERN_PARAMETERS_H
#define	EXTERN_PARAMETERS_H
#include <iostream>
#include<fstream>
#include <vector>
#include <ilcplex/ilocplex.h>
#include <ctime>
#include <sys/time.h>
#include <math.h>
#include "DCM_nodes.h"
#include "MDCM_nodes.h"
#include "NDPs.h"
#include "Infeasible_Elements.h"
//---------------------------Default setting------------------------------------
#define Algorithm_Time_limit 20000 //Algorithm Run_Time limit (seconds)
#define Time_limit_IP 20000 //Solving time limit for each IP (seconds)
#define Time_Period_For_Outputing_Results 300// How often the results should be reported (seconds)
#define CPLEX_relative_gap 1.0e-6// Relative Gap of CPLEX
#define Hybrid_eps 1.0e-6 //This is for strictly inequality constraints
#define Abs_eps_modifier 0.1//This is for the case when all variables and coefficients are integer
#define Negative_Infty -100000000// The Negative Infinity value in the code
#define Null -200000000 //this should be always strictly smaller than negative_infty parameter (Negative_Infty-Null>=2)
#define Num_threads 1// Number of Threads that CPLEX can use
#define eps_in_Denominator 1.0e-20 //epsilon in denominator of the relative gap function in MDCM
#define MDCM_Absolute_Tolerence 1.0e-6 //epsilon in denominator of the relative gap function in MDCM
//#define Using_Feasible_Solution_in_Nadir_Point_Finder

//------------------------------------------------------------------------------
//some general variables
extern char* LP_file_name;
extern int N_Variables;
extern int N_Objectives;
extern int Optimizer_Number;
extern bool All_Integer;
extern double User_Gap;
extern double* Max_z_i_x;
extern double* Min_z_i_x;
extern double Max_f_x;
extern double Min_f_x;
extern double* x_l;
extern double* x_n;
extern double* obj_x_n;
extern double f_x_n;
extern double* x_f;
extern double* obj_x_f;
extern double f_x_f;
extern double* x_u;
extern double* y_prim;
extern double* y_zegon;
extern double* Nadir_Value;
extern double* Nadir_Bound;
extern double* Nadir_Gap;
extern bool Find_DCM_NDP_is_Feasible;
extern int counter;
extern double eps_value;
extern double eps_modifier;
extern bool NDP_Is_Assigned;
extern bool Feasible_Solution_Exist;
extern bool First_Box_Done;
extern bool Satisfied;
extern int Number_IPs_Solved;
extern int Number_Infeas_IPs_Solved;
extern int N_Items_In_Tilde_Y;
extern bool Point_Is_Dominated;
extern int counter_y_prim;
extern int counter_y_zegon;
extern bool Element_Is_Assigned;
extern bool Infeasible_Relaxation_Exist;
extern bool The_Argument_is_Wrong;
extern bool Find_MDCM_LB_is_Feasible;
extern bool Search_Done;
//---------------------------------queues---------------------------------------
extern std::vector<MDCM_nodes*> MDCM_nodes_list;
extern std::vector<DCM_nodes*> DCM_nodes_list;
extern std::vector<NDPs*> NDPs_list;
extern std::vector<Infeasible_Elements*> Infeasible_Elements_list;
extern std::vector<double> NDP_times;
//------------------------------------------------------------------------------
//defining model
ILOSTLBEGIN
typedef IloArray<IloNumVarArray> NumVar2DArray;
extern IloEnv env;
extern IloModel model;
extern IloObjective cost;
extern IloRangeArray Extra_Constraints;
extern NumVar2DArray Binary_Var;
//main constraint is generating
extern IloRangeArray rngs;
extern IloSOS1Array sos1;
extern IloSOS2Array sos2;
extern IloNumVarArray dec_var;
extern IloExpr f_x;
extern IloExprArray z_i_x;
extern IloNumVarArray z_aux_var;
extern IloObjective object;
extern IloNumVarArray startVar;
extern IloNumVarArray startVarDynamicBinary_Variables;
extern IloNumArray startVal;
extern IloCplex cplex;
extern IloExpr Sum_ObjFs;
extern IloExpr Expr;
//Relaxation
extern IloModel model_relax;
extern IloCplex cplex_relax;
//------------------------------------------------------------------------------
extern ofstream Result;
extern struct timeval startTime, endTime;
extern double totalTime;
extern double clock_Run_time;
extern double clock_start_time;
extern double Termination_Time;
extern double Last_Output_Time;
extern double start_time;
extern int N_Investigated_Elements;
extern int N_Saved_Infeasible_IP;
extern int Number_Disjunctive_Constraints;
extern int Max_Number_Disjunctive_Constraints;
extern int Number_Sets_of_Disjunctive_Constraints;
extern int Max_Number_Sets_of_Disjunctive_Constraints;
#endif	/* EXTERN_PARAMETERS_H */


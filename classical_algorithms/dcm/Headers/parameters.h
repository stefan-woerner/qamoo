/* 
 * File:   parameters.h
 * Author: c3156840
 *
 * Created on 23 July 2013, 11:39 AM
 */

#ifndef PARAMETERS_H
#define	PARAMETERS_H
#include <iostream>
#include<fstream>
#include <vector>
#include <ilcplex/ilocplex.h>
#include <ctime>
#include <sys/time.h>
#include <chrono>
#include "DCM_nodes.h"
#include "MDCM_nodes.h"
#include "NDPs.h"
#include "Infeasible_Elements.h"
//some general variables
char* LP_file_name;
int N_Variables;
int N_Objectives;
int Optimizer_Number;
bool All_Integer;
double User_Gap;
double* Max_z_i_x;
double* Min_z_i_x;
double Max_f_x(0);
double Min_f_x(0);
double* x_l;
double* x_n;
double* obj_x_n;
double f_x_n;
double* x_f;
double* obj_x_f;
double f_x_f;
double* x_u;
double* y_prim;
double* y_zegon;
double* Nadir_Value;
double* Nadir_Bound;
double* Nadir_Gap;
bool Find_DCM_NDP_is_Feasible;
int counter;
double eps_value;
double eps_modifier;
bool NDP_Is_Assigned;
bool Feasible_Solution_Exist;
bool First_Box_Done;
bool Satisfied;
int Number_IPs_Solved(0);
int Number_Infeas_IPs_Solved(0);
int N_Items_In_Tilde_Y;
bool Point_Is_Dominated;
int counter_y_prim;
int counter_y_zegon;
bool Element_Is_Assigned;
bool Infeasible_Relaxation_Exist;
bool The_Argument_is_Wrong;
bool Find_MDCM_LB_is_Feasible;
bool Search_Done;
//---------------------------------queues---------------------------------------
std::vector<MDCM_nodes*> MDCM_nodes_list;
std::vector<DCM_nodes*> DCM_nodes_list;
std::vector<NDPs*> NDPs_list;
std::vector<Infeasible_Elements*> Infeasible_Elements_list;
std::vector<double> NDP_times;
//------------------------------------------------------------------------------
//Declaring CPLEX information
ILOSTLBEGIN
typedef IloArray<IloNumVarArray> NumVar2DArray;
IloEnv env;
IloModel model(env);
IloObjective cost;
IloRangeArray Extra_Constraints(env);
NumVar2DArray Binary_Var(env);
//main constraint is generating
IloRangeArray rngs(env);
IloSOS1Array sos1(env);
IloSOS2Array sos2(env);
IloNumVarArray dec_var(env);
IloExpr f_x(env);
IloExprArray z_i_x(env);
IloNumVarArray z_aux_var(env);
IloObjective object(env);
IloNumVarArray startVar(env);
IloNumVarArray startVarDynamicBinary_Variables(env);
IloNumArray startVal(env);
IloCplex cplex(env);
IloExpr Sum_ObjFs(env);
IloExpr Expr(env);
//Relaxation
IloModel model_relax(env);
IloCplex cplex_relax(env);
//------------------------------------------------------------------------------
ofstream Result;
struct timeval startTime, endTime;
double totalTime(0);
double clock_Run_time(0);
double clock_start_time;
double Termination_Time(0);
double Last_Output_Time(0);
double start_time;
int N_Investigated_Elements(0);
int N_Saved_Infeasible_IP(0);
int Number_Disjunctive_Constraints(0);
int Max_Number_Disjunctive_Constraints(0);
int Number_Sets_of_Disjunctive_Constraints(0);
int Max_Number_Sets_of_Disjunctive_Constraints(0);
//------------------------------------------------------------------------------
#endif	/* PARAMETERS_H */


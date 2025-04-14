#include "Nadir_Solver.h"
#include "extern_parameters.h"
#include "MDCM_nodes.h"

void Nadir_MDCM_Initialize_queue(int index) {
    gettimeofday(&startTime, NULL);
    clock_start_time = clock();
    MDCM_nodes* element = new MDCM_nodes;
    Nadir_Find_MDCM_LB(element, index);
    if (Find_MDCM_LB_is_Feasible == 1) {
        element->y_L_point[index] = Null;
        MDCM_nodes_list.push_back(element);
    } else {
        cout << "The Problem is Infeasible" << endl;
        exit(EXIT_FAILURE);
    }
}

double Nadir_MDCM_Finding_epsilon_value(double value_copy) {
    eps_value = Hybrid_eps;
    if (value_copy < 0) {
        value_copy = value_copy*-1;
    }
    eps_value += (value_copy * Hybrid_eps);
    return eps_value;
}

void Nadir_MDCM_Add_Sort_The_NDPs_list(double* Nondominated_Val, double* Efficient_Val, double Value_f_x) {
    NDP_Is_Assigned = 0;
    for (int i = 0; i < NDPs_list.size(); i++) {
        counter = 0;
        for (int j = 0; j < N_Objectives; j++) {
            if (NDPs_list.at(i)->Obj_Value[j] > Nondominated_Val[j] + Hybrid_eps + eps_modifier) {
                NDP_Is_Assigned = 1;
                NDPs* element = new NDPs;
                for (int k = 0; k < N_Objectives; k++) {
                    element->Obj_Value[k] = Nondominated_Val[k];
                }
                for (int k = 0; k < N_Variables; k++) {
                    element->Variables_Values[k] = Efficient_Val[k];
                }
                element->f_x_Value = Value_f_x;
                NDPs_list.insert(NDPs_list.begin() + i, element);
                for (int k = 0; k < N_Objectives; k++) {
                    Nadir_Value[k] = fmin(Nadir_Value[k], (double) - 1 * Nondominated_Val[k]);
                }
                break;
            } else {
                if (NDPs_list.at(i)->Obj_Value[j] < Nondominated_Val[j] - Hybrid_eps - eps_modifier) {
                    break;
                } else {
                    counter++;
                }
            }
        }
        if (counter == N_Objectives) {
            NDP_Is_Assigned = 1;
        }
        if (NDP_Is_Assigned == 1) {
            break;
        }
    }
    if (NDP_Is_Assigned == 0) {
        NDPs* element = new NDPs;
        for (int k = 0; k < N_Objectives; k++) {
            element->Obj_Value[k] = Nondominated_Val[k];
        }
        for (int k = 0; k < N_Variables; k++) {
            element->Variables_Values[k] = Efficient_Val[k];
        }
        element->f_x_Value = Value_f_x;
        NDPs_list.push_back(element);
        for (int k = 0; k < N_Objectives; k++) {
            Nadir_Value[k] = fmin(Nadir_Value[k], (double) - 1 * Nondominated_Val[k]);
        }
    }
}

void Nadir_MDCM_Search_For_Feasible_Solution(int index) {
    Feasible_Solution_Exist = 0;
#ifdef Using_Feasible_Solution_in_Nadir_Point_Finder
    for (int i = 0; i < NDPs_list.size(); i++) {
        Feasible_Solution_Exist = 1;
        for (int j = 0; j < N_Objectives; j++) {
            if (j != index) {
                if (NDPs_list.at(i)->Obj_Value[j] > obj_x_n[j] + Hybrid_eps) {
                    Feasible_Solution_Exist = 0;
                    break;
                }
            }
        }
        if (Feasible_Solution_Exist == 1) {
            for (int j = 0; j < N_Variables; j++) {
                x_f[j] = NDPs_list.at(i)->Variables_Values[j];
            }
            for (int j = 0; j < N_Objectives; j++) {
                obj_x_f[j] = NDPs_list.at(i)->Obj_Value[j];
            }
            f_x_f = (double) - 1 * NDPs_list.at(i)->Obj_Value[index];
            break;
        }
    }
#endif
}

void Nadir_MDCM_LB_Generate_Binary_Variables(MDCM_nodes* element_copy_copy) {
    Binary_Var.end();
    if (element_copy_copy->y_point_is_Null == 0) {
        Binary_Var = NumVar2DArray(env, element_copy_copy->Y_tilde_list.size() + 1);
        for (int i = 0; i < element_copy_copy->Y_tilde_list.size() + 1; i++) {
            Binary_Var[i] = IloNumVarArray(env, N_Objectives, 0.0, 1.0, ILOBOOL);
        }
    } else {
        Binary_Var = NumVar2DArray(env, element_copy_copy->Y_tilde_list.size());
        for (int i = 0; i < element_copy_copy->Y_tilde_list.size(); i++) {
            Binary_Var[i] = IloNumVarArray(env, N_Objectives, 0.0, 1.0, ILOBOOL);
        }
    }
}

void Nadir_MDCM_LB_Adding_Disjunctive_Constraints(MDCM_nodes* element_copy_copy) {
    //---------------------------------Adding_UB_Envelope_Constraints-------------------------
    Number_Disjunctive_Constraints = 0;
    Number_Sets_of_Disjunctive_Constraints = 0;
    for (int i = 0; i < element_copy_copy->Y_tilde_list.size(); i++) {
        Expr.end();
        Expr = IloExpr(env);
        counter = 0;
        for (int j = 0; j < N_Objectives; j++) {
            if (element_copy_copy->Y_tilde_list.at(i)->point_y[j] > Negative_Infty + Hybrid_eps) {
                counter++;
                Extra_Constraints.add(z_i_x[j]-(element_copy_copy->Y_tilde_list.at(i)->point_y[j] - Nadir_MDCM_Finding_epsilon_value(element_copy_copy->Y_tilde_list.at(i)->point_y[j]) - eps_modifier - Max_z_i_x[j]) * Binary_Var[i][j] - Max_z_i_x[j] <= 0);
                Expr += Binary_Var[i][j];
            }
        }
        if (counter > 0) {
            if (counter > 1) {
                Number_Disjunctive_Constraints += counter;
                Number_Sets_of_Disjunctive_Constraints++;
            }
            Extra_Constraints.add(Expr == 1);
        }
    }
    //------------------------------------------------------------------------------
    if (element_copy_copy->y_point_is_Null == 0) {
        Expr.end();
        Expr = IloExpr(env);
        counter = 0;
        for (int j = 0; j < N_Objectives; j++) {
            if (element_copy_copy->y_point[j] > Negative_Infty + Hybrid_eps) {
                counter++;
                Extra_Constraints.add(z_i_x[j]-(element_copy_copy->y_point[j] - Nadir_MDCM_Finding_epsilon_value(element_copy_copy->y_point[j]) - eps_modifier - Max_z_i_x[j]) * Binary_Var[element_copy_copy->Y_tilde_list.size()][j] - Max_z_i_x[j] <= 0);
                Expr += Binary_Var[element_copy_copy->Y_tilde_list.size()][j];
            }
        }
        if (counter > 0) {
            if (counter > 1) {
                Number_Disjunctive_Constraints += counter;
                Number_Sets_of_Disjunctive_Constraints++;
            }
            Extra_Constraints.add(Expr == 1);
        }
    }
    //---------------------------------Adding_LB_Constraints-------------------------
    for (int i = 0; i < N_Objectives; i++) {
        if (element_copy_copy->y_L_point[i] > Negative_Infty + Hybrid_eps) {
            Extra_Constraints.add(element_copy_copy->y_L_point[i] - z_i_x[i] <= 0);
        }
    }
    //---------------------------------Adding_LB_Constraints-------------------------
    model.add(Extra_Constraints);
    Max_Number_Disjunctive_Constraints = fmax(Max_Number_Disjunctive_Constraints, Number_Disjunctive_Constraints);
    Max_Number_Sets_of_Disjunctive_Constraints = fmax(Max_Number_Sets_of_Disjunctive_Constraints, Number_Sets_of_Disjunctive_Constraints);
}

void Nadir_Find_MDCM_LB(MDCM_nodes* element_copy, int index_copy) {
    Number_IPs_Solved++;
    Find_MDCM_LB_is_Feasible = 0;
    Nadir_MDCM_LB_Generate_Binary_Variables(element_copy);
    Nadir_MDCM_LB_Adding_Disjunctive_Constraints(element_copy);
    cost = IloMinimize(env, f_x);
    model.add(cost);
    cplex.extract(model);
    cplex.resetTime();
    cplex.setParam(IloCplex::TiLim, Time_limit_IP);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, Num_threads);
    cplex.setParam(IloCplex::EpGap, CPLEX_relative_gap);
    // cplex.exportModel("Instance.lp");
    if (cplex.solve()) {
        Find_MDCM_LB_is_Feasible = 1;
        for (int j = 0; j < N_Variables; j++) {
            element_copy->Local_x[j] = cplex.getValue(dec_var[j]);
        }
        for (int j = 0; j < N_Objectives; j++) {
            if (j != index_copy) {
                element_copy->Local_z_i_x[j] = cplex.getValue(z_i_x[j]);
            } else {
                element_copy->Local_z_i_x[j] = Null;
            }
        }
        element_copy->Local_f_x = cplex.getObjValue();
    } else {
        Number_Infeas_IPs_Solved++;
    }
    cplex.clear();
    model.remove(cost);
    model.remove(Extra_Constraints);
    cost.end();
    Extra_Constraints.clear();
    startVal.clear();
}

void Nadir_Find_MDCM_NDP(int index) {
    Number_IPs_Solved++;
    for (int i = 0; i < N_Objectives; i++) {
        if (i != index) {
            Extra_Constraints.add(z_i_x[i] <= MDCM_nodes_list.front()->Local_z_i_x[i]);
        }
    }
    model.add(Extra_Constraints);
    cost = IloMinimize(env, Sum_ObjFs);
    model.add(cost);
    cplex.extract(model);
    for (int i = 0; i < N_Variables; ++i) {
        startVal.add(MDCM_nodes_list.front()->Local_x[i]);
    }
    cplex.addMIPStart(startVar, startVal);
    cplex.resetTime();
    cplex.setParam(IloCplex::TiLim, Time_limit_IP);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, Num_threads);
    cplex.setParam(IloCplex::EpGap, CPLEX_relative_gap);
    //cplex.exportModel("Instance.lp");
    if (cplex.solve()) {
        for (int j = 0; j < N_Variables; j++) {
            x_n[j] = cplex.getValue(dec_var[j]);
        }
        for (int j = 0; j < N_Objectives; j++) {
            if (j != index) {
                obj_x_n[j] = cplex.getValue(z_i_x[j]);
            } else {
                obj_x_n[j] = Null;
            }
        }
        f_x_n = cplex.getValue(f_x);
    } else {
        cout << "Numerical Issue happened in Find_MCDM_NDP() Function" << endl;
        exit(EXIT_FAILURE);
    }
    cplex.clear();
    model.remove(cost);
    model.remove(Extra_Constraints);
    cost.end();
    Extra_Constraints.clear();
    startVal.clear();
}

void Nadir_Find_MDCM_Best_Sol(int index) {
    Number_IPs_Solved++;
    for (int i = 0; i < N_Objectives; i++) {
        if (i != index) {
            Extra_Constraints.add(z_i_x[i] <= obj_x_n[i]);
        }
    }
    model.add(Extra_Constraints);
    cost = IloMinimize(env, z_i_x[index]);
    model.add(cost);
    cplex.extract(model);
    for (int i = 0; i < N_Variables; ++i) {
        startVal.add(x_n[i]);
    }
    cplex.addMIPStart(startVar, startVal);
    cplex.resetTime();
    cplex.setParam(IloCplex::TiLim, Time_limit_IP);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, Num_threads);
    cplex.setParam(IloCplex::EpGap, CPLEX_relative_gap);
    //cplex.exportModel("Instance.lp");
    if (cplex.solve()) {
        for (int j = 0; j < N_Variables; j++) {
            x_f[j] = cplex.getValue(dec_var[j]);
        }
        for (int j = 0; j < N_Objectives; j++) {
            obj_x_f[j] = cplex.getValue(z_i_x[j]);
        }
        f_x_f = cplex.getValue(f_x);
    } else {
        cout << "Numerical Issue happened in Find_MDCM_Best_Sol() Function" << endl;
        exit(EXIT_FAILURE);
    }
    cplex.clear();
    model.remove(cost);
    model.remove(Extra_Constraints);
    cost.end();
    Extra_Constraints.clear();
    startVal.clear();
}

void Nadir_MDCM_Update_Prority_Queue(MDCM_nodes* element_Copy) {
    Element_Is_Assigned = 0;
    for (int i = 1; i < MDCM_nodes_list.size(); i++) {
        if (MDCM_nodes_list.at(i)->Local_f_x > element_Copy->Local_f_x) {
            MDCM_nodes_list.insert(MDCM_nodes_list.begin() + i, element_Copy);
            Element_Is_Assigned = 1;
            break;
        }
    }
    if (Element_Is_Assigned == 0) {
        MDCM_nodes_list.push_back(element_Copy);
    }
}

void Nadir_MDCM_Remove_Dominated_Points_of_Tilde_Y(MDCM_nodes* Element_Copy, double* Ref_Point) {
    counter = 0;
    N_Items_In_Tilde_Y = Element_Copy->Y_tilde_list.size();
    for (int i = 0; i < N_Items_In_Tilde_Y; i++) {
        Point_Is_Dominated = 1;
        for (int j = 0; j < N_Objectives; j++) {
            if (Element_Copy->Y_tilde_list.at(counter)->point_y[j] < Ref_Point[j] - Hybrid_eps - eps_modifier) {
                Point_Is_Dominated = 0;
                break;
            }
        }
        if (Point_Is_Dominated == 1) {
            delete Element_Copy->Y_tilde_list.at(counter);
            Element_Copy->Y_tilde_list.erase(Element_Copy->Y_tilde_list.begin() + counter);
        } else {
            counter++;
        }
    }
}

void Nadir_MDCM_Add_The_Internal_Element_To_The_Queue(int index) {
    MDCM_nodes* element = new MDCM_nodes;
    for (int i = 0; i < N_Objectives; i++) {
        if (i != index) {
            element->y_L_point[i] = fmax(MDCM_nodes_list.front()->y_L_point[i], fmin(MDCM_nodes_list.front()->y_point[i], obj_x_f[i]));
        } else {
            element->y_L_point[i] = Null;
        }
    }
    for (int j = 0; j < MDCM_nodes_list.front()->Y_tilde_list.size(); j++) {
        {
            Y_tilde_class* vertex = new Y_tilde_class;
            for (int i = 0; i < N_Objectives; i++) {
                vertex->point_y[i] = MDCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[i];
            }
            element->Y_tilde_list.push_back(vertex);
        }
    }
    counter_y_prim = 0;
    counter_y_zegon = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (i != index) {
            if (obj_x_f[i] > MDCM_nodes_list.front()->y_point[i] + Hybrid_eps + eps_modifier && obj_x_f[i] > element->y_L_point[i] + Hybrid_eps + eps_modifier) {
                y_prim[i] = obj_x_f[i];
            } else {
                counter_y_prim++;
                y_prim[i] = Negative_Infty;
            }
        } else {
            y_prim[i] = Null;
        }
    }
    if (counter_y_prim < N_Objectives - 1) {
        for (int i = 0; i < N_Objectives; i++) {
            if (i != index) {
                if (obj_x_f[i] < MDCM_nodes_list.front()->y_point[i] - Hybrid_eps - eps_modifier && MDCM_nodes_list.front()->y_point[i] > element->y_L_point[i] + Hybrid_eps + eps_modifier) {
                    y_zegon[i] = MDCM_nodes_list.front()->y_point[i];
                } else {
                    counter_y_zegon++;
                    y_zegon[i] = Negative_Infty;
                }
            } else {
                y_zegon[i] = Null;
            }
        }
        if (counter_y_zegon < N_Objectives - 1) {
            Nadir_MDCM_Remove_Dominated_Points_of_Tilde_Y(element, y_prim);
            Nadir_MDCM_Remove_Dominated_Points_of_Tilde_Y(element, y_zegon);
            {
                Y_tilde_class* vertex = new Y_tilde_class;
                for (int i = 0; i < N_Objectives; i++) {
                    vertex->point_y[i] = y_prim[i];
                }
                element->Y_tilde_list.push_back(vertex);
            }
            {
                Y_tilde_class* vertex = new Y_tilde_class;
                for (int i = 0; i < N_Objectives; i++) {
                    vertex->point_y[i] = y_zegon[i];
                }
                element->Y_tilde_list.push_back(vertex);
            }
            Nadir_Find_MDCM_LB(element, index);
            if (Find_MDCM_LB_is_Feasible == 1 && element->Local_f_x < Max_f_x - MDCM_Absolute_Tolerence) {
                Nadir_MDCM_Update_Prority_Queue(element);
            } else {
                element->~MDCM_nodes();
            }
        }
    }
    if (counter_y_prim == N_Objectives - 1 || counter_y_zegon == N_Objectives - 1) {
        element->~MDCM_nodes();
    }
}

void Nadir_MDCM_Add_The_External_Element_To_The_Queue(int index) {
    MDCM_nodes* element = new MDCM_nodes;
    element->y_point_is_Null = 0;
    counter = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (i != index) {
            if (MDCM_nodes_list.front()->y_point[i] < Null + 1) {
                element->y_point[i] = fmax(MDCM_nodes_list.front()->y_L_point[i], obj_x_f[i]);
            } else {
                element->y_point[i] = fmax(MDCM_nodes_list.front()->y_L_point[i], fmin(MDCM_nodes_list.front()->y_point[i], obj_x_f[i]));
            }
            if (element->y_point[i] > MDCM_nodes_list.front()->y_L_point[i] - Hybrid_eps - eps_modifier && element->y_point[i] < MDCM_nodes_list.front()->y_L_point[i] + Hybrid_eps + eps_modifier) {
                element->y_point[i] = Negative_Infty;
                counter++;
            }
        } else {
            element->y_point[i] = Null;
        }
        element->y_L_point[i] = MDCM_nodes_list.front()->y_L_point[i];
    }
    if (counter == N_Objectives - 1) {
        element->~MDCM_nodes();
    } else {
        for (int j = 0; j < MDCM_nodes_list.front()->Y_tilde_list.size(); j++) {
            {
                Y_tilde_class* vertex = new Y_tilde_class;
                for (int i = 0; i < N_Objectives; i++) {
                    vertex->point_y[i] = MDCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[i];
                }
                element->Y_tilde_list.push_back(vertex);
            }
        }
        Nadir_MDCM_Remove_Dominated_Points_of_Tilde_Y(element, element->y_point);
        Nadir_Find_MDCM_LB(element, index);
        if (Find_MDCM_LB_is_Feasible == 1 && element->Local_f_x < Max_f_x - MDCM_Absolute_Tolerence) {
            Nadir_MDCM_Update_Prority_Queue(element);
        } else {
            element->~MDCM_nodes();
        }
    }
}

void Nadir_MDCM_Write_Results() {
    gettimeofday(&endTime, NULL);
    clock_Run_time += (clock() - clock_start_time);
    totalTime += ((endTime.tv_sec - startTime.tv_sec) * 1000000L);
    totalTime += (endTime.tv_usec - startTime.tv_usec);
    Result.open("Result.txt");
    Result << LP_file_name << endl;
    Result << "Number_of_NDPs_Found: " << NDPs_list.size() << endl;
    Result << "Number_of_IPs: " << Number_IPs_Solved << endl;
    Result << "Number_of_Infeasible_IPs: " << Number_Infeas_IPs_Solved << endl;
    Result << "Wall_Time(s): " << (totalTime / 1000000L) << endl;
    Result << "CPU_Time(s): " << (clock_Run_time / CLOCKS_PER_SEC) << endl;
    Result << "Number_of_Investigated_Elements: " << N_Investigated_Elements << endl;
    Result << "Number_of_Elements_Remained_In_Priority_Queue: " << MDCM_nodes_list.size() << endl;
    Result << "Upper_Bound_for_f(x): ";
    for (int i = 0; i < N_Objectives; i++) {
        if (Nadir_Bound[i] < (-1 * Null) - 1) {
            Result << (double) - 1 * Nadir_Value[i] << " ";
        } else {
            Result << "Null" << " ";
        }
    }
    Result << endl;
    Result << "Lower_Bound_for_f(x): ";
    for (int i = 0; i < N_Objectives; i++) {
        if (Nadir_Bound[i] > Null + Hybrid_eps) {
            Result << (double) - 1 * Nadir_Bound[i] << " ";
        } else {
            Result << "Null" << " ";
        }
    }
    Result << endl;
    Result << "Relative Gap(%): ";
    for (int i = 0; i < N_Objectives; i++) {
        if (Nadir_Bound[i] > Null + Hybrid_eps) {
            Result << Nadir_Computing_Gap(Nadir_Bound[i], Nadir_Value[i]) << " ";
        } else {
            Result << "Null" << " ";
        }
    }
    Result << endl;
    Result << "Max_Number_of_BigMConstraints_added_to_a_Model: " << Max_Number_Disjunctive_Constraints << endl;
    Result << "Max_Number_of_Sets_of_Disjunctive_Constraints_added_to_a_Model: " << Max_Number_Sets_of_Disjunctive_Constraints << endl;
    Result << "-----------------------------------------------------------------" << endl;
    for (int i = 0; i < NDPs_list.size(); i++) {
        Result << "NDP " << i + 1 << ": ";
        for (int j = 0; j < N_Objectives; j++) {
            Result << NDPs_list.at(i)->Obj_Value[j] << " ";
        }
        Result << endl;
    }
    Result << "-----------------------------------------------------------------" << endl;
    for (int i = 0; i < NDPs_list.size(); i++) {
        Result << "Solution for NDP " << i + 1 << ": ";
        for (int j = 0; j < N_Variables; j++) {
            Result << dec_var[j] << "=" << NDPs_list.at(i)->Variables_Values[j] << " ";
        }
        Result << endl << endl;
    }
    Result.close();
    gettimeofday(&startTime, NULL);
    clock_start_time = clock();
    if (Num_threads == 1) {
        Last_Output_Time = (clock_Run_time / CLOCKS_PER_SEC);
    } else {
        Last_Output_Time = (totalTime / 1000000L);
    }
}

double Nadir_Computing_Gap(double Min_f_x_copy, double Max_f_x_copy) {
    if (Max_f_x_copy <= 0 && Min_f_x_copy <= 0) {
        return (double) abs(Max_f_x_copy - Min_f_x_copy)*100 / (double) (abs(Min_f_x_copy) + eps_in_Denominator);
    } else {
        if (Max_f_x_copy >= 0 && Min_f_x_copy >= 0) {
            return (double) abs(Max_f_x_copy - Min_f_x_copy)*100 / (double) (abs(Max_f_x_copy) + eps_in_Denominator);
        } else {

            return 100;
        }
    }
}

void Nadir_Optimizer_Single_Component(int comp_copy) {
    Nadir_MDCM_Initialize_queue(comp_copy);
    Search_Done = 0;
    while (Search_Done == 0 && MDCM_nodes_list.size() > 0) {
        N_Investigated_Elements++;
        Min_f_x = MDCM_nodes_list.front()->Local_f_x;
        Nadir_Bound[comp_copy] = Min_f_x;
        if (Nadir_Computing_Gap(Min_f_x, Max_f_x) > User_Gap && Max_f_x - Min_f_x > MDCM_Absolute_Tolerence && Termination_Time < Algorithm_Time_limit) {
            Nadir_Find_MDCM_NDP(comp_copy);
            Nadir_MDCM_Search_For_Feasible_Solution(comp_copy);
            if (Feasible_Solution_Exist == 0) {
                Nadir_Find_MDCM_Best_Sol(comp_copy);
                Nadir_MDCM_Add_Sort_The_NDPs_list(obj_x_f, x_f, f_x_f);
            }
            if (Max_f_x > f_x_f) {
                Max_f_x = f_x_f;
            }
            if (Min_f_x + MDCM_Absolute_Tolerence < Max_f_x) {
                if (MDCM_nodes_list.front()->y_point_is_Null == 0) {
                    Nadir_MDCM_Add_The_Internal_Element_To_The_Queue(comp_copy);
                }
                Nadir_MDCM_Add_The_External_Element_To_The_Queue(comp_copy);
            }
            delete MDCM_nodes_list.front();
            MDCM_nodes_list.erase(MDCM_nodes_list.begin());
            gettimeofday(&endTime, NULL);
            clock_Run_time += (clock() - clock_start_time);
            totalTime += ((endTime.tv_sec - startTime.tv_sec) * 1000000L);
            totalTime += (endTime.tv_usec - startTime.tv_usec);
            gettimeofday(&startTime, NULL);
            clock_start_time = clock();
            if (Num_threads == 1) {
                Termination_Time = (clock_Run_time / CLOCKS_PER_SEC);
            } else {
                Termination_Time = (totalTime / 1000000L);
            }
            if (Termination_Time - Last_Output_Time > Time_Period_For_Outputing_Results) {
                Nadir_MDCM_Write_Results();
            }
        } else {
            Search_Done = 1;
        }
    }
    if (Nadir_Computing_Gap(Min_f_x, Max_f_x) <= User_Gap || Max_f_x - Min_f_x <= MDCM_Absolute_Tolerence) {
        if (NDPs_list.size() == 0) {
            cout << "Special case is happened!" << endl;
            cout << "Finding an efficient solution will start!" << endl;
            Nadir_Find_MDCM_NDP(comp_copy);
            Nadir_Find_MDCM_Best_Sol(comp_copy);
            Max_f_x = f_x_f;
            Nadir_MDCM_Add_Sort_The_NDPs_list(obj_x_f, x_f, f_x_f);
        }
    }
    if (MDCM_nodes_list.size() == 0 && Min_f_x < Max_f_x) {
        Min_f_x = Max_f_x;
        Nadir_Bound[comp_copy] = Min_f_x;
    }
    if (Min_f_x > Max_f_x) {
        Min_f_x = Max_f_x;
        Nadir_Bound[comp_copy] = Min_f_x;
    }
    Nadir_MDCM_Write_Results();
}

void Nadir_Optimizer_All_Components() {
    for (int comp = 0; comp < N_Objectives; comp++) {
        while (MDCM_nodes_list.size()) {
            delete MDCM_nodes_list.front();
            MDCM_nodes_list.erase(MDCM_nodes_list.begin());
        }
        Min_f_x = Null;
        if (Nadir_Value[comp] < (-1 * Null) - 1) {
            Max_f_x = Nadir_Value[comp];
        } else {
            Max_f_x = (double) - 1 * Min_z_i_x[comp];
        }
        f_x.clear();
        f_x -= z_i_x[comp];
        Sum_ObjFs.clear();
        for (int i = 0; i < N_Objectives; i++) {
            if (i != comp) {
                Sum_ObjFs += z_i_x[i];
            }
        }
        Nadir_Optimizer_Single_Component(comp);
    }
    cout << "Finish!!!!" << endl;
}
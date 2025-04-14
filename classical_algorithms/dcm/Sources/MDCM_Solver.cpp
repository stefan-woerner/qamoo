#include "MDCM_Solver.h"
#include "extern_parameters.h"
#include "MDCM_nodes.h"

void MDCM_Initialize_queue() {
    gettimeofday(&startTime, NULL);
    clock_start_time = clock();
    MDCM_nodes* element = new MDCM_nodes;
    Find_MDCM_LB(element);
    if (Find_MDCM_LB_is_Feasible == 1) {
        MDCM_nodes_list.push_back(element);
    } else {
        cout << "The Problem is Infeasible" << endl;
        exit(EXIT_FAILURE);
    }
}

double MDCM_Finding_epsilon_value(double value_copy) {
    eps_value = Hybrid_eps;
    if (value_copy < 0) {
        value_copy = value_copy*-1;
    }
    eps_value += (value_copy * Hybrid_eps);
    return eps_value;
}

void MDCM_Add_Sort_The_NDPs_list(double* Nondominated_Val, double* Efficient_Val, double Value_f_x) {
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
    }
}

void MDCM_Search_For_Feasible_Solution() {
    Feasible_Solution_Exist = 0;
    for (int i = 0; i < NDPs_list.size(); i++) {
        Feasible_Solution_Exist = 1;
        if (NDPs_list.at(i)->Obj_Value[0] > MDCM_nodes_list.front()->Local_z_i_x[0] + Hybrid_eps) {
            Feasible_Solution_Exist = 0;
            break;
        }
        for (int j = 1; j < N_Objectives; j++) {
            if (NDPs_list.at(i)->Obj_Value[j] > MDCM_nodes_list.front()->Local_z_i_x[j] + Hybrid_eps) {
                Feasible_Solution_Exist = 0;
                break;
            }
        }
        if (Feasible_Solution_Exist == 1) {
            for (int j = 0; j < N_Variables; j++) {
                x_f[j] = NDPs_list.at(i)->Variables_Values[j];
            }
            for (int j = 0; j < N_Objectives; j++) {
                obj_x_f[j] = NDPs_list.at(i)->Obj_Value[j];
            }
            f_x_f = NDPs_list.at(i)->f_x_Value;
            break;
        }
    }
}

void MDCM_LB_Generate_Binary_Variables(MDCM_nodes* element_copy_copy) {
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

void MDCM_LB_Adding_Disjunctive_Constraints(MDCM_nodes* element_copy_copy) {
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
                Extra_Constraints.add(z_i_x[j]-(element_copy_copy->Y_tilde_list.at(i)->point_y[j] - MDCM_Finding_epsilon_value(element_copy_copy->Y_tilde_list.at(i)->point_y[j]) - eps_modifier - Max_z_i_x[j]) * Binary_Var[i][j] - Max_z_i_x[j] <= 0);
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
                Extra_Constraints.add(z_i_x[j]-(element_copy_copy->y_point[j] - MDCM_Finding_epsilon_value(element_copy_copy->y_point[j]) - eps_modifier - Max_z_i_x[j]) * Binary_Var[element_copy_copy->Y_tilde_list.size()][j] - Max_z_i_x[j] <= 0);
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

void Find_MDCM_LB(MDCM_nodes* element_copy) {
    Number_IPs_Solved++;
    Find_MDCM_LB_is_Feasible = 0;
    MDCM_LB_Generate_Binary_Variables(element_copy);
    MDCM_LB_Adding_Disjunctive_Constraints(element_copy);
    cost = IloMinimize(env, f_x);
    model.add(cost);
    cplex.extract(model);
    cplex.resetTime();
    cplex.setParam(IloCplex::TiLim, Time_limit_IP);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, Num_threads);
    cplex.setParam(IloCplex::EpGap, CPLEX_relative_gap);
    //cplex.exportModel("Instance.lp");
    if (cplex.solve()) {
        Find_MDCM_LB_is_Feasible = 1;
        for (int j = 0; j < N_Variables; j++) {
            element_copy->Local_x[j] = cplex.getValue(dec_var[j]);
        }
        for (int j = 0; j < N_Objectives; j++) {
            element_copy->Local_z_i_x[j] = cplex.getValue(z_i_x[j]);
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

void Find_MDCM_NDP() {
    Number_IPs_Solved++;
    for (int i = 0; i < N_Objectives; i++) {
        Extra_Constraints.add(z_i_x[i] <= MDCM_nodes_list.front()->Local_z_i_x[i]);
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
            obj_x_n[j] = cplex.getValue(z_i_x[j]);
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

void Find_MDCM_Best_Sol() {
    Number_IPs_Solved++;
    for (int i = 0; i < N_Objectives; i++) {
        Extra_Constraints.add(z_i_x[i] <= obj_x_n[i]);
    }
    model.add(Extra_Constraints);
    cost = IloMinimize(env, f_x);
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
        f_x_f = cplex.getObjValue();
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

bool Points_Are_Equal(double* Point1, double* Point2) {
    The_Argument_is_Wrong = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (Point1[i] > Point2[i] + Hybrid_eps + eps_modifier) {
            The_Argument_is_Wrong = 1;
            break;
        }
        if (Point2[i] > Point1[i] + Hybrid_eps + eps_modifier) {
            The_Argument_is_Wrong = 1;
            break;
        }
    }
    if (The_Argument_is_Wrong == 1) {
        return 0;
    } else {
        return 1;
    }
}

void Set_x_f_Equal_to_x_l_b() {
    for (int j = 0; j < N_Variables; j++) {
        x_f[j] = MDCM_nodes_list.front()->Local_x[j];
    }
    for (int j = 0; j < N_Objectives; j++) {
        obj_x_f[j] = MDCM_nodes_list.front()->Local_z_i_x[j];
    }
    f_x_f = MDCM_nodes_list.front()->Local_f_x;
}

void Set_Max_x_f_Equal_to_f_x_f() {
    for (int j = 0; j < N_Variables; j++) {
        x_u[j] = x_f[j];
    }
    Max_f_x = f_x_f;
}

void MDCM_Update_Prority_Queue(MDCM_nodes* element_Copy) {
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

void MDCM_Remove_Dominated_Points_of_Tilde_Y(MDCM_nodes* Element_Copy, double* Ref_Point) {
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

void MDCM_Add_The_Internal_Element_To_The_Queue() {
    MDCM_nodes* element = new MDCM_nodes;
    for (int i = 0; i < N_Objectives; i++) {
        element->y_L_point[i] = fmax(MDCM_nodes_list.front()->y_L_point[i], fmin(MDCM_nodes_list.front()->y_point[i], obj_x_f[i]));
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
        if (obj_x_f[i] > MDCM_nodes_list.front()->y_point[i] + Hybrid_eps + eps_modifier && obj_x_f[i] > element->y_L_point[i] + Hybrid_eps + eps_modifier) {
            y_prim[i] = obj_x_f[i];
        } else {
            counter_y_prim++;
            y_prim[i] = Negative_Infty;
        }
    }
    if (counter_y_prim < N_Objectives) {
        for (int i = 0; i < N_Objectives; i++) {
            if (obj_x_f[i] < MDCM_nodes_list.front()->y_point[i] - Hybrid_eps - eps_modifier && MDCM_nodes_list.front()->y_point[i] > element->y_L_point[i] + Hybrid_eps + eps_modifier) {
                y_zegon[i] = MDCM_nodes_list.front()->y_point[i];
            } else {
                counter_y_zegon++;
                y_zegon[i] = Negative_Infty;
            }
        }
        if (counter_y_zegon < N_Objectives) {
            MDCM_Remove_Dominated_Points_of_Tilde_Y(element, y_prim);
            MDCM_Remove_Dominated_Points_of_Tilde_Y(element, y_zegon);
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
            Find_MDCM_LB(element);
            if (Find_MDCM_LB_is_Feasible == 1 && element->Local_f_x < Max_f_x - MDCM_Absolute_Tolerence) {
                MDCM_Update_Prority_Queue(element);
            } else {
                element->~MDCM_nodes();
            }
        }
    }
    if (counter_y_prim == N_Objectives || counter_y_zegon == N_Objectives) {
        element->~MDCM_nodes();
    }
}

void MDCM_Add_The_External_Element_To_The_Queue() {
    MDCM_nodes* element = new MDCM_nodes;
    element->y_point_is_Null = 0;
    counter = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (MDCM_nodes_list.front()->y_point[i] < Null + 1) {
            element->y_point[i] = fmax(MDCM_nodes_list.front()->y_L_point[i], obj_x_f[i]);
        } else {
            element->y_point[i] = fmax(MDCM_nodes_list.front()->y_L_point[i], fmin(MDCM_nodes_list.front()->y_point[i], obj_x_f[i]));
        }
        if (element->y_point[i] > MDCM_nodes_list.front()->y_L_point[i] - Hybrid_eps - eps_modifier && element->y_point[i] < MDCM_nodes_list.front()->y_L_point[i] + Hybrid_eps + eps_modifier) {
            element->y_point[i] = Negative_Infty;
            counter++;
        }
        element->y_L_point[i] = MDCM_nodes_list.front()->y_L_point[i];
    }
    if (counter == N_Objectives) {
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
        MDCM_Remove_Dominated_Points_of_Tilde_Y(element, element->y_point);
        Find_MDCM_LB(element);
        if (Find_MDCM_LB_is_Feasible == 1 && element->Local_f_x < Max_f_x - MDCM_Absolute_Tolerence) {
            MDCM_Update_Prority_Queue(element);
        } else {
            element->~MDCM_nodes();
        }
    }
}

void MDCM_Write_Results() {
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
    Result << "Upper_Bound_for_f(x): " << Max_f_x << endl;
    Result << "Lower_Bound_for_f(x): " << Min_f_x << endl;
    Result << "Relative Gap(%): " << Computing_Gap() << endl;
    Result << "Max_Number_of_BigMConstraints_added_to_a_Model: " << Max_Number_Disjunctive_Constraints << endl;
    Result << "Max_Number_of_Sets_of_Disjunctive_Constraints_added_to_a_Model: " << Max_Number_Sets_of_Disjunctive_Constraints << endl;
    Result << "-----------------------------------------------------------------" << endl;
    for (int i = 0; i < NDPs_list.size(); i++) {
        Result << "NDP " << i + 1 << ": ";
        for (int j = 0; j < N_Objectives; j++) {
            Result << NDPs_list.at(i)->Obj_Value[j] << " ";
        }
        Result << "(f(x)= " << NDPs_list.at(i)->f_x_Value << ") ";
        Result << endl;
    }
    Result << "-----------------------------------------------------------------" << endl;
    Result << "Upper_Bound_Solution_for_f(x): ";
    if (x_u[0] == Null) {
        Result << "Null" << endl;
    } else {
        for (int j = 0; j < N_Variables; j++) {
            Result << dec_var[j] << "=" << x_u[j] << " ";
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

double Computing_Gap() {
    if (Max_f_x <= 0 && Min_f_x <= 0) {
        return (double) abs(Max_f_x - Min_f_x)*100 / (double) (abs(Min_f_x) + eps_in_Denominator);
    } else {
        if (Max_f_x >= 0 && Min_f_x >= 0) {
            return (double) abs(Max_f_x - Min_f_x)*100 / (double) (abs(Max_f_x) + eps_in_Denominator);
        } else {

            return 100;
        }
    }
}

void MDCM_Optimizer() {
    MDCM_Initialize_queue();
    Search_Done = 0;
    while (Search_Done == 0 && MDCM_nodes_list.size() > 0) {
        N_Investigated_Elements++;
        Min_f_x = MDCM_nodes_list.front()->Local_f_x;
        if (Computing_Gap() > User_Gap && Max_f_x - Min_f_x > MDCM_Absolute_Tolerence && Termination_Time < Algorithm_Time_limit) {
            MDCM_Search_For_Feasible_Solution();
            if (Feasible_Solution_Exist == 0) {
                Find_MDCM_NDP();
                if (Points_Are_Equal(MDCM_nodes_list.front()->Local_z_i_x, obj_x_n) == 1) {
                    Set_x_f_Equal_to_x_l_b();
                } else {
                    Find_MDCM_Best_Sol();
                }
                MDCM_Add_Sort_The_NDPs_list(obj_x_f, x_f, f_x_f);
                if (Max_f_x > f_x_f) {
                    Set_Max_x_f_Equal_to_f_x_f();
                }
            }
            if (Min_f_x + MDCM_Absolute_Tolerence < Max_f_x) {
                if (MDCM_nodes_list.front()->y_point_is_Null == 0) {
                    MDCM_Add_The_Internal_Element_To_The_Queue();
                }
                MDCM_Add_The_External_Element_To_The_Queue();
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
                MDCM_Write_Results();
            }
        } else {
            Search_Done = 1;
        }
    }
    if (Computing_Gap() <= User_Gap || Max_f_x - Min_f_x <= MDCM_Absolute_Tolerence) {
        if (x_u[0] == Null) {
            cout << "Special case is happened!" << endl;
            cout << "Finding an efficient solution will start!" << endl;
            Find_MDCM_NDP();
            Find_MDCM_Best_Sol();
            Set_Max_x_f_Equal_to_f_x_f();
            MDCM_Add_Sort_The_NDPs_list(obj_x_f, x_f, f_x_f);
        }
    }
    if (MDCM_nodes_list.size() == 0 && Min_f_x < Max_f_x) {
        Min_f_x = Max_f_x;
    }
    if (Min_f_x > Max_f_x) {
        Min_f_x = Max_f_x;
    }
    MDCM_Write_Results();
    cout << "Finish!!!!" << endl;
}
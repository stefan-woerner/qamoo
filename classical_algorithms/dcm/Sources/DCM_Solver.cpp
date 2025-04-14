#include"DCM_Solver.h"
#include "extern_parameters.h"
#include "DCM_nodes.h"
#include <time.h>
#include <chrono>

void Initialize_queue() {
    gettimeofday(&startTime, NULL);
    clock_start_time = clock();
    auto start = std::chrono::high_resolution_clock::now();
    start_time = std::chrono::duration<double>(start.time_since_epoch()).count();
    Infeasible_Relaxation_Exist = 0;
    Feasible_Solution_Exist = 0;
    First_Box_Done = 0;
    DCM_nodes* element = new DCM_nodes;
    element->Computing_BOX_UB_Point();
    DCM_nodes_list.push_back(element);
}

double Finding_epsilon_value(double value_copy) {
    eps_value = Hybrid_eps;
    if (value_copy < 0) {
        value_copy = value_copy*-1;
    }
    eps_value += (value_copy * Hybrid_eps);
    return eps_value;
}

void Search_For_Feasible_Solution() {
    for (int i = 0; i < NDPs_list.size(); i++) {
        Feasible_Solution_Exist = 1;
        for (int j = 0; j < DCM_nodes_list.front()->Y_tilde_list.size(); j++) {
            counter = 0;
            Satisfied = 0;
            for (int k = 0; k < N_Objectives; k++) {
                if (DCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[k] > Negative_Infty + Hybrid_eps + eps_modifier) {
                    counter++;
                    if (DCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[k] - Finding_epsilon_value(DCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[k]) - eps_modifier > NDPs_list.at(i)->Obj_Value[k]) {
                        Satisfied = 1;
                        break;
                    }
                }
            }
            if (Satisfied == 0 && counter > 0) {
                Feasible_Solution_Exist = 0;
                break;
            }
        }
        if (Feasible_Solution_Exist == 1) {
            counter = 0;
            Satisfied = 0;
            for (int k = 0; k < N_Objectives; k++) {
                if (DCM_nodes_list.front()->y_point[k] > Negative_Infty + Hybrid_eps + eps_modifier) {
                    counter++;
                    if (DCM_nodes_list.front()->y_point[k] - Finding_epsilon_value(DCM_nodes_list.front()->y_point[k]) - eps_modifier > NDPs_list.at(i)->Obj_Value[k]) {
                        Satisfied = 1;
                        break;
                    }
                }
            }
            if (Satisfied == 0 && counter > 0) {
                Feasible_Solution_Exist = 0;
            }
        }
        if (Feasible_Solution_Exist == 1) {
            for (int k = 0; k < N_Objectives; k++) {
                obj_x_n[k] = NDPs_list.at(i)->Obj_Value[k];
            }
            break;
        }
    }
}

bool FirstArg_is_Smaller_than_SecondArg(double* Point1, double* Point2) {
    The_Argument_is_Wrong = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (Point1[i] > Point2[i] + Hybrid_eps + eps_modifier) {
            The_Argument_is_Wrong = 1;
            break;
        }
    }
    if (The_Argument_is_Wrong == 0) {
        return 1;
    } else {
        return 0;
    }
}

void Searching_Among_Infeasible_Cases() {
    for (int i = 0; i < Infeasible_Elements_list.size(); i++) {
        Infeasible_Relaxation_Exist = 0;
        if (Infeasible_Elements_list.at(i)->Sum_BOX_UB_Components < DCM_nodes_list.front()->Sum_BOX_UB_Components - Hybrid_eps) {
            break;
        }
        counter = 0;
        for (int j = 0; j < N_Objectives; j++) {
            if (Infeasible_Elements_list.at(i)->Box_UB_Point[j] < DCM_nodes_list.front()->Box_UB_Point[j] - Hybrid_eps) {
                break;
            }
            counter++;
        }
        if (counter == N_Objectives) {
            Infeasible_Relaxation_Exist = 1;
            for (int j = 0; j < Infeasible_Elements_list.at(i)->Y_tilde_list.size(); j++) {
                counter = 0;
                for (int k = 0; k < DCM_nodes_list.front()->Y_tilde_list.size(); k++) {
                    if (FirstArg_is_Smaller_than_SecondArg(DCM_nodes_list.front()->Y_tilde_list.at(k)->point_y, Infeasible_Elements_list.at(i)->Y_tilde_list.at(j)->point_y) == 1) {
                        counter++;
                        break;
                    }
                }
                if (counter == 0 && DCM_nodes_list.front()->y_point[0] > Null + 1) {
                    if (FirstArg_is_Smaller_than_SecondArg(DCM_nodes_list.front()->y_point, Infeasible_Elements_list.at(i)->Y_tilde_list.at(j)->point_y) == 1) {
                        counter++;
                    }
                }
                if (counter == 0) {
                    Infeasible_Relaxation_Exist = 0;
                }
                if (Infeasible_Relaxation_Exist == 0) {
                    break;
                }
            }
        }
        if (Infeasible_Relaxation_Exist == 1) {
            N_Saved_Infeasible_IP++;
            break;
        }
    }
}

void Add_Sort_The_NDPs_list(double* Nondominated_Val, double* Efficient_Val) {
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
                NDPs_list.insert(NDPs_list.begin() + i, element);
                auto current = std::chrono::high_resolution_clock::now();
                auto current_time = std::chrono::duration<double>(current.time_since_epoch()).count();
                NDP_times.insert(NDP_times.begin() + i, current_time - start_time);
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
        NDPs_list.push_back(element);
        auto current = std::chrono::high_resolution_clock::now();
        auto current_time = std::chrono::duration<double>(current.time_since_epoch()).count();
        NDP_times.push_back(current_time - start_time);
    }
}

void Add_Infeasible_Set_To_The_List() {
    Infeasible_Elements* element = new Infeasible_Elements;
    element->Sum_BOX_UB_Components = DCM_nodes_list.front()->Sum_BOX_UB_Components;
    for (int i = 0; i < N_Objectives; i++) {
        element->Box_UB_Point[i] = DCM_nodes_list.front()->Box_UB_Point[i];
    }
    for (int i = 0; i < DCM_nodes_list.front()->Y_tilde_list.size(); i++) {
        {
            Y_tilde_class* vertex = new Y_tilde_class;
            for (int j = 0; j < N_Objectives; j++) {
                vertex->point_y[j] = DCM_nodes_list.front()->Y_tilde_list.at(i)->point_y[j];
            }
            element->Y_tilde_list.push_back(vertex);
        }
    }
    if (DCM_nodes_list.front()->y_point[0] > Null + 1) {
        {
            Y_tilde_class* vertex = new Y_tilde_class;
            for (int j = 0; j < N_Objectives; j++) {
                vertex->point_y[j] = DCM_nodes_list.front()->y_point[j];
            }
            element->Y_tilde_list.push_back(vertex);
        }
    }
    Element_Is_Assigned = 0;
    for (int i = 0; i < Infeasible_Elements_list.size(); i++) {
        if (Infeasible_Elements_list.at(i)->Sum_BOX_UB_Components < element->Sum_BOX_UB_Components) {
            Element_Is_Assigned = 1;
            Infeasible_Elements_list.insert(Infeasible_Elements_list.begin() + i, element);
            break;
        }
    }
    if (Element_Is_Assigned == 0) {
        Infeasible_Elements_list.push_back(element);
    }
}

void Generate_Binary_Variables() {
    Binary_Var.end();
    if (DCM_nodes_list.front()->y_point[0] > Null + 1) {
        Binary_Var = NumVar2DArray(env, DCM_nodes_list.front()->Y_tilde_list.size() + 1);
        for (int i = 0; i < DCM_nodes_list.front()->Y_tilde_list.size() + 1; i++) {
            Binary_Var[i] = IloNumVarArray(env, N_Objectives, 0.0, 1.0, ILOBOOL);
        }
    } else {
        Binary_Var = NumVar2DArray(env, DCM_nodes_list.front()->Y_tilde_list.size());
        for (int i = 0; i < DCM_nodes_list.front()->Y_tilde_list.size(); i++) {
            Binary_Var[i] = IloNumVarArray(env, N_Objectives, 0.0, 1.0, ILOBOOL);
        }
    }
}

void Adding_Disjunctive_Constraints() {
    Number_Disjunctive_Constraints = 0;
    Number_Sets_of_Disjunctive_Constraints = 0;
    for (int i = 0; i < DCM_nodes_list.front()->Y_tilde_list.size(); i++) {
        Expr.end();
        Expr = IloExpr(env);
        counter = 0;
        for (int j = 0; j < N_Objectives; j++) {
            if (DCM_nodes_list.front()->Y_tilde_list.at(i)->point_y[j] > Negative_Infty + Hybrid_eps) {
                counter++;
                Extra_Constraints.add(z_i_x[j]-(DCM_nodes_list.front()->Y_tilde_list.at(i)->point_y[j] - Finding_epsilon_value(DCM_nodes_list.front()->Y_tilde_list.at(i)->point_y[j]) - eps_modifier - Max_z_i_x[j]) * Binary_Var[i][j] - Max_z_i_x[j] <= 0);
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
    if (DCM_nodes_list.front()->y_point[0] > Null + 1) {
        Expr.end();
        Expr = IloExpr(env);
        counter = 0;
        for (int j = 0; j < N_Objectives; j++) {
            if (DCM_nodes_list.front()->y_point[j] > Negative_Infty + Hybrid_eps) {
                counter++;
                Extra_Constraints.add(z_i_x[j]-(DCM_nodes_list.front()->y_point[j] - Finding_epsilon_value(DCM_nodes_list.front()->y_point[j]) - eps_modifier - Max_z_i_x[j]) * Binary_Var[DCM_nodes_list.front()->Y_tilde_list.size()][j] - Max_z_i_x[j] <= 0);
                Expr += Binary_Var[DCM_nodes_list.front()->Y_tilde_list.size()][j];
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
    model.add(Extra_Constraints);
    Max_Number_Disjunctive_Constraints = fmax(Max_Number_Disjunctive_Constraints, Number_Disjunctive_Constraints);
    Max_Number_Sets_of_Disjunctive_Constraints = fmax(Max_Number_Sets_of_Disjunctive_Constraints, Number_Sets_of_Disjunctive_Constraints);
}

void Find_DCM_NDP() {
    Number_IPs_Solved++;
    Find_DCM_NDP_is_Feasible = 0;
    Generate_Binary_Variables();
    Adding_Disjunctive_Constraints();
    cost = IloMinimize(env, Sum_ObjFs);
    model.add(cost);
    cplex.extract(model);
    cplex.resetTime();
    cplex.setParam(IloCplex::TiLim, Time_limit_IP);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::Threads, Num_threads);
    cplex.setParam(IloCplex::EpGap, CPLEX_relative_gap);
    //cplex.exportModel("Instance.lp");
    if (cplex.solve()) {
        Find_DCM_NDP_is_Feasible = 1;
        for (int j = 0; j < N_Variables; j++) {
            x_n[j] = cplex.getValue(dec_var[j]);
        }
        for (int j = 0; j < N_Objectives; j++) {
            obj_x_n[j] = cplex.getValue(z_i_x[j]);
        }
        Add_Sort_The_NDPs_list(obj_x_n, x_n);
    } else {
        Number_Infeas_IPs_Solved++;
        First_Box_Done = 1;
        Add_Infeasible_Set_To_The_List();
    }
    cplex.clear();
    model.remove(cost);
    model.remove(Extra_Constraints);
    cost.end();
    Extra_Constraints.clear();
    startVal.clear();
}

void Update_Prority_Queue(DCM_nodes* Element_Copy) {
    Element_Is_Assigned = 0;
    for (int i = 1; i < DCM_nodes_list.size(); i++) {
        if (DCM_nodes_list.at(i)->Sum_BOX_UB_Components + Hybrid_eps < Element_Copy->Sum_BOX_UB_Components) {
            DCM_nodes_list.insert(DCM_nodes_list.begin() + i, Element_Copy);
            Element_Is_Assigned = 1;
            break;
        }
        if (DCM_nodes_list.at(i)->Sum_BOX_UB_Components < Element_Copy->Sum_BOX_UB_Components + Hybrid_eps && DCM_nodes_list.at(i)->Sum_BOX_UB_Components > Element_Copy->Sum_BOX_UB_Components - Hybrid_eps && DCM_nodes_list.at(i)->Sum_BOX_LB_Components > Element_Copy->Sum_BOX_LB_Components) {
            DCM_nodes_list.insert(DCM_nodes_list.begin() + i, Element_Copy);
            Element_Is_Assigned = 1;
            break;
        }
    }
    if (Element_Is_Assigned == 0) {
        DCM_nodes_list.push_back(Element_Copy);
    }
}

void Remove_Dominated_Points_of_Tilde_Y(DCM_nodes* Element_Copy, double* Ref_Point) {
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

void Add_The_Internal_Element_To_The_Queue() {
    DCM_nodes* element = new DCM_nodes;
    for (int i = 0; i < N_Objectives; i++) {
        element->y_L_point[i] = fmax(DCM_nodes_list.front()->y_L_point[i], fmin(DCM_nodes_list.front()->y_point[i], obj_x_n[i]));
    }
    for (int j = 0; j < DCM_nodes_list.front()->Y_tilde_list.size(); j++) {
        {
            Y_tilde_class* vertex = new Y_tilde_class;
            for (int i = 0; i < N_Objectives; i++) {
                vertex->point_y[i] = DCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[i];
            }
            vertex->Number_NonInfty_Components = DCM_nodes_list.front()->Y_tilde_list.at(j)->Number_NonInfty_Components;
            element->Y_tilde_list.push_back(vertex);
        }
    }
    counter_y_prim = 0;
    counter_y_zegon = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (obj_x_n[i] > DCM_nodes_list.front()->y_point[i] + Hybrid_eps + eps_modifier && obj_x_n[i] > element->y_L_point[i] + Hybrid_eps + eps_modifier) {
            y_prim[i] = obj_x_n[i];
        } else {
            counter_y_prim++;
            y_prim[i] = Negative_Infty;
        }
    }
    if (counter_y_prim < N_Objectives) {
        for (int i = 0; i < N_Objectives; i++) {
            if (obj_x_n[i] < DCM_nodes_list.front()->y_point[i] - Hybrid_eps - eps_modifier && DCM_nodes_list.front()->y_point[i] > element->y_L_point[i] + Hybrid_eps + eps_modifier) {
                y_zegon[i] = DCM_nodes_list.front()->y_point[i];
            } else {
                counter_y_zegon++;
                y_zegon[i] = Negative_Infty;
            }
        }
        if (counter_y_zegon < N_Objectives) {
            Remove_Dominated_Points_of_Tilde_Y(element, y_prim);
            Remove_Dominated_Points_of_Tilde_Y(element, y_zegon);
            {
                Y_tilde_class* vertex = new Y_tilde_class;
                for (int i = 0; i < N_Objectives; i++) {
                    vertex->point_y[i] = y_prim[i];
                }
                vertex->Compute_Number_NonInfty_Components();
                element->Y_tilde_list.push_back(vertex);
            }
            {
                Y_tilde_class* vertex = new Y_tilde_class;
                for (int i = 0; i < N_Objectives; i++) {
                    vertex->point_y[i] = y_zegon[i];
                }
                vertex->Compute_Number_NonInfty_Components();
                element->Y_tilde_list.push_back(vertex);
            }
            element->Computing_BOX_UB_Point();
            Update_Prority_Queue(element);
        }
    }
    if (counter_y_prim == N_Objectives || counter_y_zegon == N_Objectives) {
        element->~DCM_nodes();
    }
}

void Add_The_External_Element_To_The_Queue() {
    DCM_nodes* element = new DCM_nodes;
    counter = 0;
    for (int i = 0; i < N_Objectives; i++) {
        if (DCM_nodes_list.front()->y_point[i] < Null + 1) {
            element->y_point[i] = fmax(DCM_nodes_list.front()->y_L_point[i], obj_x_n[i]);
        } else {
            element->y_point[i] = fmax(DCM_nodes_list.front()->y_L_point[i], fmin(DCM_nodes_list.front()->y_point[i], obj_x_n[i]));
        }
        if (element->y_point[i] > DCM_nodes_list.front()->y_L_point[i] - Hybrid_eps - eps_modifier && element->y_point[i] < DCM_nodes_list.front()->y_L_point[i] + Hybrid_eps + eps_modifier) {
            element->y_point[i] = Negative_Infty;
            counter++;
        }
        element->y_L_point[i] = DCM_nodes_list.front()->y_L_point[i];
    }
    if (counter == N_Objectives) {
        element->~DCM_nodes();
    } else {
        for (int j = 0; j < DCM_nodes_list.front()->Y_tilde_list.size(); j++) {
            {
                Y_tilde_class* vertex = new Y_tilde_class;
                for (int i = 0; i < N_Objectives; i++) {
                    vertex->point_y[i] = DCM_nodes_list.front()->Y_tilde_list.at(j)->point_y[i];
                }
                vertex->Number_NonInfty_Components = DCM_nodes_list.front()->Y_tilde_list.at(j)->Number_NonInfty_Components;
                element->Y_tilde_list.push_back(vertex);
            }
        }
        Remove_Dominated_Points_of_Tilde_Y(element, element->y_point);
        element->Computing_BOX_UB_Point();
        Update_Prority_Queue(element);
    }
}

void Write_Results(char* output) {
    gettimeofday(&endTime, NULL);
    clock_Run_time += (clock() - clock_start_time);
    totalTime += ((endTime.tv_sec - startTime.tv_sec) * 1000000L);
    totalTime += (endTime.tv_usec - startTime.tv_usec);
    Result.open(output);
    Result << LP_file_name << endl;
    Result << "Number_of_NDPs: " << NDPs_list.size() << endl;
    Result << "Number_of_IPs: " << Number_IPs_Solved << endl;
    Result << "Number_of_Infeasible_IPs: " << Number_Infeas_IPs_Solved << endl;
    Result << "Number_of_Avoided_Infeasible_IPs: " << N_Saved_Infeasible_IP << endl;
    Result << "Wall_Time(s): " << (totalTime / 1000000L) << endl;
    Result << "CPU_Time(s): " << (clock_Run_time / CLOCKS_PER_SEC) << endl;
    Result << "Number_of_Investigated_Elements: " << N_Investigated_Elements << endl;
    Result << "Number_of_Elements_Remained_In_Priority_Queue: " << DCM_nodes_list.size() << endl;
    Result << "Max_Number_of_BigMConstraints_added_to_a_Model: " << Max_Number_Disjunctive_Constraints << endl;
    Result << "Max_Number_of_Sets_of_Disjunctive_Constraints_added_to_a_Model: " << Max_Number_Sets_of_Disjunctive_Constraints << endl;
    Result << "-----------------------------------------------------------------" << endl;
    for (int i = 0; i < NDPs_list.size(); i++) {
        Result << "NDP " << i + 1 << ": ";
        for (int j = 0; j < N_Objectives; j++) {
            Result << NDPs_list.at(i)->Obj_Value[j] << " ";
        }
        Result << "Time: " << NDP_times.at(i)<< endl;
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

void DCM_Optimizer(char* output) {
    Initialize_queue();
    while (DCM_nodes_list.size() && Termination_Time < Algorithm_Time_limit) {
        N_Investigated_Elements++;
        if (First_Box_Done == 1) {
            Infeasible_Relaxation_Exist = 0;
            Search_For_Feasible_Solution();
            if (Feasible_Solution_Exist == 1) {
                Find_DCM_NDP_is_Feasible = 1;
            } else {
                Searching_Among_Infeasible_Cases();
                if (Infeasible_Relaxation_Exist == 1) {
                    Find_DCM_NDP_is_Feasible = 0;
                }
            }
        }
        if (Feasible_Solution_Exist == 0 && Infeasible_Relaxation_Exist == 0) {
            Find_DCM_NDP();
        }
        if (Find_DCM_NDP_is_Feasible == 1) {
            if (DCM_nodes_list.front()->y_point[0] > Null + 1) {
                Add_The_Internal_Element_To_The_Queue();
            }
            Add_The_External_Element_To_The_Queue();
        }
        delete DCM_nodes_list.front();
        DCM_nodes_list.erase(DCM_nodes_list.begin());
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
            Write_Results(output);
        }
    }
    Write_Results(output);
    std::cout << "Finish!!!!" << std::endl;
}
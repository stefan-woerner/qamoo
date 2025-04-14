//This source file generate necessary global variable for CPLEX and C++ model
#include"Resetting_functions.h"
#include "extern_parameters.h"

void generating_all_variables() {
    generating_cplex_variables();
    generating_general_variables();
    Solving_LP_Relaxation_To_Obtain_Bounds();
}

void generating_cplex_variables() {
    cplex.importModel(model, LP_file_name, object, dec_var, rngs, sos1, sos2);
    N_Variables = dec_var.getSize();
    f_x.end();
    f_x = IloExpr(env);
    z_i_x.end();
    z_i_x = IloExprArray(env, N_Objectives);
    f_x = object.getExpr();
    for (int i = 0; i < N_Objectives; i++) {
        z_i_x[i] = rngs[i].getExpr();
    }
    for (int i = 0; i < N_Objectives; i++) {
        model.remove(rngs[i]);
    }
    model.remove(object);
    for (int i = 0; i < N_Variables; i++) {
        startVar.add(dec_var[i]);
    }
    Sum_ObjFs.end();
    Sum_ObjFs = IloExpr(env);
    for (int i = 0; i < N_Objectives; i++) {
        Sum_ObjFs += z_i_x[i];
    }
}

void generating_general_variables() {
    Max_z_i_x = new double[N_Objectives];
    Min_z_i_x = new double[N_Objectives];
    x_l = new double[N_Variables];
    x_f = new double[N_Variables];
    x_n = new double[N_Variables];
    x_u = new double[N_Variables];
    obj_x_n = new double[N_Objectives];
    obj_x_f = new double[N_Objectives];
    y_prim = new double[N_Objectives];
    y_zegon = new double[N_Objectives];
    Nadir_Value = new double[N_Objectives];
    Nadir_Bound = new double[N_Objectives];
    Nadir_Gap = new double[N_Objectives];
    for (int i = 0; i < N_Variables; i++) {
        x_u[i] = Null;
    }
    for (int i = 0; i < N_Objectives; i++) {
        Nadir_Value[i] = Null*-1;
        Nadir_Bound[i] = Null;
        Nadir_Gap[i] = 100;
    }
}

void Solving_LP_Relaxation_To_Obtain_Bounds() {
    model_relax.add(model);
    for (int i = 0; i < N_Variables; i++) {
        model_relax.add(IloConversion(env, dec_var[i], ILOFLOAT));
    }
    //---------------------------Min and Max for f(x)-------------------------------
    if (Optimizer_Number == 1) {
        cost = IloMaximize(env, f_x);
        model_relax.add(cost);
        cplex_relax.extract(model_relax);
        cplex_relax.setOut(env.getNullStream());
        cplex_relax.setParam(IloCplex::EpGap, CPLEX_relative_gap);
        if (cplex_relax.solve()) {
            Max_f_x = cplex_relax.getObjValue();
        } else {
            cout << "Problem is infeasible!" << endl;
            cout << "LP-relaxation returned Null!" << endl;
            exit(EXIT_FAILURE);
        }
        cplex_relax.clear();
        model_relax.remove(cost);
        cost.end();
        cost = IloMinimize(env, f_x);
        model_relax.add(cost);
        cplex_relax.extract(model_relax);
        cplex_relax.setOut(env.getNullStream());
        cplex_relax.setParam(IloCplex::EpGap, CPLEX_relative_gap);
        if (cplex_relax.solve()) {
            Min_f_x = cplex_relax.getObjValue();
        } else {
            cout << "Problem is infeasible!" << endl;
            cout << "LP-relaxation returned Null!" << endl;
            exit(EXIT_FAILURE);
        }
        cplex_relax.clear();
        model_relax.remove(cost);
        cost.end();
    }
    //-------------------------------Max for z_i(x)---------------------------------
    for (int i = 0; i < N_Objectives; i++) {
        cost = IloMaximize(env, z_i_x[i]);
        model_relax.add(cost);
        cplex_relax.extract(model_relax);
        cplex_relax.setOut(env.getNullStream());
        cplex_relax.setParam(IloCplex::EpGap, CPLEX_relative_gap);
        if (cplex_relax.solve()) {
            Max_z_i_x[i] = cplex_relax.getObjValue();
        } else {
            cout << "Problem is infeasible!" << endl;
            cout << "LP-relaxation returned Null!" << endl;
            exit(EXIT_FAILURE);
        }
        cplex_relax.clear();
        model_relax.remove(cost);
        cost.end();
    }
    //-------------------------------Max for z_i(x)---------------------------------
    if (Optimizer_Number > 1) {
        for (int i = 0; i < N_Objectives; i++) {
            cost = IloMinimize(env, z_i_x[i]);
            model_relax.add(cost);
            cplex_relax.extract(model_relax);
            cplex_relax.setOut(env.getNullStream());
            cplex_relax.setParam(IloCplex::EpGap, CPLEX_relative_gap);
            if (cplex_relax.solve()) {
                Min_z_i_x[i] = cplex_relax.getObjValue();
            } else {
                cout << "Problem is infeasible!" << endl;
                cout << "LP-relaxation returned Null!" << endl;
                exit(EXIT_FAILURE);
            }
            cplex_relax.clear();
            model_relax.remove(cost);
            cost.end();
        }
    }
    model_relax.end();
    cplex_relax.end();
}






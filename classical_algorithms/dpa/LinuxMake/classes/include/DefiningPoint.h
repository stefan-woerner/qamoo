#pragma once

#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>

#include "Efficient.h"
#include "Box.h"

#include <ilcplex/ilocplex.h>

using namespace std;

ILOSTLBEGIN


class DefiningPoint {
private:
	// fields
	bool verbose;
	int numCallsToCplex = 0;
	int numConstraints = 0;

	IloEnv env;
	IloCplex cplex;
	IloModel epsmodel;

	vector<IloExpr> objectives;
	IloConstraintArray constraints;
	IloObjective objectiveFunction;
	deque<IloRange> rangeForObjectiveFunction;
	IloExpr allObjectiveFunction;
	IloExpr augmentedObjectiveFunction;
    IloNumVarArray variables;

	int* idealPoint = new int[0];
	int* globalUpperBound = new int[0];

	deque<Efficient*> setOfNondominatedSolutions;
    deque<double> progress;
    deque<IloNumArray> solutions;
	deque<Box*> boxes;
	deque<int> indexList;

// methods
	void ExtractConstraintsAndObjectives(IloModel& inputModel);
	void ExtractNumObjectivesAndNumConstraints(IloModel& inputModel);
	void InitConstraintsAndObjectives(IloModel& inputModel);
	IloModel CreateModelObject();

	bool ComputeGlobalBounds();
	void InitListOfBoxes();
	int  ComputeNondominatedSet(bool useEconstraint, bool augmented, int indexEc, double timeout);
	
	int  SelectBoxToRefine(bool useEconstraint, int indexEc);

	// methods related to scalarization
	bool Solve(bool augmented, int indexEc, const int* rhs, int* obj);
	bool SolveAugmentedEconstraint(int indexEc, const int* rhs, int* obj);
	void ConstructAugmentedObj(int indexEc);
	bool SolveTwoStageEconstraint(int k, const int* rhs, int* obj);
	bool SolveFirstStageEconstraint(int indexEc, const int* rhs);
	void SolveSecondStageEconstraint(int * obj, int indexEc);	

	void AppendNewNondomPoint(const int* obj);

	// methods related to box management 
	void UpdateListOfBoxes(int indexEc, const int* obj, bool useEconstraint, int indexSelectedBox);
	deque<int> FindBoxesToBeUpdated(const int* obj);
	bool IsContainedInBox(const int* obj, int index);
	bool AtBoundaryOfBox(const int* obj, int index);
	int  ComputeSplitCriterion(int idx, int indexj);
	void CreateNewBox(const int* obj, int idx, int indexj, int dimension);
	void RemoveOldBoxesFromList();
	
	void LogResults(int numInterations);
	
	void DestructBoxesStructure();
	void DestructListNondominatedPoints();

public:
	int numObjectives=0;
    int solutionTime;
	bool isSenseMaximum = false;
	DefiningPoint(bool verbose);
	~DefiningPoint(void);
	void ImportProblemSpecification(const char* fileName);
	void Compute(bool useEconstraint, bool augmented, int indexEc, double timeout);
	void ExportNonDominatedPointsToFile(const std::string fileName);
};












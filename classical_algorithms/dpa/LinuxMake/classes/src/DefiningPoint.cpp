#include "DefiningPoint.h"
#include <fstream>
#include <iostream>
#include <string.h>
#include <numeric>
#include <map>
#include <sys/types.h>
#include <time.h>

using namespace std;

IloCplex createCplex(IloEnv env) {
	IloCplex result = IloCplex(env);

	// parameter settings taken from Kirlik & Sayin(2014)
	result.setParam(IloCplex::NodeLim, 1000000000);	//MIP node limit	
	result.setParam(IloCplex::TreLim, 1000000000);	//tree memory limit
	result.setParam(IloCplex::EpGap, 0.0);	//relative MIP gap tolerance
	result.setParam(IloCplex::ItLim, 1000000000);	//absolute MIP gap iteration limit
	result.setParam(IloCplex::MIPDisplay, 0);	//Display option
	result.setParam(IloCplex::Threads, 1); 		//number of threads
	result.setParam(IloCplex::AdvInd, 1);
	result.setOut(env.getNullStream());// suppress output
	return result;
}

int* createPoint(const int* obj, int dimension)
{
	int* point = new int[dimension];
	for (int k = 0; k < dimension; k++)
		point[k] = obj[k];

	return point;
}

Box* createBox(int dimension)
{
	Box* box = new Box();
	box->u = new int[dimension];
	box->listindexdefpoints.resize(dimension);
	return box;
}


// constructor / destructor
DefiningPoint::DefiningPoint(bool verboseMode) : verbose(verboseMode) {
	cplex = createCplex(env);
}

DefiningPoint::~DefiningPoint(void) {
	cplex.end();
	env.end();
	delete[] idealPoint;
	delete[] globalUpperBound;
	DestructBoxesStructure();
	DestructListNondominatedPoints();
}

void DefiningPoint::DestructListNondominatedPoints()
{
	for (int i = 0; i < (int)setOfNondominatedSolutions.size(); i++) {
		delete[] setOfNondominatedSolutions[i]->f;
		delete setOfNondominatedSolutions[i];
	}
	setOfNondominatedSolutions.clear();
}

void DefiningPoint::DestructBoxesStructure()
{
	for (int i = 0; i < (int)boxes.size(); i++) {
		delete[] boxes[i]->u;
		boxes[i]->listindexdefpoints.clear();
		delete boxes[i];
	}
	boxes.clear();
}

void DefiningPoint::ImportProblemSpecification(const char* fileName) {

	IloModel inputModel(env);
	IloObjective objective(env);
	IloRangeArray ranges(env);
	IloSOS1Array sos1(env);
	IloSOS2Array sos2(env);
    variables = IloNumVarArray(env);

    try {
        cplex.importModel(inputModel, fileName, objective, variables, ranges, sos1, sos2);
    }
    catch (exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

	isSenseMaximum = objective.getSense() == -1;


	ExtractNumObjectivesAndNumConstraints(inputModel);
	InitConstraintsAndObjectives(inputModel);

	epsmodel = CreateModelObject();
	cplex.extract(epsmodel);

	if (!verbose)
		return;

	string sense = "minimize";
	if (isSenseMaximum)
		sense = "Maximize";
	cout << "Objective: " << sense << endl;

	cout << "number of objectives  :" << numObjectives << endl;
	cout << "number of variables   :" << variables.getSize() << endl;
	cout << "number of constraints :" << numConstraints << endl;
}

void DefiningPoint::ExtractConstraintsAndObjectives(IloModel& inputModel)
{
	int j = 0;
	for (IloModel::Iterator it(inputModel); it.ok(); ++it) {
		IloExtractable extr = it.operator *();

		if (!extr.isConstraint())
			continue;

		if (j < numConstraints) {
			constraints.add(extr.asConstraint());
		}
		else {
			IloRangeI* impl = dynamic_cast<IloRangeI*>((*it).asConstraint().getImpl());
			if (impl) {
				IloRange range(impl);
				IloExpr expr = range.getExpr();

				// set factor to -1 for constraints if sense is maximize
				int factor = 1;
				if (isSenseMaximum)
					factor = -1;

				for (IloExpr::LinearIterator it2 = expr.getLinearIterator(); it2.ok(); ++it2)
					objectives[(long long)j - (long long)numConstraints] += factor * it2.getCoef() * it2.getVar();
			}
		}
		j = j + 1;
	}
}

void DefiningPoint::ExtractNumObjectivesAndNumConstraints(IloModel& inputModel)
{
	IloNum result = 0;
	int total = 0;
	for (IloModel::Iterator it(inputModel); it.ok(); ++it) {
		IloExtractable extr = it.operator *();
		if (!extr.isConstraint())
			continue;

		total++;
		IloRangeI* impl = dynamic_cast<IloRangeI*>((*it).asConstraint().getImpl());
		if (impl)
		{
			if (isSenseMaximum)
				result = impl->getLb(); // Lower bound of last constraint represents/encodes number of objectives
			else
				result = impl->getUb(); // Upper bound of last constraint represents/encodes the number of objectives
		}
	}

	numObjectives = result;
	numConstraints = total - numObjectives;
}

void DefiningPoint::InitConstraintsAndObjectives(IloModel& inputModel)
{
	objectives.resize(numObjectives);
	for (int i = 0; i < numObjectives; i++)
		objectives[i] = IloExpr(env);
	constraints = IloConstraintArray(env);

	ExtractConstraintsAndObjectives(inputModel);

	// sum of all objectives (needed for second-stage problem)
	allObjectiveFunction = IloExpr(env);
	for (int i = 0; i < numObjectives; i++)
		allObjectiveFunction += objectives[i];
}

IloModel DefiningPoint::CreateModelObject()
{
	IloModel model = IloModel(env);

	// initialize objective function
	objectiveFunction = IloMinimize(env, allObjectiveFunction);
	model.add(objectiveFunction);

	// range for the objective functions
	rangeForObjectiveFunction.resize(numObjectives);
	for (int j = 0; j < numObjectives; j++) {
		rangeForObjectiveFunction[j] = IloRange(env, -IloInfinity, objectives[j], IloInfinity);
		model.add(rangeForObjectiveFunction[j]);
	}

	// set of constraints (rows)
	for (int i = 0; i < constraints.getSize(); i++)
		model.add(constraints[i]);

	return model;
}

bool DefiningPoint::ComputeGlobalBounds() {

	idealPoint = new int[numObjectives];
	globalUpperBound = new int[numObjectives];

	// compute ideal point
	objectiveFunction.setSense(IloObjective::Minimize);
	for (int j = 0; j < numObjectives; j++) {

		objectiveFunction.setExpr(objectives[j]);
		cplex.solve();
		numCallsToCplex = numCallsToCplex + 1;

		if (cplex.getCplexStatus() == CPX_STAT_INFEASIBLE)
			return false;

		idealPoint[j] = (int)floor(cplex.getObjValue() + 0.5);
	}

	// compute global upper bound
	objectiveFunction.setSense(IloObjective::Maximize);
	for (int j = 0; j < numObjectives; j++) {
		
		objectiveFunction.setExpr(objectives[j]);
		cplex.solve();
		numCallsToCplex = numCallsToCplex + 1;

		if (cplex.getCplexStatus() == CPXMIP_UNBOUNDED) {
			// set as infinity
			globalUpperBound[j] = numeric_limits<int>::max();
		}
		else {
			globalUpperBound[j] = (int)floor(cplex.getObjValue() + 0.5);
			globalUpperBound[j] = globalUpperBound[j] + 1;
		}
	}

	objectiveFunction.setSense(IloObjective::Minimize);
	return true;
}

// Main method
void DefiningPoint::Compute(bool useEconstraint, bool augmented, int indexEc, double timeout) {
	// useEconstraint: true for using e-constraint, false: placeholder for other scalarization, currently unused
// augmented: true for augmented objective function, false for two-stage optimization
// indexEC: chooses objective function of the e-constraint model; valid indices: 0, ..., numObjectives-1

	if (!useEconstraint) throw std::invalid_argument("implementation currently only for e-constraint scalarization");
	if (indexEc >= numObjectives) throw std::invalid_argument("indexEc must be <= numObjectives");
	if (!ComputeGlobalBounds()) throw std::domain_error("Model has no feasible solution");

	InitListOfBoxes();
	int numInterations = ComputeNondominatedSet(useEconstraint, augmented, indexEc, timeout);
	LogResults(numInterations);
}

void DefiningPoint::InitListOfBoxes() {
	boxes.push_front(createBox(numObjectives));

	// initial box defined by global upper bound; defining points are dummy points (symbolized here by index 0)
	for (int j = 0; j < numObjectives; j++) {
		boxes.front()->u[j] = globalUpperBound[j];
		boxes.front()->listindexdefpoints[j].push_back(0);
	}
}

int DefiningPoint::ComputeNondominatedSet(bool useEconstraint, bool augmented, int indexEc, double timeout) {

    auto start = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::duration<double>(start.time_since_epoch()).count();

	assert(indexEc <= numObjectives);

	int* obj = new int[numObjectives];
	int* rhs = new int[numObjectives];
	int numIterations = 0;

	if (augmented)
		ConstructAugmentedObj(indexEc);

    auto current = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::duration<double>(current.time_since_epoch()).count();
    auto duration = current_time - start_time;
	// main loop: while list of boxes is not empty
	while (boxes.size() != 0 and duration <= timeout) {


		int indexSelectedBox = SelectBoxToRefine(useEconstraint, indexEc);

		// get rhs and solve model 
		for (int j = 0; j < numObjectives; j++) {
			rhs[j] = boxes[indexSelectedBox]->u[j];
		}
		bool isFeasible = Solve(augmented, indexEc, rhs, obj);
        current = std::chrono::high_resolution_clock::now();
        current_time = std::chrono::duration<double>(current.time_since_epoch()).count();
        duration = current_time - start_time;

		if (isFeasible == true) {
			AppendNewNondomPoint(obj);
			UpdateListOfBoxes(indexEc, obj, useEconstraint, indexSelectedBox);
            // get solutions
            IloNumArray x(env);
            cplex.getValues(x, variables);
            solutions.push_back(x);
            progress.push_back(duration);
		}
		else
			boxes.erase(boxes.cbegin() + indexSelectedBox);
		numIterations++;
		if (verbose) std::cout << "Iteration :" << numIterations << endl;


	}
	delete[] obj;
	delete[] rhs;
	return numIterations;
}

int DefiningPoint::SelectBoxToRefine(bool useEconstraint, int indexEc) {
	// first box in list can be used if scalarization is not e-constraint, currently not activated
	if (!useEconstraint)
		return 0;

	// find box with smallest value u[indexEc] (not necessarily unique)
	int indexSelectedBox = 0;
	int valMinU = boxes[0]->u[indexEc];

	for (int it = 1; it < boxes.size(); it++) {
		if (boxes[it]->u[indexEc] < valMinU) {
			indexSelectedBox = it;
			valMinU = boxes[it]->u[indexEc];
		}
	}

	return indexSelectedBox;
}

bool DefiningPoint::Solve(bool augmented, int indexEc, const int* rhs, int* obj) {
	if (augmented)
		return SolveAugmentedEconstraint(indexEc, rhs, obj);
	
	return SolveTwoStageEconstraint(indexEc, rhs, obj);
}

bool DefiningPoint::SolveAugmentedEconstraint(int indexEc, const int* rhs, int* obj) {

	for (int j = 0; j < numObjectives; j++) {
		if (j == indexEc) {
			rangeForObjectiveFunction[j].setUB(globalUpperBound[j]);
		}
		else {
			rangeForObjectiveFunction[j].setUB(rhs[j] - 1);
		}
	}

	objectiveFunction.setExpr(augmentedObjectiveFunction);
	numCallsToCplex = numCallsToCplex + 1;
	cplex.solve();

	// solve model and check the feasibility of the model
	if (cplex.getCplexStatus() == IloCplex::Infeasible) {
		return false;
	}

	for (int j = 0; j < numObjectives; j++) {
		double val = cplex.getValue(objectives[j]) + 0.5;
		obj[j] = (int)floor(val);
	}

	// in case of feasibility, check if new point has been found
	return obj[indexEc] < rhs[indexEc] - 0.5;
}

void DefiningPoint::ConstructAugmentedObj(int indexEc) {

	// compute rho 
	double rhoDenominator = 0.0;
	for (int r = 0; r < numObjectives; r++) {
		if (r != indexEc)
			rhoDenominator += globalUpperBound[r] - idealPoint[r] - 1;
	}

	// parameter choice: choose enumerator larger than 0 and smaller than 1
	double rho = 0.9 / rhoDenominator;

	augmentedObjectiveFunction = IloExpr(env);
	// global setting of rho, possible future improvement: adaptive selection
	augmentedObjectiveFunction = rho * allObjectiveFunction + (1 - rho) * objectives[indexEc];
}

bool DefiningPoint::SolveTwoStageEconstraint(int indexEc, const int* rhs, int* obj) {

    bool feasible = SolveFirstStageEconstraint(indexEc, rhs);
	if (!feasible) {
        return false;
    }


	// if new, feasible solution has been detected in first stage, then solve second stage problem
	SolveSecondStageEconstraint(obj, indexEc);
	return true;
}

bool DefiningPoint::SolveFirstStageEconstraint(int indexEc, const int* rhs) {
	objectiveFunction.setExpr(objectives[indexEc]);

	for (int j = 0; j < numObjectives; j++) {
		if (j == indexEc) {
            rangeForObjectiveFunction[j].setUB(globalUpperBound[j]);
        }
		else {
            rangeForObjectiveFunction[j].setUB(rhs[j] - 1);
        }

	}
	cplex.solve();
	numCallsToCplex = numCallsToCplex + 1;

	return !(cplex.getCplexStatus() == IloCplex::Infeasible || cplex.getObjValue() >= rhs[indexEc] - 0.5);
}

void DefiningPoint::SolveSecondStageEconstraint(int* obj, int indexEc) {

	// second stage is only solved if first stage is feasible and yields new point;
	// therefore, second stage is always feasible by construction

	// get objective function value from first stage
	obj[indexEc] = (int)floor(cplex.getObjValue() + 0.5);
	rangeForObjectiveFunction[indexEc].setUB(obj[indexEc]);
	objectiveFunction.setExpr(allObjectiveFunction);

	cplex.solve();
	numCallsToCplex = numCallsToCplex + 1;
	// get objective function values
	for (int j = 0; j < numObjectives; j++) {
		if (j != indexEc) {
			obj[j] = (int)floor(cplex.getValue(objectives[j]) + 0.5);
		}
	}
}

void DefiningPoint::AppendNewNondomPoint(const int* obj) {

	Efficient* point = new Efficient();
	point->f = new int[numObjectives];

	for (int j = 0; j < numObjectives; j++)
		point->f[j] = obj[j];

	setOfNondominatedSolutions.push_back(point);
}

void DefiningPoint::UpdateListOfBoxes(int indexEc, const int* obj, bool useEconstraint, int indexSelectedBox) {

	indexList = FindBoxesToBeUpdated(obj);
	for (int i = 0; i < indexList.size(); i++) {
		int index = indexList.at(i);
		for (int j = 0; j < numObjectives; j++) {
			int zmax = ComputeSplitCriterion(index, j);
			if (obj[j] > zmax + 0.5) {
				CreateNewBox(obj, index, j, numObjectives);
				if (j == indexEc && useEconstraint && index == indexSelectedBox)
					boxes.pop_back();
			}
		}
	}
	RemoveOldBoxesFromList();
	indexList.clear();
}

deque<int> DefiningPoint::FindBoxesToBeUpdated(const int* obj) {

	for (int i = 0; i < (int)boxes.size(); i++) {
		if (IsContainedInBox(obj, i)) {
			indexList.push_back(i);
			continue;
		}

		if (!AtBoundaryOfBox(obj, i))
			continue;

		for (int j = 0; j < numObjectives; j++) {
			// test for equality in component j
			if (obj[j] < boxes[i]->u[j] + 0.5 && obj[j] > boxes[i]->u[j] - 0.5)
				boxes[i]->listindexdefpoints[j].push_back(setOfNondominatedSolutions.size());
		}
	}

	return indexList;
}

bool DefiningPoint::IsContainedInBox(const int* obj, int index)
{
	for (int j = 0; j < numObjectives; j++) {
		if (obj[j] >= boxes[index]->u[j] - 0.5) {
			return false;
		}
	}
	return true;
}

bool DefiningPoint::AtBoundaryOfBox(const int* obj, int index)
{
	// note that the case that point is in box is already excluded
	for (int j = 0; j < numObjectives; j++) {
		if (obj[j] >= boxes[index]->u[j] + 0.5) {
			return false;
		}
	}
	return true;
}

int DefiningPoint::ComputeSplitCriterion(int idx, int indexj) {
	// compute "zmax-value" for split criterion according to Klamroth, Lacour, Vanderpooten (2015)
	int zmax = idealPoint[indexj] - 1;

	for (int k = 0; k < numObjectives; k++) {
		if (indexj == k)
			continue;

		int listsize = boxes[idx]->listindexdefpoints[k].size();
		int zmaxtmp = globalUpperBound[indexj];
		for (int it = 0; it < listsize; it++)
		{
			int currentindex = boxes[idx]->listindexdefpoints[k].at(it);
			int currentelementidx = idealPoint[indexj];

			if (currentindex > 0)
				currentelementidx = setOfNondominatedSolutions[(long long)currentindex - 1]->f[indexj];

			if (currentelementidx < zmaxtmp)
				zmaxtmp = currentelementidx;
		}

		if (zmaxtmp > zmax)
			zmax = zmaxtmp;
	}

	return zmax;
}

void DefiningPoint::CreateNewBox(const int* obj, int idx, int indexj, int dimension) {

	boxes.push_back(createBox(dimension));

	for (int k = 0; k < numObjectives; k++) {
		if (k != indexj) {
			boxes.back()->u[k] = boxes[idx]->u[k];

			int listsize = boxes[idx]->listindexdefpoints[k].size();
			for (int it = 0; it < listsize; it++)
			{
				int currentindex = boxes[idx]->listindexdefpoints[k].at(it);
				int currentelementidx = idealPoint[indexj];

				// check whether defining point is not a dummy point
				if (currentindex > 0)
					currentelementidx = setOfNondominatedSolutions[(long long)currentindex - 1]->f[indexj];
				if (currentelementidx < obj[indexj] - 0.5)
					boxes.back()->listindexdefpoints[k].push_back(currentindex);
			}
		}
		else {
			boxes.back()->u[indexj] = obj[indexj];
			int nsize = setOfNondominatedSolutions.size();
			boxes.back()->listindexdefpoints[k].push_back(nsize);
		}
	}
}

void DefiningPoint::RemoveOldBoxesFromList()
{
	map<int, int> indexMap;
	for (int i = 0; i < indexList.size(); i++)
		indexMap.emplace(indexList.at(i), indexList.at(i));

	deque<Box*> tmp;
	for (int i = 0; i < boxes.size(); i++) {
		bool contains = indexMap.count(i) > 0;
		if (!contains)
			tmp.push_back(boxes[i]);
	}
	boxes = tmp;
}

void DefiningPoint::LogResults(int numInterations)
{
	std::cout << "Number of nondominated points: " << setOfNondominatedSolutions.size() << endl;
	std::cout << "Iterations: " << numInterations << endl;
	std::cout << "Calls to CPLEX: " << numCallsToCplex << endl;
}

void DefiningPoint::ExportNonDominatedPointsToFile(const std::string fileName) {

	// set factor to -1 for constraints, if sense is maximize
	int factor = 1;
	if (isSenseMaximum)
		factor = -1;

    string fileObjectives = fileName + ".sol";
	ofstream outputFile(fileObjectives, ofstream::out);
	for (int i = 0; i < (int)setOfNondominatedSolutions.size(); i++) {
		for (int j = 0; j < numObjectives; j++) {
			outputFile << factor * setOfNondominatedSolutions[i]->f[j] << "\t";
		}
		outputFile << endl;
	}

	constexpr int width = 8;
	constexpr int precision = 3;
	outputFile << std::endl << "---" << std::endl;
	outputFile << std::setw(width) << std::setprecision(precision) << std::fixed;
	outputFile << numCallsToCplex << " IPs solved" << endl;
	outputFile << std::setw(width) << std::setprecision(precision) << std::fixed;
	outputFile << setOfNondominatedSolutions.size() << " Solutions found" << std::endl;
    outputFile << std::setw(width) << std::setprecision(precision) << std::fixed;
    outputFile << solutionTime << " seconds" << std::endl;

	if (verbose) std::cout << "Saved file :" << fileName << endl;
	outputFile.close();

    string fileProgress = fileName + "_progress.sol";
    ofstream outputProgress(fileProgress, ofstream::out);
    for (int i = 0; i < (int)progress.size(); i++) {
        outputProgress << progress[i] << endl;
    }
    outputProgress.close();

    string fileSolution = fileName + "_x.sol";
    ofstream outputX(fileSolution, ofstream::out);
    for (int i = 0; i < (int)solutions.size(); i++) {
        outputX << solutions[i] << endl;
    }
    outputX.close();
    cout << solutions.size() << endl;

}

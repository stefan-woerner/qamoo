//#define PRG_NAME "A Simple,  Efficient and Versatile Objective Space Algorithm for Multiobjective Integer Programming" 
//#define CPY_RGHT "Copyright © 2019-2021" 
//#define AUTHOR "Kerstin Daechert and Tino Fleuren and Kathrin Klamroth" 

#include "DefiningPoint.h"
#include <string>
#include <numeric>
#include <deque> 
#include <iostream>
#include "main.h"

using namespace std;

char* GetFileName(const char* argument) {
	char *fileName = new char[150];
	strcpy(fileName, arguments);
	return fileName;
}

int main(int argc, char** argv) {

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <file name> [-v]" << std::endl;
		std::cerr << "       " << " -v .. verbose" << std::endl;
		std::cerr << "       " << " -a .. augmented" << std::endl;
		return 1;
	}

	std::cout << "Processing: " << argv[1] << std::endl;

	const char* fileName = GetFileName(argv[1]);
	bool augmented = false;
	bool verbose = false;

	if (argc > 2 && strcmp(argv[2], "-v") == 0)
		verbose = true;
	if (argc > 2 && strcmp(argv[2], "-a") == 0)
		augmented = true;
	if (argc > 3 && strcmp(argv[3], "-v") == 0)
		verbose = true;
	if (argc > 3 && strcmp(argv[3], "-a") == 0)
		augmented = true;


	DefiningPoint df = DefiningPoint(verbose);
	try {
		// input
		df.ImportProblemSpecification(fileName);

		// arguments:
		// scalarization method (currently only econstraint-method implemented)
		// scalarization variant (augmented or not)
		// selected index for econstraint-method
		df.Compute(true, augmented, df.numObjectives - 1);


		// output
		std::string resultFileName(fileName);
		resultFileName.append(".sol");
		df.ExportNonDominatedPointsToFile(resultFileName);
	}
	catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}



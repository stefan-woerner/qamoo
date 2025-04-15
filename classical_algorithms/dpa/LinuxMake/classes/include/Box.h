#include <deque>
#include <vector>

using namespace std;


// structure for boxes:
// vector u of integers (called local upper bound) and
// vector of lists of integers (called defining points in respective component)
// note that we do not store the defining points themselves but their indices in the list of nondominated solutions 

struct Box
{
	int* u;
	vector<deque<int>> listindexdefpoints;
};





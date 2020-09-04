#include <iostream>
#include<fstream>
#include"RTree.h"
#include<time.h>
#include<cstdlib>
using std::ifstream;
using std::rand;

int main() {
	ifstream ifs("D:\\projects\\RTree\\code\\dataset\\uniform100k.txt", std::ifstream::in);
	RTree* tree = ConstructTree(100, 40);
	SetDefaultInsertStrategy(tree, 0);
	SetDefaultSplitStrategy(tree, 4);
	for (int i = 0; i < 100000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle* rectangle = InsertRec(tree, l, r, b, t);
		srand(time(NULL));
		int split_strategy = rand() % 5;
		SetDefaultSplitStrategy(tree, split_strategy);
		DefaultInsert(tree, rectangle);
	}
	ifs.close();
	ifs.open("D:\\projects\\RTree\\code\\dataset\\query1k.txt", std::ifstream::in);
	int total_access = 0;
	for (int i = 0; i < 1000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		int node_access = QueryRectangle(tree, l, r, b, t);
		total_access += node_access;
	}
	cout << "insert strategy " << tree->insert_strategy_ << " split strategy " << tree->split_strategy_ << endl;
	cout << "average node access is " << 1.0 * total_access / 1000 << endl;
	return 0;	
}

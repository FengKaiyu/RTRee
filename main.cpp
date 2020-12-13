#include <iostream>
#include<fstream>
#include"RTree.h"
#include<time.h>
#include<cstdlib>
#include<sstream>
using std::ifstream;
using std::ofstream;
using std::rand;
using std::stringstream;

void TestBaseline(int insert_strategy, int split_strategy) {
	//stringstream ss;
	//ss << "d:\\projects\\RTree\\code\\log\\" << insert_strategy << "_" << split_strategy << ".txt";
	RTree* tree = ConstructTree(50, 20);
	SetDefaultInsertStrategy(tree, insert_strategy);
	SetDefaultSplitStrategy(tree, split_strategy);
	int total_access = 0;
	ifstream ifs("./dataset/skew100k.txt", std::ifstream::in);
	for (int i = 0; i < 100000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle* rectangle = InsertRec(tree, l, r, b, t);
		//srand(time(NULL));
		//int split_strategy = rand() % 5;
		//SetDefaultSplitStrategy(tree, split_strategy);
		
		DefaultInsert(tree, rectangle);
		//TreeNode* node = RRInsert(tree, rectangle);

		//DirectSplit(tree, node);
		//cout<<"isnerted"<<endl;
		//RRSplit(tree, node);
		//cout<<"splitted"<<endl;
	}
	cout<<"average node area: "<<AverageNodeArea(tree)<<endl;
	cout<<"average node entry_num: "<<AverageNodeChildren(tree)<<endl;
	cout<<"total tree node num: "<<TotalTreeNode(tree)<<endl;
	ifs.close();
	ifs.open("./dataset/query1k.txt", std::ifstream::in);
	ofstream ofs("base.txt", std::ofstream::out);
	for (int i = 0; i < 1000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		int node_access = QueryRectangle(tree, l, r, b, t);
		total_access += node_access;
		ofs << tree->result_count << endl;
	}
	ofs.close();
	ifs.close();
	Clear(tree);
	cout << "insert strategy " << tree->insert_strategy_ << " split strategy " << tree->split_strategy_ << endl;
	cout << "average node access is " << 1.0 * total_access / 1000 << endl;
}

int main() {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 5; j++) {
			TestBaseline(i, j);
		}
	}
	return 0;	
}

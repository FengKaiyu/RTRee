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

void DataLoader(vector<Rectangle>& rectangles) {
	rectangles.resize(10000);
	ifstream ifs("d:\\projects\\RLRTree\\dataset\\uniform10k.txt", std::ifstream::in);
	for (int i = 0; i < 10000; i++) {
		double l, r, b, t;
		ifs >> rectangles[i].left_ >> rectangles[i].right_ >> rectangles[i].bottom_ >> rectangles[i].top_;
	}
	ifs.close();
}
int NaiveVerifier(vector<Rectangle>& rectangles, Rectangle& query) {
	int result = 0;
	for (int i = 0; i < rectangles.size(); i++) {
		if (rectangles[i].IsOverlap(&query)) {
			result += 1;
		}
	}
	return result;
}

void TestBaseline(int insert_strategy, int split_strategy) {
	//stringstream ss;
	//ss << "d:\\projects\\RLRTree\\dataset\\uniform10k.txt" << insert_strategy << "_" << split_strategy << ".txt";
	RTree* tree = ConstructTree(50, 20);
	SetDefaultInsertStrategy(tree, insert_strategy);
	SetDefaultSplitStrategy(tree, split_strategy);
	int total_access = 0;
	ifstream ifs("d:\\projects\\RLRTree\\dataset\\uniform10k.txt", std::ifstream::in);
	for (int i = 0; i < 10000; i++) {
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle* rectangle = InsertRec(tree, l, r, b, t);
		//srand(time(NULL));
		//int split_strategy = rand() % 5;
		//SetDefaultSplitStrategy(tree, split_strategy);
		
		DefaultInsert(tree, rectangle);
		//TreeNode* node = RRInsert(tree, rectangle);

		//DirectSplit(tree, node);
		//RRSplit(tree, node);
	}
	ifs.close();
	ifs.open("d:\\projects\\RLRTree\\dataset\\query1k.txt", std::ifstream::in);
	vector<Rectangle> rectangles;
	DataLoader(rectangles);
	ofstream ofs("verify.txt", std::ofstream::out);
	for (int i = 0; i < 1000; i++) {
		//cout<<"query "<<i<<endl;
		double l, r, b, t;
		ifs >> l >> r >> b >> t;
		Rectangle query(l, r, b, t);
		tree->Query(query);
		int result = tree->result_count;
		int verify_result = tree->VerifyQuery(query);
		int verify_result2 = NaiveVerifier(rectangles, query);
		if (result != verify_result || result != verify_result2) {
			cout << "wrong answer: " << result << " " << verify_result << endl;
		}
			ofs << "query " << i << " l: " << l << " r: " << r << " b: " << b << " t: " << t << endl;
			ofs << result << " " << verify_result << " " << verify_result2 << endl;

		//}
	}
	ofs.close();
	ifs.close();
	Clear(tree);
}

int main() {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 5; j++) {
			TestBaseline(i, j);
			exit(0);
		}
	}
	return 0;	
}

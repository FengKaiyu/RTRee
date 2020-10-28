//
// Created by Kaiyu on 2020/7/25.
//

#include "RTree.h"


int TreeNode::maximum_entry = 100;
int TreeNode::minimum_entry = 40;


Rectangle::Rectangle(const Rectangle& rectangle) {
	left_ = rectangle.Left();
	right_ = rectangle.Right();
	bottom_ = rectangle.Bottom();
	top_ = rectangle.Top();
}

void Rectangle::Include(const Rectangle &rectangle) {
	
	left_ = min(left_, rectangle.Left());
	right_ = max(right_, rectangle.Right());
	bottom_ = min(bottom_, rectangle.Bottom());
	top_ = max(top_, rectangle.Top());
}

Rectangle Rectangle::Merge(const Rectangle& rectangle) {
	double l = min(left_, rectangle.Left());
	double r = max(right_, rectangle.Right());
	double b = min(bottom_, rectangle.Bottom());
	double t = max(top_, rectangle.Top());
	Rectangle rec(l, r, b, t);
	return rec;
}

void Rectangle::Set(const Rectangle &rectangle) {
	left_ = rectangle.Left();
	right_ = rectangle.Right();
	bottom_ = rectangle.Bottom();
	top_ = rectangle.Top();
}


Rectangle Merge(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	return Rectangle(min(rectangle1.Left(), rectangle2.Left()), max(rectangle1.Right(), rectangle2.Right()), min(rectangle1.Bottom(), rectangle2.Bottom()), max(rectangle1.Top(), rectangle2.Top()));
}

void Stats::Reset() {
	node_access = 0;
	for (int i = 0; i < 5; i++) {
		action_history[i] = 0;
	}
}

void Stats::TakeAction(int action) {
	action_history[action] += 1;
}

/*
double Overlap(const Rectangle& rectangle1, const Rectangle& rectangle2){
	double left = max(rectangle1.left, rectangle2.left);
	double right = min(rectangle1.right, rectangle2.right);
	double bottom = max(rectangle1.bottom, rectangle2.bottom);
	double top = min(rectangle1.top, rectangle2.top);
	if(left < right && bottom < top){
		return (top - bottom) * (right - left);
	}
	else{
		return 0;
	}
}
*/




/*
double Area(const Rectangle& rectangle){
	return (rectangle.top - rectangle.bottom) * (rectangle.right - rectangle.left);
}

double Perimeter(const Rectangle& rectangle){
	return 2 * (rectangle.top - rectangle.bottom + rectangle.right - rectangle.left);
}
*/
double SplitArea(const Rectangle& rectangle1, const Rectangle& rectangle2) {

	return rectangle1.Area() + rectangle2.Area();
}

//template<class T>
//double SplitArea(T* t1, T* t2) {
//	return t1->Area() + t2->Area();
//}


double SplitPerimeter(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	return rectangle1.Perimeter() + rectangle2.Perimeter();
}

//template<class T>
//double SlitPerimeter(T* t1, T* t2) {
//	return t1->Perimeter() + t2->Perimeter();
//}

double SplitOverlap(const Rectangle& rectangle1, const Rectangle& rectangle2) {
	double left = max(rectangle1.Left(), rectangle2.Left());
	double right = min(rectangle1.Right(), rectangle2.Right());
	double bottom = max(rectangle1.Bottom(), rectangle2.Bottom());
	double top = min(rectangle1.Top(), rectangle2.Top());
	if (left < right && bottom < top) {
		return (top - bottom) * (right - left);
	}
	else {
		return 0;
	}
}

double MarginOvlpPerim(const Rectangle* rectangle1, const Rectangle* obj, const Rectangle* rectangle2){
	Rectangle r;
	r.Set(*rectangle1);
	r.Include(*obj);
	double left = max(r.Left(), rectangle2->Left());
	double right = min(r.Right(), rectangle2->Right());
	double bottom = max(r.Bottom(), rectangle2->Bottom());
	double top = min(r.Top(), rectangle2->Top());
	double perim1 = 0, perim2 = 0;
	if(left < right && bottom <top){
		perim1 = (right - left) * 2 + (top - bottom) * 2;
	}
	left = max(rectangle1->Left(), rectangle2->Left());
	right = min(rectangle1->Right(), rectangle2->Right());
	bottom = max(rectangle1->Bottom(), rectangle2->Bottom());
	top = min(rectangle1->Top(), rectangle2->Top());
	if(left < right && bottom < top){
		perim2 = (right - left) * 2 + (top - bottom) * 2;
	}
	return perim1 - perim2;
}

double MarginOvlpArea(const Rectangle* rectangle1, const Rectangle* obj, const Rectangle* rectangle2){
	Rectangle r;
	r.Set(*rectangle1);
	r.Include(*obj);
	double left = max(r.Left(), rectangle2->Left());
	double right = min(r.Right(), rectangle2->Right());
	double bottom = max(r.Bottom(), rectangle2->Bottom());
	double top = min(r.Top(), rectangle2->Top());
	double area1 = 0, area2 = 0;
	if(left < right && bottom < top){
		area1 = (right - left) * (top - bottom);
	}
	left = max(rectangle1->Left(), rectangle2->Left());
	right = min(rectangle1->Right(), rectangle2->Right());
	bottom = max(rectangle1->Bottom(), rectangle2->Bottom());
	top = min(rectangle1->Top(), rectangle2->Top());
	if(left < right && bottom < top){
		area2 = (right - left) * (top - bottom);
	}
	return area1 - area2;
}



//template<class T>
//double SplitOverlap(T* t1, T* t2) {
//	double left = max(t1->Left(), t2->Left());
//	double right = min(t1->Right(), t2->Right());
//	double bottom = max(t1->Bottom(), t2->Bottom()); 
//	double top = min(t1->Top(), t2->Top());
//	if (left < right && bottom < top) {
//		return (top - bottom) * (right - left);
//	}
//	else {
//		return 0;
//	}
//}

bool Rectangle::Contains(Rectangle* rec) {
	if (left_ <= rec->left_ && rec->right_ <= right_ && bottom_ <= rec->bottom_ && top_ >= rec->top_) {
		return true;
	}
	else {
		return false;
	}
}


//bool IsContained(const Rectangle& rectangle1, const Rectangle& rectangle2){
//    if(rectangle1.right >= rectangle2.right && rectangle1.left <= rectangle2.left &&
//    rectangle1.top >= rectangle2.top && rectangle1.bottom <= rectangle2.bottom){
//        return true;
//    }
//    else{
//        return false;
//    }
//}

bool Rectangle::IsOverlap(Rectangle* rec) {
	double left = max(left_, rec->left_);
	double right = min(right_, rec->right_);
	double bottom = max(bottom_, rec->bottom_);
	double top = max(top_, rec->top_);
	if (left < right && bottom < top) {
		return true;
	}
	else {
		return false;
	}
}

//bool IsOverlapped(const Rectangle& rectangle1, const Rectangle& rectangle2){
//    double left = max(rectangle1.left, rectangle2.left);
//    double right = min(rectangle1.right, rectangle2.right);
//    double bottom = max(rectangle1.bottom, rectangle2.bottom);
//    double top = min(rectangle1.top, rectangle2.top);
//    if(left < right && bottom < top){
//        return true;
//    }
//    else{
//        return false;
//    }
//}



bool SortedByLeft(const Rectangle* rec1, const Rectangle* rec2) {
	return rec1->left_ < rec2->left_;
}



bool SortedByRight(const Rectangle* rec1, const Rectangle* rec2) {
	return rec1->right_ < rec2->right_;
}



bool SortedByTop(const Rectangle* rec1, const Rectangle* rec2) {
	return rec1->top_ < rec2->top_;
}



bool SortedByBottom(const Rectangle* rec1, const Rectangle* rec2) {
	return rec1->bottom_ < rec2->bottom_;
}


TreeNode::TreeNode() {
	entry_num = 0;
	is_overflow = false;
	children = vector<int>(TreeNode::maximum_entry + 1);
	father = -1;
}

TreeNode::TreeNode(TreeNode* node) {
	entry_num = node->entry_num;
	is_overflow = node->is_overflow;
	is_leaf = node->is_leaf;
	children = vector<int>(TreeNode::maximum_entry + 1);
	for (int i = 0; i < entry_num; i++) {
		children[i] = node->children[i];
	}
	father = node->father;
	left_ = node->left_;
	right_ = node->right_;
	bottom_ = node->bottom_;
	top_ = node->top_;
	id_ = node->id_;
}

bool TreeNode::AddChildren(int node_id) {
	children[entry_num] = node_id;
	entry_num += 1;
	if (entry_num > TreeNode::maximum_entry) {
		is_overflow = true;
	}
	return is_overflow;
}

bool TreeNode::AddChildren(TreeNode *node) {
	children[entry_num] = node->id_;
	entry_num += 1;
	if (entry_num > TreeNode::maximum_entry) {
		is_overflow = true;
	}
	return is_overflow;
}

bool TreeNode::CopyChildren(const vector<int>& nodes) {
	if (nodes.size() >= TreeNode::maximum_entry)return false;
	int idx = 0;
	for (int idx = 0; idx < nodes.size(); idx++) {
		children[idx] = nodes[idx];
	}
	entry_num = nodes.size();
	is_overflow = false;
	return true;
}

bool TreeNode::CopyChildren(const vector<TreeNode *> &nodes) {
	if (nodes.size() >= TreeNode::maximum_entry)return false;
	for (int idx = 0; idx < nodes.size(); idx++) {
		children[idx] = nodes[idx]->id_;
	}
	entry_num = nodes.size();
	is_overflow = false;
	return true;
}

Rectangle* RTree::InsertRectangle(double left, double right, double bottom, double top) {
	Rectangle* rectangle = new Rectangle(left, right, bottom, top);
	rectangle->id_ = objects_.size();
	objects_.push_back(rectangle);
	return rectangle;
}

TreeNode* RTree::InsertStepByStep(const Rectangle *rectangle, TreeNode *tree_node) {
	return InsertStepByStep(rectangle, tree_node, insert_strategy_);
}

void RTree::PrintEntryNum() {
	TreeNode* iter = nullptr;
	list<TreeNode*> queue;
	queue.push_back(tree_nodes_[root_]);
	while (!queue.empty()) {
		iter = queue.front();
		queue.pop_front();
		cout << iter->entry_num << " ";
		if (!iter->is_leaf) {
			for (int i = 0; i < iter->entry_num; i++) {
				int child_id = iter->children[i];
				queue.push_back(tree_nodes_[child_id]);
			}
		}
	}
}

void RTree::Print() {
	TreeNode* iter = nullptr;
	list<TreeNode*> queue;
	queue.push_back(tree_nodes_[root_]);
	while (!queue.empty()) {
		iter = queue.front();
		queue.pop_front();
		cout << "node: [" << iter->left_ << ", " << iter->right_ << ", " << iter->bottom_ << ", " << iter->top_ << "]";
		cout << " " << iter->entry_num << " children";
		for (int i = 0; i < iter->entry_num; i++) {
			if (iter->is_leaf) {
				int child_idx = iter->children[i];
				Rectangle* r_iter = objects_[child_idx];
				cout << " object [" << r_iter->left_ << ", " << r_iter->right_ << ", " << r_iter->bottom_ << ", " << r_iter->top_ << "]";
			}
			else {
				int child_idx = iter->children[i];
				TreeNode* t_iter = tree_nodes_[child_idx];
				cout << " node[" << t_iter->left_ << ", " << t_iter->right_ << ", " << t_iter->bottom_ << ", " << t_iter->top_ << "]";
				queue.push_back(t_iter);
			}
		}
		cout << endl;
	}
	cout << "#######################"<<endl;
}

TreeNode* RTree::SplitStepByStep(TreeNode *tree_node) {
	return SplitStepByStep(tree_node, split_strategy_);
}


template<class T>
Rectangle MergeRange(const vector<T*>& entries, const int start_idx, const int end_idx) {
	Rectangle rectangle(*entries[start_idx]);
	for (int idx = start_idx + 1; idx < end_idx; ++idx) {
		rectangle.Include(*entries[idx]);
	}
	return rectangle;
}

template<class T>
pair<double, bool> SplitPerimSum(const vector<T*>& entries){
	Rectangle prefix = MergeRange<T>(entries, 0, TreeNode::minimum_entry - 1);
	Rectangle suffix = MergeRange<T>(entries, TreeNode::maximum_entry - TreeNode::minimum_entry+1, entries.size());
	double perim_sum = 0;
	Rectangle rec_remaining;
	bool is_overlap = true;
	for(int idx = TreeNode::minimum_entry - 1; idx < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; idx++){
		prefix.Include(*entries[idx]);
		rec_remaining.Set(suffix);
		for(int i = idx + 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			rec_remaining.Include(*entries[i]);
		}
		if(!prefix.IsOverlap(&rec_remaining)){
			is_overlap = false;
		}
		perim_sum += prefix.Perimeter() + rec_remaining.Perimeter();
	}
	pair<double, bool> result;
	result.first = perim_sum;
	result.second = is_overlap;
	return result;
}

//Rectangle RTree::MergeRange(const vector<Rectangle*>& entries, const int start_idx, const int end_idx) {
//	Rectangle rectangle(*entries[0]);
//	for (int idx = start_idx + 1; idx < end_idx; ++idx) {
//		rectangle.Include(*entries[idx]);
//	}
//	return rectangle;
//}
//Rectangle RTree::MergeRange(const vector<TreeNode*> &entry_list, const int start_idx, const int end_idx) {
//    Rectangle rectangle(*entry_list[start_idx]);
//    for(int idx = start_idx+1; idx < end_idx; ++idx){
//        rectangle.Include(*entry_list[idx]);
//    }
//    return rectangle;
//}

template<class T>
int FindMinimumSplit(const vector<T*>& entries, double(*score_func1)(const Rectangle &, const Rectangle &),
	double(*score_func2)(const Rectangle &, const Rectangle &), double& min_value1, double& min_value2, Rectangle& rec1, Rectangle& rec2) {
	Rectangle rec_prefix = MergeRange<T>(entries, 0, TreeNode::minimum_entry - 1);
	Rectangle rec_suffix = MergeRange<T>(entries, TreeNode::maximum_entry - TreeNode::minimum_entry + 1, entries.size());
	int optimal_split = -1;
	for (int idx = TreeNode::minimum_entry - 1; idx < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; idx++) {
		rec_prefix.Include(*entries[idx]);
		Rectangle rec_remaining(rec_suffix);
		for (int i = idx + 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++) {
			rec_remaining.Include(*entries[i]);
		}
		double value1 = score_func1(rec_prefix, rec_remaining);
		double value2 = score_func2(rec_prefix, rec_remaining);
		if (value1 < min_value1 || (value1 == min_value1 && value2 < min_value2)) {
			min_value1 = value1;
			min_value2 = value2;
			rec1.Set(rec_prefix);
			rec2.Set(rec_remaining);
			optimal_split = idx;
		}
	}
	return optimal_split;
}

template<class T>
int FindMinimumSplitRR(const vector<T*>& entries, double ys, double y1, double miu, double delta, double perim_max, Rectangle& rec1, Rectangle& rec2){
	Rectangle rec_prefix = MergeRange<T>(entries, 0, TreeNode::minimum_entry - 1);
	Rectangle rec_suffix = MergeRange<T>(entries, TreeNode::maximum_entry - TreeNode::minimum_entry +1, entries.size());
	int optimal_split = -1;
	Rectangle rec_remaining;
	double min_score = DBL_MAX;
	for(int idx = TreeNode::minimum_entry - 1; idx < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; idx++){
		rec_prefix.Include(*entries[idx]);
		rec_remaining.Set(rec_suffix);
		for(int i = idx + 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry + 1; i++){
			rec_remaining.Include(*entries[i]);
		}
		double wg = 0;
		if(rec_prefix.IsOverlap(&rec_remaining)){
			wg = SplitOverlap(rec_prefix, rec_remaining);
		}
		else{
			wg = rec_prefix.Perimeter() + rec_remaining.Perimeter() - perim_max;
		}
		
		double x = (2.0 * idx) / (TreeNode::maximum_entry + 1) - 1;
		double wf = ys * (exp(-1.0 * (x - miu) * (x-miu) / delta /delta) - y1);
		double score = 0;
		if(rec_prefix.IsOverlap(&rec_remaining)){
			score = wg / wf;
		}
		else{
			score = wg * wf;
		}
		if(score < min_score){
			min_score = score;
			optimal_split = idx;
			rec1.Set(rec_prefix);
			rec2.Set(rec_remaining);
		}
	}
	return optimal_split;
}

//void RTree::FindMinimumSplit(const vector<Rectangle*> &entry_list, double(*score_func1)(const Rectangle &, const Rectangle &), double(*score_func2)(const Rectangle &, const Rectangle &), double &min_value1, double &min_value2, vector<Rectangle> &child1, vector<Rectangle> &child2) {
//	Rectangle rec_prefix = MergeRange(entry_list, 0, TreeNode::minimum_entry);
//	Rectangle rec_suffix = MergeRange(entry_list, TreeNode::maximum_entry + 1, entry_list.size());
//	int optimal_split = -1;
//	for (int idx = TreeNode::minimum_entry; idx <= TreeNode::maximum_entry; idx++) {
//		rec_prefix.Include(entry_list[idx]);
//		Rectangle rec_remaining(rec_suffix);
//		for (int i = idx + 1; i < TreeNode::maximum_entry + 1; i++) {
//			rec_remaining.Include(entry_list[i]);
//		}
//		double value1 = score_func1(rec_prefix, rec_remaining);
//		double value2 = score_func2(rec_prefix, rec_remaining);
//		if (value1 < min_value1 || (value1 == min_value1 && value2 < min_value2)) {
//			min_value1 = value1;
//			min_value2 = value2;
//			optimal_split = idx;
//			//child1.assign(entry_list.begin, entry_list.begin(0))
//		}
//	}
//	if (optimal_split > 0) {
//		child1.assign(entry_list.begin(), entry_list.begin() + optimal_split + 1);
//		child2.assign(entry_list.begin() + optimal_split + 1, entry_list.end());
//	}
//}

//void RTree::FindMinimumSplit(const vector<TreeNode *> &entry_list, double (*score_func1)(const Rectangle &, const Rectangle &), double (*score_func2)(const Rectangle &, const Rectangle &), double &min_value1, double &min_value2, vector<TreeNode *> &child1, vector<TreeNode *> &child2) {
//    Rectangle rec_prefix = MergeRange(entry_list, 0, TreeNode::minimum_entry);
//    Rectangle rec_suffix = MergeRange(entry_list, TreeNode::maximum_entry+1, entry_list.size());
//	int optimal_split = -1;
//    for(int idx = TreeNode::minimum_entry; idx <= TreeNode::maximum_entry; idx++){
//        rec_prefix.Include(entry_list[idx]->bounding_box);
//        Rectangle rec_remaining(rec_suffix);
//        for(int i = idx+1; i<TreeNode::maximum_entry+1; i++){
//            rec_remaining.Include(entry_list[i]->bounding_box);
//        }
//        double value1 = score_func1(rec_prefix, rec_remaining);
//        double value2 = score_func2(rec_prefix, rec_remaining);
//        if(value1 < min_value1 || (value1 == min_value1 && value2 < min_value2)){
//            min_value1 = value1;
//            min_value2 = value2;
//			optimal_split = idx;
//            //child1.assign(entry_list.begin(), entry_list.begin() + idx + 1);
//            //child2.assign(entry_list.begin() + idx + 1, entry_list.end());
//        }
//    }
//	if (optimal_split > 0) {
//		child1.assign(entry_list.begin(), entry_list.begin() + optimal_split + 1);
//		child2.assign(entry_list.begin() + optimal_split + 1, entry_list.end());
//	}
//}


RTree::RTree() {
	TreeNode* root = CreateNode();
	height_ = 1;
	root->is_leaf = true;
	root_ = 0;

	RR_s = 0.5;
	RR_y1 = exp(-1 / RR_s / RR_s);
	RR_ys = 1.0 / (1.0 - RR_y1);
}

TreeNode* RTree::Root() {
	return tree_nodes_[root_];
}

void RTree::Recover(RTree* rtree) {
	for (auto it = history.begin(); it != history.end(); ++it) {
		int node_id = *it;
		tree_nodes_[node_id]->Set(*(rtree->tree_nodes_[node_id]));
		tree_nodes_[node_id]->entry_num = rtree->tree_nodes_[node_id]->entry_num;
		for (int i = 0; i < tree_nodes_[node_id]->entry_num; i++) {
			tree_nodes_[node_id]->children[i] = rtree->tree_nodes_[node_id]->children[i];
			int child = tree_nodes_[node_id]->children[i];
			tree_nodes_[child]->father = node_id;
		}
	}
}

TreeNode* SplitWithLoc(RTree* tree, TreeNode* tree_node, int loc) {
	TreeNode* node = tree->SplitInLoc(tree_node, loc);
	return node;
}

TreeNode* InsertWithLoc(RTree* tree, TreeNode* tree_node, int loc, Rectangle* rec){
	//cout<<"insert with loc invoked"<<endl;
	TreeNode* next_node = tree->InsertInLoc(tree_node, loc, rec);
	return next_node;
}


TreeNode* RTree::InsertInLoc(TreeNode *tree_node, int loc, Rectangle *rec){
	TreeNode* next_node = nullptr;
	if(tree_node->is_leaf){
		//cout<<"is leaf"<<endl;
		if(tree_node->entry_num == 0){
			tree_node->Set(*rec);
		}
		else{
			tree_node->Include(*rec);
		}
		tree_node->AddChildren(rec->id_);
	}
	else{
		tree_node->Include(*rec);
		int next_node_id = tree_node->children[loc];
		next_node = tree_nodes_[next_node_id];
	}
	return next_node;
}

TreeNode* RTree::SplitInLoc(TreeNode* tree_node, int loc) {
	TreeNode* next_node = nullptr;
	int size_per_dim = TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2;
	int dimension = loc / size_per_dim;
	int idx = loc % size_per_dim + TreeNode::minimum_entry - 1;
	vector<int> new_child1;
	vector<int> new_child2;
	Rectangle bounding_box1;
	Rectangle bounding_box2;
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			int obj_id = tree_node->children[i];
			recs[i] = objects_[obj_id];
		}
		switch (dimension)
		{
		case 0: {
			sort(recs.begin(), recs.end(), SortedByLeft);
			break;
		}
		case 1: {
			sort(recs.begin(), recs.end(), SortedByRight);
			break;
		}
		case 2: {
			sort(recs.begin(), recs.end(), SortedByBottom);
			break;
		}
		case 3: {
			sort(recs.begin(), recs.end(), SortedByTop);
			break;
		}
		default:
			break;
		}
		new_child1.resize(idx + 1);
		new_child2.resize(TreeNode::maximum_entry - idx);
		for (int i = 0; i < idx + 1; i++) {
			new_child1[i] = recs[i]->id_;
		}
		for (int i = idx + 1; i < recs.size(); i++){
			new_child2[i - idx - 1] = recs[i]->id_;
		}
		bounding_box1.Set(*recs[0]);
		for (int i = 1; i < idx + 1; i++) {
			bounding_box1.Include(*recs[i]);
		}
		bounding_box2.Set(*recs[idx + 1]);
		for (int i = idx + 2; i < recs.size(); i++) {
			bounding_box2.Include(*recs[i]);
		}
	}
	else {
		vector<TreeNode*> nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			int node_id = tree_node->children[i];
			nodes[i] = tree_nodes_[node_id];
		}
		switch (dimension)
		{
		case 0: {
			sort(nodes.begin(), nodes.end(), SortedByLeft);
			break;
		}
		case 1: {
			sort(nodes.begin(), nodes.end(), SortedByRight);
			break;
		}
		case 2: {
			sort(nodes.begin(), nodes.end(), SortedByBottom);
			break;
		}
		case 3: {
			sort(nodes.begin(), nodes.end(), SortedByTop);
			break;
		}
		default:
			break;
		}
		new_child1.resize(idx + 1);
		new_child2.resize(TreeNode::maximum_entry - idx);
		for (int i = 0; i < idx + 1; i++) {
			new_child1[i] = nodes[i]->id_;
		}
		for (int i = idx+1; i < nodes.size(); i++) {
			new_child2[i - idx - 1] = nodes[i]->id_;
		}
		bounding_box1.Set(*nodes[0]);
		for (int i = 1; i < idx + 1; i++) {
			bounding_box1.Include(*nodes[i]);
		}
		bounding_box2.Set(*nodes[idx + 1]);
		for (int i = idx + 2; i < nodes.size(); i++) {
			bounding_box2.Include(*nodes[i]);
		}
	}
	TreeNode* sibling = CreateNode();
	sibling->is_leaf = tree_node->is_leaf;
	sibling->CopyChildren(new_child2);
	if (!sibling->is_leaf) {
		for (int i = 0; i < new_child2.size(); i++) {
			tree_nodes_[new_child2[i]]->father = sibling->id_;
		}
	}
	sibling->Set(bounding_box2);
	tree_node->CopyChildren(new_child1);
	if (!tree_node->is_leaf) {
		for (int i = 0; i < new_child1.size(); i++) {
			tree_nodes_[new_child1[i]]->father = tree_node->id_;
		}
	}
	tree_node->Set(bounding_box1);
	if (tree_node->father >= 0) {
		tree_nodes_[tree_node->father]->AddChildren(sibling);
		tree_nodes_[tree_node->father]->Include(bounding_box2);
		sibling->father = tree_node->father;
	}
	else {
		TreeNode* new_root = CreateNode();
		new_root->is_leaf = false;
		new_root->AddChildren(tree_node);
		new_root->AddChildren(sibling);
		new_root->Set(bounding_box1);
		new_root->Include(bounding_box2);
		root_ = new_root->id_;
		tree_node->father = new_root->id_;
		sibling->father = new_root->id_;
		height_ += 1;
	}
	next_node = tree_nodes_[tree_node->father];
	return next_node;
}

TreeNode* RTree::RRSplit(TreeNode* tree_node){
	TreeNode* next_node = nullptr;
	if(tree_node->is_overflow){
		//determine split dimension
		int split_dim = 0;
		double delta = 0;
		double new_center[2];
		new_center[0] = 0.5 * (tree_node->Right() + tree_node->Left());
		new_center[1] = 0.5 * (tree_node->Top() + tree_node->Bottom());
		double length[2];
		length[0] = tree_node->Right() - tree_node->Left();
		length[1] = tree_node->Top() - tree_node->Bottom();

		Rectangle bounding_box1;
		Rectangle bounding_box2;
		vector<int> new_child1;
		vector<int> new_child2;

		double perim_max = 2.0 * (length[0] + length[1]) - min(length[0], length[1]);
		double perim_sum[2];
		bool exist_nonoverlap[2];
		if(tree_node->is_leaf){			
			vector<Rectangle*> children(tree_node->entry_num);
			for(int i=0; i<tree_node->entry_num; i++){
				int obj_id = tree_node->children[i];
				children[i] = objects_[obj_id];
			}
			sort(children.begin(), children.end(), SortedByLeft);
			pair<double, bool> split_x = SplitPerimSum<Rectangle>(children);
			perim_sum[0] = split_x.first;
			exist_nonoverlap[0] = split_x.second;
			sort(children.begin(), children.end(), SortedByBottom);
			pair<double, bool> split_y = SplitPerimSum<Rectangle>(children);
			perim_sum[1] = split_y.first;
			exist_nonoverlap[1] = split_y.second;
			split_dim = perim_sum[0] < perim_sum[1] ? 0 : 1;

			cout<<"perim: x: "<<perim_sum[0]<<" y: "<<perim_sum[1]<<" split axis: "<<split_dim<<endl;
			getchar();

			double miu = 2.0 * (new_center[split_dim] - tree_node->origin_center[split_dim]) / length[split_dim] * (1 - 2 * TreeNode::minimum_entry / (TreeNode::maximum_entry + 1));		
			delta = RR_s * (1.0 + abs(miu));
			int split = FindMinimumSplitRR<Rectangle>(children, RR_ys, RR_y1, miu, delta, perim_max, bounding_box1, bounding_box2);
			cout<<"split position: "<<split<<endl;
			getchar();
			vector<Rectangle*> child_rec1;
			vector<Rectangle*> child_rec2;
			child_rec1.assign(children.begin(), children.begin() + split + 1);
			child_rec2.assign(children.begin() + split + 1, children.end());
			new_child1.resize(child_rec1.size());
			new_child2.resize(child_rec2.size());
			for (int i = 0; i < new_child1.size(); i++) {
				new_child1[i] = child_rec1[i]->id_; 
			}
			for (int i = 0; i < new_child2.size(); i++) {
				new_child2[i] = child_rec2[i]->id_;
			}
		}
		else{
			vector<TreeNode*> children(tree_node->entry_num);
			for(int i=0; i<tree_node->entry_num; i++){
				int node_id = tree_node->children[i];
				children[i] = tree_nodes_[node_id];
			}
			sort(children.begin(), children.end(), SortedByLeft);
			pair<double, bool> split_x = SplitPerimSum<TreeNode>(children);
			perim_sum[0] = split_x.first;
			exist_nonoverlap[0] = split_x.second;
			sort(children.begin(), children.end(), SortedByBottom);
			pair<double, bool> split_y = SplitPerimSum<TreeNode>(children);
			perim_sum[1] = split_y.first;
			exist_nonoverlap[1] = split_y.second;
			split_dim = perim_sum[0] < perim_sum[1] ? 0 : 1;
			double miu = 2.0 * (new_center[split_dim] - tree_node->origin_center[split_dim]) / length[split_dim] * (1 - 2 * TreeNode::minimum_entry / (TreeNode::maximum_entry + 1));		
			delta = RR_s * (1.0 + abs(miu));
			int split = FindMinimumSplitRR<TreeNode>(children, RR_ys, RR_y1, miu, delta, perim_max, bounding_box1, bounding_box2);
			vector<TreeNode*> child_rec1;
			vector<TreeNode*> child_rec2;
			child_rec1.assign(children.begin(), children.begin() + split + 1);
			child_rec2.assign(children.begin() + split + 1, children.end());
			new_child1.resize(child_rec1.size());
			new_child2.resize(child_rec2.size());
			for (int i = 0; i < new_child1.size(); i++) {
				new_child1[i] = child_rec1[i]->id_; 
			}
			for (int i = 0; i < new_child2.size(); i++) {
				new_child2[i] = child_rec2[i]->id_;
			}
		}
		TreeNode* sibling = CreateNode();
		sibling->is_leaf = tree_node->is_leaf;
		sibling->CopyChildren(new_child2);
		if(!sibling->is_leaf){
			for(int i=0; i < new_child2.size(); i++){
				tree_nodes_[new_child2[i]]->father = sibling->id_;
			}
		}
		sibling->Set(bounding_box2);
		sibling->origin_center[0] = 0.5 * (bounding_box2.Left() + bounding_box2.Right());
		sibling->origin_center[1] = 0.5 * (bounding_box2.Top() + bounding_box2.Bottom());

		tree_node->CopyChildren(new_child1);
		if(!tree_node->is_leaf){
			for(int i=0; i < new_child1.size(); i++){
				tree_nodes_[new_child1[i]]->father = tree_node->id_;
			}
		}
		tree_node->Set(bounding_box1);
		tree_node->origin_center[0] = 0.5 * (bounding_box1.Left() + bounding_box1.Right());
		tree_node->origin_center[1] = 0.5 * (bounding_box1.Bottom() + bounding_box1.Top());

		if (tree_node->father >= 0) {
			tree_nodes_[tree_node->father]->AddChildren(sibling);
			tree_nodes_[tree_node->father]->Include(bounding_box2);
			sibling->father = tree_node->father;
			//tree_node->father->AddChildren(sibling);
			//tree_node->father->Include(bounding_box2);
		}
		else {
			TreeNode* new_root = CreateNode();
			new_root->is_leaf = false;
			new_root->AddChildren(tree_node);
			new_root->AddChildren(sibling);
			new_root->Set(bounding_box1);
			new_root->Include(bounding_box2);
			root_ = new_root->id_;
			tree_node->father = new_root->id_;
			sibling->father = new_root->id_;
			height_ += 1;
		}
		next_node = tree_nodes_[tree_node->father];
	}
	return next_node;
}


TreeNode* RTree::SplitStepByStep(TreeNode *tree_node, SPLIT_STRATEGY strategy) {
	TreeNode* next_node = nullptr;
	if (tree_node->is_overflow) {
		//vector<TreeNode*> new_child1;
		//vector<TreeNode*> new_child2;
		//vector<Rectangle*> new_child1_rec;
		//vector<Rectangle*> new_child2_rec;
		vector<int> new_child1;
		vector<int> new_child2;
		Rectangle bounding_box1;
		Rectangle bounding_box2;
		switch (strategy) {
		case SPL_MIN_AREA: {

			double minimum_area = DBL_MAX;
			double minimum_overlap = DBL_MAX;

			//choose the split with the minimum total area, break the tie by preferring the split with smaller overlap
			if (tree_node->is_leaf) {
				vector<Rectangle*> recs(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					recs[i] = objects_[obj_id];
				}
				vector<Rectangle*> child_rec1;
				vector<Rectangle*> child_rec2;
				//sort by left
				sort(recs.begin(), recs.end(), SortedByLeft);
				Rectangle rec1;
				Rectangle rec2;
				int split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}

				//sort by right
				sort(recs.begin(), recs.end(), SortedByRight);
				split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}

				//sort by bottom
				sort(recs.begin(), recs.end(), SortedByBottom);
				split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}

				//sort by top
				sort(recs.begin(), recs.end(), SortedByTop);
				split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_rec1.size());
				new_child2.resize(child_rec2.size());
				for (int i = 0; i < new_child1.size(); i++) {
					new_child1[i] = child_rec1[i]->id_; 
				}
				for (int i = 0; i < new_child2.size(); i++) {
					new_child2[i] = child_rec2[i]->id_;
				}
			}
			else {
				vector<TreeNode*> children(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					children[i] = tree_nodes_[obj_id];
				}
				vector<TreeNode*> child_treenode1;
				vector<TreeNode*> child_treenode2;
				//sort by left
				sort(children.begin(), children.end(), SortedByLeft);
				Rectangle rec1;
				Rectangle rec2;
				
				int split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(children.begin(), children.end(), SortedByRight);
				split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(children.begin(), children.end(), SortedByBottom);
				split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(children.begin(), children.end(), SortedByTop);
				split = FindMinimumSplit<TreeNode>(children, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_treenode1.size());
				new_child2.resize(child_treenode2.size());
				for (int i = 0; i < child_treenode1.size(); i++) {
					new_child1[i] = child_treenode1[i]->id_;
				}
				for (int i = 0; i < child_treenode2.size(); i++) {
					new_child2[i] = child_treenode2[i]->id_;
				}

			}
			break;
		}
		case SPL_MIN_MARGIN: {
			double minimum_perimeter = DBL_MAX;
			double minimum_overlap = DBL_MAX;
			//choose the split with the minimum total perimeter, break the tie by preferring the split with smaller overlap

			if (tree_node->is_leaf) {
				vector<Rectangle*> recs(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					recs[i] = objects_[obj_id];
				}
				vector<Rectangle*> child_rec1;
				vector<Rectangle*> child_rec2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(recs.begin(), recs.end(), SortedByLeft);
				int split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(recs.begin(), recs.end(), SortedByRight);
				split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom 
				sort(recs.begin(), recs.end(), SortedByBottom);
				split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(recs.begin(), recs.end(), SortedByTop);
				split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
                new_child1.resize(child_rec1.size());
                new_child2.resize(child_rec2.size());
                for (int i = 0; i < new_child1.size(); i++) {
					new_child1[i] = child_rec1[i]->id_;
                }
                for (int i = 0; i < new_child2.size(); i++) {
                    new_child2[i] = child_rec2[i]->id_;
                }
			}
			else {
				vector<TreeNode*> children(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int child_id = tree_node->children[i];
					children[i] = tree_nodes_[child_id];
				}
				vector<TreeNode*> child_treenode1;
				vector<TreeNode*> child_treenode2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(children.begin(), children.end(), SortedByLeft);
				int split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(children.begin(), children.end(), SortedByRight);
				split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(children.begin(), children.end(), SortedByBottom);
				split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(children.begin(), children.end(), SortedByTop);
				split = FindMinimumSplit<TreeNode>(children, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_treenode1.size());
				new_child2.resize(child_treenode2.size());
				for (int i = 0; i < child_treenode1.size(); i++) {
					new_child1[i] = child_treenode1[i]->id_;
				}
				for (int i = 0; i < child_treenode2.size(); i++) {
					new_child2[i] = child_treenode2[i]->id_;
				}
			}
			break;
		}
		case SPL_MIN_OVERLAP: {
			double minimum_overlap = DBL_MAX;
			double minimum_area = DBL_MAX;
			//choose the split with the minimum overlap, break the tie by preferring the split with smaller total area

			if (tree_node->is_leaf) {
				vector<Rectangle*> recs(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int obj_id = tree_node->children[i];
					recs[i] = objects_[obj_id]; 
				}
				vector<Rectangle*> child_rec1;
				vector<Rectangle*> child_rec2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(recs.begin(), recs.end(), SortedByLeft);
				int split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(recs.begin(), recs.end(), SortedByRight);
				split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(recs.begin(), recs.end(), SortedByBottom);
				split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(recs.begin(), recs.end(), SortedByTop);
				split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_rec1.assign(recs.begin(), recs.begin() + split + 1);
					child_rec2.assign(recs.begin() + split + 1, recs.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
                new_child1.resize(child_rec1.size());
                new_child2.resize(child_rec2.size());
                for (int i = 0; i < child_rec1.size(); i++) {
					new_child1[i] = child_rec1[i]->id_;
                }
                for (int i = 0; i < child_rec2.size(); i++) {
					new_child2[i] = child_rec2[i]->id_;
                }
			}
			else {
				vector<TreeNode*> children(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					int child_id = tree_node->children[i];
					children[i] = tree_nodes_[child_id];
				}
				vector<TreeNode*> child_treenode1;
				vector<TreeNode*> child_treenode2;
				Rectangle rec1;
				Rectangle rec2;
				//sort by left
				sort(children.begin(), children.end(), SortedByLeft);
				int split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by right
				sort(children.begin(), children.end(), SortedByRight);
				split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by bottom
				sort(children.begin(), children.end(), SortedByBottom);
				split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				//sort by top
				sort(children.begin(), children.end(), SortedByTop);
				split = FindMinimumSplit<TreeNode>(children, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
				if (split >= 0) {
					child_treenode1.assign(children.begin(), children.begin() + split + 1);
					child_treenode2.assign(children.begin() + split + 1, children.end());
					bounding_box1.Set(rec1);
					bounding_box2.Set(rec2);
				}
				new_child1.resize(child_treenode1.size());
				new_child2.resize(child_treenode2.size());
				for (int i = 0; i < child_treenode1.size(); i++) {
					new_child1[i] = child_treenode1[i]->id_;
				}
				for (int i = 0; i < child_treenode2.size(); i++) {
					new_child2[i] = child_treenode2[i]->id_;
				}
			}
			break;
		}
		case SPL_QUADRATIC: {
			int seed1 = -1;
			int seed2 = -1;
			int seed_idx1 = -1, seed_idx2 = -1;
			double max_waste = -DBL_MAX;
			//find the pair of children that waste the most area were they to be inserted in the same node
			for (int i = 0; i < tree_node->entry_num - 1; i++) {
				for (int j = i + 1; j < tree_node->entry_num; j++) {
					double waste = 0;
					unsigned int id1 = tree_node->children[i];
					unsigned int id2 = tree_node->children[j];
					if (tree_node->is_leaf) {
						waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
						//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
					}
					else {
						waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
						//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
					}
					if (waste > max_waste) {
						max_waste = waste;
						seed1 = id1;
						seed2 = id2;
						seed_idx1 = i;
						seed_idx2 = j;
					}
				}
			}
			//list<TreeNode*> child1;
			//list<TreeNode*> child2;
			list<int> child1;
			list<int> child2;
			child1.push_back(seed1);
			child2.push_back(seed2);
			if (tree_node->is_leaf) {
				bounding_box1.Set(*objects_[seed1]);
				bounding_box2.Set(*objects_[seed2]);
			}
			else {
				bounding_box1.Set(*tree_nodes_[seed1]);
				bounding_box2.Set(*tree_nodes_[seed2]);
			}
			list<int> unassigned_entry;
			for (int i = 0; i < tree_node->entry_num; i++) {
				if (i == seed_idx1 || i == seed_idx2)continue;
				unassigned_entry.push_back(tree_node->children[i]);
			}
			while (!unassigned_entry.empty()) {
				//make sure the two child nodes are balanced.
				if (unassigned_entry.size() + child1.size() == TreeNode::minimum_entry) {
					for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
						child1.push_back(*it);
						if (tree_node->is_leaf) {
							Rectangle* rec_ptr = objects_[*it];
							bounding_box1.Include(*rec_ptr);
						}
						else {
							TreeNode* node_ptr = tree_nodes_[*it];
							bounding_box1.Include(*node_ptr);
						}
						
					}
					break;
				}
				if (unassigned_entry.size() + child2.size() == TreeNode::minimum_entry) {
					for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
						child2.push_back(*it);
						if (tree_node->is_leaf) {
							Rectangle* rec_ptr = objects_[*it];
							bounding_box2.Include(*rec_ptr);
						}
						else {
							TreeNode* node_ptr = tree_nodes_[*it];
							bounding_box2.Include(*node_ptr);
						}
						
					}
					break;
				}
				//pick next: pick an unassigned entry that maximizes the difference between adding into different groups
				double max_difference = - DBL_MAX;
				double new_area1 = 0, new_area2 = 0;
				list<int>::iterator iter;
				int next_entry;
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					double d1 = 0, d2 = 0;
					if (tree_node->is_leaf) {
						d1 = bounding_box1.Merge(*objects_[*it]).Area();
						d2 = bounding_box2.Merge(*objects_[*it]).Area();
					}
					else {
						d1 = bounding_box1.Merge(*tree_nodes_[*it]).Area();
						d2 = bounding_box2.Merge(*tree_nodes_[*it]).Area();
					}
					double difference = d1 > d2 ? d1 - d2 : d2 - d1;
					if (difference > max_difference) {
						max_difference = difference;
						iter = it;
						next_entry = *it;
						new_area1 = d1;
						new_area2 = d2;
					}
				}
				unassigned_entry.erase(iter);
				//add the entry to the group with smaller area
				Rectangle *chosen_bounding_box = nullptr;
				if (new_area1 < new_area2) {
					child1.push_back(next_entry);
					chosen_bounding_box = &bounding_box1;
				}
				else if (new_area1 > new_area2) {
					child2.push_back(next_entry);
					chosen_bounding_box = &bounding_box2;
				}
				else {
					if (child1.size() < child2.size()) {
						child1.push_back(next_entry);
						chosen_bounding_box = &bounding_box1;
					}
					else {
						child2.push_back(next_entry);
						chosen_bounding_box = &bounding_box2;
					}
				}
				if (tree_node->is_leaf) {
					chosen_bounding_box->Include(*objects_[next_entry]);
				}
				else {
					chosen_bounding_box->Include(*tree_nodes_[next_entry]);
				}
			}
			new_child1.assign(child1.begin(), child1.end());
			new_child2.assign(child2.begin(), child2.end());
			break;
		}

		case SPL_GREENE: {
			int seed1 = -1;
			int seed2 = -1;
			double max_waste = - DBL_MAX;
			for (int i = 0; i < tree_node->entry_num - 1; i++) {
				for (int j = i + 1; j < tree_node->entry_num; j++) {
					double waste = 0;
					int id1 = tree_node->children[i];
					int id2 = tree_node->children[j];
					if (tree_node->is_leaf) {
						waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
						//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
					}
					else {
						waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
						//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
					}
					if (waste > max_waste) {
						max_waste = waste;
						seed1 = id1;
						seed2 = id2;
					}
				}
			}

			double max_seed_left, min_seed_right, max_seed_bottom, min_seed_top;
			if (tree_node->is_leaf) {
				max_seed_left = max(objects_[seed1]->Left(), objects_[seed2]->Left());
				min_seed_right = min(objects_[seed1]->Right(), objects_[seed2]->Right());
				max_seed_bottom = max(objects_[seed1]->Bottom(), objects_[seed2]->Bottom());
				min_seed_top = min(objects_[seed1]->Top(), objects_[seed2]->Top());
			}
			else {
				max_seed_left = max(tree_nodes_[seed1]->Left(), tree_nodes_[seed2]->Left());
				min_seed_right = min(tree_nodes_[seed1]->Right(), tree_nodes_[seed2]->Right());
				max_seed_bottom = max(tree_nodes_[seed1]->Bottom(), tree_nodes_[seed2]->Bottom());
				min_seed_top = min(tree_nodes_[seed1]->Top(), tree_nodes_[seed2]->Top());
			}
			double x_seperation = min_seed_right > max_seed_left ? (min_seed_right - max_seed_left) : (max_seed_left - min_seed_right);
			double y_seperation = min_seed_top > max_seed_bottom ? (min_seed_top - max_seed_bottom) : (max_seed_bottom - min_seed_top);

			x_seperation = x_seperation / (tree_node->Right() - tree_node->Left());
			y_seperation = y_seperation / (tree_node->Top() - tree_node->Bottom());
			
			vector<Rectangle*>	recs;
			vector<TreeNode*> child_nodes;

			if (tree_node->is_leaf) {
				recs.resize(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					recs[i] = objects_[tree_node->children[i]]; 
				}
			}
			else {
				child_nodes.resize(tree_node->entry_num);
				for (int i = 0; i < tree_node->entry_num; i++) {
					child_nodes[i] = tree_nodes_[tree_node->children[i]];
				}
			}

			if (x_seperation < y_seperation) {
				if (tree_node->is_leaf) {
					sort(recs.begin(), recs.end(), SortedByBottom);
				}
				else {
					sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
				}
			}
			else {
				if (tree_node->is_leaf) {
					sort(recs.begin(), recs.end(), SortedByLeft);
				}
				else {
					sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
				}
			}
			if (tree_node->is_leaf) {
				new_child1.resize(tree_node->entry_num / 2);
				new_child2.resize(tree_node->entry_num / 2);
				for (int i = 0; i < tree_node->entry_num / 2; i++) {
					new_child1[i] = recs[i]->id_;
					new_child2[i] = recs[tree_node->entry_num - 1 - i]->id_;
				}
				bounding_box1 = MergeRange<Rectangle>(recs, 0, tree_node->entry_num / 2);
				bounding_box2 = MergeRange<Rectangle>(recs, tree_node->entry_num - tree_node->entry_num / 2, tree_node->entry_num);
				if (tree_node->entry_num % 2 == 1) {
					Rectangle rec1 = bounding_box1.Merge(*recs[tree_node->entry_num / 2]);
					Rectangle rec2 = bounding_box2.Merge(*recs[tree_node->entry_num / 2]);
					double area_increase1 = rec1.Area() - bounding_box1.Area();
					double area_increase2 = rec2.Area() - bounding_box2.Area();
					if (area_increase1 < area_increase2) {
						new_child1.push_back(recs[tree_node->entry_num / 2]->id_);
						bounding_box1 = rec1;
					}
					else {
						new_child2.push_back(recs[tree_node->entry_num / 2]->id_);
						bounding_box2 = rec2;
					}
				}
			}
			else {
				new_child1.resize(tree_node->entry_num / 2);
				new_child2.resize(tree_node->entry_num / 2);
				for (int i = 0; i < tree_node->entry_num / 2; i++) {
					new_child1[i] = child_nodes[i]->id_;
					new_child2[i] = child_nodes[tree_node->entry_num - 1 - i]->id_;
				}
				
				bounding_box1 = MergeRange<TreeNode>(child_nodes, 0, tree_node->entry_num / 2);
				bounding_box2 = MergeRange<TreeNode>(child_nodes, tree_node->entry_num - tree_node->entry_num/2, tree_node->entry_num);
				if (tree_node->entry_num % 2 == 1) {
					Rectangle rec1 = bounding_box1.Merge(*child_nodes[tree_node->entry_num / 2]);
					Rectangle rec2 = bounding_box2.Merge(*child_nodes[tree_node->entry_num / 2]);
					double area_increase1 = rec1.Area() - bounding_box1.Area();
					double area_increase2 = rec2.Area() - bounding_box2.Area();
					if (area_increase1 < area_increase2) {
						new_child1.push_back(child_nodes[tree_node->entry_num / 2]->id_);
						bounding_box1 = rec1;
					}
					else {
						new_child2.push_back(child_nodes[tree_node->entry_num / 2]->id_);
						bounding_box2 = rec2;
					}
				}
			}

			break;
		}
		}
		
		TreeNode* sibling = CreateNode();
		sibling->is_leaf = tree_node->is_leaf;
		sibling->CopyChildren(new_child2);
		if(!sibling->is_leaf){
            for (int i = 0; i < new_child2.size(); i++) {
				tree_nodes_[new_child2[i]]->father = sibling->id_;
                //new_child2[i]->father = sibling;
            }
		}
		sibling->Set(bounding_box2);

		tree_node->CopyChildren(new_child1);
		if(!tree_node->is_leaf){
            for (int i = 0; i < new_child1.size(); i++) {
				tree_nodes_[new_child1[i]]->father = tree_node->id_;
                //new_child1[i]->father = tree_node;
            }
		}
		tree_node->Set(bounding_box1);

		if (tree_node->father >= 0) {
			tree_nodes_[tree_node->father]->AddChildren(sibling);
			tree_nodes_[tree_node->father]->Include(bounding_box2);
			sibling->father = tree_node->father;
			//tree_node->father->AddChildren(sibling);
			//tree_node->father->Include(bounding_box2);
		}
		else {
			TreeNode* new_root = CreateNode();
			new_root->is_leaf = false;
			new_root->AddChildren(tree_node);
			new_root->AddChildren(sibling);
			new_root->Set(bounding_box1);
			new_root->Include(bounding_box2);
			root_ = new_root->id_;
			tree_node->father = new_root->id_;
			sibling->father = new_root->id_;
			height_ += 1;
		}
		next_node = tree_nodes_[tree_node->father];
	}

	return next_node;
}


int GetQueryResult(RTree* rtree) {
	return rtree->result_count;
}

int RTree::Query(Rectangle& rectangle) {
	result_count = 0;
	list<TreeNode*> queue;
	queue.push_back(tree_nodes_[root_]);
	stats_.Reset();
	TreeNode* iter = tree_nodes_[root_];
	if (!iter->IsOverlap(&rectangle)) {
		return 0;
	}
	while (!queue.empty()) {
		iter = queue.front();
		stats_.node_access += 1;
		queue.pop_front();
		if (iter->is_leaf) {
			for (int i = 0; i < iter->entry_num; i++) {
				Rectangle* rec_iter = objects_[iter->children[i]];
				if (rec_iter->IsOverlap(&rectangle)) {
					result_count += 1;
				}
			}
		}
		else {
			for (int i = 0; i < iter->entry_num; i++) {
				TreeNode* node = tree_nodes_[iter->children[i]];
				if (node->IsOverlap(&rectangle)) {
					queue.push_back(node);
				}
			}
		}
	}
	return 1;
}

TreeNode* RTree::CreateNode() {
	TreeNode* node = new TreeNode();
	node->id_ = tree_nodes_.size();
	tree_nodes_.push_back(node);
	assert(tree_nodes_[node->id_]->id_ == node->id_);
	return node;
}


TreeNode* RTree::TryInsertStepByStep(const Rectangle* rectangle, TreeNode* tree_node) {
	TreeNode* next_node = nullptr;
	if (!tree_node->is_leaf) {
		switch (insert_strategy_) {
		case INS_AREA: {
			double min_area_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double area_increase = new_rectangle.Area() - it->Area();
				if (area_increase < min_area_increase) {
					//choose the subtree with the smaller area increase
					min_area_increase = area_increase;
					next_node = it;
				}
				else if (area_increase == min_area_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_MARGIN: {
			double min_margin_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double margin_increase = new_rectangle.Perimeter() - it->Perimeter();
				if (margin_increase < min_margin_increase) {
					//choose the subtree with smaller perimeter increase
					next_node = it;
					min_margin_increase = margin_increase;
				}
				else if (margin_increase == min_margin_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_OVERLAP: {
			double min_overlap_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double overlap_increase = 0;
				for (int idx2 = 0; idx2 < tree_node->entry_num; idx2++) {
					if (idx == idx2)continue;
					TreeNode* it2 = tree_nodes_[tree_node->children[idx2]];
					//overlap_increase += Overlap(new_rectangle, it2->bounding_box) - Overlap(it->bounding_box, it2->bounding_box);
					overlap_increase += SplitOverlap(new_rectangle, *it2) - SplitOverlap(*it, *it2);
				}
				if (overlap_increase < min_overlap_increase) {
					//choose the subtree with smaller overlap increase
					min_overlap_increase = overlap_increase;
					next_node = it;
				}
				else if (overlap_increase == min_overlap_increase) {
					//break the tie by favoring the one with smaller area increase
					//double area_increase = Area(Merge(rectangle, it->bounding_box)) - Area(it->bounding_box);
					double area_increase = it->Merge(*rectangle).Area() - it->Area();
					if (area_increase < next_node->Merge(*rectangle).Area() - next_node->Area()) {
						next_node = it;
					}

				}
			}
			break;
		}
		case INS_RANDOM: {
			int chosen_child = rand() % tree_node->entry_num;
			next_node = tree_nodes_[tree_node->children[chosen_child]];
			break;
		}
		}
	}
	return next_node;
}

TreeNode* RTree::RRInsert(Rectangle* rectangle, TreeNode* tree_node){
	TreeNode* next_node = nullptr;
	if(tree_node->is_leaf){
		if(objects_.size() == 0){
			tree_node->origin_center[0] = 0.5 * (rectangle->Left() + rectangle->Right());
			tree_node->origin_center[1] = 0.5 * (rectangle->Bottom() + rectangle->Top());
		}
		if(tree_node->entry_num == 0){
			tree_node->Set(*rectangle);
		}
		else{
			tree_node->Include(*rectangle);
		}
		tree_node->AddChildren(rectangle->id_);
	}
	else{
		tree_node->Include(*rectangle);
		list<int> COV;
		for(int i=0; i < tree_node->entry_num; i++){
			int node_id = tree_node->children[i];
			if(tree_nodes_[node_id]->Contains(rectangle)){
				COV.push_back(node_id);
			}
		}
		if(COV.empty()){
			vector<pair<double, int> > sequence(tree_node->entry_num);
			Rectangle r;
			for(int i=0; i<tree_node->entry_num; i++){
				int node_id = tree_node->children[i];
				r.Set(*tree_nodes_[node_id]);
				r.Include(*rectangle);
				sequence[i].first = r.Perimeter() - tree_nodes_[node_id]->Perimeter();
				sequence[i].second = node_id;
			}
			sort(sequence.begin(), sequence.end());
			vector<double> margin_ovlp_perim(tree_node->entry_num, 0);
			double total_ovlp_perim = 0;
			for(int i=0; i<tree_node->entry_num; i++){
				margin_ovlp_perim[i] = MarginOvlpPerim(tree_nodes_[sequence[0].second], rectangle, tree_nodes_[sequence[i].second]);
				total_ovlp_perim += margin_ovlp_perim[i];
			}
			if(total_ovlp_perim == 0){
				next_node = tree_nodes_[sequence[0].second];
				return next_node;
			}
			int p = 1;
			for(int i = 1; i<tree_node->entry_num; i++){
				if(margin_ovlp_perim[i] > margin_ovlp_perim[p]){
					p = i;
				}
			}
			vector<int> cand;
			vector<bool> is_in_cand(p+1, false);
			cand.push_back(0);
			is_in_cand[0] = true;
			int idx = 0;
			vector<double> margin_ovlp_area(p+1, 0);
			while(idx < cand.size()){
				int t = cand[idx];
				for(int j = 0; j <= p; j++){
					if(j != t)continue;
					double ovlp_tj = MarginOvlpArea(tree_nodes_[sequence[t].second], rectangle, tree_nodes_[sequence[j].second]);
					margin_ovlp_area[t] += ovlp_tj;
					if(ovlp_tj != 0 && (!is_in_cand[j])){
						cand.push_back(j);
						is_in_cand[j] = true;
					}
				}
				idx += 1;
			}
			for(int i = cand.size() - 1; i >= 0; i = i - 1){
				int t = cand[i];
				if(margin_ovlp_area[t] == 0){
					next_node = tree_nodes_[sequence[t].second];
					return next_node;
				}
			}
			int c = cand[0];
			for(int i = 1; i < cand.size(); i++){
				if(margin_ovlp_area[sequence[c].second] > margin_ovlp_area[sequence[cand[i]].second]){
					c = cand[i];
				}
			}
			next_node = tree_nodes_[sequence[c].second];
			return next_node;

		}
		else{
			double min_volume = DBL_MAX;
			for(auto it = COV.begin(); it != COV.end(); ++it){
				int node_id = *it;
				if(tree_nodes_[node_id]->Area() < min_volume){
					min_volume = tree_nodes_[node_id]->Area();
					next_node = tree_nodes_[node_id];
				}
			}
			return next_node;
		}
	}
	return next_node;
}

TreeNode* RTree::InsertStepByStep(const Rectangle *rectangle, TreeNode *tree_node, INSERT_STRATEGY strategy) {
	TreeNode* next_node = nullptr;
	if (tree_node->is_leaf) {
		//this tree node is a leaf node
		if (tree_node->entry_num == 0) {
			tree_node->Set(*rectangle);
		}
		else {
			tree_node->Include(*rectangle);
		}
		tree_node->AddChildren(rectangle->id_);
	}
	else {
		//this tree node is an internal node
		switch (strategy) {
		case INS_AREA: {
			double min_area_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double area_increase = new_rectangle.Area() - it->Area();
				if (area_increase < min_area_increase) {
					//choose the subtree with the smaller area increase
					min_area_increase = area_increase;
					next_node = it;
				}
				else if (area_increase == min_area_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_MARGIN: {
			double min_margin_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double margin_increase = new_rectangle.Perimeter() - it->Perimeter();
				if (margin_increase < min_margin_increase) {
					//choose the subtree with smaller perimeter increase
					next_node = it;
					min_margin_increase = margin_increase;
				}
				else if (margin_increase == min_margin_increase) {
					//break the tie by favoring the smaller MBR
					if (next_node->Area() > it->Area()) {
						next_node = it;
					}
				}
			}
			break;
		}
		case INS_OVERLAP: {
			double min_overlap_increase = DBL_MAX;
			for (int idx = 0; idx < tree_node->entry_num; idx++) {
				TreeNode* it = tree_nodes_[tree_node->children[idx]];
				Rectangle new_rectangle = it->Merge(*rectangle);
				double overlap_increase = 0;
				for (int idx2 = 0; idx2 < tree_node->entry_num; idx2++) {
					if (idx == idx2)continue;
					TreeNode* it2 = tree_nodes_[tree_node->children[idx2]];
					//overlap_increase += Overlap(new_rectangle, it2->bounding_box) - Overlap(it->bounding_box, it2->bounding_box);
					overlap_increase += SplitOverlap(new_rectangle, *it2) - SplitOverlap(*it, *it2);
				}
				if (overlap_increase < min_overlap_increase) {
					//choose the subtree with smaller overlap increase
					min_overlap_increase = overlap_increase;
					next_node = it;
				}
				else if (overlap_increase == min_overlap_increase) {
					//break the tie by favoring the one with smaller area increase
					//double area_increase = Area(Merge(rectangle, it->bounding_box)) - Area(it->bounding_box);
					double area_increase = it->Merge(*rectangle).Area() - it->Area();
					if (area_increase < next_node->Merge(*rectangle).Area() - next_node->Area()) {
						next_node = it;
					}

				}
			}
			break;
		}
		case INS_RANDOM: {
			int chosen_child = rand() % tree_node->entry_num;
			next_node = tree_nodes_[tree_node->children[chosen_child]];
			break;
		}
		}
		tree_node->Set(tree_node->Merge(*rectangle));
	}

	return next_node;
}

void RTree::GetInsertStates6(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 6 * TreeNode::maximum_entry;
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	double max_area = 0;
	double max_perimeter = 0;
	double max_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 6;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
		if(old_area > max_area){
			max_area = old_area;
		}
		if(old_perimeter > max_perimeter){
			max_perimeter = old_perimeter;
		}
		if(old_overlap > max_overlap){
			max_overlap = old_overlap;
		}
		states[pos] = child->Area();
		states[pos+1] = child->Perimeter();
		states[pos+2] = old_overlap;
		states[pos+3] = new_area - old_area;
		states[pos+4] = new_perimeter - old_perimeter;
		states[pos+5] = new_overlap - old_overlap;
	}
	for(int i=tree_node->entry_num; i < TreeNode::maximum_entry; i++){
		int loc = (i - tree_node->entry_num) % tree_node->entry_num;
		for(int j=0; j<6; j++){
			states[i * 6 + j] = states[loc*6 +j];
		}
	}
	for(int i=0; i<size; i++){
		switch(i%6){
			case 0:{
				states[i] = states[i] / (max_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_overlap + 0.001);
				break;
			}
			case 3:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 4:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 5:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}

void RTree::GetInsertStates3(TreeNode *tree_node, Rectangle *rec, double *states){
	int size = 3 * TreeNode::maximum_entry;
	Rectangle new_rectangle;
	double max_delta_area = 0;
	double max_delta_perimeter = 0;
	double max_delta_overlap = 0;
	for(int i = 0; i < tree_node->entry_num; i++){
		int pos = i * 3;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		double old_area = child->Area();
		double new_area = new_rectangle.Area();

		double old_perimeter = child->Perimeter();
		double new_perimeter = new_rectangle.Perimeter();

		double old_overlap = 0;
		double new_overlap = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_overlap += SplitOverlap(*child, *other_child);
			new_overlap += SplitOverlap(new_rectangle, *other_child);
		}
		states[pos] = new_area - old_area;
		states[pos+1] = new_perimeter - old_perimeter;
		states[pos+2] = new_overlap - old_overlap;
		if(new_area - old_area > max_delta_area){
			max_delta_area = new_area - old_area;
		}
		if(new_perimeter - old_perimeter > max_delta_perimeter){
			max_delta_perimeter = new_perimeter - old_perimeter;
		}
		if(new_overlap - old_overlap > max_delta_overlap){
			max_delta_overlap = new_overlap - old_overlap;
		}
	}
	for(int i = tree_node->entry_num; i < TreeNode::maximum_entry; i++){
		int loc = (i - tree_node->entry_num) % tree_node->entry_num;
		for(int j=0; j<3; j++){
			states[i*3 + j] = states[loc*3 + j];
		}
	}
	for(int i=0; i<size; i++){
		switch(i%3){
			case 0:{
				states[i] = states[i] / (max_delta_area + 0.001);
				break;
			}
			case 1:{
				states[i] = states[i] / (max_delta_perimeter + 0.001);
				break;
			}
			case 2:{
				states[i] = states[i] / (max_delta_overlap + 0.001);
				break;
			}
		}
	}
}
void RTree::GetInsertStates(TreeNode *tree_node, Rectangle* rec, double *states){
	int size = 6 + 9 * TreeNode::maximum_entry;
	for(int i=0; i < size; i++){
		states[i] = 0;
	}
	states[0] = rec->Left();
	states[1] = rec->Bottom();
	states[2] = rec->Right() - rec->Left();
	states[3] = rec->Top() - rec->Bottom();
	states[4] = states[3] / states[2];
	states[5] = states[3] * states[2];
	Rectangle new_rectangle;
	for(int i=0; i < tree_node->entry_num; i++){
		int pos = 6 + i * 9;
		int child_id = tree_node->children[i];
		TreeNode* child = tree_nodes_[child_id];
		new_rectangle.Set(*child);
		new_rectangle.Include(*rec);
		states[pos] = child->Left();
		states[pos+1] = child->Bottom();
		states[pos+2] = child->Right() - child->Left();
		states[pos+3] = child->Top() - child->Bottom();
		states[pos+4] = states[pos+3] / states[pos+2];
		states[pos+5] = child->Area();
		states[pos+6] = new_rectangle.Area() - child->Area();
		states[pos+7] = new_rectangle.Perimeter() - child->Perimeter();
		double old_ovlp = 0;
		double new_ovlp = 0;
		for(int j=0; j<tree_node->entry_num; j++){
			if(i == j)continue;
			TreeNode* other_child = tree_nodes_[tree_node->children[j]];
			old_ovlp += SplitOverlap(*child, *other_child);
			new_ovlp += SplitOverlap(new_rectangle, *other_child);
		} 
		states[pos+8] = new_ovlp - old_ovlp;
	}
}

void RTree::GetSplitStates(TreeNode* tree_node, double* states) {
	int size_per_dim = TreeNode::maximum_entry - 2 * TreeNode::minimum_entry + 2;
	//cout << "size_per_dim " << size_per_dim << endl;
	double max_area = -DBL_MAX;
	double max_perimeter = -DBL_MAX;
	double max_overlap = -DBL_MAX;
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		//cout << "minimum_entry-1: " << TreeNode::minimum_entry - 1 << " maximum_entry - minimum_entry+1: " << TreeNode::maximum_entry - TreeNode::minimum_entry + 1 << endl;
		int loc = 0;
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		
		}
		
		sort(recs.begin(), recs.end(), SortedByRight);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
		sort(recs.begin(), recs.end(), SortedByBottom);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
		sort(recs.begin(), recs.end(), SortedByTop);
		prefix = MergeRange<Rectangle>(recs, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<Rectangle>(recs, TreeNode::maximum_entry - TreeNode::minimum_entry+1, recs.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*recs[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*recs[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
		
	}
	else {
		vector<TreeNode*> nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		int loc = 0;
		sort(nodes.begin(), nodes.end(), SortedByLeft);
		Rectangle prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		Rectangle suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
		sort(nodes.begin(), nodes.end(), SortedByRight);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
		sort(nodes.begin(), nodes.end(), SortedByBottom);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
		sort(nodes.begin(), nodes.end(), SortedByTop);
		prefix = MergeRange<TreeNode>(nodes, 0, TreeNode::minimum_entry - 1);
		suffix = MergeRange<TreeNode>(nodes, TreeNode::maximum_entry - TreeNode::minimum_entry+1, nodes.size());
		for (int i = TreeNode::minimum_entry - 1; i < TreeNode::maximum_entry - TreeNode::minimum_entry+1; i++) {
			prefix.Include(*nodes[i]);
			Rectangle remaining(suffix);
			for (int j = i + 1; j < TreeNode::maximum_entry - TreeNode::minimum_entry+1; j++) {
				remaining.Include(*nodes[j]);
			}
			states[loc] = prefix.Area();
			//cout << "states[" << loc << "] = " << prefix.Area() << endl;;
			loc += 1;
			states[loc] = remaining.Area();
			//cout << "states[" << loc << "] = " << remaining.Area() << endl;;
			loc += 1;
			states[loc] = prefix.Perimeter();
			//cout << "states[" << loc << "] = " << prefix.Perimeter() << endl;;
			loc += 1;
			states[loc] = remaining.Perimeter();
			//cout << "states[" << loc << "] = " << remaining.Perimeter() << endl;;
			loc += 1;
			states[loc] = SplitOverlap(prefix, remaining);
			//cout << "states[" << loc << "] = " << SplitOverlap(prefix, remaining) << endl;;
			loc += 1;
			max_area = max(max_area, max(prefix.Area(), remaining.Area()));
			max_perimeter = max(max_perimeter, max(prefix.Perimeter(), remaining.Perimeter()));
			max_overlap = max(max_overlap, SplitOverlap(prefix, remaining));
		}
	}
	for (int i = 0; i < 240; i++) {
		switch (i % 5) {
		case 0: {
			states[i] = states[i] / (max_area + 0.01);
			break;
		}
		case 1: {
			states[i] = states[i] / (max_area + 0.01);
			break;
		}
		case 2: {
			states[i] = states[i] / (max_perimeter + 0.01);
			break;
		}
		case 3: {
			states[i] = states[i] / (max_perimeter + 0.01);
			break;
		}
		case 4: {
			states[i] = states[i] / (max_overlap + 0.01);
			break;
		}

		}

	}
}

void RTree::SplitAREACost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	double minimum_area = DBL_MAX;
	double minimum_overlap = DBL_MAX;
	//choose the split with the minimum total area, break the tie by preferring the split with smaller overlap
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		//sort by left
		sort(recs.begin(), recs.end(), SortedByLeft);
		Rectangle rec1;
		Rectangle rec2;
		int split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

		//sort by right
		sort(recs.begin(), recs.end(), SortedByRight);
		split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

		//sort by bottom
		sort(recs.begin(), recs.end(), SortedByBottom);
		split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

		//sort by top
		sort(recs.begin(), recs.end(), SortedByTop);
		split = FindMinimumSplit<Rectangle>(recs, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	else {
		vector<TreeNode*> child_nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		//sort by left
		sort(child_nodes.begin(), child_nodes.begin() + tree_node->entry_num, SortedByLeft);
		Rectangle rec1;
		Rectangle rec2;
		int split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(child_nodes.begin(), child_nodes.end(), SortedByRight);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(child_nodes.begin(), child_nodes.end(), SortedByTop);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitArea, SplitOverlap, minimum_area, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}

	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}


void RTree::SplitMARGINCost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	double minimum_perimeter = DBL_MAX;
	double minimum_overlap = DBL_MAX;
	//choose the split with the minimum total perimeter, break the tie by preferring the split with smaller overlap

	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(recs.begin(), recs.end(), SortedByLeft);
		int split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(recs.begin(), recs.end(), SortedByRight);
		split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom 
		sort(recs.begin(), recs.end(), SortedByBottom);
		split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(recs.begin(), recs.end(), SortedByTop);
		split = FindMinimumSplit<Rectangle>(recs, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	else {
		vector<TreeNode*> child_nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
		int split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(child_nodes.begin(), child_nodes.end(), SortedByRight);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(child_nodes.begin(), child_nodes.end(), SortedByTop);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitPerimeter, SplitOverlap, minimum_perimeter, minimum_overlap, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}

void RTree::SplitOVERLAPCost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	double minimum_overlap = DBL_MAX;
	double minimum_area = DBL_MAX;
	//choose the split with the minimum overlap, break the tie by preferring the split with smaller total area
	if (tree_node->is_leaf) {
		vector<Rectangle*> recs(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(recs.begin(), recs.end(), SortedByLeft);
		int split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(recs.begin(), recs.end(), SortedByRight);
		split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(recs.begin(), recs.end(), SortedByBottom);
		split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(recs.begin(), recs.end(), SortedByTop);
		split = FindMinimumSplit<Rectangle>(recs, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	else {
		vector<TreeNode*> child_nodes(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
		Rectangle rec1;
		Rectangle rec2;
		//sort by left
		sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
		int split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by right
		sort(child_nodes.begin(), child_nodes.end(), SortedByRight);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by bottom
		sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
		//sort by top
		sort(child_nodes.begin(), child_nodes.end(), SortedByTop);
		split = FindMinimumSplit<TreeNode>(child_nodes, SplitOverlap, SplitArea, minimum_overlap, minimum_area, rec1, rec2);
		if (split >= 0) {
			bounding_box1.Set(rec1);
			bounding_box2.Set(rec2);
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}

void RTree::SplitGREENECost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	int seed1 = -1;
	int seed2 = -1;
	double max_waste = - DBL_MAX;
	for (int i = 0; i < tree_node->entry_num - 1; i++) {
		for (int j = i + 1; j < tree_node->entry_num; j++) {
			double waste = 0;
			int id1 = tree_node->children[i];
			int id2 = tree_node->children[j];
			if (tree_node->is_leaf) {
				/*if(debug){
					cout << "id1 " << id1 << " id2 " << id2 << endl;
					cout << objects_[id1]->Area() << " " << objects_[id2]->Area() << " " << objects_[id1]->Merge(*objects_[id2]).Area() << endl;
					cout << objects_[id1]->left_ << " " << objects_[id1]->right_ << " " << objects_[id1]->bottom_ << " " << objects_[id1]->top_ << endl;
					cout << objects_[id2]->left_ << " " << objects_[id2]->right_ << " " << objects_[id2]->bottom_ << " " << objects_[id2]->top_ << endl;
					Rectangle r = objects_[id1]->Merge(*objects_[id2]);
					cout << r.left_ << " " << r.right_ << " " << r.bottom_ << " " << r.top_ << endl;
				}*/
				waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
				/*if (debug) {
					cout << "waste: " << waste << "max waste: "<<max_waste<<" "<<(waste > max_waste)<<" "<<(waste < max_waste)<<endl;
				}*/
				//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
			}
			else {
				waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
				//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
			}
			/*cout << "waste: " << waste << endl;
			getchar();*/
			if (waste > max_waste) {
				max_waste = waste;
				seed1 = id1;
				seed2 = id2;
			}
		}
	}
	/*if (debug) {
		cout << "seeds found" << " " << seed1 << " " << seed2 << endl;
	}*/
	//cout << "seeds found" <<" "<<seed1<<" "<<seed2<< endl;
	double max_seed_left, min_seed_right, max_seed_bottom, min_seed_top;
	if (tree_node->is_leaf) {
		max_seed_left = max(objects_[seed1]->Left(), objects_[seed2]->Left());
		min_seed_right = min(objects_[seed1]->Right(), objects_[seed2]->Right());
		max_seed_bottom = max(objects_[seed1]->Bottom(), objects_[seed2]->Bottom());
		min_seed_top = min(objects_[seed1]->Top(), objects_[seed2]->Top());
	}
	else {
		max_seed_left = max(tree_nodes_[seed1]->Left(), tree_nodes_[seed2]->Left());
		min_seed_right = min(tree_nodes_[seed1]->Right(), tree_nodes_[seed2]->Right());
		max_seed_bottom = max(tree_nodes_[seed1]->Bottom(), tree_nodes_[seed2]->Bottom());
		min_seed_top = min(tree_nodes_[seed1]->Top(), tree_nodes_[seed2]->Top());
	}
	//cout << max_seed_left << " " << min_seed_right << " " << max_seed_bottom << " " << min_seed_top << endl;
	double x_seperation = min_seed_right > max_seed_left ? (min_seed_right - max_seed_left) : (max_seed_left - min_seed_right);
	double y_seperation = min_seed_top > max_seed_bottom ? (min_seed_top - max_seed_bottom) : (max_seed_bottom - min_seed_top);

	x_seperation = x_seperation / (tree_node->Right() - tree_node->Left());
	y_seperation = y_seperation / (tree_node->Top() - tree_node->Bottom());
	//cout << "seperation computed" << endl;
	vector<Rectangle*> recs;
	vector<TreeNode*> child_nodes;
	if (tree_node->is_leaf) {
		recs.resize(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			recs[i] = objects_[tree_node->children[i]]; 
		}
	}
	else {
		child_nodes.resize(tree_node->entry_num);
		for (int i = 0; i < tree_node->entry_num; i++) {
			child_nodes[i] = tree_nodes_[tree_node->children[i]];
		}
	}
	if (x_seperation < y_seperation) {
		if (tree_node->is_leaf) {
			sort(recs.begin(), recs.end(), SortedByBottom);
		}
		else {
			sort(child_nodes.begin(), child_nodes.end(), SortedByBottom);
		}
	}
	else {
		if (tree_node->is_leaf) {
			sort(recs.begin(), recs.end(), SortedByLeft);
		}
		else {
			sort(child_nodes.begin(), child_nodes.end(), SortedByLeft);
		}
	}
	if (tree_node->is_leaf) {
		bounding_box1 = MergeRange<Rectangle>(recs, 0, tree_node->entry_num / 2);
		bounding_box2 = MergeRange<Rectangle>(recs, tree_node->entry_num - tree_node->entry_num / 2, tree_node->entry_num);
		if (tree_node->entry_num % 2 == 1) {
			Rectangle rec1 = bounding_box1.Merge(*recs[tree_node->entry_num / 2]);
			Rectangle rec2 = bounding_box2.Merge(*recs[tree_node->entry_num / 2]);
			double area_increase1 = rec1.Area() - bounding_box1.Area();
			double area_increase2 = rec2.Area() - bounding_box2.Area();
			if (area_increase1 < area_increase2) {
				bounding_box1 = rec1;
			}
			else {
				bounding_box2 = rec2;
			}
		}
	}
	else {
		bounding_box1 = MergeRange<TreeNode>(child_nodes, 0, tree_node->entry_num / 2);
		bounding_box2 = MergeRange<TreeNode>(child_nodes, tree_node->entry_num - tree_node->entry_num / 2, tree_node->entry_num);
		if (tree_node->entry_num % 2 == 1) {
			Rectangle rec1 = bounding_box1.Merge(*child_nodes[tree_node->entry_num / 2]);
			Rectangle rec2 = bounding_box2.Merge(*child_nodes[tree_node->entry_num / 2]);
			double area_increase1 = rec1.Area() - bounding_box1.Area();
			double area_increase2 = rec2.Area() - bounding_box2.Area();
			if (area_increase1 < area_increase2) {
				bounding_box1 = rec1;
			}
			else {
				bounding_box2 = rec2;
			}
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();
}

void RTree::SplitQUADRATICCost(TreeNode* tree_node, vector<double>& values, Rectangle& bounding_box1, Rectangle& bounding_box2) {
	int seed1 = -1;
	int seed2 = -1;
	int seed_idx1 = -1, seed_idx2 = -1;
	double max_waste = - DBL_MAX;
	//find the pair of children that waste the most area were they to be inserted in the same node
	for (int i = 0; i < tree_node->entry_num - 1; i++) {
		for (int j = i + 1; j < tree_node->entry_num; j++) {
			double waste = 0;
			int id1 = tree_node->children[i];
			int id2 = tree_node->children[j];
			if (tree_node->is_leaf) {
				waste = objects_[id1]->Merge(*objects_[id2]).Area() - objects_[id1]->Area() - objects_[id2]->Area();
				//waste = ((Rectangle*)tree_node->children[i])->Merge(*((Rectangle*)tree_node->children[j])).Area() - ((Rectangle*)tree_node->children[i])->Area() - ((Rectangle*)tree_node->children[j])->Area();
			}
			else {
				waste = tree_nodes_[id1]->Merge(*tree_nodes_[id2]).Area() - tree_nodes_[id1]->Area() - tree_nodes_[id2]->Area();
				//waste = tree_node->children[i]->Merge(*tree_node->children[j]).Area() - tree_node->children[i]->Area() - tree_node->children[j]->Area();
			}
			if (waste > max_waste) {
				max_waste = waste;
				seed1 = id1;
				seed2 = id2;
				seed_idx1 = i;
				seed_idx2 = j;
			}
		}
	}
	int entry_count1 = 1;
	int entry_count2 = 1;
	if (tree_node->is_leaf) {
		bounding_box1.Set(*objects_[seed1]);
		bounding_box2.Set(*objects_[seed2]);
	}
	else {
		bounding_box1.Set(*tree_nodes_[seed1]);
		bounding_box2.Set(*tree_nodes_[seed2]);
	}
	//list<TreeNode*> unassigned_entry;
	list<int> unassigned_entry;
	for (int i = 0; i < tree_node->entry_num; i++) {
		if (i == seed_idx1 || i == seed_idx2)continue;
		unassigned_entry.push_back(tree_node->children[i]);
	}
	while (!unassigned_entry.empty()) {
		//make sure the two child nodes are balanced.
		if (unassigned_entry.size() + entry_count1 == TreeNode::minimum_entry) {
			if (tree_node->is_leaf) {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					Rectangle* rec_ptr = objects_[*it];
					bounding_box1.Include(*rec_ptr);
				}
			}
			else {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					TreeNode* node_ptr = tree_nodes_[*it];
					bounding_box1.Include(*node_ptr);
				}
			}
			break;
		}
		if (unassigned_entry.size() + entry_count2 == TreeNode::minimum_entry) {
			if (tree_node->is_leaf) {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					Rectangle* rec_ptr = objects_[*it];
					bounding_box2.Include(*rec_ptr);
				}
			}
			else {
				for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
					TreeNode* node_ptr = tree_nodes_[*it];
					bounding_box2.Include(*node_ptr);
				}
			}
			break;
		}
		//pick next: pick an unassigned entry that maximizes the difference between adding into different groups
		double max_difference = - DBL_MAX;
		double new_area1 = 0, new_area2 = 0;
		list<int>::iterator iter;
		int next_entry = -1;
		for (auto it = unassigned_entry.begin(); it != unassigned_entry.end(); ++it) {
			double d1 = 0, d2 = 0;
			if (tree_node->is_leaf) {
				d1 = bounding_box1.Merge(*objects_[*it]).Area();
				d2 = bounding_box2.Merge(*objects_[*it]).Area();
			}
			else {
				d1 = bounding_box1.Merge(*tree_nodes_[*it]).Area();
				d2 = bounding_box2.Merge(*tree_nodes_[*it]).Area();
			}
			double difference = d1 > d2 ? d1 - d2 : d2 - d1;
			if (difference > max_difference) {
				max_difference = difference;
				iter = it;
				next_entry = *it;
				new_area1 = d1;
				new_area2 = d2;
			}
		}
		unassigned_entry.erase(iter);
		//add the entry to the group with smaller area
		Rectangle *chosen_bounding_box = nullptr;
		if (new_area1 < new_area2) {
			chosen_bounding_box = &bounding_box1;
			entry_count1 += 1;
		}
		else if (new_area1 > new_area2) {
			chosen_bounding_box = &bounding_box2;
			entry_count2 += 1;
		}
		else {
			if (entry_count1 < entry_count2) {
				entry_count1 += 1;
				chosen_bounding_box = &bounding_box1;
			}
			else {
				entry_count2 += 1;
				chosen_bounding_box = &bounding_box2;
			}
		}
		if (tree_node->is_leaf) {
			chosen_bounding_box->Include(*objects_[next_entry]);
		}
		else {
			chosen_bounding_box->Include(*tree_nodes_[next_entry]);
		}
	}
	if (values.size() != 5) {
		values.resize(5);
	}
	values[0] = bounding_box1.Perimeter() / tree_node->Perimeter();
	values[1] = bounding_box2.Perimeter() / tree_node->Perimeter();
	values[2] = bounding_box1.Area() / tree_node->Area();
	values[3] = bounding_box2.Area() / tree_node->Area();
	values[4] = SplitOverlap(bounding_box1, bounding_box2) / tree_node->Area();

}


RTree* ConstructTree(int max_entry, int min_entry) {
	TreeNode::maximum_entry = max_entry;
	TreeNode::minimum_entry = min_entry;
	RTree* rtree = new RTree();
	return rtree;
}

void SetDefaultInsertStrategy(RTree* rtree, int strategy) {
	switch (strategy) {
	case 0: {
		rtree->insert_strategy_ = INS_AREA;
		break;
	}
	case 1: {
		rtree->insert_strategy_ = INS_MARGIN;
		break;
	}
	case 2: {
		rtree->insert_strategy_ = INS_OVERLAP;
		break;
	}
	case 3: {
		rtree->insert_strategy_ = INS_RANDOM;
		break;
	}
	}	
}
void SetDefaultSplitStrategy(RTree* rtree, int strategy) {
	switch (strategy) {
	case 0: {
		rtree->split_strategy_ = SPL_MIN_AREA;
		break;
	}
	case 1: {
		rtree->split_strategy_ = SPL_MIN_MARGIN;
		break;
	}
	case 2: {
		rtree->split_strategy_ = SPL_MIN_OVERLAP;
		break;
	}
	case 3: {
		rtree->split_strategy_ = SPL_QUADRATIC;
		break;
	}
	case 4: {
		rtree->split_strategy_ = SPL_GREENE;
	}
	}
}

int QueryRectangle(RTree* rtree, double left, double right, double bottom, double top) {
	Rectangle rec(left, right, bottom, top);
	rtree->Query(rec);
	int node_access = rtree->stats_.node_access;
	return node_access;
}

TreeNode* GetRoot(RTree* rtree) {
	return rtree->tree_nodes_[rtree->root_];
}

void SetDebug(RTree* rtree, int value) {
	rtree->debug = value;
}

int IsLeaf(TreeNode* node) {
	if (node->is_leaf) {
		return 1;
	}
	else {
		return 0;
	}
}

Rectangle* InsertRec(RTree* rtree, double left, double right, double bottom, double top) {
	Rectangle* rec = rtree->InsertRectangle(left, right, bottom, top);
	return rec;
}

TreeNode* InsertOneStep(RTree* rtree, Rectangle* rec, TreeNode* node, int strategy) {
	INSERT_STRATEGY ins_strat;
	switch (strategy) {
	case 0: {
		ins_strat = INS_AREA;
		break;
	}
	case 1: {
		ins_strat = INS_MARGIN;
		break;
	}
	case 2: {
		ins_strat = INS_OVERLAP;
		break;
	}
	case 3: {
		ins_strat = INS_RANDOM;
		break;
	}
	}
	TreeNode* next_iter = rtree->InsertStepByStep(rec, node, ins_strat);
	return next_iter;
}



TreeNode* SplitOneStep(RTree* rtree, TreeNode* node, int strategy) {
	SPLIT_STRATEGY spl_strat;
	switch (strategy) {
	case 0: {
		spl_strat = SPL_MIN_AREA;
		break;
	}
	case 1: {
		spl_strat = SPL_MIN_MARGIN;
		break;
	}
	case 2: {
		spl_strat = SPL_MIN_OVERLAP;
		break;
	}
	case 3: {
		spl_strat = SPL_QUADRATIC;
		break;
	}
	case 4: {
		spl_strat = SPL_GREENE;
	}
	}
	TreeNode* next_node = rtree->SplitStepByStep(node, spl_strat);
	return next_node;
}

int IsOverflow(TreeNode* node) {
	if (node->is_overflow) {
		return 1;
	}
	else {
		return 0;
	}
}

void Swap(Rectangle& rec1, Rectangle& rec2){
    Rectangle tmp(rec1);
    rec1.Set(rec2);
    rec2.Set(tmp);
}

void CheckOrder(Rectangle& rec1, Rectangle& rec2){
    //check the order of rec1 and rec2, sorted by left, right, bottom, top
    if(rec1.left_ < rec2.left_){
        return;
    }
    if(rec1.left_ > rec2.left_){
        Swap(rec1, rec2);
        return;
    }
    if(rec1.right_ < rec2.right_){
        return;
    }
    if(rec1.right_ > rec2.right_){
        Swap(rec1, rec2);
        return;
    }
    if(rec1.bottom_ < rec2.bottom_){
        return;
    }
    if(rec1.bottom_ > rec2.bottom_){
        Swap(rec1, rec2);
        return;
    }
    if(rec1.top_ < rec2.top_){
        return;
    }
    if(rec1.top_ > rec2.top_){
        Swap(rec1, rec2);
        return;
    }
    return;
}

bool IsSameSplit(Rectangle& rec00, Rectangle& rec01, Rectangle& rec10, Rectangle& rec11){
    bool is_same1 = false, is_same2 = false;
    if(rec00.left_ == rec10.left_ && rec00.right_ == rec10.right_ && rec00.bottom_ == rec10.bottom_ && rec00.top_ == rec10.top_){
        is_same1 = true;
    }
    if(rec01.left_ == rec11.left_ && rec01.right_ == rec11.right_ && rec01.bottom_ == rec11.bottom_ && rec01.top_ == rec11.top_){
        is_same2 = true;
    }
    return (is_same1 && is_same2);
}



void RetrieveSpecialStates(RTree* tree, TreeNode* tree_node, double* states) {
	tree->GetSplitStates(tree_node, states);
}

void RetrieveSpecialInsertStates(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates(tree_node, rec, states);
}

void RetrieveSpecialInsertStates3(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates3(tree_node, rec, states);
}

void RetrieveSpecialInsertStates6(RTree* tree, TreeNode* tree_node, Rectangle* rec, double* states){
	tree->GetInsertStates6(tree_node, rec, states);
}


int RetrieveStates(RTree* tree, TreeNode* tree_node, double* states) {
	vector<double> values(5, 0);
	Rectangle rec[5][2];
	/*if (tree->debug) {
		cout << "tree_node->id=" << tree_node->id_ << " is_leaf: " << tree_node->is_leaf << " is_overflow: " << tree_node->is_overflow <<" entry_num: "<<tree_node->entry_num<< endl;
		cout << "retrieving states" << endl;
	}*/
	//cout << "tree_node->id=" << tree_node->id_ << " is_leaf: " << tree_node->is_leaf << " is_overflow: " << tree_node->is_overflow <<" entry_num: "<<tree_node->entry_num<< endl;
	//cout << "retrieving states" << endl;
	tree->SplitAREACost(tree_node, values, rec[0][0], rec[0][1]);
	for (int i = 0; i < 5; i++) {
		states[i] = values[i];
	}
	/*if (tree->debug) {
		cout << 1 << endl;
	}*/
	//cout << "1" << endl;
	tree->SplitMARGINCost(tree_node, values, rec[1][0], rec[1][1]);
	for (int i = 0; i < 5; i++) {
		states[5 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 2 << endl;
	}*/
	//cout << "2" << endl;
	tree->SplitOVERLAPCost(tree_node, values, rec[2][0], rec[2][1]);
	for (int i = 0; i < 5; i++) {
		states[10 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 3 << endl;
	}*/
	//cout << "3" << endl;
	tree->SplitGREENECost(tree_node, values, rec[3][0], rec[3][1]);
	for (int i = 0; i < 5; i++) {
		states[15 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 4 << endl;
	}*/
	//cout << "4" << endl;
	tree->SplitQUADRATICCost(tree_node, values, rec[4][0], rec[4][1]);
	for (int i = 0; i < 5; i++) {
		states[20 + i] = values[i];
	}
	/*if (tree->debug) {
		cout << 5 << endl;
	}*/
	//cout << "5" << endl;
	int is_valid = 0;
	Rectangle tmp;
    CheckOrder(rec[0][0], rec[0][1]);
	for(int i=1; i<5; i++){
        CheckOrder(rec[1][0], rec[1][1]);
        if(!IsSameSplit(rec[0][0], rec[0][1], rec[i][0], rec[i][1])){
            is_valid = 1;
            break;
        }
	}
	return is_valid;
}

int GetChildNum(TreeNode* node){
	return node->entry_num;
}

void GetMBR(RTree* rtree, double* boundary){
	TreeNode* root = rtree->tree_nodes_[rtree->root_];
    boundary[0] = root->left_;
    boundary[1] = root->right_;
    boundary[2] = root->bottom_;
    boundary[3] = root->top_;
}

void PrintTree(RTree* rtree) {
	rtree->Print();
}

void PrintEntryNum(RTree* rtree) {
	rtree->PrintEntryNum();
}


TreeNode* DirectInsert(RTree* rtree, Rectangle* rec) {
	TreeNode* iter = rtree->Root();
	while (true) {
		TreeNode* next_iter = rtree->InsertStepByStep(rec, iter);
		if (next_iter != nullptr) {
			iter = next_iter;
		}
		else {
			break;
		}
	}
	return iter;
}



TreeNode* RRInsert(RTree* rtree, Rectangle* rec){
	TreeNode* iter = rtree->Root();
	while(true){
		TreeNode* next_iter = rtree->RRInsert(rec, iter);
		if(next_iter != nullptr){
			iter = next_iter;
		}
		else{
			break;
		}
	}
	return iter;
}

void RRSplit(RTree* rtree, TreeNode* node){
	TreeNode* iter = node;
	while(iter->is_overflow){
		iter = rtree->RRSplit(iter);
	}	
}

void DirectSplit(RTree* rtree, TreeNode* node) {
	TreeNode* iter = node;
	while (iter->is_overflow) {
		iter = rtree->SplitStepByStep(iter);
	}
}

int TryInsert(RTree* rtree, Rectangle* rec) {
	TreeNode* iter = rtree->Root();
	while (true) {
		TreeNode* next_iter = rtree->TryInsertStepByStep(rec, iter);
		if (next_iter != nullptr) {
			iter = next_iter;
		}
		else {
			if (iter->entry_num < TreeNode::maximum_entry) {
				if (iter->entry_num == 0) {
					iter->Set(*rec);
				}
				else {
					iter->Include(*rec);
				}
				iter->AddChildren(rec->id_);
				while (iter->father >= 0) {
					next_iter = rtree->tree_nodes_[iter->father];
					next_iter->Include(*rec);
					iter = next_iter;
				}
				return 1;
			}
			else {
				return 0;
			}
		}
	}
}

void DefaultSplit(RTree* rtree, TreeNode* tree_node){
	TreeNode* iter = tree_node;
	while(iter->is_overflow){
		iter = rtree->SplitStepByStep(iter);

	}
}

void DefaultInsert(RTree* rtree, Rectangle* rec) {
	TreeNode* iter = rtree->Root();
	
	while (true) {
		TreeNode* next_iter = rtree->InsertStepByStep(rec, iter);
		if (next_iter != nullptr) {
			iter = next_iter;
		}
		else {
			break;
		}
	}
	while (iter->is_overflow) {
		iter = rtree->SplitStepByStep(iter);
	}
}

int TotalTreeNode(RTree* rtree) {
	return (int)rtree->tree_nodes_.size();
}

int TreeHeight(RTree* rtree){
	return rtree->height_;
}

void Clear(RTree* rtree) {
	for(int i=0; i<rtree->objects_.size(); i++){
		delete rtree->objects_[i];
	}
	for(int i=0; i<rtree->tree_nodes_.size(); i++){
		delete rtree->tree_nodes_[i];
	}
	rtree->height_ = 1;
	rtree->tree_nodes_.clear();
	rtree->objects_.clear();
	rtree->root_ = rtree->CreateNode()->id_;
	rtree->tree_nodes_[rtree->root_]->is_leaf = true;
}

void RTree::Copy(RTree* tree) {
	for (int i = 0; i < objects_.size(); i++) {
		delete objects_[i];
	}
	for (int i = 0; i < tree_nodes_.size(); i++) {
		delete tree_nodes_[i];
	}
	objects_.resize(tree->objects_.size());
	tree_nodes_.resize(tree->tree_nodes_.size());
	for (int i = 0; i < objects_.size(); i++) {
		objects_[i] = new Rectangle(*tree->objects_[i]);
		objects_[i]->id_ = tree->objects_[i]->id_;
		assert(objects_[i]->id_ == i);
	}
	height_ = tree->height_;
	for (int i = 0; i < tree_nodes_.size(); i++) {
		tree_nodes_[i] = new TreeNode(tree->tree_nodes_[i]);
		assert(tree_nodes_[i]->id_ == i);
	}
	root_ = tree->root_;
}

void CopyTree(RTree* tree, RTree* from_tree) {
	Clear(tree);
	tree->Copy(from_tree);
}
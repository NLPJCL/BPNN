#include<iostream>
#include<vector>
#include <algorithm>  
#include <Eigen/Dense>
#include"corpus.h"
#include<tuple>
#include"BPNN.h"
using namespace Eigen;
namespace test
{
	void test_column()
	{
		VectorXd v(VectorXd::Constant(3,1));
		std::cout << v << std::endl;
		std::cout << v.transpose() << std::endl;
		std::cout << v << std::endl;

	}
	void test_matrix()
	{
		Eigen::MatrixXd w(2, 2),z(2,2);
		w << 1, 1, 1, 1;
		z << 1, 1, 1, 1;
		std::cout << w.array()*z.array() << std::endl;

	}
	void test_()
	{
		std::vector<std::pair<std::vector<int>, std::vector<int>>>z;
		const std::vector<std::pair<std::vector<int>, std::vector<int>>> w=z;
	}


}
using pre_data = std::vector<std::pair<std::vector<int>, std::vector<int>>>;
int main()
{
	//test::test_matrix();
	//配置文件
	
	
	std::string embed_file = "../data/embed.txt";
	std::string train_file = "../data/ctb5/train.conll";
	int input_size = 100;//预训练向量维度
	int window = 5;
	int hidden_size = 300;
	int epochs = 2;
	int bach_size = 50;
	double eta = 0.5;

	corpus data;
	data.fit(embed_file,train_file);
	pre_data train=data.load(train_file,window);
	pre_data dev= data.load(train_file, window);
	std::vector<int> sizes{ input_size*window, hidden_size, int(data.tags.size()) };
	BPNN bp(sizes,data.extend_embed);

	bp.SGD(train,dev,epochs,bach_size,eta);
	system("pause");
	return 0;
}
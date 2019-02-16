#include<iostream>
#include<vector>
#include <algorithm>  
#include <Eigen/Dense>
#include"corpus.h"
#include"BPNN.h"
using namespace Eigen;
namespace test
{

	Eigen::VectorXd  exp_(Eigen::VectorXd &z)
	{
		return z.array().exp();
	}
	void test_column()
	{
		//std::vector<int> q{ 
		VectorXd q(4);
		q << 1, 1,1,1;
		//q=q.array().exp();
		q =exp_(q).array() *(1-q.array().exp());
		//std::vector<int> &w = q;
		//Eigen::VectorXi y = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(w.data(), w.size());
		
		std::cout << q << std::endl;
	}

	void test_matrix()
	{
		Eigen::MatrixXd w(2, 2),z(2,2);
		w << 1, 1, 1, 1;
		z << 1, 1, 1, 1;
		std::cout << w.array()*z.array() << std::endl;

	}
	void test_move()
	{
		std::vector<std::pair<std::vector<int>, std::vector<int>>>z;
		steady_clock::time_point t1 = steady_clock::now();

		for (int i = 0; i < 10000000; i++)
		{
			std::pair < std::vector<int>, std::vector<int>> q{ {1},{2} };
			z.emplace_back(q);
			std::vector<int> w (q.first);
			std::vector<int> w0(q.second);
		}
		steady_clock::time_point t2 = steady_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		std::cout << "It took me " << time_span.count() << " seconds.";


	}
	void test_vector()
	{
		std::vector<int> train{3,2,1};
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(train.begin(), train.end(), std::default_random_engine(seed));


		for (int i = 0; i < train.size(); i++)
		{
			std::cout << train[i] << " ";

		}
		std::cout << std::endl;
	}




}
using pre_data = std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>;
int main()
{
	//test::test_matrix();
	//配置文件
//	test::test_move();
	//test::test_column();
	//test::test_vector();
	
	std::string embed_file = "../data/embed.txt";
	std::string train_file = "../data/ctb5/train.conll";
	std::string dev_file = "../data/ctb5/dev.conll";
	int input_size = 100;//预训练向量维度
	int window = 5;
	int hidden_size = 300;
	int epochs = 2;
	int bach_size = 50;
	double eta = 0.5;

	corpus data;
	data.fit(embed_file,train_file);
	pre_data train=data.load(train_file,window);
	pre_data dev= data.load(dev_file, window);
	std::vector<int> sizes{ input_size*window, hidden_size, int(data.tags.size()) };
	BPNN bp(sizes,data.extend_embed_matrix);

	bp.SGD(train,dev,epochs,bach_size,eta);
	system("pause");
	return 0;
}
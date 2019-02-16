#pragma once
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include<iostream>
#include <Eigen/Dense>
#include<algorithm>
#include<chrono>
#include<unordered_map>
#include<random>
//using namespace Eigen;
#include<tuple>
using pre_data = std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>;
using pair_x_y = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
using tuple_w_b_x =std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>, Eigen::VectorXd>;
using namespace std::chrono;

class BPNN
{
private:
	//输入层，隐藏层，输出层大小
	std::vector<int> sizes;
	int input_size;
	std::vector<Eigen::MatrixXd> weight;
	std::vector<Eigen::VectorXd> biases;
	Eigen::MatrixXd extend_embed;
	int nl;
public:
	//初始化w和b
	BPNN(const std::vector<int> &sizes_, Eigen::MatrixXd &extend_embed_):sizes(sizes_),extend_embed(extend_embed_),input_size(sizes_[0])
	{
		nl = sizes.size();
		//初始化b
		for (int i = 1; i < nl; ++i)
		{
			Eigen::VectorXd a = Eigen::VectorXd::Random(sizes[i]);
			biases.emplace_back(std::move(a));	
		}
		//初始化w
		for (int i = 1; i < nl; ++i)
		{
			Eigen::MatrixXd a= Eigen::MatrixXd::Random(sizes[i],sizes[i-1]);
			weight.emplace_back(std::move(a));
		}
	}
	void SGD(  pre_data &train,  pre_data &dev,int epochs,int bach_size,double eta)
	{
		double max_dev_precision=0.0, max_train_precision=0.0;

		std::cout << train.size() << std::endl;
		for (int i = 0; i < epochs; i++)
		{
			std::cout << "iteration" << i << std::endl;
			steady_clock::time_point t1 = steady_clock::now();

			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::shuffle(train.begin(), train.end(), std::default_random_engine(seed));
			for (int j= 0; j < train.size();j=j + bach_size)
			{
				if (j <= train.size() - 50)
				{
					pre_data bach(train.begin() + j, train.begin() + j + bach_size);
					update(bach, eta);
				}
				else
				{
					pre_data bach(train.begin() + j, train.end());
					update(bach, eta);
				}
			}
			steady_clock::time_point t2 = steady_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			std::cout << "It took me " << time_span.count() << " seconds.";


			double train_precision = evaluate(train);
			std::cout << "train precision is" << train_precision << std::endl;
			
			double dev_precision = evaluate(dev);
			std::cout << "dev precision is" << dev_precision << std::endl;	
		}	
	}
	double evaluate( pre_data &x_y)
	{
		std::vector<Eigen::VectorXd> z;
		int max, correct=0;
		for (int i = 0; i < x_y.size(); ++i)
		{
			z = forward(x_y[i].first);
			z[z.size()-1].maxCoeff(&max);
			if (x_y[i].second(max) == 1.0)
			{
				correct++;
			}
		}
		return correct / double(x_y.size());

	}

	void update( pre_data &bach,const double &eta)
	{
		std::vector<Eigen::MatrixXd> nabla_w(init_w()),delta_nabla_w;
		std::vector<Eigen::VectorXd> nabla_b(init_b()),delta_nabla_b;
		std::unordered_map<int, Eigen::RowVectorXd> nabla_x;
		Eigen::VectorXd delta_nabla_x;

		tuple_w_b_x delta_nabla_w_b_x;
		//Eigen::RowVectorXd
		for (int i = 0; i < bach.size(); ++i)
		{
			delta_nabla_w_b_x = backprop(bach[i]);

			delta_nabla_w = std::move(std::get<0>(delta_nabla_w_b_x));
			delta_nabla_b = std::move(std::get<1>(delta_nabla_w_b_x));
			delta_nabla_x = std::move(std::get<2>(delta_nabla_w_b_x));
			for (int q = 0; q < delta_nabla_w.size(); q++)
			{
				nabla_w[q] += delta_nabla_w[q];
				nabla_b[q] += delta_nabla_b[q];
			}
			int k = 0;
			for (int j = 0; j< bach[i].first.size(); j++)
			{

				if (nabla_x.find(bach[i].first(j)) == nabla_x.end())
				{
					nabla_x[bach[i].first(j)]= delta_nabla_x.segment(k, 100);
				}
				else
				{
					nabla_x[bach[i].first(j)] += delta_nabla_x.segment(k, 100).transpose();
				}
				k = k + 100;
			}
		}
		//更新梯度
		for (int i = 0; i < weight.size(); i++)
		{
			biases[i] -= nabla_b[i]*(eta / bach.size());
			weight[i] -= nabla_w[i] * (eta / bach.size());
		}
		for (auto w=nabla_x.begin(); w != nabla_x.end(); ++w)
		{
				extend_embed.row(w->first) -= w->second*(eta / bach.size());
		}
	}
	tuple_w_b_x backprop( pair_x_y  &x_y)
	{
		std::vector<Eigen::MatrixXd> nabla_w (init_w());
		std::vector<Eigen::VectorXd> nabla_b(init_b());

		//前向算法
		std::vector<Eigen::VectorXd> zs, activations;
		//将输入经过词嵌入矩阵转化为词向量。
		Eigen::VectorXd a(500), z;
		int j = 0;
		std::vector<double> line;

		//没找到好的api
		for (int i = 0; i < x_y.first.size(); i++)
		{
			for (int k = 0; k < extend_embed.row(x_y.first[i]).size(); k++)
			{
				a(j) = extend_embed(x_y.first(i), k);
				j++;
			}
		}

		//前向计算、

		activations.emplace_back(a);
		for (int i = 0; i < weight.size() - 1; i++)
		{
			z = weight[i] * a + biases[i];
			zs.emplace_back(z);
			a = sigmoid(z);
			activations.emplace_back(a);
		}
		z = weight[weight.size() - 1] * a + biases[biases.size() - 1];
		zs.emplace_back(z);
		//输出用softmax()
		a = softmax(z);
		activations.emplace_back(a);

		//std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> z_a(forward(x_y.first));
		//后向用cross_entropy
	//	std::vector<Eigen::VectorXd> zs( std::move(z_a.first)), activations (std::move( z_a.second));
		Eigen::VectorXd delta;
		delta = activations[activations.size()-1] - x_y.second;
		nabla_b[nabla_b.size()-1] = delta;
		nabla_w[nabla_w.size()-1] = delta*activations[activations.size()-2].transpose();

		for (int i = nabla_b.size()- 2; i >=0;i--)
		{
			z = std::move(zs[i]);
			delta = (weight[i + 1].transpose()*delta).array()*sigmoid_prime(z).array();
			nabla_b[i] = delta;
			nabla_w[i] = delta * activations[i].transpose();
		}
		Eigen::VectorXd nabla_x = weight[0].transpose()*delta;
		return tuple_w_b_x{ nabla_w,nabla_b,nabla_x};
	}
	std::vector<Eigen::VectorXd> forward(Eigen::VectorXd &x)
	{
		std::vector<Eigen::VectorXd> zs, activations;
		//将输入经过词嵌入矩阵转化为词向量。
		Eigen::VectorXd a(500),z;
		int j = 0;
		std::vector<double> line;
		for (int i = 0; i < x.size(); i++)
		{
			for (int k = 0; k < extend_embed.row(x[i]).size(); k++)
			{
				a(j) = extend_embed(x(i), k);
				j++;
			}
		}

		//前向计算、

		activations.emplace_back(a);
		for (int i = 0; i < weight.size() - 1; i++)
		{
			z = weight[i]*a+biases[i];
			zs.emplace_back(z);
			a = sigmoid(z);
			activations.emplace_back(a);
		}
		z = weight[weight.size()-1]*a+biases[biases.size()-1];
		zs.emplace_back(z);
		//输出用softmax()
		a = softmax(z);
		activations.emplace_back(a);
		return activations;
	}
	Eigen::VectorXd sigmoid(const Eigen::VectorXd &z)
	{
		Eigen::VectorXd  a(z.rows());
		for (int i = 0; i < z.rows(); i++)
		{
			a(i) = sigmoid_(z(i));
		}
		return a;
	}
	Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd &z)
	{
		Eigen::VectorXd  a(z.rows());
		for (int i = 0; i < z.rows(); i++)
		{
			a(i) = sigmoid_(z(i))*(1 - sigmoid_(z(i)));
		}
		return a;
	}
	double sigmoid_(const double &z)
	{

		return 1.0 / (1.0 + std::exp(-z));
	}

	Eigen::VectorXd softmax(const Eigen::VectorXd &z)
	{
		Eigen::VectorXd  z_(z - Eigen::VectorXd::Constant(z.rows(), z.maxCoeff()));
		for (int i = 0; i < z.rows(); i++)
		{
			z_(i) = std::exp(z_(i));
		}
		double denominator = z_.sum();
		for (int i = 0; i < z.rows(); i++)
		{
			z_(i) = z_(i) / denominator;
		}
		return z_;
	}
	/*
	Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd &z)
	{
		return sigmoid(z).array()*(1 - sigmoid(z).array());
	}
	 Eigen::VectorXd  sigmoid(const Eigen::VectorXd & z)
	{
		 return 1.0 / (1.0 + z.array().exp());
	}

	Eigen::VectorXd softmax(const Eigen::VectorXd &z)
	{
		Eigen::VectorXd  z_(z - Eigen::VectorXd::Constant(z.rows(), z.maxCoeff()));
		z_=z_.array().exp();
		return z_.array() / z_.sum();
	}
	*/
	std::vector<Eigen::MatrixXd> init_w()
	{
		std::vector<Eigen::MatrixXd> nabla_w;
		for (int i = 0; i < weight.size(); ++i)
		{
			nabla_w.emplace_back(Eigen::MatrixXd::Zero(weight[i].rows(), weight[i].cols()));
		}
		return nabla_w;
	}

	std::vector<Eigen::VectorXd> init_b()
	{
		std::vector<Eigen::VectorXd> nabla_b;
		for (int i = 0; i < biases.size(); ++i)
		{
			nabla_b.emplace_back(Eigen::VectorXd::Zero(biases[i].rows(), biases[i].cols()));
		}
		return nabla_b;
	}

};

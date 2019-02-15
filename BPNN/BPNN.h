#pragma once
#include<iostream>
#include <Eigen/Dense>
#include<algorithm>
#include<chrono>
//using namespace Eigen;
using pre_data = std::vector<std::pair<std::vector<int>, std::vector<int>>>;
using pair_x_y = std::pair<std::vector<int>, std::vector<int>>;
using pair_w_b =std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd>>;
using namespace std::chrono;
class BPNN
{
private:
	//输入层，隐藏层，输出层大小
	std::vector<int> sizes;
	int input_size;
	std::vector<Eigen::MatrixXd> weight;
	std::vector<Eigen::VectorXd> biases;
	std::vector<std::vector<double_t>> extend_embed;
	int nl;
public:
	//初始化w和b
	BPNN(std::vector<int>sizes_, std::vector<std::vector<double_t>> extend_embed_):sizes(sizes_),extend_embed(extend_embed_),input_size(sizes_[0])
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
		for (int i = 0; i < epochs; i++)
		{
			std::cout << "iteration" << i << std::endl;
			steady_clock::time_point t1 = steady_clock::now();
			std::random_shuffle(train.begin(), train.end());
			for (int j= 0; j < train.size();j=j + bach_size)
			{
				pre_data bach(train.begin()+j,train.begin()+j+bach_size);	
				update(bach,eta);
				std::cout << j<<std::endl;
			}

			steady_clock::time_point t2 = steady_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			std::cout << "It took me " << time_span.count() << " seconds.";

			double train_precision = evaluate(train);
			std::cout << "train precision is" << train_precision << std::endl;
		

			double dev_precision = evaluate(train);
			std::cout << "train precision is" << dev_precision << std::endl;
		
		
		}
	}
	double evaluate( pre_data x_y)
	{
		std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> z;
		int max, correct;
		for (int i = 0; i < x_y.size(); ++i)
		{
			z = forward(x_y[i].first);
			z.second[-1].maxCoeff(&max);
			if (x_y[i].second[max] == 1)
			{
				correct++;
			}
		}
		
		return correct / double(x_y.size());

	}

	void update( pre_data &bach,const double &eta)
	{
		std::vector<Eigen::MatrixXd> nabla_w=init_w(),delta_nabla_w;
		std::vector<Eigen::VectorXd> nabla_b=init_b(),delta_nabla_b;
		pair_w_b delta_nabla_w_b;
		for (int i = 0; i < bach.size(); ++i)
		{
			delta_nabla_w_b = backprop(bach[i]);
			delta_nabla_w = delta_nabla_w_b.first;
			delta_nabla_b = delta_nabla_w_b.second;
			for (int i = 0; i < delta_nabla_w.size(); i++)
			{
				nabla_w[i] += delta_nabla_w[i];
				nabla_b[i] += delta_nabla_b[i];
			}
		}

		//更新梯度
		for (int i = 0; i < weight.size(); i++)
		{
			biases[i] -= nabla_b[i]*(eta / bach.size());
			weight[i] -= nabla_w[i] * (eta / bach.size());
		}
	}
	pair_w_b backprop( pair_x_y  &x_y)
	{
		std::vector<Eigen::MatrixXd> nabla_w = init_w();
		std::vector<Eigen::VectorXd> nabla_b=init_b();

		Eigen::VectorXd y(x_y.second.size());
		for (int i = 0; i < x_y.second.size(); i++)
		{
			y(i) = x_y.second[i];
		}
		//前向
		std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> z_a=forward(x_y.first);


		//后向用cross_entropy
		std::vector<Eigen::VectorXd> zs = z_a.first, activations = z_a.second;
		Eigen::VectorXd delta, z;
		delta = activations[activations.size()-1] - y;
		nabla_b[nabla_b.size()-1] = delta;
		nabla_w[nabla_w.size()-1] = delta*activations[activations.size()-2].transpose();

		for (int i = nabla_b.size()- 2; i >=0;i--)
		{
			z = zs[i];
			delta = (weight[i + 1].transpose()*delta).array()*sigmoid_prime(z).array();
			nabla_b[i] = delta;
			nabla_w[i] = delta * activations[i].transpose();
		}

		return pair_w_b{ nabla_w,nabla_b };
	}
	std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> forward(const std::vector<int> &x)
	{
		std::vector<Eigen::VectorXd> zs, activations;
		//将输入经过词嵌入矩阵转化为词向量。
		Eigen::VectorXd a(input_size),z;
		int j = 0;
		for (int i = 0; i < x.size(); i++)
		{
			std::vector<double_t> line = extend_embed[x[i]];
			for (int k = 0; k < line.size(); k++)
			{
				a(j) = line[k];
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
		return std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd >>{zs,activations};
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
			a(i) = sigmoid_(z(i))*(1- sigmoid_(z(i)));
		}
		return a;
	}
	double sigmoid_(const double &z)
	{

		return 1.0 / (1.0 + std::exp(-z));
	}

	Eigen::VectorXd softmax(const Eigen::VectorXd &z)
	{
		Eigen::VectorXd  z_(std::move(z - Eigen::VectorXd::Constant(z.rows(), z.maxCoeff())));
		for (int i = 0; i < z.rows(); i++)
		{
			z_(i) = std::exp(z_(i));
		}
		double denominator = z_.sum();
		for (int i = 0; i < z.rows(); i++)
		{
			z_(i) = z_(i)/denominator;
		}
		return z_;
	}

	std::vector<Eigen::MatrixXd> init_w()
	{
		std::vector<Eigen::MatrixXd> nabla_w;
		for (int i = 0; i < weight.size(); ++i)
		{
			Eigen::MatrixXd a = Eigen::MatrixXd::Zero(weight[i].rows(), weight[i].cols());
			nabla_w.emplace_back(std::move(a));
		}
		return nabla_w;
	}

	std::vector<Eigen::VectorXd> init_b()
	{
		std::vector<Eigen::VectorXd> nabla_b;
		for (int i = 0; i < biases.size(); ++i)
		{
			Eigen::VectorXd a = Eigen::VectorXd::Zero(biases[i].rows(), biases[i].cols());
			nabla_b.emplace_back(std::move(a));
		}
		return nabla_b;
	}

};

#include<fstream>
#include<unordered_map>
#include<vector>
#include<assert.h>
#include <codecvt>
#include<iostream>
#include<random>
#include <Eigen/Dense>
struct sentence
{
	std::vector<std::wstring> word;
	std::vector<std::wstring> tag;
};
class corpus
{

	std::wstring SOS = L"<SOS>";
	std::wstring EOS = L"<EOS>";
	std::wstring UNK = L"<UNK>";
public:
	std::unordered_map<std::wstring, int> words;
	std::unordered_map<std::wstring, int> tags;
	std::vector<std::vector<double_t>> extend_embed;
	Eigen::MatrixXd extend_embed_matrix;
	//构造嵌入矩阵。
	void fit(const std::string &embed_name, const std::string &train_name)
	{
		//处理预训练词向量的数据
		embed(embed_name);
		//处理训练集train的数据
		std::vector<sentence> train = read_data(train_name);
		//获取词一起词性。
		int unk_words = 0;
		for (int i = 0; i < train.size(); ++i)
		{
			for (int j = 0; j < train[i].word.size(); ++j)
			{
				if (tags.find(train[i].tag[j]) == tags.end())
				{
					tags.emplace(train[i].tag[j],tags.size());
				}
				if (words.find(train[i].word[j]) == words.end())
				{
					words.emplace(train[i].word[j], words.size());
					unk_words++;
				}
			}
		}
		extend_embed.reserve(extend_embed.size() + unk_words+3);
		//随机初始化嵌入矩阵。
		for (int i = 0; i < unk_words; ++i)
		{
			extend_embed.emplace_back(std::move(good_randVec()));
		}
		add_word(SOS); //增加词以及对应嵌入矩阵。
		add_word(EOS);
		add_word(UNK);
		extend_embed_matrix.resize(extend_embed.size(), 100);
		for (int i = 0; i < extend_embed.size(); i++)
		{
			for (int j = 0; j < extend_embed[i].size(); j++)
			{
				extend_embed_matrix(i, j) = extend_embed[i][j];
			}
		}
		std::cout << words.size() << std::endl;
		std::cout << tags.size() << std::endl;
	}
	void add_word(const std::wstring &word)
	{
		if (words.find(word) == words.end())
		{
			words.emplace(word, words.size());
			extend_embed.emplace_back(std::move(good_randVec()));
		}
	}
	std::vector<double_t> good_randVec()
	{
		static std::default_random_engine e;
		static std::uniform_real_distribution<double_t> u(-1, 1);
		std::vector<double_t> ret;
		ret.reserve(100);
		for (int i = 0; i < 100; ++i)
		{
			ret.emplace_back(u(e));
		}
		return ret;
	}
	void embed(const std::string &embed_name)
	{
		std::wstring_convert<std::codecvt_utf8<wchar_t >> codec;

		std::ifstream  embed_file(embed_name);
		assert(embed_file);
		std::string line;
		std::wstring wline, word;
		int t0, t1;
		std::vector<double_t> values;
		values.reserve(100);
		while (getline(embed_file, line))
		{
			wline = codec.from_bytes(line);
			t0 = wline.find(L" ");
			words.emplace( wline.substr(0,t0),words.size() );
			t1 = wline.find(L" ", t0 + 1);
			while (t1 != wline.npos)
			{
				values.emplace_back(std::stod(wline.substr(t0 + 1, t1 - (t0 + 1))));
				t0 = t1;
				t1 = wline.find(L" ", t0 + 1);
			}
			values.emplace_back(std::stod(wline.substr(t0 + 1)));
			extend_embed.emplace_back(values);
			values.clear();
		}
	}
	std::vector<sentence> read_data(const std::string & file_name)
	{
		//	std::locale::global(std::locale("Chinese-simplified"));
		std::wstring_convert<std::codecvt_utf8<wchar_t>> codec;
		std::ifstream file(file_name);
		assert(file);
		std::vector<sentence> sentences;
		std::string line;
		sentence sen;
		int t0, t1, t2, t3;
		std::wstring word, tag;
		std::wstring wline;
		while (getline(file, line))
		{
			wline = codec.from_bytes(line);
			if (line.size() == 0)
			{
				sentences.emplace_back(std::move(sen));
				sen.~sentence();
				continue;
			}
			t0 = wline.find(L"\t") + 1;
			t1 = wline.find(L"\t", t0);
			word = wline.substr(t0, t1 - t0);
			t2 = wline.find(L"\t", t1 + 1) + 1;
			t3 = wline.find(L"\t", t2);
			tag = wline.substr(t2, t3 - t2);
			sen.word.emplace_back(std::move(word));
			sen.tag.emplace_back(std::move(tag));
			//构建词的单个元素。
		}
		return sentences;
	}
	//加载各个数据集，
	std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> load(const std::string &file_name,int windows)
	{
		std::vector<sentence> sentences = read_data(file_name);
		std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> data;
		for (int i = 0; i < sentences.size(); ++i)
		{
			std::vector<std::wstring> word;
			word.emplace_back(SOS);
			word.emplace_back(SOS);
			word.insert(word.end(),sentences[i].word.begin(),sentences[i].word.end());
			word.emplace_back(EOS);
			word.emplace_back(EOS);
			for (int j = 0; j <word.size()-4; j++)
			{
				Eigen::VectorXd wseq(5);
				int count = 0;
				Eigen::VectorXd tseq = Eigen::VectorXd::Zero(tags.size());
				for (int k = j; k < j + 5; k++)
				{
					if (words.find(word[k]) != words.end())
					{
						wseq(count)=words.at(word[k]);
					}
					else
					{
						wseq(count)=words.at(UNK);
					}
					count++;
				}
				tseq(tags.at(sentences[i].tag[j])) = 1.0;
				data.emplace_back( std::move(wseq),std::move(tseq));
			}
		}
		return data;
	}
};
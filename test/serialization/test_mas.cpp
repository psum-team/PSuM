#include <psum/serialization.hpp>
#include <iostream>

using namespace std;
using namespace psum::serialization;

int main(){

	cout << "=== unit test: masIO(basic) ===" << endl;
    
	mas_file fp("test.mas");
	fp.clear();

	size_t size = 100;
	double data1[size];
	int data2[size];
	float data3[size];
	for (int i = 0; i < size; i++)
	{
		data1[i] = i;
		data2[i] = i;
		data3[i] = i;
	}

	fp.writeData("data1", data1, {size, 1});
	fp.writeData("data2", data2, {size, 1});
	fp.writeData("data3", data3, {size, 1});

	// fp.writeData("data4", data4, { size,1 });
	fp.readHead();
	auto read1 = Cast<double>(fp.readData("data1"));
	auto read2 = Cast<int>(fp.readData("data2"));
	auto read3 = Cast<float>(fp.readData("data3"));
	// long long int* read4 = Cast<long long int>(fp.readData("data4"));

	bool same = true;
	for (int i = 0; i < size; i++)
	{
		if (data1[i] != read1[i] //){
			|| data2[i] != read2[i] || data3[i] != read3[i])
		{
			//|| data4[i] != read4[i]) {
			same = false;
			break;
		}
	}
	cout << "check:" << (same==true?"pass":"failed") << endl;

	for (int i = 0; i < size; i++)
	{
		data1[i] = i * 2;
	}
	fp.readHead();
	fp.replaceData("data1", data1, {size, 1});
	fp.readHead();
	read1 = Cast<double>(fp.readData("data1"));
	same = true;
	for (int i = 0; i < size; i++)
	{
		if (data1[i] != read1[i])
		{
			same = false;
			break;
		}
	}
	cout << "check:" << (same==true?"pass":"failed") << endl;

	fp.smashData("data1");
	auto read1_after_smash = fp.readData("data1");
	cout << "check:" << (read1_after_smash.info.blockName.empty() ? "pass" : "failed") << endl;

	{
		bool thrown = false;
		try {
			// same name same type
			fp.writeData("data1", data1, {size, 1});
			fp.writeData("data1", data1, {size, 1});
		}
		catch (const std::exception& e) {
			thrown = true;
		}
		cout << "check:" << (thrown==true?"pass":"failed") << endl;
	}

	{
		bool thrown = false;
		try {
			fp.writeData("double_item", data1, {size, 1});
			auto read = Cast<int>(fp.readData("double_item"));
		}
		catch (const std::exception& e) {
			thrown = true;
		}
		cout << "check:" << (thrown==true?"pass":"failed") << endl;

	}

	fp.writeData("words", "hello world");
	fp.readHead();
	auto read_words = Cast<std::string>(fp.readData("words"));
	cout << "check:" << (read_words == "hello world"? "pass" : "failed") << endl;

	save(fp, "pi", 3.14159);
	auto pi = load<double>(fp, "pi");
	cout << "check:" << (pi==3.14159?"pass":"failed") << endl;

	save(fp, "words(2)", "hello?");
	auto s = load<std::string>(fp, "words(2)");
	cout << "check:" << (s=="hello?"?"pass":"failed") << endl;

	Eigen::Matrix<double, -1, -1, Eigen::ColMajor> mat23;
	mat23.resize(2,3);
	mat23 << 1, 2, 3, 666, 555, 444;
	save(fp, "eigen_mat", mat23);
	auto mat = load<Eigen::MatrixXd>(fp, "eigen_mat");
	cout << "check:" << (mat==mat23?"pass":"failed") << endl;

	Eigen::RowVectorXd vec4, vec;
	vec4.resize(4);
	vec4 << 1, 2, 3, 666;
	save(fp, "eigen_vec", vec4);
	load<Eigen::RowVectorXd>(fp, "eigen_vec", vec);
	cout << "check:" << (vec==vec4?"pass":"failed") << endl;

	return 0;
}

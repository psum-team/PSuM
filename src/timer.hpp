#ifndef PSUM_TIMER_HPP
#define PSUM_TIMER_HPP

#include <map>
#include <ctime>
#include <vector>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <filesystem>
#include "table/ptable.hpp"

template <class... Ts>
using VariadicTable = psum::table::VariadicTable<Ts...>;

namespace psum {

class timer {

	struct _timer_item {
		std::chrono::steady_clock::time_point start_time;
		double elapsed_time;
		int stack_depth;
	};

	typedef std::map<std::string, std::vector<_timer_item>> SVT;
	std::map<std::string, double> nametable;
	std::map<std::string, std::chrono::steady_clock::time_point> timeStart;
	SVT times;
	std::vector<std::string> lastKeyStack;

public:
	timer() {};
	~timer() {};
	enum print_mode {
		SumMode,
		MaxkMode,
		MaxAveMode,
		DetailMode
	};

	void inline tic(std::string key) {
		lastKeyStack.push_back(key);
		if (times.find(key) == times.end())
			nametable[key + "ALL"] = 0;
		timeStart[key] = std::chrono::steady_clock::now();
	}

	void inline tocTic(std::string key)
	{
		toc();
		tic(key);
	}

	double inline toc(std::string key) {
		if(lastKeyStack.size()>0)
			if(key==lastKeyStack[lastKeyStack.size()-1]) 
				lastKeyStack.pop_back();
		double ans = 0;
		if (timeStart.find(key) != timeStart.end()) {
			auto end_time = std::chrono::steady_clock::now();
            auto duration = end_time - timeStart[key];
            ans = std::chrono::duration<double>(duration).count();
			if (ans < 0)
				throw std::runtime_error("Error: negative time deteced in timer.");
			times[key].push_back({timeStart[key], ans, (int)lastKeyStack.size()});
			nametable[key + "ALL"] += ans;
		}
		return ans;
	}

	double inline toc()
	{
		if(lastKeyStack.size()==0) return 0;
		return toc(lastKeyStack[lastKeyStack.size()-1]);
	}

	void inline stop_all() {
		for (auto it = timeStart.begin(); it != timeStart.end(); it++) {
			toc((*it).first);
		}
		lastKeyStack.clear();
	}

	void inline print2file(const std::filesystem::path& file_path, print_mode mode = print_mode::SumMode, int k = 20) {
        // check path
        auto parent = file_path.parent_path();
        if (!parent.empty() && !std::filesystem::exists(parent)) {
            throw std::runtime_error("Error: invalid file path in timer.");
        }
        std::ofstream ofs(file_path.string());
        if (!ofs.is_open()) {
            throw std::runtime_error("Error: cannot open file in timer.");
        }
        print2stream(ofs, mode, k);
        ofs.close();
    }

	void inline print2screen(print_mode mode = print_mode::SumMode, int k = 20) {
		print2stream(std::cout, mode, k);
	}

	void inline print2stream(std::ostream& in, print_mode mode = print_mode::SumMode, int k = 20) {
		if(mode == print_mode::DetailMode) {
			typedef VariadicTable<std::string, std::string, std::string> tableType;
			tableType t_sum({"item","","time"});
			for (auto& i : times) {
				t_sum.addRow(i.first, "all", std::to_string(nametable[i.first + "ALL"]));
			}
			t_sum.print(in);
			tableType t({"detail item", "count", "time"});
			typedef std::tuple<std::string, _timer_item, std::pair<int, int>> item_tuple;
			std::vector<item_tuple> all_item;
			for (auto& i : times) {
				for (size_t count = 0; count < i.second.size(); count++)
					all_item.push_back({i.first, i.second[count], {count, i.second.size()}});
			}
			std::sort(all_item.begin(), all_item.end(), [](item_tuple &a, item_tuple &b)
					  { return std::get<1>(a).start_time < std::get<1>(b).start_time; });
			bool first_col = true;
			for (auto &i : all_item) {
				auto repeat_string = [](const std::string &in, int n)
				{
					std::string ans;
					for (int i = 0; i < n; i++)
						ans += in;
					return ans;
				};
				if(first_col)
					first_col = false;
				else{
					t.addRow(repeat_string("  |", std::get<1>(i).stack_depth) + "-" + std::string(std::get<0>(i).size(), '-'),
							 "",
							 "");
				}
				t.addRow(repeat_string("  |", std::get<1>(i).stack_depth) + "-" + std::get<0>(i),
						 std::to_string(std::get<2>(i).first + 1) + "/" + std::to_string(std::get<2>(i).second),
						 std::to_string(std::get<1>(i).elapsed_time));
			}
			t.print(in);
		}
		else if (mode == print_mode::SumMode) {
			typedef VariadicTable<std::string, std::string, std::string> tableType;
			tableType t({"item", "counter", "time"});
			for (auto& i : times)
			{
				t.addRow(i.first, std::to_string(i.second.size()), std::to_string(nametable[i.first + "ALL"]));
			}
			t.print(in);
		}
		else if (mode == print_mode::MaxkMode) {
			typedef VariadicTable<std::string, std::string, std::string, std::string> tableType;
			tableType t({"item", "order", "counter","time"});
			std::vector<std::pair<std::string, double>> time_vec;
			for (auto &i : times)
			{
				time_vec.push_back({i.first, nametable[i.first + "ALL"]});
			}
			std::sort(time_vec.begin(), time_vec.end(),
					  [](const std::pair<std::string, double> &a, const std::pair<std::string, double> &b)
					  {
						  return a.second > b.second;
					  });
			for (size_t i = 0; i < time_vec.size() && i < size_t(k); i++)
			{
				t.addRow(time_vec[i].first, std::to_string(i + 1),
				std::to_string(times[time_vec[i].first].size()),
				std::to_string(time_vec[i].second));
			}
			t.print(in);
		}
		else if (mode == print_mode::MaxAveMode) {
			typedef VariadicTable<std::string, std::string, std::string, std::string> tableType;
			tableType t({"item", "order", "time", "aveTime"});
			std::vector<std::tuple<std::string, double, int>> time_vec;
			for (auto &i : times)
			{
				time_vec.push_back({i.first, nametable[i.first + "ALL"], times[i.first].size()});
			}
			std::sort(time_vec.begin(), time_vec.end(),
					  [](const std::tuple<std::string, double, int> &a,
						 const std::tuple<std::string, double, int> &b)
					  {
						  return (std::get<1>(a) / std::get<2>(a)) > (std::get<1>(b) / std::get<2>(b));
					  });
			for (size_t i = 0; i < time_vec.size() && i < size_t(k); i++)
			{
				t.addRow(std::get<0>(time_vec[i]),
						 std::to_string(i + 1),
						 std::to_string(std::get<1>(time_vec[i])),
						 std::to_string(std::get<1>(time_vec[i]) / std::get<2>(time_vec[i])) +
							" x "+ std::to_string(std::get<2>(time_vec[i])));
			}
			t.print(in);
		}
	}

    double inline time_used(std::string key) {
		if (nametable.find(key + "ALL") != nametable.end()) {
			return nametable[key + "ALL"];
		}
		else return 0;
	}

	void inline clear() {
		nametable.clear();
		timeStart.clear();
		times.clear();
		lastKeyStack.clear();
	}

	static std::string inline current_time_string(std::string form = "%y%m%d_%H%M%S") {
		auto now = std::chrono::system_clock::now();
		std::time_t time = std::chrono::system_clock::to_time_t(now);
		std::stringstream ss;
		ss << std::put_time(std::localtime(&time), form.c_str());
		return ss.str();
	}
};

struct TimerSingleton {
	timer globalTimer;

	static TimerSingleton& getInstance() {
        static TimerSingleton instance;
		return instance;
	}

    TimerSingleton(const TimerSingleton &) = delete;
    TimerSingleton& operator=(const TimerSingleton &) = delete;

private:
	TimerSingleton() = default;
	~TimerSingleton() = default;
};

class RAII_Timer {
	std::string key_name;
public:
	inline RAII_Timer(std::string name):key_name(name) {
		TimerSingleton::getInstance().globalTimer.tic(key_name);
	}
	inline ~RAII_Timer() {
		TimerSingleton::getInstance().globalTimer.toc(key_name);
    }
};

#define Tic(x) psum::TimerSingleton::getInstance().globalTimer.tic(x);
#define TocTic(x) psum::TimerSingleton::getInstance().globalTimer.tocTic(x);
#define Toc psum::TimerSingleton::getInstance().globalTimer.toc();
#define Toc_(x) psum::TimerSingleton::getInstance().globalTimer.toc(x);
#define PrintTimer psum::TimerSingleton::getInstance().globalTimer.print2screen();
#define PrintTimer_(x) psum::TimerSingleton::getInstance().globalTimer.print2screen(x);
#define PrintTimer__(x,y) psum::TimerSingleton::getInstance().globalTimer.print2screen(x,y);
#define WriteTimer(x) psum::TimerSingleton::getInstance().globalTimer.print2file(x);
#define TimeUsed(x) psum::TimerSingleton::getInstance().globalTimer.time_used(x)
#define ClearTimer psum::TimerSingleton::getInstance().globalTimer.clear()
#define CurrentTimeStr psum::TimerSingleton::getInstance().globalTimer.current_time_string()
#define ScopeTic(x, ...) Tic(x) __VA_ARGS__ Toc_(x)
#define GlobalTimer psum::TimerSingleton::getInstance().globalTimer

void inline gt_print2screen(timer::print_mode mode = timer::print_mode::SumMode, int k = 20) {
	GlobalTimer.print2screen(mode, k);
}

void inline gt_print2file(const std::filesystem::path& fname, timer::print_mode mode = timer::print_mode::SumMode, int k = 20) {
	GlobalTimer.print2file(fname, mode, k);
}

#define TIMER__XSTR(a) TIMER__STR(a)
#define TIMER__STR(a) #a
#define TIMER__LINE_AS_STR TIMER__XSTR(__LINE__)
#define FILE_AND_LINE __FILE__ ":" TIMER__LINE_AS_STR
#define ScopeTime(...) Tic(FILE_AND_LINE) __VA_ARGS__ Toc_(FILE_AND_LINE)

#define FUNC_FILE_LINE std::string(__FILE__) +"("+ TIMER__LINE_AS_STR + "):" + "["+std::string(__func__) +"]"
#define TIMER_CONCATENATE_D(x, y) x##y
#define TIMER_CONCATENATE(x, y) TIMER_CONCATENATE_D(x, y)
#define TIMER_UNIQUE_NAME(base) TIMER_CONCATENATE(base, __LINE__)

#define __FuncTimeActive psum::RAII_Timer TIMER_UNIQUE_NAME(__t) (FUNC_FILE_LINE);
#define __FuncTimeInactive
#define FuncTime __FuncTimeActive
#ifdef GENERAL_TIMER
#define __FuncTime __FuncTimeActive
#else
#define __FuncTime __FuncTimeInactive
#endif

}

#endif
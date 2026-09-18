/*
    Masker is a lightweight mas viewer and editor.
*/

#include <iostream>
#include "mas.hpp"
#include "../table/ptable.hpp"

template <class... Ts>
using VariadicTable = psum::table::VariadicTable<Ts...>;
#include <filesystem>
#include <variant>
#include <unordered_set>
#include <readline/readline.h>
#include <readline/history.h>
#include <type_traits>
#include "../ui/getKey.hpp"

using namespace psum::serialization;
using namespace std;

template <typename scalar_type>
struct plotable_vec;

template <typename scalar_type>
static void operator<<(string &out, const plotable_vec<scalar_type> &A)
{
    if(A.begin()==A.end())
        out = "";
    else if constexpr(std::is_same_v<scalar_type, char>)
    {
        std::vector<char> in_str = A;
        out = "\""+std::string(in_str.begin(), in_str.end())+"\"";
    }
    else
    {
        stringstream ss;
        ss << A.beginSymbol;
        for (auto i = A.begin(); i < A.end(); i++)
        {
            ss << *i;
            if (i != A.end() - 1)
                ss << ",";
        }
        ss << A.endSymbol;
        out = ss.str();
    }
}

template <typename scalar_type>
static ostream &operator<<(ostream &out, const plotable_vec<scalar_type> &A)
{
    string str;
    str << A;
    out << str;
    return out;
}

template <typename scalar_type>
struct plotable_vec: public vector<scalar_type>
{
    plotable_vec(const vector<scalar_type> &in) : vector<scalar_type>(in){};
    plotable_vec(scalar_type *_begin, scalar_type *_end) : vector<scalar_type>(_begin, _end){};
    string beginSymbol = "[";
    string endSymbol = "]";
    size_t size() const
    {
        stringstream ss;
        ss << *(this);
        return ss.str().size();
    }
    void set_curly_braces(){
        beginSymbol = "{";
        endSymbol = "}";
    }
    void set_non_braces(){
        beginSymbol = "";
        endSymbol = "";
    }
};

typedef VariadicTable<std::string, std::string, std::string, plotable_vec<size_t>, plotable_vec<size_t>, std::string> tableType;

vector<string> table_content_str;
vector<string> header_str;
string dots_str;
bool need_reload = true;

void make_mas_view(mas_file &fp, std::ostream& os, string name_chose, int idx_chose, int max_show_length, int max_entry_length, int max_name_length)
{
    auto &heads = fp.readHead();
    tableType t({"index", "name", "type", "shape", "offset", "content"});
    int counter = 0;
    for (int _idx = 0; _idx < heads.size();_idx++)
    {
        auto &i = heads[_idx];
        string name = i.info.blockName;
        if(name_chose!="*") {
            // name does not contain the keyword
            if (name.find(name_chose) == string::npos)
                continue;
        }
        if(idx_chose!=0&&(idx_chose!=_idx+1)) continue;
        string type = i.info.blockType;
        plotable_vec<size_t> shape = i.info.blockShape;
        plotable_vec<size_t> ofs({i.position});
        ofs.set_non_braces();
        string content_str;
        auto content = fp.readData(_idx);
        int size_read = 1;
        for (auto i = shape.begin(); i < shape.end(); i++)
            size_read *= (*i);
        bool over_size = max_show_length < size_read;

        visit([&size_read, &over_size, &content_str, max_show_length](auto& v)
              {
                if constexpr(std::is_same_v<typename std::decay_t<decltype(v)>::element_type, char>)
                {
                    over_size = max_show_length * 2 - 1 < size_read;
                    size_read = over_size ? (max_show_length * 2 - 1) : size_read;
                }
                else{
                    size_read = over_size ? max_show_length : size_read;
                }
                std::vector<typename std::decay_t<decltype(v)>::element_type> temp_data;
                temp_data.resize(size_read);
                for (int i = 0; i < size_read; i++)
                    temp_data[i] = v[i];
                plotable_vec<typename std::decay_t<decltype(v)>::element_type> temp(temp_data);
                temp.set_curly_braces();
                content_str << temp; },
              content.content);

        if(over_size) content_str.insert(content_str.size() - 1, "...");
        if(
            (name.size()<=max_entry_length)&&
            (content_str.size()<=max_entry_length)
        ){
            t.addRow(to_string(_idx+1), name, type, shape, ofs, content_str);
        }
        else{
            if (name.size() > max_name_length) {
                // 'abcdefgh' -> 'abd...fgh'
                string name_front = name.substr(0, max_name_length / 2 - 4) + "... ...";
                string name_back = name.substr(name.size() - max_name_length / 2 + 4);
                name = name_front + name_back;
            }
            vector<string> name_sliced(1);
            for (int i = 0; i < name.size(); )
            {
                if (name_sliced.back().size() < max_entry_length)
                {
                    name_sliced.back().push_back(name[i]);
                    i++;
                }
                else{
                    name_sliced.push_back(string());
                }
            }
            vector<string> content_split(1);
            for (auto i : content_str)
            {
                content_split.back().push_back(i);
                if (i == ',')
                    content_split.push_back(string());
            }
            vector<string> content_sliced(1);
            for (auto& i : content_split)
            {
                if (content_sliced.back().size() + i.size() >= max_entry_length - 6)
                    content_sliced.push_back("  ");
                content_sliced.back() += i;
            }
            if(name_sliced.size()<content_sliced.size())
                name_sliced.resize(content_sliced.size());
            else
                content_sliced.resize(name_sliced.size());
            for (int i = 0; i < content_sliced.size(); i++)
            {
                if(i==0)
                    t.addRow(to_string(_idx+1), name_sliced[0], type, shape, ofs, content_sliced[0]);
                else
                    t.addRow("", name_sliced[i], "", vector<size_t>(), vector<size_t>(), content_sliced[i]);
            }
        }
        counter++;
    }
    t.addRow("...", "...", "...", vector<size_t>(), vector<size_t>(), " ...");
    t.print(os);
};

void view_mas(mas_file &fp, std::ostream& os, int start_row, string name_chose = "*", int idx_chose = 0, int max_num_row = 16, int max_show_length = 4, int max_entry_length = 50, int max_name_length = 150)
{
    stringstream ss;
    if (need_reload)
    {
        make_mas_view(fp, ss, name_chose, idx_chose, max_show_length, max_entry_length, max_name_length);
        header_str.clear();
        table_content_str.clear();
        std::string line;
        int row_counter = 0;
        while (std::getline(ss, line, '\n'))
        {
            if (row_counter < 3)
                header_str.push_back(line);
            else
                table_content_str.push_back(line);
            row_counter++;
        }
        dots_str = table_content_str[table_content_str.size() - 2];
        string end_row = table_content_str.back();
        table_content_str.pop_back();
        table_content_str.back() = end_row;

        need_reload = false;
    }

    for (auto& s: header_str)
        os << s << endl;

    int end_row = start_row + max_num_row;
    end_row = end_row < table_content_str.size() ? end_row : table_content_str.size();
    bool print_dots = end_row != table_content_str.size();

    for (int i = start_row; i < end_row; i++)
        os << table_content_str[i] << endl;

    if (print_dots) {
        os << dots_str << endl;
        os << table_content_str.back() << endl;
    }
}

struct multiple_string: public unordered_set<string>
{
    using unordered_set<string>::unordered_set;
};
bool operator==(const multiple_string &a, const string &b)
{
    return a.count(b) != 0;
}
bool operator==(const string &b, const multiple_string &a)
{
    return a.count(b) != 0;
}

multiple_string _clear = {"clearall","clearAll"};
multiple_string _help = {"help","-h"};
multiple_string _add = {"add","+","-a"};
multiple_string _add_g = {"add","+","-a","-+","replace"};
multiple_string _replace = {"-+","replace"};
multiple_string _delete = {"delete","-","-d"};
multiple_string _view = {"view","-v","print"};
multiple_string _quit = {"quit","-q"};
multiple_string _export = {"export","-e"};
multiple_string _print_table = {"table","-t"};

unordered_map<string, string> type_str_map{
    {"-f", "float"},
    {"-d", "double"},
    {"-b", "bool"},
    {"-i", "int32_t"},
    {"-c", "char"},
    {"-l", "int64_t"},
};

bool is_int(const string& str)
{
    std::istringstream iss(str);
    int num_print;
    iss >> num_print;
    if (!iss.fail() && iss.eof())
        return true;
    else
        return false;
}

int main(int argc, char* argv[])
{
    bool _c_Mode = (argc > 1 && string(argv[1]) == "-c");
    using_history();
    rl_variable_bind("enable-keypad", "off");
    filesystem::path filepath;
    if (argc == 1 || _c_Mode)
    {
        auto dummy = freopen("/dev/null", "w", stdout); 
        cout << "input mas file path:";
        string f;
        cin >> f;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        filepath = f;
    }
    else
        filepath = string(argv[1]);
    
    filepath = filesystem::absolute(filepath);
    if(!filesystem::exists(filepath)){
        cout << "file does not exist." << endl;
        cout << "path input = " << filepath << endl;
        if (!filesystem ::exists(filepath.parent_path()))
        {
            cout << "file directory does not exist.\ntry to create directory and file." << endl;
            mas_file fp(filepath.string());
            if(!std::filesystem::exists(std::filesystem::path(filepath)))
            {
                cout << "creating fail;masker exit." << endl;
                return 0;
            }
        }
        else{
            cout << "try to create file." << endl;
			FILE* fp = fopen(filepath.c_str(), "ab");
            if(fp!=NULL)
                fclose(fp);
        }
    }
    std::cout << "\e[?1049h";   // Enter alternate screen buffer
    std::cout << "\e[1;1H\e[2J";// Move cursor position and clear screen
    mas_file fp(filepath.string());
    view_mas(fp, std::cout, 0);
    bool need_fresh = false;
    bool flow_view = false; // Use direction key to switch page
    int view_start_row = 0;
    while (true)
    {
        std::string input;
        std::string beginer;
        if(!_c_Mode){
            if (flow_view == false)
            {
                beginer = string("[") + filepath.filename().string() + "]operation:";
                char *input_c = readline(beginer.c_str());
                if (input_c == nullptr) {
                    break;  // EOF
                }
                if (*input_c) {
                    add_history(input_c);
                }
                input = input_c;
                free(input_c);
            }
        }
        else{
            getline(cin, input);
        }
        std::istringstream iss(input);
        std::vector<std::string> words;
        std::string word;
        while (iss >> word)
            words.push_back(word);
        if(words.size()==0) {
            need_fresh = true;
        }
        else if(words[0]==_quit) break;
        else if(words[0]==_help) {
            int result = _c_Mode?0:system("clear");
            if (result != 0) {
                std::cerr << "Failed to clear the screen. Error code: " << result << std::endl;
            }
            std::cout << "Here are the supported commands and their usage:" << std::endl;
            std::cout << std::endl;
            std::cout << "'help' or '-h'" << std::endl;
            std::cout << "    Show this help message." << std::endl;
            std::cout << "'clearall' or 'clearAll'" << std::endl;
            std::cout << "    Clear the data in this file." << std::endl;
            std::cout << "'view' or '-v' or 'print' + [item_name | item_index] [num_items]" << std::endl;
            std::cout << "    View data items. If 'item_name' or 'item_index' is provided, view that item." << std::endl;
            std::cout << "    If 'num_items' is provided, only the first num_items lines will be displayed." << std::endl;
            std::cout << "'add' or '+' or '-a' + [item_name] [data_type] [shape] [data]" << std::endl;
            std::cout << "    Add new data item. 'data_type' must be a valid fundamental type supported by mas_file system." << std::endl;
            std::cout << "      '-f' -> float     '-d' -> double      '-b' -> bool" << std::endl;
            std::cout << "      '-i' -> int32     '-c' -> char        '-l' -> int64" << std::endl;
            std::cout << "    'shape' is a comma-separated list of dimensions. 'data' is a string or comma-separated values." << std::endl;
            std::cout << "    If 'shape' is 0, the actual data size will be determined by the following content." << std::endl;
            std::cout << "'delete' or '-' or '-d' + [item_name | item_index]" << std::endl;
            std::cout << "    Delete a data item by name or index." << std::endl;
            std::cout << "'replace' or '-+' + [item_name] [data_type] [shape] [data]" << std::endl;
            std::cout << "    Delete the selected item first and then set new value." << std::endl;
            std::cout << "'export' or '-e' + [item_name | item_index] [file] [modifier]" << std::endl;
            std::cout << "    Export a selected item to file. 'modifier' can be '-c', '-r', '-t' or missing." << std::endl;
            std::cout << "'quit' or '-q'" << std::endl;
            std::cout << "    Exit." << std::endl;
            cout << "Press any key to return ..." << endl;
            cin.get();
            need_fresh = true;
        }
        else if(words[0]==_clear) {
            fp.clear();
            need_fresh = true;
            need_reload = true;
        }
        else if(words[0]==_view)
        {
            need_fresh = true;
        }
        else if(words[0]==_add_g)
        {
            if (words.size() <= 4){
                cout << "invalid input for 'add' command." << endl;
                continue;
            }
            string item_name = words[1];
            string item_shape_str = words[3];
            std::vector<size_t> shape;
            
            std::stringstream ss(item_shape_str);
            std::string token;
            try
            {
                while (std::getline(ss, token, ','))
                    shape.push_back(stoi(token));
            }
            catch(const std::exception& e)
            {
                shape.resize(0);
            };
            size_t length = 1;
            for(auto i: shape)
                length *= i;
            if(type_str_map.count(words[2])!=0)
                words[2] = type_str_map[words[2]];
            if (shape.size() != 0 && length >= 0 && foundation::vPtr_typestr2_setmem().count(words[2]) == 1)
            {
                if (shape.size() == 1)
                    shape.push_back(1);
                if (length == 0)
                {
                    if(words[2]=="char")
                    {
                        length = words[4].size();
                    }
                    else{
                        std::stringstream iss(words[4]);
                        std::string token;
                        while (std::getline(iss, token, ','))
                        {
                            length++;
                        }
                    }
                    shape = {length, 1};
                }
                foundation::vPtr buffer_write = foundation::vPtr_typestr2_setmem().at(words[2])(length);
                if(words[0]==_replace) fp.smashData(words[1]);
                visit([&words, &length, &fp, &shape, &need_fresh](auto& v)
                    {
                        std::stringstream iss(words[4]);
                        std::string token;
                        try
                        {
                            if constexpr (std::is_same_v<typename std::decay_t<decltype(v)>::element_type, char>){
                                for (int i = 0; i < length; i++)
                                {
                                    v[i] = words[4][i];
                                }
                            }
                            else
                            {
                                int counter = 0;
                                while (std::getline(iss, token, ',') && counter < length)
                                {
                                    std::istringstream(token) >> v[counter];
                                    counter++;
                                }
                            }
                            need_fresh = true;
                            need_reload = true;
                        }
                        catch (const std::exception &e)
                        {
                            cout << "input cannot be convert to corresponding type." << endl;
                        }; 
                        fp.writeData(words[1], v.get(),shape);
                    }, buffer_write);
            }
            else if(foundation::vPtr_typestr2_setmem().count(words[2]) != 1)
            {
                cout << "invalid type parameter. available type:" << endl;
                for(auto pair: foundation::vPtr_typestr2_setmem())
                    cout << pair.first << " ";
                cout << endl;
            }
            else
                cout << "invalid input for 'add' command." << endl;
        }
        else if(words[0]==_delete){
            if (words.size() == 2)
            {
                auto heads = fp.readHead();

                std::stringstream iss(words[1]);
                std::string token;
                while (std::getline(iss, token, ','))
                {
                    int chosedIdx = -1;
                    bool success = false;
                    if (is_int(token))
                        chosedIdx = stoi(token);
                    if(chosedIdx>0 && chosedIdx<=heads.size())
                        success = fp.smashData(heads[chosedIdx - 1].info.blockName);
                    else
                        success = fp.smashData(token);
                    cout << "delete " << token << (success ? " success." : " fail.") << endl;
                }
                cout << "Press any key to continue ..." << endl;
                cin.get();
                need_fresh = true;
                need_reload = true;
            }
            else
                cout << "invalid input for 'delete' command." << endl;
        }
        else if(words[0]==_export)
        {
            if (words.size() != 3 && words.size() != 4)
                cout << "invalid input for 'export' command." << endl;
            else{
                auto heads = fp.getHeads();
                std::string item_name = words[1];
                if (is_int(words[1]) && stoi(words[1]) <= heads.size() && stoi(words[1]) > 0)
                    item_name = heads[stoi(words[1]) - 1].info.blockName;
                if(fp.getHeadmap().count(item_name)!=0)
                {
                    std::filesystem::path out_file(words[2]);
                    out_file = filesystem::absolute(out_file);
                    std::ofstream o_fp;

                    if (!filesystem::exists(out_file))
                    {
                        if (!filesystem ::exists(out_file.parent_path()))
                        {
                            cout << "file directory does not exist and try to create directory and file." << endl;
                            if (!std::filesystem::create_directory(out_file))
                            {
                                std::cout << "Directory creation failed." << std::endl;
                            }
                            else{
                                o_fp.open(out_file);
                            }
                        }
                        else
                        {
                            o_fp.open(out_file);
                        }
                    }
                    else
                    {
                        o_fp.open(out_file);
                    }
                    if (o_fp.is_open())
                    {
                        auto data = fp.readData(item_name);
                        size_t reline_interval = 1;
                        if (data.info.blockShape.size() == 2) // 2d data will be exported in a matrix-like format.
                            reline_interval = data.info.blockShape[1];
                        size_t out_size = 1;
                        for (auto s : data.info.blockShape)
                            out_size *= s;
                        if(words.size()==4)
                        {
                            if(words[3] == "-c")
                                reline_interval = 1;
                            if(words[3] == "-r")
                                reline_interval = out_size;
                            if(words[3] == "-t" && data.info.blockShape.size() == 2)
                                reline_interval = out_size / reline_interval;
                        }
                        std::visit([out_size, reline_interval, &o_fp](auto& v)
                                {
                                    for (size_t i = 0; i < out_size; i++)
                                    {
                                        o_fp << v[i];
                                        if (reline_interval != 1)
                                            o_fp << " ";
                                        if ((i + 1) % reline_interval == 0)
                                            o_fp << "\n";
                                    }
                                }, data.content);
                        need_fresh = true;
                        o_fp.close();
                    }
                    else std::cout << "The target file cannot be created." << std::endl;
                }
                else std::cout << "This item was not found." << std::endl;
            }
        }
        else if(words[0] == _print_table) {
            std::ofstream table_file("table.txt");
            need_reload = true;
            view_mas(fp, table_file, 0, "*", 0, RAND_MAX, 40, RAND_MAX, RAND_MAX);
            cout << "table file generated in 'table.txt'. Press any key to continue ..." << endl;
            cin.get();
            need_fresh = true;
            need_reload = true;
        }
        else
        {
            cout << "Invalid operation. Press any key to continue ..." << endl;
            cin.get();
            need_fresh = true;
        }
        if(need_fresh){
            int result =  _c_Mode?0:system("clear");
            if (result != 0) {
                std::cerr << "Failed to clear the screen. Error code: " << result << std::endl;
            }
            need_fresh = false;

            view_mas(fp, std::cout, view_start_row);
            if (flow_view == true)
            {
                std::cout << string("[") + filepath.filename().string() + "]viewing; you can press 'up' or 'down' to switch page, 'q' or 'Enter' to quit." << std::endl;
                using namespace psum::ui;
                Key key = KeyboardInput::get_key();
                if (key == Key::UpArrow || key == Key::UpArrowNumPad)
                    view_start_row = (view_start_row - 1) < 0 ? 0 : view_start_row - 1;
                else if (key == Key::DownArrow || key == Key::DownArrowNumPad)
                    view_start_row = (view_start_row + 1) > int(table_content_str.size()) - 6 ? view_start_row : view_start_row + 1;
                else if (key == Key::Q || key == Key::Enter)
                {
                    need_reload = true;
                    view_start_row = 0;
                    flow_view = false;
                    int result =  _c_Mode?0:system("clear");
                    if (result != 0) {
                        std::cerr << "Failed to clear the screen. Error code: " << result << std::endl;
                    }
                    view_mas(fp, std::cout, view_start_row);
                }
            }
        }
        if(words.size()>=1 && words[0]==_view)
        {
            if (words.size() < 2 || words.size() >= 4)
            {
                if (words.size() >= 4)
                    cout << "invalid input for 'view' or 'print' command." << endl;
                else if (_c_Mode==false && flow_view == false)
                {
                    flow_view = true;
                }
            }
            else
            {
                cout << beginer + input << endl;
                string chosedname = "*";
                int chosedIdx = 0;
                if (is_int(words[1])) chosedIdx = stoi(words[1]);
                else chosedname = words[1];

                if (words.size() == 2)
                {
                    need_reload = true;
                    flow_view = true;
                    view_mas(fp, std::cout, 0, chosedname, chosedIdx, 16);
                }
                else
                {
                    if (is_int(words[2]))
                    {
                        int num_print = stoi(words[2]);
                        need_reload = true;
                        flow_view = true;
                        view_mas(fp, std::cout, 0, chosedname, chosedIdx, 16, num_print < 1000 ? num_print : 1000);
                    }
                    else cout << "invalid input for 'view' or 'print' command." << endl;
                }
            }
        }
    }
    std::cout << "\e[?1049l";  // Leave alternate screen buffer
    return 0;
}
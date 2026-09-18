#include "interface.h"
#include <map>
#include <string>
#include <stdexcept>

static std::map<std::string, func_table>& get_registry() {
    static std::map<std::string, func_table> registry;
    return registry;
}

extern "C" {
    void register_solver(const char* name, func_table func) {
        get_registry()[name] = func;
    }

    func_table get_solver_funcs(const char* name) {
        func_table ans;
        ans.init = nullptr;
        auto &reg = get_registry();
        if (reg.find(name) != reg.end()) return reg[name];
        return ans;
    }
}
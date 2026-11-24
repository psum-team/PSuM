#ifndef PSUM_RUNTIME_LAST_FAILURE_HPP
#define PSUM_RUNTIME_LAST_FAILURE_HPP

// The most recent failure of an engine. The engine keeps it and hands it to
// its managers: whoever knows which object a failure is about writes it here.

#include <string>

namespace psum_runtime {

    struct last_failure {
        int status = 0;              // the code
        std::string name;            // the object it concerns; "" when none is known
        unsigned writes = 0;         // lets the engine tell an entry of this call from an older one
        void write(int code, const char* object) {
            status = code; 
            name = object ? object : "";
            writes++; 
        }
    };

} // namespace psum_runtime

#endif

#ifndef PSUM_RUNTIME_STATUS_HPP
#define PSUM_RUNTIME_STATUS_HPP

typedef enum psum_status {
    PSUM_OK = 0,
    PSUM_ERR_ARG = -1,                  /* a null pointer or an out-of-range argument */
    PSUM_ERR_NOT_FOUND = -2,            /* no object under the name */
    PSUM_ERR_EXISTS = -3,               /* an identical object exists */
    PSUM_ERR_UNSUPPORTED = -4,          /* no such type in this build; also a solver backend that failed to build */
    PSUM_ERR_BACKEND = -5,              /* reserved, never returned: an unregistered backend degrades to the native solver */
    PSUM_ERR_OVERFLOW = -6,             /* capacity exceeded in certain device_vector */
    PSUM_ERR_IO = -7,                   /* .mas save/load failure */
    PSUM_ERR_CONFLICT_TYPE = -8,        /* same name, but the TYPE invariants of the descriptor differ */
    PSUM_ERR_CONFLICT_VALUE = -9        /* same name and TYPE invariants, but the VALUE invariants differ */
} psum_status;

/* the segments: which family reported */
typedef enum psum_segment {
    PSUM_SEGMENT_NONE = 0,
    PSUM_SEGMENT_GRID = 100,
    PSUM_SEGMENT_FIELD = 200,
    PSUM_SEGMENT_SPECIES = 300
} psum_segment;

/* ---- putting a code together, and taking it apart ---- */
static inline int psum_code(int base, int segment) { return base - segment; }

/* what happened, whichever family reported it */
static inline psum_status psum_status_base(int status) { return (psum_status)(status % 100); }

/* which family reported it: a PSUM_SEGMENT_* value */
static inline int psum_status_segment(int status) { return status % 100 - status; }

/* the name of a code's base, "?" when it is none of them */
static inline const char* psum_status_name(int status) {
    switch (psum_status_base(status)) {
        case PSUM_OK: return "PSUM_OK";
        case PSUM_ERR_ARG: return "PSUM_ERR_ARG";
        case PSUM_ERR_NOT_FOUND: return "PSUM_ERR_NOT_FOUND";
        case PSUM_ERR_EXISTS: return "PSUM_ERR_EXISTS";
        case PSUM_ERR_UNSUPPORTED: return "PSUM_ERR_UNSUPPORTED";
        case PSUM_ERR_BACKEND: return "PSUM_ERR_BACKEND";
        case PSUM_ERR_OVERFLOW: return "PSUM_ERR_OVERFLOW";
        case PSUM_ERR_IO: return "PSUM_ERR_IO";
        case PSUM_ERR_CONFLICT_TYPE: return "PSUM_ERR_CONFLICT_TYPE";
        case PSUM_ERR_CONFLICT_VALUE: return "PSUM_ERR_CONFLICT_VALUE";
        default: return "?";
    }
}

/* the name of a segment, "" for none */
static inline const char* psum_segment_name(int segment) {
    switch (segment) {
        case PSUM_SEGMENT_GRID: return "grid";
        case PSUM_SEGMENT_FIELD: return "field";
        case PSUM_SEGMENT_SPECIES: return "species";
        default: return "";
    }
}

#endif

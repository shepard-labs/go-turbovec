// turbovec-ffi/include/cabi.h
// C-ABI surface for go-turbovec. Consumed by cgo via #cgo CFLAGS.
// Hand-written; keep in sync with turbovec-ffi/src/lib.rs.

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Opaque handle types. The shim owns a Box<...> behind each one.
typedef struct TqHandle TqHandle;
typedef struct IdHandle IdHandle;

// Search result structs. Buffers are flat, row-major (nq, k).
// Caller frees with tq_results_free / id_results_free.
typedef struct {
    float  *scores;   // length nq * k
    int64_t *indices; // length nq * k
    size_t   nq;
    size_t   k;
} TqSearchResults;

typedef struct {
    float   *scores;  // length nq * k
    uint64_t *ids;     // length nq * k
    size_t    nq;
    size_t    k;
} IdSearchResults;

// TurboQuantIndex
TqHandle *tq_new(size_t dim, uint32_t bit_width, char **err);
TqHandle *tq_new_lazy(uint32_t bit_width, char **err);
void      tq_drop(TqHandle *h);
void      tq_add(TqHandle *h, const float *vectors, size_t n, size_t dim, char **err);
TqSearchResults tq_search(const TqHandle *h, const float *queries, size_t nq,
                          size_t dim, size_t k, const uint8_t *mask /* nullable */);
size_t    tq_swap_remove(TqHandle *h, size_t idx, char **err);
void      tq_prepare(const TqHandle *h);
int32_t   tq_write(const TqHandle *h, const char *path);
TqHandle *tq_load(const char *path, char **err);
size_t    tq_len(const TqHandle *h);
size_t    tq_dim(const TqHandle *h);            // 0 = uncommitted lazy
uint32_t  tq_bit_width(const TqHandle *h);
void      tq_results_free(TqSearchResults r);

// IdMapIndex
IdHandle  *id_new(size_t dim, uint32_t bit_width, char **err);
IdHandle  *id_new_lazy(uint32_t bit_width, char **err);
void       id_drop(IdHandle *h);
void       id_add_with_ids(IdHandle *h, const float *vectors, size_t n,
                           size_t dim, const uint64_t *ids, char **err);
bool       id_remove(IdHandle *h, uint64_t id);
bool       id_contains(const IdHandle *h, uint64_t id);
IdSearchResults id_search(const IdHandle *h, const float *queries, size_t nq,
                          size_t dim, size_t k,
                          const uint64_t *allowlist, size_t allowlist_len);
void       id_prepare(const IdHandle *h);
int32_t    id_write(const IdHandle *h, const char *path);
IdHandle  *id_load(const char *path, char **err);
size_t     id_len(const IdHandle *h);
size_t     id_dim(const IdHandle *h);
uint32_t   id_bit_width(const IdHandle *h);
void       id_results_free(IdSearchResults r);

// Cross-cutting
void err_free(char *err);

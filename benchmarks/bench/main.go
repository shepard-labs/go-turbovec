// Command bench measures the Go binding on the same shapes the upstream
// turbovec benchmark suite uses: 100K corpus vectors, 1,000 queries, k=64,
// median of 5 runs.
//
// Inputs are raw little-endian float32 files, row-major (n, dim), produced by
// benchmarks/prepare_data.py.
package main

import (
	"container/heap"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"sync"
	"time"
	"unsafe"

	"github.com/divy/go-turbovec/turbovec"
)

var (
	corpusPath = flag.String("corpus", "", "raw float32 corpus file (n, dim)")
	queryPath  = flag.String("queries", "", "raw float32 query file (nq, dim) used for timing")
	recallPath = flag.String("recall-queries", "", "query file for recall (default: -queries)")
	dim        = flag.Int("dim", 200, "vector dimensionality")
	n          = flag.Int("n", 100000, "corpus vectors to load")
	nq         = flag.Int("nq", 1000, "query vectors to load")
	k          = flag.Int("k", 64, "results per query")
	runs       = flag.Int("runs", 5, "timed runs per measurement")
	exactRuns  = flag.Int("exact-runs", 3, "timed runs for the exact float32 baseline")
	exactThr   = flag.Int("exact-threads", 0, "threads for the exact baseline (0 = NumCPU)")
	dataset    = flag.String("dataset", "glove-200-angular", "dataset name recorded in the output")
	machine    = flag.String("machine", "", "machine description recorded in the output")
	outPath    = flag.String("out", "bench.json", "output JSON path")
)

func main() {
	flag.Parse()
	if *corpusPath == "" || *queryPath == "" {
		fatalf("-corpus and -queries are required")
	}

	corpus := loadF32(*corpusPath, *n, *dim)
	queries := loadF32(*queryPath, *nq, *dim)
	recallQueries := queries
	if *recallPath != "" {
		recallQueries = loadF32(*recallPath, 0, *dim)
	}
	nVec, nQuery := len(corpus)/(*dim), len(queries)/(*dim)
	nRecall := len(recallQueries) / (*dim)

	res := Results{
		Dataset: *dataset,
		Dim:     *dim,
		N:       nVec,
		Nq:      nQuery,
		NRecall: nRecall,
		K:       *k,
		Machine: *machine,
		Env: Env{
			GoVersion:    runtime.Version(),
			GOOS:         runtime.GOOS,
			GOARCH:       runtime.GOARCH,
			NumCPU:       runtime.NumCPU(),
			GOMAXPROCS:   runtime.GOMAXPROCS(0),
			RayonThreads: os.Getenv("RAYON_NUM_THREADS"),
		},
	}

	fmt.Printf("dataset=%s n=%d nq=%d k=%d dim=%d rayon=%q\n",
		res.Dataset, nVec, nQuery, *k, *dim, res.Env.RayonThreads)

	// Exact float32 top-1 per recall query: the ground truth recall is scored
	// against. Computed once, untimed.
	truth, _ := exactSearch(corpus, recallQueries, *dim, *k, threads(*exactThr))
	fmt.Printf("recall computed over %d queries\n", nRecall)

	// The same exact scan, timed: the float32 baseline TurboQuant replaces.
	exactTimes := make([]float64, *exactRuns)
	for r := range exactTimes {
		_, d := exactSearch(corpus, queries, *dim, *k, threads(*exactThr))
		exactTimes[r] = ms(d) / float64(nQuery)
	}
	res.Exact.SearchMSPerQuery = round(median(exactTimes), 3)
	fmt.Printf("exact top-%d float32 scan: %.3f ms/query\n", *k, res.Exact.SearchMSPerQuery)

	for _, bits := range []int{2, 4} {
		b := measureBitWidth(corpus, queries, recallQueries, *dim, bits, *k, *runs, truth)
		res.BitWidths = append(res.BitWidths, b)
		fmt.Printf("%d-bit: recall@1@1=%.4f search=%.3f ms/query\n",
			bits, b.RecallAt1InTopK["1"], b.SearchMSPerQuery)
	}

	res.Remove = measureRemove(corpus, *dim)
	fmt.Printf("remove: %.2f us/op (n=1), %.2f us/op (n=100)\n",
		res.Remove.SingleUSPerOp, res.Remove.Batch100USPerOp)

	writeJSON(*outPath, res)
	fmt.Printf("wrote %s\n", *outPath)
}

type Results struct {
	Dataset   string       `json:"dataset"`
	Dim       int          `json:"dim"`
	N         int          `json:"n"`
	Nq        int          `json:"nq"`
	NRecall   int          `json:"n_recall_queries"`
	K         int          `json:"k"`
	Machine   string       `json:"machine,omitempty"`
	Env       Env          `json:"env"`
	Exact     ExactResult  `json:"exact"`
	BitWidths []BitResult  `json:"bit_widths"`
	Remove    RemoveResult `json:"remove"`
}

type Env struct {
	GoVersion    string `json:"go_version"`
	GOOS         string `json:"goos"`
	GOARCH       string `json:"goarch"`
	NumCPU       int    `json:"num_cpu"`
	GOMAXPROCS   int    `json:"gomaxprocs"`
	RayonThreads string `json:"rayon_num_threads,omitempty"`
}

type ExactResult struct {
	SearchMSPerQuery float64 `json:"search_ms_per_query"`
}

type BitResult struct {
	BitWidth                  int                `json:"bit_width"`
	BuildMS                   float64            `json:"build_ms"`
	PrepareMS                 float64            `json:"prepare_ms"`
	RecallAt1InTopK           map[string]float64 `json:"recall_at_1_in_top_k"`
	SearchMSPerQuery          float64            `json:"search_ms_per_query"`
	InsertSingleUS            float64            `json:"insert_single_us"`
	InsertBatch100USPerVector float64            `json:"insert_batch100_us_per_vector"`
	WriteMS                   float64            `json:"write_ms"`
	LoadFirstSearchMS         float64            `json:"load_first_search_ms"`
	FileMB                    float64            `json:"file_mb"`
	FP32MB                    float64            `json:"fp32_mb"`
	CompressionRatio          float64            `json:"compression_ratio"`
}

type RemoveResult struct {
	SingleUSPerOp   float64 `json:"single_us_per_op"`
	Batch100USPerOp float64 `json:"batch100_us_per_op"`
}

func measureBitWidth(corpus, queries, recallQueries []float32, dim, bits, k, runs int, truth []int32) BitResult {
	nVec := len(corpus) / dim
	nQuery := len(queries) / dim
	nRecall := len(recallQueries) / dim
	b := BitResult{BitWidth: bits, RecallAt1InTopK: map[string]float64{}}

	t0 := time.Now()
	idx, err := turbovec.NewTurboQuantIndex(dim, bits)
	check(err)
	check(idx.Add(corpus))
	b.BuildMS = ms(time.Since(t0))

	t0 = time.Now()
	idx.Prepare()
	b.PrepareMS = ms(time.Since(t0))

	// Recall@1 within the first k' results, k' in 1..k, as upstream measures
	// it. One search at k=64 answers every k'.
	res, err := idx.Search(recallQueries, k)
	check(err)
	for _, kk := range []int{1, 2, 4, 8, 16, 32, 64} {
		if kk > res.K {
			continue
		}
		hits := 0
		for q := 0; q < nRecall; q++ {
			for _, got := range res.IndicesForQuery(q)[:kk] {
				if int32(got) == truth[q] {
					hits++
					break
				}
			}
		}
		b.RecallAt1InTopK[fmt.Sprint(kk)] = round(float64(hits)/float64(nRecall), 4)
	}

	// Warm up the lazy path, then time full 1,000-query batches.
	_, err = idx.Search(queries[:dim], k)
	check(err)
	times := make([]float64, runs)
	for r := range times {
		t0 = time.Now()
		_, err = idx.Search(queries, k)
		check(err)
		times[r] = ms(time.Since(t0)) / float64(nQuery)
	}
	b.SearchMSPerQuery = round(median(times), 3)

	// Whole-file save of the pristine 100K index, then load -> first search,
	// so the file size below is the index size with nothing appended.
	path := filepath.Join(os.TempDir(), fmt.Sprintf("turbovec-bench-%dbit.tv", bits))
	defer os.Remove(path)
	writes := make([]float64, runs)
	for r := range writes {
		t0 = time.Now()
		check(idx.Write(path))
		writes[r] = ms(time.Since(t0))
	}
	b.WriteMS = round(median(writes), 3)

	loads := make([]float64, runs)
	for r := range loads {
		t0 = time.Now()
		loaded, err := turbovec.LoadTurboQuantIndex(path)
		check(err)
		_, err = loaded.Search(queries[:dim], k)
		check(err)
		loads[r] = ms(time.Since(t0))
		check(loaded.Close())
	}
	b.LoadFirstSearchMS = round(median(loads), 3)

	st, err := os.Stat(path)
	check(err)
	b.FileMB = round(float64(st.Size())/(1024*1024), 1)
	b.FP32MB = round(float64(nVec*dim*4)/(1024*1024), 1)
	b.CompressionRatio = round(b.FP32MB/b.FileMB, 1)

	// Single-vector and 100-vector inserts into the warm, populated index.
	single := make([]float64, 200)
	for i := range single {
		row := corpus[i*dim : (i+1)*dim]
		t0 = time.Now()
		check(idx.Add(row))
		single[i] = us(time.Since(t0))
	}
	b.InsertSingleUS = round(median(single), 2)

	batch := make([]float64, 20)
	const batchN = 100
	for i := range batch {
		vecs := corpus[i*batchN*dim : (i+1)*batchN*dim]
		t0 = time.Now()
		check(idx.Add(vecs))
		batch[i] = us(time.Since(t0)) / batchN
	}
	b.InsertBatch100USPerVector = round(median(batch), 2)

	check(idx.Close())
	return b
}

func measureRemove(corpus []float32, dim int) RemoveResult {
	nVec := len(corpus) / dim
	build := func() *turbovec.IdMapIndex {
		idx, err := turbovec.NewIdMapIndex(dim, 4)
		check(err)
		ids := make([]uint64, nVec)
		for i := range ids {
			ids[i] = uint64(i)
		}
		check(idx.AddWithIDs(corpus, ids))
		idx.Prepare()
		return idx
	}

	idx := build()
	t0 := time.Now()
	for i := 0; i < 1000; i++ {
		if !idx.Remove(uint64(i)) {
			fatalf("remove(%d) returned false", i)
		}
	}
	single := us(time.Since(t0)) / 1000
	check(idx.Close())

	idx = build()
	t0 = time.Now()
	for i := 0; i < 100; i++ {
		if !idx.Remove(uint64(i)) {
			fatalf("remove(%d) returned false", i)
		}
	}
	batch := us(time.Since(t0)) / 100
	check(idx.Close())

	return RemoveResult{SingleUSPerOp: round(single, 2), Batch100USPerOp: round(batch, 2)}
}

// exactSearch returns the exact top-k slots for each query and how long the
// pass took. The compressed index replaces this scan, so it is the baseline
// every TurboQuant number in the README is measured against.
func exactSearch(corpus, queries []float32, dim, k, nthreads int) ([]int32, time.Duration) {
	nQuery := len(queries) / dim
	top1 := make([]int32, nQuery)
	t0 := time.Now()
	parallelFor(nQuery, nthreads, func(q int) {
		row := queries[q*dim : (q+1)*dim]
		h := make(topK, 0, k)
		best, bestIdx := float32(math.Inf(-1)), int32(0)
		for i := 0; i+dim <= len(corpus); i += dim {
			s := dot(row, corpus[i:i+dim])
			if s > best {
				best, bestIdx = s, int32(i/dim)
			}
			if len(h) < k {
				heap.Push(&h, hit{s, int32(i / dim)})
			} else if s > h[0].score {
				h[0] = hit{s, int32(i / dim)}
				heap.Fix(&h, 0)
			}
		}
		top1[q] = bestIdx
	})
	return top1, time.Since(t0)
}

type hit struct {
	score float32
	idx   int32
}

// topK is a min-heap of the k best hits seen so far: h[0] is the worst of them.
type topK []hit

func (h topK) Len() int           { return len(h) }
func (h topK) Less(i, j int) bool { return h[i].score < h[j].score }
func (h topK) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *topK) Push(x any)        { *h = append(*h, x.(hit)) }
func (h *topK) Pop() any          { old := *h; x := old[len(old)-1]; *h = old[:len(old)-1]; return x }

func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func parallelFor(n, nthreads int, fn func(i int)) {
	if nthreads <= 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (n + nthreads - 1) / nthreads
	for w := 0; w < nthreads; w++ {
		lo := w * chunk
		hi := min(lo+chunk, n)
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				fn(i)
			}
		}(lo, hi)
	}
	wg.Wait()
}

func threads(n int) int {
	if n > 0 {
		return n
	}
	return runtime.NumCPU()
}

func loadF32(path string, maxRows, dim int) []float32 {
	b, err := os.ReadFile(path)
	check(err)
	if len(b)%(4*dim) != 0 {
		fatalf("%s: size %d is not a multiple of 4*dim=%d", path, len(b), 4*dim)
	}
	rows := len(b) / (4 * dim)
	if maxRows > 0 && rows > maxRows {
		rows = maxRows
	}
	out := make([]float32, rows*dim)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), rows*dim*4), b)
	return out
}

func median(xs []float64) float64 {
	s := slices.Clone(xs)
	sort.Float64s(s)
	return s[len(s)/2]
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
func us(d time.Duration) float64 { return float64(d) / float64(time.Microsecond) }

func round(v float64, places int) float64 {
	p := math.Pow(10, float64(places))
	return math.Round(v*p) / p
}

func writeJSON(path string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	check(err)
	check(os.MkdirAll(filepath.Dir(path), 0o755))
	check(os.WriteFile(path, append(b, '\n'), 0o644))
}

func check(err error) {
	if err != nil {
		fatalf("%v", err)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "bench: "+format+"\n", args...)
	os.Exit(1)
}

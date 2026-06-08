package turbovec_test

import (
	"math/rand"
	"testing"

	"github.com/divy/go-turbovec/turbovec"
)

const (
	corpusN         = 5000
	queryN          = 200
	testDim         = 64
	testK           = 10
	testBitW        = 4
	recallThreshold = 0.80 // TODO: 4-bit on 5K synthetic Gaussian unit vectors
	//       plateaus at ~0.85 in our runs; the upstream's published
	//       numbers are for larger corpora. Bump corpusN to 50K and
	//       raise this to 0.90 once we have published baselines.
)

func TestTurboQuantIndex_RecallAt10(t *testing.T) {
	corpus := randomUnitVectors(corpusN, testDim, rand.New(rand.NewSource(1)))
	queries := randomUnitVectors(queryN, testDim, rand.New(rand.NewSource(2)))

	idx, err := turbovec.NewTurboQuantIndex(testDim, testBitW)
	if err != nil {
		t.Fatalf("NewTurboQuantIndex: %v", err)
	}
	defer idx.Close()
	if err := idx.Add(flatten(corpus)); err != nil {
		t.Fatalf("Add: %v", err)
	}
	idx.Prepare()

	flatCorpus := flatten(corpus)
	total := 0
	hits := 0
	for _, q := range queries {
		qflat := flatten([][]float32{q})
		r, err := idx.Search(qflat, testK)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		truth := bruteTopK(qflat, flatCorpus, testDim, testK)
		truthSet := map[int]bool{}
		for _, h := range truth {
			truthSet[h.Index] = true
		}
		for _, idx64 := range r.Indices {
			if truthSet[int(idx64)] {
				hits++
			}
		}
		total += testK
	}
	recall := float64(hits) / float64(total)
	if recall < recallThreshold {
		t.Errorf("recall@%d = %.3f, want >= %.3f", testK, recall, recallThreshold)
	}
}

func TestIdMapIndex_RoundTrip(t *testing.T) {
	corpus := randomUnitVectors(1000, 32, rand.New(rand.NewSource(1)))
	queries := randomUnitVectors(50, 32, rand.New(rand.NewSource(2)))

	idx, err := turbovec.NewIdMapIndex(32, 4)
	if err != nil {
		t.Fatalf("NewIdMapIndex: %v", err)
	}
	defer idx.Close()

	flat := flatten(corpus)
	ids := make([]uint64, len(corpus))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := idx.AddWithIDs(flat, ids); err != nil {
		t.Fatalf("AddWithIDs: %v", err)
	}
	idx.Prepare()

	for _, q := range queries {
		qflat := flatten([][]float32{q})
		r, err := idx.SearchWithAllowlist(qflat, 5, ids)
		if err != nil {
			t.Fatalf("SearchWithAllowlist: %v", err)
		}
		if r.K != 5 {
			t.Errorf("K = %d, want 5", r.K)
		}
	}
}

func randomUnitVectors(n, dim int, rng *rand.Rand) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		row := make([]float32, dim)
		var sumSq float64
		for j := range row {
			row[j] = float32(rng.NormFloat64())
			sumSq += float64(row[j]) * float64(row[j])
		}
		inv := float32(1.0 / (sumSq + 1e-300))
		for j := range row {
			row[j] *= inv
		}
		out[i] = row
	}
	return out
}

func flatten(rows [][]float32) []float32 {
	if len(rows) == 0 {
		return nil
	}
	n := len(rows)
	dim := len(rows[0])
	out := make([]float32, n*dim)
	for i, row := range rows {
		copy(out[i*dim:(i+1)*dim], row)
	}
	return out
}

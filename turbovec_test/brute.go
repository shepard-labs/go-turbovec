package turbovec_test

import (
	"container/heap"
)

// bruteTopK returns the top-k highest-scoring (score, index) pairs from
// corpus, scored against query. Both query and corpus rows have the
// given dim. Uses inner product as the score (matches turbovec's
// default inner-product search).
func bruteTopK(query []float32, corpus []float32, dim int, k int) []bruteHit {
	pq := make(scoredPQ, 0, k)
	n := len(corpus) / dim
	for i := 0; i < n; i++ {
		row := corpus[i*dim : (i+1)*dim]
		var s float32
		for j := 0; j < dim; j++ {
			s += query[j] * row[j]
		}
		if pq.Len() < k {
			heap.Push(&pq, bruteHit{Score: s, Index: i})
		} else if pq.Len() > 0 && s > pq[0].Score {
			heap.Pop(&pq)
			heap.Push(&pq, bruteHit{Score: s, Index: i})
		}
	}
	out := make([]bruteHit, pq.Len())
	for j := pq.Len() - 1; j >= 0; j-- {
		out[j] = bruteHit{Score: pq[0].Score, Index: pq[0].Index}
		heap.Pop(&pq)
	}
	return out
}

type bruteHit struct {
	Score float32
	Index int
}

type scoredPQ []bruteHit

func (p scoredPQ) Len() int            { return len(p) }
func (p scoredPQ) Less(i, j int) bool  { return p[i].Score < p[j].Score }
func (p scoredPQ) Swap(i, j int)       { p[i], p[j] = p[j], p[i] }
func (p *scoredPQ) Push(x interface{}) { *p = append(*p, x.(bruteHit)) }
func (p *scoredPQ) Pop() interface{} {
	old := *p
	n := len(old)
	x := old[n-1]
	*p = old[:n-1]
	return x
}

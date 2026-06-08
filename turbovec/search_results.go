package turbovec

// Results holds the output of a TurboQuantIndex search. Slices are
// row-major (nq, k); use ScoresForQuery / IndicesForQuery for per-row
// access.
type Results struct {
	Scores  []float32 // length Nq * K
	Indices []int64   // length Nq * K
	Nq, K   int
}

// ScoresForQuery returns the scores for query qi.
func (r Results) ScoresForQuery(qi int) []float32 {
	return r.Scores[qi*r.K : (qi+1)*r.K]
}

// IndicesForQuery returns the indices for query qi.
func (r Results) IndicesForQuery(qi int) []int64 {
	return r.Indices[qi*r.K : (qi+1)*r.K]
}

// IdResults holds the output of an IdMapIndex search. Same layout as
// Results, but indices are external uint64 ids.
type IdResults struct {
	Scores []float32
	IDs    []uint64
	Nq, K  int
}

// ScoresForQuery returns the scores for query qi.
func (r IdResults) ScoresForQuery(qi int) []float32 {
	return r.Scores[qi*r.K : (qi+1)*r.K]
}

// IDsForQuery returns the external ids for query qi.
func (r IdResults) IDsForQuery(qi int) []uint64 {
	return r.IDs[qi*r.K : (qi+1)*r.K]
}

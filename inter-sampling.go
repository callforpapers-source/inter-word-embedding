
package main

import (
	"fmt"
	"os"
	"bufio"
	"strings"
	// "time"
	"math/rand"
	"math"
	"sync"
)

var (
	DATASET_FILE = "data/b.txt"
	WINDOW = 5
	WINDOW_FLOAT = 3.0
	MAX_RATE = 8.0
	DIMENSION = 3
	SAMPLE_ACCURACY = 8
	SAMPLE_WORDS []string
	MODEL_FILE = fmt.Sprintf("%s-AC-MODEL-%d", DATASET_FILE, SAMPLE_ACCURACY)
	CONTEXT = make(map[string]map[string][]float64)
	SAMPLE [][]float64
	CONTEXT_LEN int
	PIECES []int
	WORDS []string
	lock sync.Mutex
)

func max(i int, j int) (int) {
	if i < j {
		return j
	} else {
		return i
	}
}

func min(i int, j int) (int) {
	if j > i {
		return i
	} else {
		return j
	}
}

func counter_update(list, newlist map[string][]float64) (map[string][]float64) {
	for x,_ := range newlist {
		if _, ok := list[x]; ok {
			list[x][0] = WINDOW_FLOAT - (list[x][0] + newlist[x][0]) / 2
			list[x][1] += newlist[x][1]
		} else {
			list[x] = newlist[x]
		}
	}
	return list
}

func makeContentDest() {
	fopen, err := os.Open(DATASET_FILE)
	if err != nil {
		panic(err)
	}
	defer fopen.Close()
	reader := bufio.NewScanner(fopen)
	var window map[string][]float64
	var line []string
	var doclen, pr int
	var p float64
	for reader.Scan() {
		line = strings.Split(reader.Text(), " ")
		doclen = len(line)
		for i,k := range line {
			window = make(map[string][]float64)
			prev, next := line[max(i - WINDOW, 0):i], line[i + 1:min(1 + WINDOW + i, doclen)]
			for mines,j := range [2][]string{prev, next} {
				pr = 0
				nlen := len(j)
				for w,x := range j {
					if mines == 0 {
						p = float64((nlen - w) - pr)
					} else {
						p = float64((w + 1) - pr)
					}
					if x == k {
						pr += 1
					} else {
						if wx, ok := window[x]; ok {
							window[x][0] = WINDOW_FLOAT - ((wx[0] + p) / (wx[1] + 1))
						} else {
							window[x] = []float64{p, 0, 0}
						}
						window[x][1] += 1
					}
				}
			}
			if _,ok := CONTEXT[k]; ok {
				CONTEXT[k] = counter_update(CONTEXT[k], window)
			} else {
				WORDS = append(WORDS, k)
				CONTEXT[k] = window
			}
		}
	}
}

func perc(p float64, it float64, mx float64) (float64){
	if p == 0.0 || it == 0.0 || mx == 0.0 {
		return 0.0
	}
	return (p/it)*mx
}

func euclidean(x []float64, y []float64, mx float64) (float64) {
	sigma := 0.0
	sum := 0.0
	for i := range x {
		sigma += math.Pow(x[i] - y[i], 2)
		sum += x[i] + y[i]
	}
	squad := math.Sqrt(sigma)
	return mx-perc(squad, sum, mx)
}

func euclidean_word(x, y [][]float64) (float64) {
	sigma := 0.0
	sum := 0.0
	for i,k := range x {
		for j := range k {
			sigma += math.Pow(x[i][j] - y[i][j], 2)
			sum += x[i][j] + y[i][j]
		}
	}
	squad := math.Sqrt(sigma)
	return 4-perc(squad, sum, 4)
}

func contextSimilarity(x, y map[string][]float64) (float64) {
	inter := 0
	var value1 [][]float64
	var value2 [][]float64
	for k,_ := range x {
		for k1,_ := range y {
			if k == k1 {
				inter += 1
				value1 = append(value1, x[k])
				value2 = append(value2, y[k])
				break
			}
		}
	}
	dist := euclidean_word(value1, value2)
	return perc(float64(inter * 2), float64(len(x)) + float64(len(y)), 4) + dist
}

func randFloats(min, max float64) []float64 {
    res := make([]float64, CONTEXT_LEN)
    for i := range res {
        res[i] = min + rand.Float64() * (max - min)
    }
    return res
}

func samples() {
	if SAMPLE_ACCURACY == CONTEXT_LEN {
		SAMPLE_WORDS = WORDS
	} else {
		division := CONTEXT_LEN/SAMPLE_ACCURACY
		r := 0
		for i := 0; i < SAMPLE_ACCURACY; i++ {
			SAMPLE_WORDS = append(SAMPLE_WORDS, WORDS[r])
			r += division
		}
	}
}

func pieceOfSlices() []int{
	l := []int{}
	cond := SAMPLE_ACCURACY % DIMENSION == 0
	difloat := float64(DIMENSION)
	div := float64(SAMPLE_ACCURACY) / difloat
	var n1, n2 float64
	if cond {
		for x := 0; x < DIMENSION; x++ {
			l = append(l, int(div))
		}
	} else {
		first := int(math.Ceil(float64(div)))
		second := int(div)
		mul := float64(div - float64(second))
		if mul == 0.5 {
			n1 = mul * difloat
			n2 = mul * difloat
		} else if mul > 0.5 {
			n2 = mul * difloat
			n1 = (1 - mul) * difloat
		} else if mul < 0.5{
			n2 = (1 - mul) * difloat
			n1 = mul * difloat
		}
		for i := 0; i < int(n1); i++ {
			l = append(l, first)
		}
		for i := 0; i < int(n2); i++ {
			l = append(l, second)
		}
	}
	return l
}

func sliceSplit(slice []float64) ([][]float64){
	start := 0
	pieces := [][]float64{}
	for _,i := range PIECES {
		pieces = append(pieces, slice[start:start+i])
		start += i
	}
	return pieces
}

func createVectors(slice []float64) []float64 {
	vectors := []float64{}
	splitted_slice := sliceSplit(slice)
	for k,i := range splitted_slice {
		vectors = append(vectors, euclidean(i, SAMPLE[k], 8))
	}
	return vectors
}

func main() {
	makeContentDest()
	CONTEXT_LEN = len(CONTEXT)-1
	if SAMPLE_ACCURACY > CONTEXT_LEN || SAMPLE_ACCURACY == -1 {
		SAMPLE_ACCURACY = CONTEXT_LEN
	}
	if DIMENSION > SAMPLE_ACCURACY {
		fmt.Printf("dimension size must be <= sample accuracy: %d\n", CONTEXT_LEN)
		os.Exit(1)
	}
	fopen, err := os.Create(MODEL_FILE)
	if err != nil {
		panic(err)
	}
	defer fopen.Close()
	// rand.Seed(time.Now().UnixNano())
	// PIECES = pieceOfSlices()
	// SAMPLE = sliceSplit(randFloats(0, MAX_RATE))
	samples()
	for _,k := range WORDS {
		_, sims := make([]float64, len(SAMPLE_WORDS)), make([]float64, len(SAMPLE_WORDS))
		context := CONTEXT[k]
		sMap := make([][]float64, len(SAMPLE_WORDS))
		activeThreads := 0
		finishedTasks := make(chan bool)
		for i,k1 := range SAMPLE_WORDS {
			go func(i int, k1 string) {
				var num float64
				if k == k1 {
					num = MAX_RATE
				} else {
					num = contextSimilarity(context, CONTEXT[k1])
				}
				sMap[i] = []float64{float64(i), num}
				finishedTasks <- true
			}(i, k1)
			activeThreads++
			if activeThreads > 50 {
				for activeThreads > 50 {
					<- finishedTasks
					activeThreads--
				}
			}
		}
		for activeThreads > 0 {
			<- finishedTasks
			activeThreads--
		}
		for _,j := range sMap {
			sims[int(j[0])] = j[1]
		}
		// vectors = createVectors(sims)
		fmt.Fprintf(fopen, "%s", k)
		fmt.Fprintf(fopen, " %v", sims)
		fmt.Fprintf(fopen, "\n")
		fmt.Printf("%s\r",k)
	}
}

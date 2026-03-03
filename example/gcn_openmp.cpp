#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <immintrin.h>  // AVX/SSE

using namespace std;

struct CSRMatrix {
    int n_rows, n_cols, nnz;
    vector<int> row_ptr;
    vector<int> col_idx;
    vector<float> values;
    
	CSRMatrix() = default;

    CSRMatrix(int rows, int cols) : n_rows(rows), n_cols(cols), nnz(0) {
        row_ptr.resize(rows + 1, 0);
    }
};


typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

CSRMatrix graph;

vector<vector<int>> edge_index;
vector<vector<float>> edge_val;
vector<int> degree;
vector<int> raw_graph;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;

void readGraph(char *fname)
{
	ifstream infile(fname);

	int source;
	int end;

	infile >> v_num >> e_num;

	// raw_graph.resize(e_num * 2);

	while (!infile.eof())
	{
		infile >> source >> end;
		if (infile.peek() == EOF)
			break;
		raw_graph.push_back(source);
		raw_graph.push_back(end);
	}
}

void raw_graph_to_AdjacencyList()
{

	int src;
	int dst;

	edge_index.resize(v_num);
	edge_val.resize(v_num);
	degree.resize(v_num, 0);

	for (int i = 0; i < raw_graph.size() / 2; i++)
	{
		src = raw_graph[2*i];
		dst = raw_graph[2*i + 1];
		edge_index[dst].push_back(src);
		degree[src]++;
	}
}

void raw_graph_to_CSR()
{
	// count dst edges (each vertex has at least one self-loop)
	vector<int> row_sizes(v_num, 0);
	
	for (int i = 0; i < raw_graph.size() / 2; i++)
	{
		int src = raw_graph[2*i];
		int dst = raw_graph[2*i + 1];
		row_sizes[dst]++;
	}

	// construct row_ptr
	graph.row_ptr.resize(v_num + 1, 0);
	for (int i = 0; i < v_num; i++)
	{
		graph.row_ptr[i + 1] = graph.row_ptr[i] + row_sizes[i];
	}

	// allocate col_idx and values
	graph.col_idx.resize(graph.row_ptr[v_num], 0);
	graph.values.resize(graph.row_ptr[v_num], 1.0f);

	// fill col_idx and values
	vector<int> current_pos(v_num, 0);
	

	// add edges from raw_graph
	for (int i = 0; i < raw_graph.size() / 2; i++)
	{
		int src = raw_graph[2*i];
		int dst = raw_graph[2*i + 1];
		int pos = graph.row_ptr[dst] + current_pos[dst];
		graph.col_idx[pos] = src;
		current_pos[dst]++;
	}
}

void edgeNormalization()
{
	for (int i = 0; i < v_num; i++)
	{
		for (int j = 0; j < edge_index[i].size(); j++)
		{
			float val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
			edge_val[i].push_back(val);
		}
	}
}

void edgeNormalizationCSR()
{
    // calculate degree for each vertex
    vector<float> degree(v_num, 0.0f);
    
    for (int i = 0; i < v_num; i++)
    {
        for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++)
        {
            degree[i] += graph.values[j];
        }
    }
    
    // normalize edge values using the degree of source and destination vertices
    for (int i = 0; i < v_num; i++)
    {
        // calculate the normalization factor for vertex i
        float norm_i = 0.0f;
        if (degree[i] > 0.0f) {
            norm_i = 1.0f / sqrtf(degree[i]);
        }
        
        for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++)
        {
            int nbr = graph.col_idx[j];
            
            // calculate the normalization factor for the neighbor vertex
            float norm_nbr = 0.0f;
            if (degree[nbr] > 0.0f) {
                norm_nbr = 1.0f / sqrtf(degree[nbr]);
            }
            
            // apply normalization to the edge value
            graph.values[j] *= norm_i * norm_nbr;
        }
    }
}


void readFloat(char *fname, float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	FILE *fp = fopen(fname, "rb");
	fread(dst, num * sizeof(float), 1, fp);
	fclose(fp);
}

void initFloat(float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	memset(dst, 0, num * sizeof(float));
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
	float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
	float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
	float(*tmp_W)[out_dim] = (float(*)[out_dim])W;

	#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		for (int k = 0; k < in_dim; k++)
		{
			float a_ik = tmp_in_X[i][k];
            
            // SIMD
            #ifdef __AVX2__
            // get the pointer to the k-th row of W and the i-th row of out_X
            float* w_row = tmp_W[k];
            float* out_row = tmp_out_X[i];
            
            __m256 a_vec = _mm256_set1_ps(a_ik);
            int j = 0;
            for (; j + 7 < out_dim; j += 8) {
                // load 8 floats from W and out_X
                __m256 w_vec = _mm256_loadu_ps(&w_row[j]);
                __m256 out_vec = _mm256_loadu_ps(&out_row[j]);
                
                // out_vec = out_vec + a_vec * w_vec
                out_vec = _mm256_fmadd_ps(a_vec, w_vec, out_vec);
                
                // store the result back to out_X
                _mm256_storeu_ps(&out_row[j], out_vec);
            }
            
            // deal with the remaining elements
            for (; j < out_dim; j++) {
                tmp_out_X[i][j] += a_ik * tmp_W[k][j];
            }
            #else
            // original code without SIMD
            for (int j = 0; j < out_dim; j++) {
                tmp_out_X[i][j] += a_ik * tmp_W[k][j];
            }
            #endif
		}
	}
	
}

void AX(int dim, float *in_X, float *out_X)
{
	float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
	float(*tmp_out_X)[dim] = (float(*)[dim])out_X;


	for (int i = 0; i < v_num; i++)
	{
		vector<int> &nlist = edge_index[i];
		for (int j = 0; j < nlist.size(); j++)
		{
			int nbr = nlist[j];
			for (int k = 0; k < dim; k++)
			{
				tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_val[i][j];
			}
		}
	}
}

void AX_CSR(int dim, float *in_X, float *out_X)
{
	float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
	float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

	#pragma omp parallel for
	for (int i = 0; i < v_num; i++)
	{
		for (int j = graph.row_ptr[i]; j < graph.row_ptr[i + 1]; j++)
		{
			int nbr = graph.col_idx[j];
			float edge_weight = graph.values[j];
			for (int k = 0; k < dim; k++)
			{
				tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_weight;
			}
		}
	}
}

void ReLU(int dim, float *X)
{
	for (int i = 0; i < v_num * dim; i++)
		if (X[i] < 0)
			X[i] = 0;
}

void LogSoftmax(int dim, float *X)
{
	float(*tmp_X)[dim] = (float(*)[dim])X;

	for (int i = 0; i < v_num; i++)
	{
		float max = tmp_X[i][0];
		for (int j = 1; j < dim; j++)
		{
			if (tmp_X[i][j] > max)
				max = tmp_X[i][j];
		}

		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += exp(tmp_X[i][j] - max);
		}
		sum = log(sum);

		for (int j = 0; j < dim; j++)
		{
			tmp_X[i][j] = tmp_X[i][j] - max - sum;
		}
	}
}

float MaxRowSum(float *X, int dim)
{
	float(*tmp_X)[dim] = (float(*)[dim])X;
	float max = -__FLT_MAX__;

	for (int i = 0; i < v_num; i++)
	{
		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += tmp_X[i][j];
		}
		if (sum > max)
			max = sum;
	}
	return max;
}

void freeFloats()
{
	free(X0);
	free(W1);
	free(W2);
	free(X1);
	free(X2);
	free(X1_inter);
	free(X2_inter);
}

void somePreprocessing()
{
	//The graph  will be transformed into adjacency list ,you can use other data structure such as CSR
	// raw_graph_to_AdjacencyList();
	raw_graph_to_CSR();
}

int main(int argc, char **argv)
{
	printf("GCN Example_openmp\n");
	#ifdef __AVX2__
	printf("Using AVX2 optimizations\n");
	#else
	printf("AVX2 not supported, using original code\n");
	#endif
	// Do NOT count the time of reading files, malloc, and memset
	F0 = atoi(argv[1]);
	F1 = atoi(argv[2]);
	F2 = atoi(argv[3]);

	readGraph(argv[4]);
	readFloat(argv[5], X0, v_num * F0);
	readFloat(argv[6], W1, F0 * F1);
	readFloat(argv[7], W2, F1 * F2);

	initFloat(X1, v_num * F1);
	initFloat(X1_inter, v_num * F1);
	initFloat(X2, v_num * F2);
	initFloat(X2_inter, v_num * F2);

	// Time point at the start of the computation
	TimePoint start = chrono::steady_clock::now();

	// Preprocessing time should be included

	TimePoint prepross_start = chrono::steady_clock::now();
	somePreprocessing();
	TimePoint prepross_end = chrono::steady_clock::now();
	chrono::duration<double> prepross_ = prepross_end - prepross_start;
	double prepross_time = prepross_.count() * 1e3;
	printf("prepross_time: %.8lf\n", prepross_time);

	TimePoint edgeNorm_start = chrono::steady_clock::now();
	// edgeNormalization();
	edgeNormalizationCSR();
	TimePoint edgeNorm_end = chrono::steady_clock::now();
	chrono::duration<double> edgeNorm_ = edgeNorm_end - edgeNorm_start;
	double edgeNorm_time = edgeNorm_.count() * 1e3;
	printf("edgeNorm_time: %.8lf\n", edgeNorm_time);


	// printf("Layer1 XW\n");
	TimePoint XW1_start = chrono::steady_clock::now();
	XW(F0, F1, X0, X1_inter, W1);
	TimePoint XW1_end = chrono::steady_clock::now();
	chrono::duration<double> XW1_ = XW1_end - XW1_start;
	double XW1_time = XW1_.count() * 1e3;
	printf("XW1_time: %.8lf\n", XW1_time);

	

	// printf("Layer1 AX\n");
	TimePoint AX1_start = chrono::steady_clock::now();
	// AX(F1, X1_inter, X1);
	AX_CSR(F1, X1_inter, X1);
	TimePoint AX1_end = chrono::steady_clock::now();
	chrono::duration<double> AX1_ = AX1_end - AX1_start;
	double AX1_time = AX1_.count() * 1e3;
	printf("AX1_time: %.8lf\n", AX1_time);

	// printf("Layer1 ReLU\n");
	TimePoint ReLU_start = chrono::steady_clock::now();
	ReLU(F1, X1);
	TimePoint ReLU_end = chrono::steady_clock::now();
	chrono::duration<double> ReLU_ = ReLU_end - ReLU_start;
	double ReLU_time = ReLU_.count() * 1e3;
	printf("ReLU_time: %.8lf\n", ReLU_time);

	// printf("Layer2 XW\n");	
	TimePoint XW2_start = chrono::steady_clock::now();
	XW(F1, F2, X1, X2_inter, W2);
	TimePoint XW2_end = chrono::steady_clock::now();
	chrono::duration<double> XW2_ = XW2_end - XW2_start;
	double XW2_time = XW2_.count() * 1e3;
	printf("XW2_time: %.8lf\n", XW2_time);

	// printf("Layer2 AX\n");
	TimePoint AX2_start = chrono::steady_clock::now();
	// AX(F2, X2_inter, X2);
	AX_CSR(F2, X2_inter, X2);
	TimePoint AX2_end = chrono::steady_clock::now();
	chrono::duration<double> AX2_ = AX2_end - AX2_start;
	double AX2_time = AX2_.count() * 1e3;
	printf("AX2_time: %.8lf\n", AX2_time);

	// printf("Layer2 LogSoftmax\n");
	TimePoint LogSoftmax_start = chrono::steady_clock::now();
	LogSoftmax(F2, X2);
	TimePoint LogSoftmax_end = chrono::steady_clock::now();
	chrono::duration<double> LogSoftmax_ = LogSoftmax_end - LogSoftmax_start;
	double LogSoftmax_time = LogSoftmax_.count() * 1e3;
	printf("LogSoftmax_time: %.8lf\n", LogSoftmax_time);

	// You need to compute the max row sum for result verification
	TimePoint max_sum_start = chrono::steady_clock::now();
	float max_sum = MaxRowSum(X2, F2);
	TimePoint max_sum_end = chrono::steady_clock::now();
	chrono::duration<double> max_sum_ = max_sum_end - max_sum_start;
	double max_sum_time = max_sum_.count() * 1e3;
	printf("max_sum_time: %.8lf\n", max_sum_time);

	// Time point at the end of the computation
	TimePoint end = chrono::steady_clock::now();
	chrono::duration<double> l_durationSec = end - start;
	double l_timeMs = l_durationSec.count() * 1e3;

	// Finally, the max row sum and the computing time
	// should be print to the terminal in the following format
	printf("%.8f\n", max_sum);
	printf("total time: %.8lf\n\n", l_timeMs);

	// Remember to free your allocated memory
	freeFloats();
}
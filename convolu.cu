#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "CImg.h"

#define a_row 299
#define a_col 10
#define ab_colrow 325
#define b_col 10
#define f_dim 3
#define tam f_dim/2
#define BLOCK_SIZE 16

using namespace std;
using namespace cimg_library;


void fill_Matrix(double **&matrix, int m, int n){
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++) {
      matrix[i][j] = 2;//rand()%2+1;
  }
}

void fill_filter5(double **&matrix){
  matrix[0][0]=1.0 ; matrix[0][1]=4.0 ; matrix[0][2]=6.0; matrix[0][3]=4.0; matrix[0][4]=1.0;
  matrix[1][0]=4.0 ; matrix[1][1]=16.0 ; matrix[1][2]=24.0; matrix[1][3]=16.0; matrix[1][4]=4.0;
  matrix[2][0]=6.0 ; matrix[2][1]=24.0 ; matrix[2][2]=-476.0; matrix[2][3]=24.0; matrix[2][4]=6.0;
  matrix[3][0]=4.0 ; matrix[3][1]=16.0 ; matrix[3][2]=24.0; matrix[3][3]=16.0; matrix[3][4]=4.0;
  matrix[4][0]=1.0 ; matrix[4][1]=4.0 ; matrix[4][2]=6.0; matrix[4][3]=4.0; matrix[4][4]=1.0;
}

void fill_filter3(double **&matrix){
  matrix[0][0]=0.0 ; matrix[0][1]=-1.0 ; matrix[0][2]=0.0;
  matrix[1][0]=-1.0 ; matrix[1][1]=5 ; matrix[1][2]=-1.0;
  matrix[2][0]=0.0 ; matrix[2][1]=-1.0 ; matrix[2][2]=0.0;
}
  /*
  void fill_filter3(double **&matrix){
    matrix[0][0]=1 ; matrix[0][1]=1 ; matrix[0][2]=1;
    matrix[1][0]=1 ; matrix[1][1]=1 ; matrix[1][2]=1;
    matrix[2][0]=1 ; matrix[2][1]=1 ; matrix[2][2]=1;
  }*/


void print_Matrix(double **&matrix, int m, int n){
  for (size_t i = 0; i < m; i++){
    for (size_t j = 0; j < n; j++)
      printf("%.2f\t", matrix[i][j] );
    printf("\n");
  }
}

void createSimpleMatrix(double **&matrix, int row, int col){
  matrix = (double**)malloc(row*sizeof(double));
  for(size_t i=0; i<row; ++i)
    matrix[i] = (double*)malloc(col*sizeof(double));
}

void createMatrixs(double **&matrix_h, double **&matrix_d, double **&matrix_dd, size_t totalsize, int m, int n){
  size_t row_size = sizeof(double*)*m;
  // PARA EL HOST
  matrix_h = (double**)malloc(row_size);
  matrix_h[0] = (double*)malloc(totalsize);
  // PARA EL DEVICE
  cudaMalloc((void**)&matrix_d, row_size);
  matrix_dd = (double**)malloc(row_size);
  cudaMalloc((void**)&matrix_dd[0],totalsize);

  for (size_t i = 1; i < m; i++) {
    matrix_h[i] = matrix_h[i-1] + n;
    matrix_dd[i] = matrix_dd[i-1] + n;
  }
  cudaMemcpy(matrix_d, matrix_dd, row_size, cudaMemcpyHostToDevice);
}

void convolucion(double **matrix, int m_row, int m_col, double **filter, double fil_const, double fil_dim, double  **&out){

  for (int i = 0; i < m_row; i++) {
    for (int j = 0; j < m_col; j++) {
      double res = 0.0;
      int radio_i = -fil_dim/2;
      for (int i_f = 0; (i_f<fil_dim) && (radio_i<=(fil_dim/2)); ++i_f, ++radio_i) {
        int radio_j = -fil_dim/2;
        for (int j_f = 0; (j_f<fil_dim) &&(radio_j<=(fil_dim/2)); ++j_f, ++radio_j) {
          if( m_row >(i+radio_i) && (i+radio_i)>=0 && m_col>(j+radio_j) && (j+radio_j)>=0){
            res += matrix[i+radio_i][j+radio_j] * filter[i_f][j_f];
          }
        }
      }
      out[i][j] = res/fil_const;
    }
  }
}

__global__ void  convolucion_cuda(double **matrix, int m_row, int m_col,double **filter, double fil_const, int fil_dim , double **out){
  int row = threadIdx.x + blockIdx.x*blockDim.x;
  int col = threadIdx.y + blockIdx.y*blockDim.y;
  if( row<m_row && col < m_col){
    double res = 0.0;
    int radio_i = -fil_dim/2;
    for (int i = 0; i < fil_dim; ++i){
      if(radio_i <= fil_dim/2){
        int radio_j = -fil_dim/2;
        for (int j = 0; j < fil_dim ; ++j){
          if(radio_j<=fil_dim/2){
            if( m_row>(row+radio_i) && (row+radio_i)>=0 && m_col>(col+radio_j) && (col+radio_j)>=0){
              res += matrix[row+radio_i][col+radio_j] * filter[i][j];
            }
          }
          radio_j++;
        }
      }
      radio_i++;
    }
    out[row][col] = res/fil_const ;
  }
}

void colorTOgrayMatrix( CImg<unsigned char> &color, double **&gray){
    for(int i=0;i<color.height(); ++i){
        for(int j=0; j<color.width(); ++j){
            int R = (int)color(i,j,0,0);
            int G = (int)color(i,j,0,1);
            int B = (int)color(i,j,0,2);
            int grayValue = (int)(0.33*R + 0.33*G + 0.33*B);
            //tmp(x, y, 0,0 ) = grayValue;
            gray[i][j] = grayValue;
        }
    }
}

void print_RGB(CImg<unsigned char> color){
    for(int i=0;i<color.height(); ++i){
        for(int j=0; j<color.width(); ++j){
            printf("%d\t%d\t\t%d\t%d\t%d\n",i,j,(int)color(i,j,0,0),(int)color(i,j,0,1), (int)color(i,j,0,2));
        }
    }
}

CImg<unsigned char> matrixTOgray( double **matrix, int row, int col){
    CImg<unsigned char> tmp(col,row);
    for(int i=0; i<row; ++i)
        for(int j=0;j<col; ++j){
            tmp(i,j,0,0) = (int)matrix[i][j];
        }
    return tmp;
}

void grayConvolucion(CImg<unsigned char> &imagen, double fil_const, int fil_dim){
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((imagen.height()-1)/BLOCK_SIZE+1,((imagen.width()-1)/BLOCK_SIZE)+1);
    
    size_t total_size = sizeof(double)*imagen.width()*imagen.height();
    size_t filter_size = sizeof(double)*fil_dim*fil_dim;
    int row = imagen.width(), col = imagen.height();
    
    double **in, **de_1, **tmp_1;
    double **out, **de_2, **tmp_2;
    double **filter, **de_f, **tmp_f;
    
    createMatrixs(in, de_1, tmp_1, total_size, row, col);
    createMatrixs(out, de_2, tmp_2, total_size, row, col);
    createMatrixs(filter, de_f, tmp_f, filter_size, fil_dim, fil_dim);
    
    colorTOgrayMatrix(imagen, in);
    
    if(fil_dim == 3){
        fill_filter3(filter);
    }else if(fil_dim == 5)
        fill_filter5(filter);
    else{
        printf("Error en la dimension del filtro\n");
        return;
    }
    cudaMemcpy(tmp_1[0], in[0], total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_f[0], filter[0], filter_size, cudaMemcpyHostToDevice);
    
    convolucion_cuda<<<blocksPerGrid,threadsPerBlock>>>(de_1, row, col, de_f, fil_const, fil_dim, de_2);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out[0], tmp_2[0], total_size, cudaMemcpyDeviceToHost);
    
    CImg<unsigned char> salida;
    salida = matrixTOgray(out, row, col);
    salida.display("salida");
}


void grayConvolucionCPU(CImg<unsigned char> &imagen, double fil_const, int fil_dim){

    double **mat , **filter, **out;
    int row = imagen.width(), col = imagen.height();
    createSimpleMatrix(mat, row, col);
    createSimpleMatrix(filter, fil_dim, fil_dim);
    createSimpleMatrix(out, row, col);

    colorTOgrayMatrix(imagen, mat);

    fill_filter3(filter);

    clock_t t0, t1;
    t0 = clock();
    convolucion(mat,row,col,filter,fil_const,fil_dim,out);
    t1 = clock();
    printf("Tiempo  %.2fs\n",((double)(t1-t0)/CLOCKS_PER_SEC));
    CImg<unsigned char> salida;
    salida = matrixTOgray(out, row, col);
    salida.display("Bye!");
}



int main(int argc, char const *argv[]) {

    CImg<unsigned char> image("ironman.bmp");
    grayConvolucionCPU(image,1 , 3);
    grayConvolucion(image,1 , 3);
    return 0;
}

// nvcc convolu.cu -lX11 -w
// ./a.out
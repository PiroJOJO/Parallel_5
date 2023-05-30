//Для компиляции mpic++ MPI_1.cu -o mpi.pg -O2 
//Для прогона time mpiexec -n 1 ./mpi.pg ac _ it _ n _

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

// Функция, обновляющая граничные значения сетки
__global__ void update_boundaries(double* arr, double* new_arr, size_t n, size_t sizeForOne)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n - 1 && i > 0)
    {
		new_arr[n + i] = 0.25 * (arr[n + i - 1] + arr[i] + arr[2 * n + i] + arr[n + i + 1]);
		new_arr[(sizeForOne - 2) * n + i] = 0.25 * (arr[(sizeForOne - 2) * n + i - 1] + arr[(sizeForOne - 3) * n + i] + arr[(sizeForOne - 1)  * n + i] + arr[(sizeForOne - 2) * n + i + 1]);
	}
}

//Функция, которая высчитывает средние значения для обновления сетки(внутренняя часть)
__global__ void update(double* arr, double* new_arr, int n, int sizeForOne)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i > 1 && i < (sizeForOne - 1) && j > 0 && j < n - 1)
    {
        new_arr[i*n + j] = 0.25 * (arr[i*n + j - 1] + arr[i*n + j + 1] + arr[(i - 1)*n + j] + arr[(i + 1)*n + j]);
    }
}

//Функция, которая высчитывает разницу между двумя массивами 
__global__ void my_sub(double* arr, double* new_arr, double* c, size_t n, size_t sizeForOne)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//Индекс для обращения к элементу массива 
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;//Высчитывается из логики - Номер блока на размер блока(кол-во поток) плюс номер потока
    if(i > 0 && i < sizeForOne - 1  && j > 0 && j < n - 1)
    {
        c[i*n + j] = fabs(new_arr[i*n + j] - arr[i*n + j]);//Обращение по индексу по логике преобразования матрицы в одномерный массив
    }
} 


//Функция, которая заполняет сетку начальными значениям - границами
void fill(double* arr, double* new_arr, int n)
{
    arr[0] = new_arr[0] = 10;
    arr[n - 1]= new_arr[n - 1] = 20;
    arr[n * n - 1] = new_arr[n * n - 1] = 30;
    arr[n * (n - 1)] = new_arr[n * (n - 1)] = 20;
    for(int i = 0; i < n -1; i++)
    {
        arr[i] = new_arr[i] = arr[0] + (10.0 / (n-1)) * i;
        arr[n*(n-1) + i] = new_arr[n*(n-1) + i] = arr[n - 1] + 10.0 / (n-1) * i;
        arr[n*i]= new_arr[n*i] = arr[0] + 10.0 / (n-1) * i;
        arr[n*i + n - 1] = new_arr[n*i + n - 1] = arr[n-1] + 10.0 / (n-1) * i;
    }
} 

//Функция вывода матрицы для проверки решения
void print_matrix(double* vec, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout<<vec[n*i + j]<<' ';
        }
        std::cout<<std::endl;
    }
}

int main(int argc, char* argv[]){

    //Объявление и инициализация переменных rank и count_ranks работы с MPI.
    int rank, count_ranks;
    MPI_Init(&argc, &argv);
    // Получение ранга текущего процесса.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего числа процессов.
    MPI_Comm_size(MPI_COMM_WORLD, &count_ranks);

    if(count_ranks < 1 || count_ranks > 4)
    {
        std::cout<<"Wrong number of processes. Value must be between 1 and 4.";
        exit(0);
    }

    // Установка текущего устройства CUDA, соответствующего рангу процесса
	cudaSetDevice(rank);

    //Считывание значений с командной строки
    double error = std::stod(argv[2]);//Значение ошибки
    size_t max_iter = std::stoi(argv[4]);//Количество итераций
    size_t n = std::stoi(argv[6]);//Размер сетки 

    //Инициализация переменных для Host
    double* vec= new double[n * n];
    double* new_vec = new double[n * n];
    double max_error = error + 1;
    int it = 0; 

    //Инициализация переменных для определения количества данных
    size_t start_index = n / count_ranks * rank;
	size_t sizeForOne = n / count_ranks;

    //Инициализация переменных для Device
    double* vec_d;
    double* new_vec_d;
    double* tmp_d;

	//Выделение памяти на процессоре
    cudaMallocHost(&vec, sizeof(double) * n * n);
    cudaMallocHost(&new_vec, sizeof(double) * n * n);

    //Заполнение массивов начальными граничными значениями
	fill(vec, new_vec, n);

	//Выделяем необходимую для процесса память на GPU
	if(count_ranks!=1)
    {
		if (rank != 0 && rank != count_ranks - 1)
        { 
            sizeForOne += 2;
        }
		else 
        {
            sizeForOne += 1;
        }
	 }
	cudaMalloc((void**)&vec_d, n * sizeForOne * sizeof(double));
	cudaMalloc((void**)&new_vec_d, n * sizeForOne * sizeof(double));
	cudaMalloc((void**)&tmp_d, n * sizeForOne * sizeof(double));

	size_t offset = 0;
    if (rank != 0)
    {
        offset = n;
    }
    //Копирование массивов на с Host на Device
 	cudaMemcpy(vec_d, (vec + (start_index * n) - offset), sizeof(double) * n * sizeForOne, cudaMemcpyHostToDevice);
	cudaMemcpy(new_vec_d, new_vec + (start_index * n) - offset, sizeof(double) * n * sizeForOne, cudaMemcpyHostToDevice);

	//Создаем потоки и назначаем приоритет
	int littlePriority = 0;
    int bigPriority = 0;
	cudaStream_t stream_boundaries, stream_inner;
	cudaDeviceGetStreamPriorityRange(&littlePriority, &bigPriority);
	cudaStreamCreateWithPriority(&stream_boundaries, cudaStreamDefault, bigPriority);
	cudaStreamCreateWithPriority(&stream_inner, cudaStreamDefault, littlePriority);

    //Задаем размер блока и сетки 
    dim3 BLOCK_SIZE((n<1024)? n:1024, 1);//Размер блока - количество потоков
    dim3 GRID_SIZE(n / ((n<1024)? n:1024), sizeForOne);//Размер сетки - количество блоков

    //Также инициализируем переменную для расчета максимальной ошибки на cuda
	double* max_error_d;
    cudaMalloc(&max_error_d, sizeof(double));

    //Переменные для работы с библиотекой cub
    void* store = NULL;//Доступное устройство выделения временного хранилища. 
    //При NULL требуемый размер выделения записывается в bytes, и никакая работа не выполняется.
    size_t bytes = 0;//Ссылка на размер в байтах распределения store
    cub::DeviceReduce::Max(store, bytes, vec, max_error_d, n * n);
    // Allocate temporary storage
	cudaMalloc(&store, bytes);
    
    //Основной цикл алгоритма 
	while((max_error > error) && (iter < max_iter))
    {
		it += 1;
		
		//Обновление границ с
		update_boundaries<<<n, 1, 0, stream_boundaries>>>(vec_d, new_vec_d, n, sizeForOne);
		cudaStreamSynchronize(stream_boundaries);
		//Обновление внутренних значений сетки
		update<<<BLOCK_SIZE, GRID_SIZE, 0, stream_inner>>>(vec_d, new_vec_d, n, sizeForOne);

		//Расчет ошибки
		if (it % 500 == 0)
        {
			my_sub<<<BLOCK_SIZE, GRID_SIZE, 0, stream_inner>>>(vec_d, new_vec_d, tmp_d, n, sizeForOne);
			cub::DeviceReduce::Max(store, bytes, tmp_d, max_error_d, n * sizeForOne, stream_inner);
			cudaStreamSynchronize(stream_inner);
			MPI_Allreduce((void*)max_error_d, (void*)max_error_d, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			cudaMemcpyAsync(&max_error, max_error_d, sizeof(double), cudaMemcpyDeviceToHost, stream_inner);
		}
		
		//Вверхняя граница
        if (rank != 0)
        {
		    MPI_Sendrecv(new_vec_d + n + 1, n - 2, MPI_DOUBLE, rank - 1, 0, 
					new_vec_d + 1, n - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		//Нижняя граница
		if (rank != count_ranks - 1)
        {
		    MPI_Sendrecv(new_vec_d + (sizeForOne - 2) * n + 1, n - 2, MPI_DOUBLE, rank + 1, 0,
					new_vec_d + (sizeForOne - 1) * n + 1, 
					n - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		cudaStreamSynchronize(stream_inner);

		//Копирование массивов
		double* swap = vec_d;
		vec_d = new_vec_d;
		new_vec_d = swap;
	}
    cudaMemcpy(vec, vec_d, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(new_vec, new_vec_d, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    print_matrix(vec, n);

    //Очищение памяти
    delete [] vec;
    delete [] new_vec;
    cudaFree(vec_d);
    cudaFree(new_vec_d);
    cudaFree(max_error_d);
    cudaFree(tmp_d);
    cudaFree(store);
    MPI_Finalize();
	return 0;
}
//Для компиляции mpic++ MPI_1.cu -o mpi.pg -O2 
//Для прогона time mpiexec -n 1 ./mpi.pg ac _ it _ n _

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

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
__constant__ double add;

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

int main(int argc, char* argv[]){

	auto begin1 = std::chrono::steady_clock::now();
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
    double error = std::atof(argv[2]);//Значение ошибки
	int iter_max = std::atoi(argv[4]);//Количество итераций
    int size = std::atoi(argv[6]);//Размер сетки 

    //Инициализация переменных для Host
	double* vec = new double[size * size];
	double* new_vec = new double[size * size];
	double max_error = error + 1;
	int it = 0; 

    //Инициализация переменных для Device
	double* vec_d = NULL;
	double* new_vec_d = NULL;
	double* tmp_d;

    //Инициализация переменных для определения количества данных
	size_t sizeForOne = size / count_ranks;
	size_t start_index = size / count_ranks * rank;

	//Выделение памяти на процессоре
    cudaMallocHost(&vec, sizeof(double) * size * size);
    cudaMallocHost(&new_vec, sizeof(double) * size * size);

    //Заполнение массивов начальными граничными значениями
	fill(vec,new_vec,size);

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

	cudaMalloc((void**)&vec_d, size * sizeForOne * sizeof(double));
	cudaMalloc((void**)&new_vec_d, size * sizeForOne * sizeof(double));
	cudaMalloc((void**)&tmp_d, size * sizeForOne * sizeof(double));

	size_t offset = 0;
    if (rank != 0)
    {
        offset = size;
    }

    //Копирование массивов на с Host на Device
 	cudaMemcpy(vec_d, (vec + (start_index * size) - offset), sizeof(double) * size * sizeForOne, cudaMemcpyHostToDevice);
	cudaMemcpy(new_vec_d, new_vec + (start_index * size) - offset, sizeof(double) * size * sizeForOne, cudaMemcpyHostToDevice);

	//Создаем потоки и назначаем приоритет
	int leastPriority = 0;
    int greatestPriority = leastPriority;
	cudaStream_t stream_boundaries, stream_inner;
	cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
	cudaStreamCreateWithPriority(&stream_boundaries, cudaStreamDefault, greatestPriority);
	cudaStreamCreateWithPriority(&stream_inner, cudaStreamDefault, leastPriority);

    //Задаем размер блока и сетки 
    dim3 BLOCK_SIZE((size<1024)? size:1024, 1);//Размер блока - количество потоков
    dim3 GRID_SIZE(size / ((size<1024)? size:1024), sizeForOne);//Размер сетки - количество блоков

    //Также инициализируем переменную для расчета максимальной ошибки на cuda
	double* max_error_d;
    cudaMalloc(&max_error_d, sizeof(double));

	//Переменные для работы с библиотекой cub
    void* store = NULL;//Доступное устройство выделения временного хранилища. 
    //При NULL требуемый размер выделения записывается в bytes, и никакая работа не выполняется.
    size_t bytes = 0;//Ссылка на размер в байтах распределения store
    cub::DeviceReduce::Max(store, bytes, tmp_d, max_error_d, size*size);
    cudaMalloc(&store, bytes);

    //Основной цикл алгоритма 
	while((max_error > error) && (it < iter_max)){
		it += 1;
		
		//Обновление границ 
		update_boundaries<<<size, 1, 0, stream_boundaries>>>(vec_d, new_vec_d, size, sizeForOne);
		cudaStreamSynchronize(stream_boundaries);
		//Обновление внутренних значений сетки
		update<<<BLOCK_SIZE, GRID_SIZE, 0, stream_inner>>>(vec_d, new_vec_d, size, sizeForOne);

		//Расчет ошибки
		if (it % 500 == 0)
		{
			my_sub<<<BLOCK_SIZE, GRID_SIZE, 0, stream_inner>>>(vec_d, new_vec_d, tmp_d, size, sizeForOne);
			cub::DeviceReduce::Max(store, bytes, tmp_d, max_error_d, size * sizeForOne, stream_inner);
			cudaStreamSynchronize(stream_inner);
			MPI_Allreduce((void*)max_error_d, (void*)max_error_d, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			cudaMemcpyAsync(&max_error, max_error_d, sizeof(double), cudaMemcpyDeviceToHost, stream_inner);
		}
		
		//Вверхняя граница
        if (rank != 0)
		{
		    MPI_Sendrecv(new_vec_d + size + 1, size - 2, MPI_DOUBLE, rank - 1, 0, 
					new_vec_d + 1, size - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		//Нижняя граница
		if (rank != count_ranks - 1)
		{
		    MPI_Sendrecv(new_vec_d + (sizeForOne - 2) * size + 1, size - 2, MPI_DOUBLE, rank + 1, 0,
					new_vec_d + (sizeForOne - 1) * size + 1, 
					size - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		cudaStreamSynchronize(stream_inner);

		//Копирование массивов
		double* swap = vec_d;
		vec_d = new_vec_d;
		new_vec_d = swap;
	}
    // cudaMemcpy(vec, vec_d, sizeof(double)*size*size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(new_vec, new_vec_d, sizeof(double)*size*size, cudaMemcpyDeviceToHost);
    // print_matrix(vec, size);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin1);
    if(rank == 0)
	{
        std::cout<<"Error: "<<max_error<<std::endl;
    	std::cout<<"time: "<<elapsed_ms.count()<<" mcs\n";
    	std::cout<<"Iterations: "<<it<<std::endl;
    }

    //Очищение памяти
    cudaFree(vec_d);
    cudaFree(new_vec_d);
    cudaFree(max_error_d);
    cudaFree(tmp_d);
    cudaFree(store);
    MPI_Finalize();

	return 0;
}
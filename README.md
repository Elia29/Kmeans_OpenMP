# K-Means Clustering con OpenMP

Questo progetto implementa l'algoritmo di clustering **K-Means** utilizzando un approccio di layout dati **Structure of Arrays (SoA)**, in linguaggio C++, parallelizzato utilizzando **OpenMP** per migliorare le prestazioni su architetture multi-core.

## Descrizione
L'algoritmo K-Means è uno dei più noti metodi di apprendimento non supervisionato per il partizionamento di un set di dati in $K$ gruppi (cluster). 
Questa implementazione sfrutta il parallelismo a livello di thread per velocizzare:
1. Il calcolo delle distanze tra punti e centroidi.
2. L'assegnazione dei punti al cluster più vicino.
3. L'aggiornamento della posizione dei centroidi.

## Prerequisiti
Per compilare ed eseguire il progetto [kmeans2.cpp](kmeans2.cpp), è necessario avere installato:
1. Un compilatore C++ che supporti OpenMP (come `gcc` o `clang`).
2. Libreria di OpenMP

## Setup e Utilizzo (Windows)
### 1. Clonare la repository
```bash
git clone Elia29/Kmeans-OpenMP
cd Kmeans-OpenMP
```
### 2. Compilare il programma
```bash
g++ -O3 -fopenmp kmeans2.cpp -o kmeans_OpenMP
```
Dove: `g++` è il compilatore; `-03` attiva il massimo livello di ottimizzazione. Senza questo, le istruzioni **#pragma omp simd** e il layout SoA non verrebbero sfruttati appieno; `-fopenmp` attiva la libreria OpenMP; `-o kmeans_OpenMP` nome del file eseguibile finale.

### 3. Eseguire il programma
```bash
.\kmeans_OpenMP.exe
```

> [!NOTE]
> Al termine dell'esecuzione, Il programma crea automaticamente tre diversi file:
> - `table_points_threads.csv`: Lo speedup calcolato su dataset di dimensioni crescenti
> - `table_dims_threads.csv`: Lo speedup per dimensioni crescenti
> - `table_clusters_threads.csv`: Lo speedup variando il numero di cluster

## Fonti
Il dettaglio teorico dell'algoritmo e i risultati dei test di performance sono consultabili qui:
**[Leggi l'articolo (PDF)](kmeansOpenMP.pdf)**

## Licenza
Questo progetto è distribuito sotto la licenza [MIT](LICENSE).


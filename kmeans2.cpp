#include <algorithm> // per funzioni come std::fill e std::numeric_limits
#include <vector> // per l'uso del contenitore std::vector (memoria dimanica)
#include <random> // per la generazione di numeri casuali (mersenne twister)
#include <iostream> // per l'output a video (std::cout)
#include <fstream> // per scrivere i risultati nel file .csv
#include <omp.h> // libreria per OpenMP (parallelismo e timing)
#include <limits> // per definire il valore massimo di un float (minima distanza iniziale)
#include <string> // per la gestione delle stringhe

// --- STRUTTURE DATI ---
struct Dataset {
    size_t n_points; // memorizza il num tot di punti nel dataset
    size_t n_dims; // memorizza il num di dimensioni (coordinate) di ogni punto
    std::vector<float> data; // crea un vettore continuo che contiene tutti i valori

    void resize(size_t points, size_t dims) {
        n_points = points;
        n_dims = dims;
        data.resize(points * dims);
    } // moltiplica num.punti per le dim. per preparare lo spazio giusto nel vettore "data"

    float& at(size_t dim, size_t point) { return data[dim * n_points + point]; } // calcola l'indice delle diverse coordinate
    const float& at(size_t dim, size_t point) const { return data[dim * n_points + point]; } // versione in sola lettura
};

struct Centroids {
    size_t k{}; // imposta il numero di cluster
    size_t n_dims{}; // inidica le dimensioni di ogni centroide
    std::vector<size_t> counts; // vettore che conta quanti punti sono stati assegnati ad ogni cluster
    std::vector<float> data; // vettore contenente coordinate dei centoridi

    void resize(size_t clusters, size_t dims) {
        k = clusters;
        n_dims = dims;
        data.resize(clusters * dims); // alloca lo spazio per i dati dei centroidi
        counts.resize(clusters); // vettore con una cella per ogni cluster
    }

    float& at(size_t dim, size_t cluster) { return data[dim * k + cluster]; } // accesso SoA per i centroidi
    const float& at(size_t dim, size_t cluster) const { return data[dim * k + cluster]; } // versione in sola lettura
};

// --- GENERAZIONE DATI SINTETICI ---
Dataset generate_synthetic_data(size_t points, size_t dims) { // crea un dataset sintetico
    Dataset data; // richiama "data"
    data.resize(points, dims); // alloca spazio in memoria per i punti e le dimensioni
    std::mt19937 gen(42); // inizializza il generatore di numeri casuali
    std::uniform_real_distribution<float> dis(0.0f, 100.0f); // definisce l'intervallo dei numeri casuali
    
    for (size_t i = 0; i < points * dims; ++i) { // ciclo che attraversa il vettore dei dati
        data.data[i] = dis(gen); // assegna a ogni cella del vettore un valore casuale generato
    }
    return data;
}

Centroids initialize_centroids(const Dataset& data, size_t k) { // scelta della posizione iniziale dei centroidi
    std::mt19937 gen(42); // stessa generazione causale dei dati
    std::uniform_int_distribution<> dis(0, data.n_points - 1); // sceglie un indice casuale che vada da 0 al numero tot di punti nel dataset
    Centroids centroids; // oggetto che ospita i centroidi
    centroids.resize(k, data.n_dims); // alloca spazio per k cluster
    for (size_t i = 0; i < k; ++i) { // ciclo che inizializza ogni k cluster
        size_t idx = dis(gen); // sceglie l'indice di un punto a caso nel dataset
        for (size_t d = 0; d < data.n_dims; ++d) { // ciclo che attraversa ogni dim del punto scelto
            centroids.at(d, i) = data.at(d, idx); // copia le coordinate dal punto del dataset scelto al centroide
        }
    }
    return centroids;
}

// --- CORE ALGORITHMS (SoA) ---
inline float compute_distance(const Dataset& data, size_t point_idx, const Centroids& centroids, size_t centroid_idx) { // inline inserisce il corpo funzione nel punto invece di fare un salto di memoria -> no overhead
    float dist = 0.0f; // inizializza a zero la variabile distanza tra i punti
    const size_t n_dims = data.n_dims; // salva i valori in varibili locali
    const size_t n_points = data.n_points;
    const size_t k = centroids.k;
    const float* p_ptr = data.data.data(); // estrae il puntatore all'inizio del vettore dei punti
    const float* c_ptr = centroids.data.data(); // estare il puntatore dei centroidi

    #pragma omp simd reduction(+:dist) // istruzione SIMD -> carica blocchi di coordinate e le processa in unico ciclo di clock
    for (size_t d = 0; d < n_dims; ++d) { // ciclo che scorre sulle dim del punto
        float diff = p_ptr[d * n_points + point_idx] - c_ptr[d * k + centroid_idx]; // salta di n_points per velocizzare
        dist += diff * diff; // eleva al quadrato la diff e la somma, basta la distanza qudratica evitanto spreco di tempo per la radice quadrata
    }
    return dist;
}

void kmeans_sequential(const Dataset& data, Centroids& centroids, std::vector<int>& assignments, int max_iter) {
    std::vector<float> new_centroid_data(centroids.k * data.n_dims, 0.0f); // crea un vettore temp per accumulare le nuove posizioni dei centroidi
    for (int iter = 0; iter < max_iter; ++iter) { // ciclo delle iterazioni
        for (size_t i = 0; i < data.n_points; ++i) { // ciclo su ogni punto
            float min_dist = std::numeric_limits<float>::max(); // inizializza la distanza minima al valore più alto possibile per un float
            int best_c = 0;
            for (size_t j = 0; j < centroids.k; ++j) { // ciclo su tutti i k centroidi
                float d = compute_distance(data, i, centroids, j); // chiama la funzione che calcola la distanza tra punti
                if (d < min_dist) { min_dist = d; best_c = j; } // se la distanza è la minore trovata aggiorna min_dist e salva l'indice del centroide in best_c
            }
            assignments[i] = best_c; // quando a confrontato con tutti i centro salva a quale cluster appartiene il punto
        }
        std::fill(new_centroid_data.begin(), new_centroid_data.end(), 0.0f); // azzera i dati dei nuovi centroidi 
        std::fill(centroids.counts.begin(), centroids.counts.end(), 0); // azzera i contatori dei punti di ogni cluster
        for (size_t i = 0; i < data.n_points; ++i) { // nuovo ciclo su tutti i punti
            int c = assignments[i]; // recupera a quale cluster appartiene il punto
            centroids.counts[c]++; // incrementa il numero di punti del cluster
            for (size_t d = 0; d < data.n_dims; ++d) // ciclo sulle dimensioni
                new_centroid_data[d * centroids.k + c] += data.at(d, i); // somma la coordinata della dimensione e del punto al toale del cluster
        }
        for (size_t j = 0; j < centroids.k; ++j) { // ciclo su ogni cluster
            if (centroids.counts[j] > 0) // se un cluster non ha punti asseganti non si può dividere per zero
                for (size_t d = 0; d < data.n_dims; ++d) // divide la somma totale delle coord. per il numero di punti dati al cluster -> media aritm. per definire la nuova posizione del centroide
                    centroids.at(d, j) = new_centroid_data[d * centroids.k + j] / centroids.counts[j];
        }
    }
}

void kmeans_parallel(const Dataset& data, Centroids& centroids, std::vector<int>& assignments, int max_iter) {
    std::vector<float> global_new_data(centroids.k * data.n_dims); // alloca il vettore del conteggio della somma globale delle coordinate dei punti, diviso per cluster
    for (int iter = 0; iter < max_iter; ++iter) { // ciclo delle iterazioni
        std::fill(global_new_data.begin(), global_new_data.end(), 0.0f); // all'inizio della iterazione, azzera i dati globali e i contatori dei punti
        std::fill(centroids.counts.begin(), centroids.counts.end(), 0);

        #pragma omp parallel default(none) shared(data, centroids, assignments, global_new_data) // crea un insieme di thread
        {
            std::vector<float> local_new_data(centroids.k * data.n_dims, 0.0f);
            std::vector<size_t> local_counts(centroids.k, 0); // ogni thread riceve la propria copia dei vettori

            #pragma omp for schedule(static) // prende il ciclo dei punti e lo divide in blocchi uguali nei vari core
            for (size_t i = 0; i < data.n_points; ++i) {
                float min_d = std::numeric_limits<float>::max();
                int best_c = 0;
                for (size_t j = 0; j < centroids.k; ++j) {
                    float d = compute_distance(data, i, centroids, j);
                    if (d < min_d) { min_d = d; best_c = j; }
                }
                assignments[i] = best_c;
                local_counts[best_c]++; // aggiornano solo la memoria del thread attuale
                for (size_t d = 0; d < data.n_dims; ++d)
                    local_new_data[d * centroids.k + best_c] += data.at(d, i);
            }

            #pragma omp critical // quando un thread ha finito riversa i risultati locali in quelli gloabali (avviene un sola volta per thread e non per ogni singolo punto)
            {
                for (size_t j = 0; j < centroids.k; ++j) {
                    centroids.counts[j] += local_counts[j];
                    for (size_t d = 0; d < data.n_dims; ++d)
                        global_new_data[d * centroids.k + j] += local_new_data[d * centroids.k + j];
                }
            }
        }
        for (size_t j = 0; j < centroids.k; ++j) {
            if (centroids.counts[j] > 0)
                for (size_t d = 0; d < data.n_dims; ++d)
                    centroids.at(d, j) = global_new_data[d * centroids.k + j] / centroids.counts[j];
        }
    }
}

// --- MAIN ---
int main() {
    // Impostazioni base
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 24, 32, 48};
    int max_iter = 5; // limita le iterazioni per non aspettare la convergenza per la confrontabilità

    // 1. MATRICE: SPEEDUP vs POINTS
    {
        std::vector<size_t> test_points = {1000, 10000, 100000, 1000000, 10000000}; // definisco i dataset da testare
        size_t d = 8; size_t k = 16; // fisso dimensioni e cluster
        std::ofstream csv("table_points_threads.csv"); // apre un file per scrivere i risultati
        csv << "Points,T1,T2,T4,T8,T16,T24,T32,T48\n"; // scrive l'intestazione del csv

        for (size_t N : test_points) { // cicla su ogni dimensione del dataset
            Dataset data = generate_synthetic_data(N, d); // crea i punti casuali
            std::vector<int> assign(N);
            Centroids c_seq = initialize_centroids(data, k); // inizializza i centroidi
            double t0 = omp_get_wtime(); // prende il tempo iniziale
            kmeans_sequential(data, c_seq, assign, max_iter); // esegue il k-means su un solo core
            double seq_ms = (omp_get_wtime() - t0) * 1000.0; // calcola il tempo impiegato

            csv << N;
            for (int T : thread_counts) { // per lo stesso dataset prova tutte le configurazione di thread
                omp_set_num_threads(T); // dice a OpenMP quanti processori logici usare
                Centroids c_par = initialize_centroids(data, k);
                double pt0 = omp_get_wtime();
                kmeans_parallel(data, c_par, assign, max_iter); // esegue k-means sfruttando T thread
                double speedup = seq_ms / ((omp_get_wtime() - pt0) * 1000.0); // calcola speedup (Tseq/Tpar)
                csv << "," << speedup; // aggiunge il risultato alla riga del csv
            }
            csv << "\n"; // una volta testati tutti i thread impostati va a capo per iniziare con una nuova riga
        }
    }

    // 2. MATRICE: SPEEDUP vs DIMENSIONS
    {
        std::vector<size_t> test_dims = {2, 4, 8, 16, 32, 64, 128};
        size_t N = 1000000; size_t k = 16;
        std::ofstream csv("table_dims_threads.csv");
        csv << "Dim,T1,T2,T4,T8,T16,T24,T32,T48\n";

        for (size_t D : test_dims) {
            Dataset data = generate_synthetic_data(N, D);
            std::vector<int> assign(N);
            Centroids c_seq = initialize_centroids(data, k);
            double t0 = omp_get_wtime();
            kmeans_sequential(data, c_seq, assign, max_iter);
            double seq_ms = (omp_get_wtime() - t0) * 1000.0;

            csv << D << "D";
            for (int T : thread_counts) {
                omp_set_num_threads(T);
                Centroids c_par = initialize_centroids(data, k);
                double pt0 = omp_get_wtime();
                kmeans_parallel(data, c_par, assign, max_iter);
                double speedup = seq_ms / ((omp_get_wtime() - pt0) * 1000.0);
                csv << "," << speedup;
            }
            csv << "\n";
        }
    }

    // 3. MATRICE: SPEEDUP vs CLUSTERS
    {
        std::vector<size_t> test_ks = {2, 4, 8, 16, 32, 64, 128};
        size_t N = 1000000; size_t d = 8;
        std::ofstream csv("table_clusters_threads.csv");
        csv << "Clusters,T1,T2,T4,T8,T16,T24,T32,T48\n";

        for (size_t K : test_ks) {
            Dataset data = generate_synthetic_data(N, d);
            std::vector<int> assign(N);
            Centroids c_seq = initialize_centroids(data, K);
            double t0 = omp_get_wtime();
            kmeans_sequential(data, c_seq, assign, max_iter);
            double seq_ms = (omp_get_wtime() - t0) * 1000.0;

            csv << K;
            for (int T : thread_counts) {
                omp_set_num_threads(T);
                Centroids c_par = initialize_centroids(data, K);
                double pt0 = omp_get_wtime();
                kmeans_parallel(data, c_par, assign, max_iter);
                double speedup = seq_ms / ((omp_get_wtime() - pt0) * 1000.0);
                csv << "," << speedup;
            }
            csv << "\n";
        }
    }

    std::cout << "Tutti i file CSV sono stati generati" << std::endl;
    return 0;
}
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// nvcc -std=c++17 -O3 breeder.cu -o breeder.exe
// nvcc -Xcompiler /W4 -std=c++17 -g -G -O1 breeder.cu -o breeder.exe

typedef uint16_t Plant;

enum Gene {
    GeneX,
    GeneW,
    GeneG,
    GeneY,
    GeneH,

    GeneCount,
    GeneGood = GeneG,
    GeneInvalid = GeneCount,
};

constexpr Plant maximumEncodedPlant = GeneCount * GeneCount * GeneCount * GeneCount * GeneCount * GeneCount;

__host__ __device__ __forceinline__
static constexpr Plant encodePlant(const Gene& g0, const Gene& g1, const Gene& g2, const Gene& g3, const Gene& g4, const Gene& g5) {
    Plant x = 0;
    x = x * static_cast<Plant>(GeneCount) + static_cast<Plant>(g0);
    x = x * static_cast<Plant>(GeneCount) + static_cast<Plant>(g1);
    x = x * static_cast<Plant>(GeneCount) + static_cast<Plant>(g2);
    x = x * static_cast<Plant>(GeneCount) + static_cast<Plant>(g3);
    x = x * static_cast<Plant>(GeneCount) + static_cast<Plant>(g4);
    x = x * static_cast<Plant>(GeneCount) + static_cast<Plant>(g5);
    return x;
}

__host__ __device__ __forceinline__
static constexpr void decodePlant(Plant x, Gene& g0, Gene& g1, Gene& g2, Gene& g3, Gene& g4, Gene& g5) {
    g5 = static_cast<Gene>(x % GeneCount); x /= GeneCount;
    g4 = static_cast<Gene>(x % GeneCount); x /= GeneCount;
    g3 = static_cast<Gene>(x % GeneCount); x /= GeneCount;
    g2 = static_cast<Gene>(x % GeneCount); x /= GeneCount;
    g1 = static_cast<Gene>(x % GeneCount); x /= GeneCount;
    g0 = static_cast<Gene>(x % GeneCount);
}

__host__ __device__ __forceinline__
static constexpr void decodePlant(Plant x, Gene genes[6]) {
    decodePlant(x, genes[0], genes[1], genes[2], genes[3], genes[4], genes[5]);
}

__host__ __device__ __forceinline__
static constexpr Plant encodePlant(const Gene genes[6]) {
    return encodePlant(genes[0], genes[1], genes[2], genes[3], genes[4], genes[5]);
}

__host__
static void plantToString(const Gene decoded[6], char text[7]) {
    constexpr char lut[] = "XWGYH?";

    text[0] = lut[decoded[0]];
    text[1] = lut[decoded[1]];
    text[2] = lut[decoded[2]];
    text[3] = lut[decoded[3]];
    text[4] = lut[decoded[4]];
    text[5] = lut[decoded[5]];
    text[6] = 0;
}

__host__
static void plantToString(Plant plant, char text[7]) {
    Gene decoded[6];
    decodePlant(plant, decoded);
    plantToString(decoded, text);
}

__host__
static bool plantFromString(std::string_view text, Plant& plant) {
    if (text.size() != 6) {
        return false;
    }

    Gene genes[6];

    for (int i = 0; i < 6; i++) {
        switch (text[i]) {
            case 'X':
            case 'x':
                genes[i] = GeneX;
                break;
            case 'W':
            case 'w':
                genes[i] = GeneW;
                break;
            case 'G':
            case 'g':
                genes[i] = GeneG;
                break;
            case 'Y':
            case 'y':
                genes[i] = GeneY;
                break;
            case 'H':
            case 'h':
                genes[i] = GeneH;
                break;
            default:
                return false;
        }
    }

    plant = encodePlant(genes);
    return true;
}

constexpr auto maximumPlantInputs = 5;

struct Member {
    Plant inputs[maximumPlantInputs];

    bool universal;
    uint8_t count;
    uint32_t lock;

    volatile uint32_t score;
};

struct Generation {
    Member members[maximumEncodedPlant];

    uint64_t iterations;
};

constexpr uint32_t crossbreedWeightGood = 60;
constexpr uint32_t crossbreedWeightBad = 100;

__host__ __device__ __forceinline__
static uint32_t crossbreedWeight(Gene gene) {
    return (gene >= GeneGood) ? crossbreedWeightGood : crossbreedWeightBad;
}

__host__ __device__ __forceinline__
static bool computeGenesGeneric(const Gene* genes, int N, Gene& resultingPrimary, Gene& resultingSecondary, uint32_t& resultingWeight) {
    uint32_t weights[GeneCount] = {};
    uint32_t winner = 0;
    bool hasSecondaryGene = false;

    Gene primaryGene = GeneX;
    Gene secondaryGene = GeneX;

#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; i++) {
        auto gene = genes[i];
        auto& weight = weights[static_cast<int>(gene)];

        weight += crossbreedWeight(gene);

        if (weight > winner) {
            winner = weight;
            primaryGene = gene;
            secondaryGene = gene;
            hasSecondaryGene = false;
        } else if (weight == winner) {
            if (hasSecondaryGene) {
                return false;
            }

            hasSecondaryGene = true;
            secondaryGene = gene;
        }
    }

    resultingPrimary = primaryGene;
    resultingSecondary = secondaryGene;
    resultingWeight = winner;

    return true;
}

template <int N>
__device__ __forceinline__
static bool computePlantsDevice(const Plant (&plants)[N], Plant& resultingPrimary, Plant& resultingSecondary) {
    Gene matrix[6][N];
    Gene primary[6];
    Gene secondary[6];

    #pragma unroll
    for (int i = 0; i < N; i++) {
        decodePlant(
            plants[i],
            matrix[0][i],
            matrix[1][i],
            matrix[2][i],
            matrix[3][i],
            matrix[4][i],
            matrix[5][i]
        );
    }

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        uint32_t _;
        if (!computeGenesGeneric(matrix[i], N, primary[i], secondary[i], _)) {
            return false;
        }
    }

    resultingPrimary = encodePlant(primary);
    resultingSecondary = encodePlant(secondary);

    return true;
}

template <int N>
__device__ __forceinline__
static void insertIntoGeneration(Generation* generation, Plant plant, const Plant (&inputs)[N], uint32_t score) {
    static_assert(N <= maximumPlantInputs, "tried to insert plant with input chain that is too large");

    auto& member = generation->members[plant];

    if (score < member.score) { // This is a very race condition prone check, in theory it might be possible for us to get a value that isnt either the old or new value (so the worst case impact is we loose out on a specific plant) however i have not observed this in practice
        while (atomicCAS(&member.lock, 0u, 1u) != 0u) {
            __nanosleep(64);
        } 

        __threadfence();

        if (score < member.score)  {
            member.score = score;
            member.count = static_cast<uint8_t>(N);

            #pragma unroll
            for (int i = 0; i < N; i++) {
                member.inputs[i] = inputs[i];
            }
        }
        
        __threadfence();
        atomicExch(&member.lock, 0u);
    }
}

template <int N>
__device__
static bool computeAndInsertIntoGeneration(const Plant (&plants)[N], Generation* currentGeneration, const Generation* previousGeneration) {
    Plant primary;
    Plant secondary;

    if (!computePlantsDevice(plants, primary, secondary)) {
        return false;
    }

    uint32_t score = 0;

    #pragma unroll
    for (int i = 0; i < N; i++) {
        score += previousGeneration->members[plants[i]].score;
    }

    if (primary == secondary) {
        insertIntoGeneration(currentGeneration, primary, plants, score);

        return false;
    } else {
        insertIntoGeneration(currentGeneration, primary, plants, score * 2);
        insertIntoGeneration(currentGeneration, secondary, plants, score * 2);

        return true;
    }
}

__host__ __device__ __forceinline__
static uint64_t computeBinomialSlow(uint32_t n, uint32_t k) {
    if (k > n) {
        return 0;
    } else if (k == 0 || k == n) {
        return 1;
    }

    uint32_t kk = k;
    uint64_t result = 1;

    if (kk > n - kk) {
        kk = n - kk;
    }

    for (uint32_t i = 1u; i <= kk; ++i) {
        result = (result * static_cast<uint64_t>(n - kk + i)) / static_cast<uint64_t>(i);
    }

    return result;
}

template <int K>
__device__ __forceinline__
static void unrankCombKSlow(uint64_t rank, uint32_t n, uint32_t (&idx)[K]) {
    uint32_t x = 0;

    #pragma unroll
    for (int i = 0; i < K; i++) {
        uint32_t remaining = static_cast<uint32_t>(K - i);

         for (; x <= n - remaining; ++x) {
            uint64_t c = computeBinomialSlow(n - x - 1u, remaining - 1u);

            if (rank < c) {
                break;
            } else {
                rank -= c;
            }
        }

        idx[i] = x;
        x++;
    }
}

template <int K>
__device__ __forceinline__
static bool nextCombK(uint32_t (&idx)[K], uint32_t n) {
    for (int i = K - 1; i >= 0; --i) {
        uint32_t limit = n - static_cast<uint32_t>(K - i);

        if (idx[i] < limit) {
            idx[i]++;

            for (int j = i + 1; j < K; j++) {
                idx[j] = static_cast<uint32_t>(idx[j - 1] + 1u);
            }

            return true;
        }
    }

    return false;
}

constexpr int incomingPlantLimit = 180;

__constant__ Plant incomingPlantBuffer[incomingPlantLimit];

#define _COMB_SWAP_COMPUTE(a, b) \
    do { \
        auto tmp = p[a]; p[a] = p[b]; p[b] = tmp; \
        computeAndInsertIntoGeneration<K>(p, currentGeneration, previousGeneration); \
    } while(0);

template <int K>
__global__
static void plantKernelCombN(uint32_t n, Generation* currentGeneration, const Generation* previousGeneration) {
    if (n > incomingPlantLimit) {
        return; // Should never happen
    } else if (n < K) {
        return; // When very small amount of input plants given
    }

    uint64_t total = computeBinomialSlow(n, static_cast<uint32_t>(K));
    uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x) + static_cast<uint64_t>(threadIdx.x);
    uint64_t threads = static_cast<uint64_t>(gridDim.x) * static_cast<uint64_t>(blockDim.x);

    uint64_t start = (total * tid) / threads;
    uint64_t end = (total * (tid + 1u)) / threads;

    if (start >= end) {
        return; // Happens when total < threads
    }

    // We unrank the starting combination, and then iterate across them by using nextCombK

    uint64_t iterations = end - start;
    uint32_t idx[K];
    unrankCombKSlow<K>(start, n, idx);

    for (uint64_t r = start; r < end; ++r) {
        Plant p[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            p[i] = incomingPlantBuffer[idx[i]];
        }

        if (computeAndInsertIntoGeneration<K>(p, currentGeneration, previousGeneration)) { // If there were ties, then we iterate permutations to find best plants
            // TODO: Possible optimization is to only shuffle plants that participate in the tie, however that will be most of them anyways, and due to this branch ties dont get checked super often
            // This is equivalent to checking ~500mil extra plants per generation, its considered fast enough
            
            if constexpr(K == 2) { // All permutations for N = 2
                iterations += 1;

                _COMB_SWAP_COMPUTE(0, 1);
            } else if constexpr(K == 3) { // All permutations for N = 3
                iterations += 5;

                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
            } else if constexpr(K == 4) { // All permutations for N = 4
                iterations += 23;

                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 3);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(1, 3);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(2, 3);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
                _COMB_SWAP_COMPUTE(0, 2);
                _COMB_SWAP_COMPUTE(0, 1);
            }
        }

        if (!nextCombK<K>(idx, n)) {
            break;
        }
    }

    atomicAdd(&currentGeneration->iterations, iterations);
}

#ifdef _WIN32

#include <Windows.h>
#pragma comment(lib, "user32.lib")

__host__
bool platformReadClipboard(std::string& result) {
    if (!OpenClipboard(nullptr)) {
        return false;
    }

    auto data = GetClipboardData(CF_TEXT);
    if (!data) {
        CloseClipboard();
        return false;
    }

    auto memory = static_cast<const char*>(GlobalLock(data));
    if (!memory) {
        CloseClipboard();
        return false;
    }
    
    result.clear();
    result.assign(memory);

    GlobalUnlock(data);
    CloseClipboard();

    return true;
}

#else
#error "Platform is not supported, needs platformReadClipboard"
#endif

template <typename F>
__host__
static void iterateOverLines(std::string_view text, F&& callback) {
    do {
        auto pos = text.find('\n');

        if (pos == std::string_view::npos) {
            callback(text);
            break;
        } else {
            callback(text.substr(0, pos));
            text = text.substr(pos + 1);
        }
    } while (!text.empty());
}

__host__
static void getPlantsFromClipboard(std::vector<Plant>& plants) {
    plants.clear();

    std::string text;
    if (!platformReadClipboard(text)) {
        return;
    }

    iterateOverLines(text, [&](std::string_view line) {
        Plant plant;

        while (line.size() > 6 && line.back() == '\r') {
            line.remove_suffix(1);
        }

        if (plantFromString(line.substr(0, 6), plant)) {
            auto it = std::lower_bound(plants.begin(), plants.end(), plant);

            if (it == plants.end() || *it != plant) {
                plants.insert(it, plant);
            }
        }
    });
}

#define _CUDA_CHECK_ERROR(label) \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            auto name = cudaGetErrorName(err); \
            auto str = cudaGetErrorString(err); \
            if (!name) name = "(null)"; \
            if (!str) str = "(null)"; \
            std::cerr << "Cuda: Error during " << label << ": " << name << " - " << str << "\n"; \
            return 1; \
        } \
    } while (0);

static constexpr Plant plant2G4Y[] = {
    encodePlant(GeneG, GeneG, GeneY, GeneY, GeneY, GeneY),
    encodePlant(GeneG, GeneY, GeneG, GeneY, GeneY, GeneY),
    encodePlant(GeneG, GeneY, GeneY, GeneG, GeneY, GeneY),
    encodePlant(GeneG, GeneY, GeneY, GeneY, GeneG, GeneY),
    encodePlant(GeneG, GeneY, GeneY, GeneY, GeneY, GeneG),
    encodePlant(GeneY, GeneG, GeneG, GeneY, GeneY, GeneY),
    encodePlant(GeneY, GeneG, GeneY, GeneG, GeneY, GeneY),
    encodePlant(GeneY, GeneG, GeneY, GeneY, GeneG, GeneY),
    encodePlant(GeneY, GeneG, GeneY, GeneY, GeneY, GeneG),
    encodePlant(GeneY, GeneY, GeneG, GeneG, GeneY, GeneY),
    encodePlant(GeneY, GeneY, GeneG, GeneY, GeneG, GeneY),
    encodePlant(GeneY, GeneY, GeneG, GeneY, GeneY, GeneG),
    encodePlant(GeneY, GeneY, GeneY, GeneG, GeneG, GeneY),
    encodePlant(GeneY, GeneY, GeneY, GeneG, GeneY, GeneG),
    encodePlant(GeneY, GeneY, GeneY, GeneY, GeneG, GeneG),
};

static constexpr Plant plant3G3Y[] = {
    encodePlant(GeneG, GeneG, GeneG, GeneY, GeneY, GeneY),
    encodePlant(GeneG, GeneG, GeneY, GeneG, GeneY, GeneY),
    encodePlant(GeneG, GeneG, GeneY, GeneY, GeneG, GeneY),
    encodePlant(GeneG, GeneG, GeneY, GeneY, GeneY, GeneG),
    encodePlant(GeneG, GeneY, GeneG, GeneG, GeneY, GeneY),
    encodePlant(GeneG, GeneY, GeneG, GeneY, GeneG, GeneY),
    encodePlant(GeneG, GeneY, GeneG, GeneY, GeneY, GeneG),
    encodePlant(GeneG, GeneY, GeneY, GeneG, GeneG, GeneY),
    encodePlant(GeneG, GeneY, GeneY, GeneG, GeneY, GeneG),
    encodePlant(GeneG, GeneY, GeneY, GeneY, GeneG, GeneG),
    encodePlant(GeneY, GeneG, GeneG, GeneG, GeneY, GeneY),
    encodePlant(GeneY, GeneG, GeneG, GeneY, GeneG, GeneY),
    encodePlant(GeneY, GeneG, GeneG, GeneY, GeneY, GeneG),
    encodePlant(GeneY, GeneG, GeneY, GeneG, GeneG, GeneY),
    encodePlant(GeneY, GeneG, GeneY, GeneG, GeneY, GeneG),
    encodePlant(GeneY, GeneG, GeneY, GeneY, GeneG, GeneG),
    encodePlant(GeneY, GeneY, GeneG, GeneG, GeneG, GeneY),
    encodePlant(GeneY, GeneY, GeneG, GeneG, GeneY, GeneG),
    encodePlant(GeneY, GeneY, GeneG, GeneY, GeneG, GeneG),
    encodePlant(GeneY, GeneY, GeneY, GeneG, GeneG, GeneG),
};

static constexpr Plant plant6W[] = {
    encodePlant(GeneW, GeneW, GeneW, GeneW, GeneW, GeneW),
};

static constexpr Plant plant6X[] = {
    encodePlant(GeneX, GeneX, GeneX, GeneX, GeneX, GeneX),
};

template <int N>
__host__
static bool findBestInGeneration(const Generation* generation, const Plant (&plants)[N], Plant& resultingPlant) {
    Plant bestPlant = 0;
    uint32_t bestPlantScore = UINT32_MAX;

    for (auto& plant : plants) {
        if (generation->members[plant].count != 0 && generation->members[plant].score < bestPlantScore) {
            bestPlantScore = generation->members[plant].score;
            bestPlant = plant;
        }
    }

    if (bestPlantScore != UINT32_MAX) {
        resultingPlant = bestPlant;
        return true;
    } else {
        return false;
    }
}

__host__
static void resetGeneration(Generation* generation) {
    for (int i = 0; i < maximumEncodedPlant; i++) {
        auto& m = generation->members[i];
        m.count = 0;
        m.lock  = 0;
        m.score = UINT32_MAX;
    }
}

constexpr int8_t wwwwwwGeneHueristic[GeneCount] = {
    2, // GeneX
    4, // GeneW
    1, // GeneG
    1, // GeneY
    1, // GeneH
};

constexpr int8_t xxxxxxGeneHueristic[GeneCount] = {
    4, // GeneX
    2, // GeneW
    1, // GeneG
    1, // GeneY
    1, // GeneH
};

constexpr int8_t godGeneHueristic[GeneCount] = {
    0, // GeneX
    0, // GeneW
    2, // GeneG
    2, // GeneY
    1, // GeneH
};

enum Target {
    TargetWWWWWW,
    TargetXXXXXX,
    TargetGGGYYY,
    TargetGGYYYY,
    TargetBest
};

__host__ __forceinline__
static int8_t hueristicForPlant(Plant plant, const int8_t hueristic[GeneCount]) {
    Gene genes[6];
    decodePlant(plant, genes);

    return
        hueristic[genes[0]] +
        hueristic[genes[1]] +
        hueristic[genes[2]] +
        hueristic[genes[3]] +
        hueristic[genes[4]] +
        hueristic[genes[5]];
}

__host__ __forceinline__
static const int8_t* hueristicTableFor(Target target) {
    if (target == TargetWWWWWW) {
        return wwwwwwGeneHueristic;
    } else if (target == TargetXXXXXX) {
        return xxxxxxGeneHueristic;
    } else {
        return godGeneHueristic;
    }
}

enum Direction {
    DirectionAny,
    DirectionEast,
    DirectionWest,
};

struct TutorialInput {
    Plant plant;
    Direction direction;
};

struct Tutorial {
    Gene target[6];
    TutorialInput inputs[maximumPlantInputs + 1]; // One extra for a duplicate center

    uint32_t count;
    uint32_t score;
    bool center;
};

constexpr auto maximumPlantsEast = 3;
constexpr auto maximumPlantsWest = 3;

__host__ __forceinline__
static bool bucketHasGene(const Gene* genes, uint32_t n, Gene gene) {
    for (uint32_t i = 0; i < n; i++) {
        if (genes[i] == gene) {
            return true;
        }
    }

    return false;
}

__host__
static bool isGuaranteedWinner(Gene target, Gene center, const Gene* genes, uint32_t east, uint32_t m) {
    if (east > 0 && (east > maximumPlantsEast || m - east > maximumPlantsWest)) {
        return false; // Too many plants in either bucket, we allow the case where all plants are in the same bucket as it will get turned into an any placement
    }

    Gene primary;
    Gene secondary;
    uint32_t weight;

    if (!computeGenesGeneric(genes, m, primary, secondary, weight)) {
        return false; // Three way tie
    }
    
    if (center != GeneInvalid) {
        if (weight <= crossbreedWeight(center)) {
            return center == target; // If neighbors dont beat center, center is kept
        }
    } else {
        if (weight <= crossbreedWeightBad) {
            return false; // In case of a universal center if we dont always win then consider it a failure
        }
    }

    if (primary == secondary) {
        return primary == target; // If there is no tie, winner is just primary
    } else if (target != primary && target != secondary) {
        return false; // Tie was between unrelated genes
    }

    auto competitor = primary == target ? secondary : primary;

    auto targetInEast = bucketHasGene(genes + 0, east, target);
    auto competitorInEast = bucketHasGene(genes + 0, east, competitor);

    if (targetInEast && competitorInEast) {
        return false; // Order is not guaranteed
    } else if (targetInEast) {
        return true; // Target will always be seen first
    } else if (competitorInEast) {
        return false; // Competitor will always be seen first
    }

    auto targetInWest = bucketHasGene(genes + east, m - east, target);
    auto competitorInWest = bucketHasGene(genes + east, m - east, competitor);

    if (targetInWest && competitorInWest) {
        return false; // Order is not guaranteed 
    } else if (targetInWest) {
        return true; // Target will always be seen first
    } else if (competitorInWest) {
        return false; // Competitor will always be seen first
    }

    return false; // Should never actually happen
}

__host__
static bool solveOrderingConstraints(Tutorial& target) {
    // This method selects center saplings and applied east/west ordering to guarantee that we can get our desired plants

    Plant plants[maximumPlantInputs];

    auto n = target.count;
    auto start = plants;
    auto end = plants + n;

    for (uint32_t i = 0; i < n; i++) {
        plants[i] = target.inputs[i].plant;
    }

    // First we scan permutations that can lead to no duplicate plants in the input buffer (using an existing plant as center, trying to use a universal center)
   
    std::sort(start, end);

    do {
        Gene matrix[6][maximumPlantInputs];

        for (uint32_t i = 0; i < n; i++) {
            decodePlant(
                plants[i],
                matrix[0][i],
                matrix[1][i],
                matrix[2][i],
                matrix[3][i],
                matrix[4][i],
                matrix[5][i]
            );
        }

        for (uint32_t east = 0; east < n; east++) {
            auto winnerWithFixedCenter = east > 0;
            auto winnerUniversalCenter = true;

            for (uint32_t i = 0; i < 6; i++) {
                auto genes = matrix[i];

                if (winnerWithFixedCenter && !isGuaranteedWinner(target.target[i], genes[0], genes + 1, east - 1, n - 1)) {
                    winnerWithFixedCenter = false;
                }

                if (winnerUniversalCenter && !isGuaranteedWinner(target.target[i], GeneInvalid, genes, east, n)) {
                    winnerUniversalCenter = false;
                }
            }

            if (!winnerWithFixedCenter && !winnerUniversalCenter) {
                continue;
            }

            auto allAreEast = true;
            auto allAreWest = true;

            target.center = winnerWithFixedCenter;

            for (uint32_t i = 0; i < n; i++) {
                target.inputs[i].plant = plants[i];

                if (i == 0 && target.center) {
                    target.inputs[i].direction = DirectionAny;
                } else if (i < east) {
                    target.inputs[i].direction = DirectionEast;
                    allAreWest = false;
                } else {
                    target.inputs[i].direction = DirectionWest;
                    allAreEast = false;
                }
            }

            if (allAreEast || allAreWest) {
                for (uint32_t i = 0; i < n; i++) {
                    target.inputs[i].direction = DirectionAny;
                }
            }

            return true;
        }
    } while(std::next_permutation(start, end));

    // Then if finding a zero cost order fails, try using duplicate plant centers (this is not ideal as this cost is not accounted for by the GPU bruteforcer)

    std::sort(start, end);

    do {
        Gene matrix[6][maximumPlantInputs];

        for (uint32_t i = 0; i < n; i++) {
            decodePlant(
                plants[i],
                matrix[0][i],
                matrix[1][i],
                matrix[2][i],
                matrix[3][i],
                matrix[4][i],
                matrix[5][i]
            );
        }

        for (uint32_t east = 0; east < n; east++) {
            for (uint32_t center = 0; center < n; center++) {
                auto winnerDuplicateCenter = true;

                for (uint32_t i = 0; i < 6; i++) {
                    auto genes = matrix[i];

                    if (!isGuaranteedWinner(target.target[i], genes[center], genes, east, n)) {
                        winnerDuplicateCenter = false;
                        break;
                    }
                }

                if (!winnerDuplicateCenter) {
                    continue;
                }

                target.center = true;
                target.count = n + 1;

                target.inputs[0].plant = plants[center];
                target.inputs[0].direction = DirectionAny;

                auto allAreEast = true;
                auto allAreWest = true;

                for (uint32_t i = 0; i < n; i++) {
                    target.inputs[i + 1].plant = plants[i];

                    if (i <= east) {
                        target.inputs[i + 1].direction = DirectionEast;
                        allAreWest = false;
                    } else {
                        target.inputs[i + 1].direction = DirectionWest;
                        allAreEast = false;
                    }
                }

                if (allAreEast || allAreWest) {
                    for (uint32_t i = 0; i < n; i++) {
                        target.inputs[i + 1].direction = DirectionAny;
                    }
                }

                return true;
            }
        }
    } while(std::next_permutation(start, end));

    return false;
}

__host__
static void printTutorialFor(Plant plant, const Generation* generation) {
    std::vector<Tutorial> tutorials;
    std::vector<Plant> seen;
    std::vector<Plant> queue;

    // Convert all plants we will ever need to make into tutorial nodes (this is to make printing logic seperate from calculation logic)

    queue.push_back(plant);

    while (!queue.empty()) {
        auto current = queue.back();
        auto it = std::lower_bound(seen.begin(), seen.end(), current);
        queue.pop_back();

        if (it != seen.end() && *it == current) {
            continue;
        } else {
            seen.insert(it, current);
        }

        auto& member = generation->members[current];
        if (member.count <= 1) {
            continue;
        }

        for (int i = 0; i < member.count; i++) {
            queue.push_back(member.inputs[i]);
        }

        auto& tutorial = tutorials.emplace_back();
        tutorial.count = member.count;
        tutorial.score = member.score;

        for (int i = 0; i < member.count; i++) {
            tutorial.inputs[i].plant = member.inputs[i];
        }

        decodePlant(current, tutorial.target);
        
        if (!solveOrderingConstraints(tutorial)) {
            char s[7];

            std::cout << "Failure!! Failed to solve constraints on plant inputs. This is a bug.\n";
            std::cout << "Out = " << (plantToString(current, s), s) << "\n";

            for (int i = 0; i < member.count; i++) {
                std::cout << "In = " << (plantToString(member.inputs[i], s), s) << "\n";
            }

            return;
        }
    }

    // Print tutorial for each plant

    for (auto& tutorial : tutorials) {
        char name[7];
        plantToString(tutorial.target, name);

        std::cout << "| Making " << name << " (difficulty " << tutorial.score << ")\n";
        std::cout << "| ================\n";

        for (uint32_t i = 0; i < tutorial.count; i++) {
            plantToString(tutorial.inputs[i].plant, name);

            auto isFromInput = generation->members[tutorial.inputs[i].plant].count == 1;
            auto isFromCenter = i == 0 && tutorial.center;

            if (isFromCenter) {
                std::cout << "| Center = " << name;
            } else if (isFromInput) {
                std::cout << "| Input = " << name;
            } else {
                std::cout << "| Breed = " << name;
            }

            if (tutorial.inputs[i].direction == DirectionEast) {
                std::cout << " G";
            } else if (tutorial.inputs[i].direction == DirectionWest) {
                std::cout << " R";
            }

            std::cout << "\n";
        }

        std::cout << "| ================\n";
        std::cout << "|";

        for (uint32_t i = 0; i < tutorial.count; i++) {
            plantToString(tutorial.inputs[i].plant, name);
            std::cout << " " << name;
        }

        std::cout << "\n\n";
    }
}

int main(int argc, const char* argv[]) {
    auto target = TargetBest;
    auto showHelp = false;

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];

        if (arg == "--wwwwww") {
            target = TargetWWWWWW;
        } else if (arg == "--xxxxxx") {
            target = TargetXXXXXX;
        } else if (arg == "--gggyyy") {
            target = TargetGGGYYY;
        } else if (arg == "--ggyyyy") {
            target = TargetGGYYYY;
        } else if (arg == "--help") {
            showHelp = true;
            break;
        } else {
            showHelp = true;
            std::cerr << "Unrecognized argument: " << arg << "\n";
            break;
        }
    }

    if (showHelp) {
        std::cerr <<
            "Usage: breeder [flags]\n"
            "Flags:\n"
            "  --wwwwww\n"
            "  --xxxxxx\n"
            "  --gggyyy\n"
            "  --ggyyyy\t"
            "\n";
        return 1;
    }

    std::vector<Plant> plants;
    getPlantsFromClipboard(plants);

    if (plants.size() < 2) {
        std::cerr << "Error: Need at least 2 plants to get any meaningful results\n";
        return 1;
    }

    Generation* currentGeneration = nullptr;
    Generation* previousGeneration = nullptr;

    cudaMallocManaged(&currentGeneration, sizeof(Generation));
    cudaMallocManaged(&previousGeneration, sizeof(Generation));
    _CUDA_CHECK_ERROR("initial memory allocation");

    resetGeneration(currentGeneration);
    resetGeneration(previousGeneration);

    for (auto& plant : plants) {
        currentGeneration->members[plant].score = 1;
        currentGeneration->members[plant].count = 1;
        currentGeneration->members[plant].inputs[0] = plant;

        previousGeneration->members[plant].score = 1;
        previousGeneration->members[plant].count = 1;
        previousGeneration->members[plant].inputs[0] = plant;
    }
    
    Plant bestPlant;
    auto hasBestPlant = false;
    auto heuristicTable = hueristicTableFor(target);

    auto findBestPlant = [&]() {
        if (target == TargetWWWWWW) {
            hasBestPlant = findBestInGeneration(currentGeneration, plant6W, bestPlant);
        } else if (target == TargetXXXXXX) {
            hasBestPlant = findBestInGeneration(currentGeneration, plant6X, bestPlant);
        } else if (target == TargetGGYYYY) {
            hasBestPlant = findBestInGeneration(currentGeneration, plant2G4Y, bestPlant);
        } else if (target == TargetGGGYYY) {
            hasBestPlant = findBestInGeneration(currentGeneration, plant3G3Y, bestPlant);
        } else if (target == TargetBest) {
            hasBestPlant = findBestInGeneration(currentGeneration, plant3G3Y, bestPlant) || findBestInGeneration(currentGeneration, plant2G4Y, bestPlant);
        }

        return hasBestPlant;
    };

    if (findBestPlant()) {
        char name[7];
        plantToString(bestPlant, name);
        std::cout << "| Best plant already in input! " << name << "\n";
        return 0;
    }

    for (int g = 1; g <= 2; g++) {
        std::sort(plants.begin(), plants.end(), [&](Plant lhs, Plant rhs) {
            auto ls = hueristicForPlant(lhs, heuristicTable);
            auto rs = hueristicForPlant(rhs, heuristicTable);

            if (ls != rs) {
                return ls > rs;
            } else {
                return lhs < rhs;
            }
        });

        if (plants.size() > incomingPlantLimit) {
            plants.resize(incomingPlantLimit);
        }

        std::cout << "Generation " << g << ": We have " << plants.size() << " plants as inputs\n";
                
        cudaMemcpyToSymbol(incomingPlantBuffer, plants.data(), plants.size() * sizeof(Plant), 0, cudaMemcpyHostToDevice);
        _CUDA_CHECK_ERROR("generation launch prepare");

        auto start = std::chrono::high_resolution_clock::now();

        currentGeneration->iterations = 0;

        plantKernelCombN<2><<<36 * 1 * 8, 256>>>(static_cast<uint32_t>(plants.size()), currentGeneration, previousGeneration);
        plantKernelCombN<3><<<36 * 1 * 8, 256>>>(static_cast<uint32_t>(plants.size()), currentGeneration, previousGeneration);
        plantKernelCombN<4><<<36 * 2 * 8, 256>>>(static_cast<uint32_t>(plants.size()), currentGeneration, previousGeneration);
        plantKernelCombN<5><<<36 * 3 * 8, 256>>>(static_cast<uint32_t>(plants.size()), currentGeneration, previousGeneration);

        cudaDeviceSynchronize();
        _CUDA_CHECK_ERROR("generation launch sync");

        auto end = std::chrono::high_resolution_clock::now();
        auto took = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto work = currentGeneration->iterations;

        std::cout << "Generation " << g << ":   Took " << took << "ms to execute over " << work << " combinations\n";
        std::cout << "Generation " << g << ":   Rate is " << static_cast<uint64_t>(static_cast<double>(work) / (static_cast<double>(took) / 1000)) << " combinations/sec\n";

        for (Plant i = 0; i < maximumEncodedPlant; i++) {
            if (previousGeneration->members[i].count == 0 && currentGeneration->members[i].count != 0) {
                // This is a plant we only just learned how to make this generation, its guaranteed to be unique, append it to our list to permute over during the next generation
                plants.push_back(i);
            }
        }

        std::cout << "Generation " << g << ":   Now up to " << plants.size() << " plants before prune\n\n";

        memcpy(previousGeneration, currentGeneration, sizeof(Generation));

        if (findBestPlant()) {
            break;
        }
    }

    if (hasBestPlant) {
        printTutorialFor(bestPlant, currentGeneration);
    }

    cudaFree(currentGeneration);
    cudaFree(previousGeneration);

    return 0;
}

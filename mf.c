#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define a type for large counts
typedef long long ll;

// Function to process a folding by incrementing the count
void process(int *a, int *b, int n, ll *mCount) {
    *mCount += n;
}

// The main folding algorithm
void foldings(int *p, int dim, int flag, int res, int mod, ll *mCount) {
    // Calculate the total number of leaves
    int n = 1;
    for(int i = 0; i < dim; i++) {
        n *= p[i];
    }

    // Allocate and initialize necessary arrays
    int *a = (int*) calloc(n + 1, sizeof(int));
    int *b_arr = (int*) calloc(n + 1, sizeof(int));
    int *count = (int*) calloc(n + 1, sizeof(int));
    int *gapter = (int*) calloc(n + 1, sizeof(int));
    int gap_size = n * n + 1;
    int *gap = (int*) calloc(gap_size, sizeof(int));

    // Calculate bigP array
    int *bigP = (int*) malloc((dim + 1) * sizeof(int));
    bigP[0] = 1;
    for(int i = 1; i <= dim; i++) {
        bigP[i] = bigP[i - 1] * p[i - 1];
    }

    // Initialize c array
    int **c = (int**) malloc((dim + 1) * sizeof(int*));
    for(int i = 0; i <= dim; i++) {
        c[i] = (int*) malloc((n + 1) * sizeof(int));
    }
    for(int i = 1; i <= dim; i++) {
        for(int m = 1; m <= n; m++) {
            c[i][m] = (m - 1) / bigP[i - 1] - ((m - 1) / bigP[i]) * p[i - 1] + 1;
        }
    }

    // Initialize d array
    int ***d = (int***) malloc((dim + 1) * sizeof(int**));
    for(int i = 0; i <= dim; i++) {
        d[i] = (int**) malloc((n + 1) * sizeof(int*));
        for(int m = 0; m <= n; m++) {
            d[i][m] = (int*) malloc((n + 1) * sizeof(int));
        }
    }
    for(int i = 1; i <= dim; i++) {
        for(int l = 1; l <= n; l++) {
            for(int m = 1; m <= l; m++) {
                int delta = c[i][l] - c[i][m];
                if( (delta & 1) == 0 ) {
                    if(c[i][m] == 1 ) {
                        d[i][l][m] = m;
                    } else {
                        d[i][l][m] = m - bigP[i - 1];
                    }
                }
                else {
                    if(c[i][m] == p[i - 1] || m + bigP[i - 1] > l) {
                        d[i][l][m] = m;
                    } else {
                        d[i][l][m] = m + bigP[i - 1];
                    }
                }
            }
        }
    }

    // Initialize variables for the main loop
    int g = 0;
    int l_leaf = 1;

    // Main backtrack loop
    while(l_leaf > 0) {
        if(!flag || l_leaf <= 1 || b_arr[0] == 1) { // Filter normal case
            if(l_leaf > n) {
                process(a, b_arr, n, mCount);
            }
            else {
                int dd = 0;
                int gg = gapter[l_leaf - 1];
                g = gg;

                // Append potential gaps for leaf l in each section
                for(int i = 1; i <= dim; i++) {
                    if(d[i][l_leaf][l_leaf] == l_leaf) {
                        dd++;
                    }
                    else {
                        int m = d[i][l_leaf][l_leaf];
                        while(m != l_leaf) {
                            if(mod == 0 || l_leaf != mod || m % mod == res) {
                                gap[gg] = m;
                                if(count[m]++ == 0) {
                                    gg++;
                                }
                            }
                            m = d[i][l_leaf][b_arr[m]];
                        }
                    }
                }

                // Discard gaps not common to all sections
                if(dd == dim) {
                    for(int m = 0; m < l_leaf; m++) {
                        gap[gg++] = m;
                    }
                }

                for(int j = g; j < gg; j++) {
                    gap[g] = gap[j];
                    if(count[gap[j]] == dim - dd) {
                        g++;
                    }
                    count[gap[j]] = 0;
                }
            }
        }

        // Detach leaf l and retreat if no more gaps
        while(l_leaf > 0 && g == gapter[l_leaf - 1]) {
            l_leaf--;
            b_arr[a[l_leaf]] = b_arr[l_leaf];
            a[b_arr[l_leaf]] = a[l_leaf];
        }

        // Insert leaf l and advance to next level
        if(l_leaf > 0) {
            a[l_leaf] = gap[--g];
            b_arr[l_leaf] = b_arr[a[l_leaf]];
            b_arr[a[l_leaf]] = l_leaf;
            a[b_arr[l_leaf]] = l_leaf;
            gapter[l_leaf++] = g;
        }
    }

    // Free allocated memory
    free(a);
    free(b_arr);
    free(count);
    free(gapter);
    free(gap);
    free(bigP);
    for(int i = 0; i <= dim; i++) {
        free(c[i]);
    }
    free(c);
    for(int i = 0; i <= dim; i++) {
        for(int m = 0; m <= n; m++) {
            free(d[i][m]);
        }
        free(d[i]);
    }
    free(d);
}

int main(int argc, char *argv[]) {
    // Check for minimum arguments
    if(argc < 2) {
        printf("Usage: [res/mod] dimension...\n");
        return 1;
    }

    int res = 0, mod = 0;
    int argsUsed = 0;

    // Check if the first argument contains '/'
    char *slash = strchr(argv[1], '/');
    if(slash != NULL) {
        *slash = '\0'; // Split the string at '/'
        res = atoi(argv[1]);
        mod = atoi(slash + 1);
        argsUsed = 1;
    }

    // Calculate the number of dimensions
    int dim = argc - 1 - argsUsed;
    if(dim <= 0) {
        printf("No dimensions provided.\n");
        return 1;
    }

    // Parse dimensions
    int *d = (int*) malloc(dim * sizeof(int));
    for(int k = 0; k < dim; k++) {
        d[k] = atoi(argv[k + 1 + argsUsed]);
    }

    // Initialize count and perform foldings
    ll mCount = 0;
    foldings(d, dim, 1, res, mod, &mCount);
    printf("%lld\n", mCount);

    // Free allocated memory for dimensions
    free(d);
    return 0;
}

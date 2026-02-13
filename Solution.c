#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

#define EPSILON 0.001
#define PI 3.14159265358979323846
#define MAX_ITERATIONS 100

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ============================================
// UTILITY
// ============================================

void swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// ============================================
// MATH FUNCTIONS
// ============================================

int log2_int(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;  // Divide by 2
        log++;
    }
    return log;
}

float mysin(float x) {

    // Normalize x to [-PI, PI] range
    x = fmod(x, 2 * PI);
    if (x > PI) x -= 2 * PI;
    if (x < -PI) x += 2 * PI;
    
    float term = x;
    float res = x;
    int n = 1;
    
    while (fabs(term) > EPSILON) {
        term *= -x * x / ((2 * n) * (2 * n + 1));
        res += term;
        n++;
        
        // Safety check to prevent infinite loops
        if (n > MAX_ITERATIONS) break;
    }
    
    return res;
}

float mycos(float x) {

    // Normalize x to [-PI, PI] range
    x = fmod(x, 2 * PI);
    if (x > PI) x -= 2 * PI;
    if (x < -PI) x += 2 * PI;
    
    float term = 1.0;
    float res = 1.0;
    int n = 1;
    
    while (fabs(term) > EPSILON) {
        term *= -x * x / ((2 * n - 1) * (2 * n));
        res += term;
        n++;
        
        // Safety check to prevent infinite loops
        if (n > MAX_ITERATIONS) break;
    }
    
    return res;
}

float myexp(const float x) {

    // For large negative x
    // Use exp(-x) = 1/exp(x)
    // For better convergence
    if (x < -1) {
        return 1.0 / myexp(-x);
    }
    
    float res = 1.0;
    float term = 1.0;
    int i = 1;
    
    while (fabs(term) > EPSILON) {
        term *= x / i;
        res += term;
        i++;
        
        // Safety check
        if (i > 100) break;
    }
    
    return res;
}

// ============================================
// TWIDDLE FACTORS
// ============================================

void generate_twiddle_factors(float* real, float* imag, int n) {
    real[0] = 1.0;
    imag[0] = 0.0;

    // W[1] = e^(-2πi/n)
    float angle = -2.0 * PI / n;
    float w_real = mycos(angle);
    float w_imag = mysin(angle);

    const int m = n / 2;

    // Each W[k] = W[k-1] * W[1]
    for (int k = 1; k < m; k++) {
        real[k] = real[k-1] * w_real - imag[k-1] * w_imag;
        imag[k] = real[k-1] * w_imag + imag[k-1] * w_real;
    }
}

void generate_inverse_twiddle_factors(float* real, float* imag, int n) {
    real[0] = 1.0;
    imag[0] = 0.0;

    // Positive angle for inverse
    float angle = 2.0 * PI / n;
    float w_real = mycos(angle);
    float w_imag = mysin(angle);

    const int m = n / 2;

    for (int k = 1; k < m; k++) {
        real[k] = real[k-1] * w_real - imag[k-1] * w_imag;
        imag[k] = real[k-1] * w_imag + imag[k-1] * w_real;
    }
}

// ============================================
// BIT REVERSE
// ============================================

unsigned int reverse_bits(unsigned int x, int log_n) {
    unsigned int reversed = 0;
    for (int i = 0; i < log_n; i++) {
        unsigned int bit = x & 1;           // Get rightmost bit
        reversed = (reversed << 1) | bit;   // Add to reversed
        x >>= 1;                            // Move to next bit
    }
    return reversed;
}

void bit_reverse_array(float* x_real, float* x_imag, int n) {
    int log_n = log2_int(n);
    for (int i = 0; i < n; i++) {
        unsigned int j = reverse_bits(i, log_n);
        if (i < j) {
            swap(&x_real[i], &x_real[j]);
            swap(&x_imag[i], &x_imag[j]);
        }
    }
}

// ============================================
// 1D FFT - ITERATIVE VERSION
// ============================================

void butterfly_iterative(float* x_real, float* x_imag, float* twiddle_real, 
                         float* twiddle_imag, int n) {
    
    // Process stages: 2-point, 4-point, 8-point, ..., n-point
    for (int block_size = 2; block_size <= n; block_size *= 2) {
        int half_block = block_size / 2;
        
        for (int block_start = 0; block_start < n; block_start += block_size) {
            for (int k = 0; k < half_block; k++) {
                int i = block_start + k;
                int j = i + half_block;
                
                // Twiddle Factor index
                int t_index = k * n / block_size;
                float w_real = twiddle_real[t_index];
                float w_imag = twiddle_imag[t_index];
                
                // Complex Multiplication: t = W * x[j]
                float t_real = w_real * x_real[j] - w_imag * x_imag[j];
                float t_imag = w_real * x_imag[j] + w_imag * x_real[j];
                
                // Butterfly operations
                float u_real = x_real[i];
                float u_imag = x_imag[i];
                
                // x[i] = u + t
                x_real[i] = u_real + t_real;
                x_imag[i] = u_imag + t_imag;
                
                // x[j] = u - t
                x_real[j] = u_real - t_real;
                x_imag[j] = u_imag - t_imag;
            }
        }
    }
}

void fft_1d_iterative(float* x_real, float* x_imag, float* twiddle_real, 
                     float* twiddle_imag, int n) {    
    bit_reverse_array(x_real, x_imag, n);
    butterfly_iterative(x_real, x_imag, twiddle_real, twiddle_imag, n);
}

// ============================================
// 1D FFT - RECURSIVE VERSION
// ============================================

void butterfly_recursive_helper(float* x_real, float* x_imag, float* twiddle_real, 
                                float* twiddle_imag, int n, int N_original) {
    // Base case
    if (n <= 1) return;
    
    const int m = n / 2;
    
    // Recursively process first half and second half
    butterfly_recursive_helper(x_real, x_imag, twiddle_real, twiddle_imag, m, N_original);
    butterfly_recursive_helper(x_real + m, x_imag + m, twiddle_real, twiddle_imag, m, N_original);
    
    // Combine results
    for (int k = 0; k < m; k++) {

        // Twiddle factor index
        int t_index = k * (N_original / n);
        
        float w_real = twiddle_real[t_index];
        float w_imag = twiddle_imag[t_index];
        
        // Complex Multiplication: t = W * x[k + m]
        float t_real = w_real * x_real[k + m] - w_imag * x_imag[k + m];
        float t_imag = w_real * x_imag[k + m] + w_imag * x_real[k + m];
        
        // Butterfly operations
        float u_real = x_real[k];
        float u_imag = x_imag[k];
        
        // x[k] = u + t
        x_real[k] = u_real + t_real;
        x_imag[k] = u_imag + t_imag;
        
        // x[k + m] = u - t
        x_real[k + m] = u_real - t_real;
        x_imag[k + m] = u_imag - t_imag;
    }
}

void butterfly_recursive(float* x_real, float* x_imag, float* twiddle_real, 
                        float* twiddle_imag, int n) {    
    butterfly_recursive_helper(x_real, x_imag, twiddle_real, twiddle_imag, n, n);
}

void fft_1d_recursive(float* x_real, float* x_imag, float* twiddle_real, 
                     float* twiddle_imag, int n) {    
    bit_reverse_array(x_real, x_imag, n);
    butterfly_recursive(x_real, x_imag, twiddle_real, twiddle_imag, n);
}

// ============================================
// 2D FFT
// ============================================

void transpose(float** matrix, int rows, int cols) {
    float** temp = (float**)malloc(cols * sizeof(float*));

    for (int i = 0; i < cols; i++)
        temp[i] = (float*)malloc(rows * sizeof(float));
    
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            temp[j][i] = matrix[i][j];
    
    for (int i = 0; i < rows && i < cols; i++)
        for (int j = 0; j < cols && j < rows; j++)
            matrix[i][j] = temp[i][j];
    
    for (int i = 0; i < cols; i++)
        free(temp[i]);
    free(temp);
}

void fft_2d(float** x_real, float** x_imag, float* twiddle_real, float* twiddle_imag, int rows, int cols, bool test, bool inverse) {
    if (inverse) {
        generate_inverse_twiddle_factors(twiddle_real, twiddle_imag, MAX(rows, cols));
    } else {
        generate_twiddle_factors(twiddle_real, twiddle_imag, MAX(rows, cols));
    }
    
    for (int i = 0; i < rows; i++)
        fft_1d_iterative(x_real[i], x_imag[i], twiddle_real, twiddle_imag, cols);
    
    if (test) {
        printf("After row FFT:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("(%.2f+%.2fi) ", x_real[i][j], x_imag[i][j]);
            }
            printf("\n");
        }
    }

    // Transpose
    transpose(x_real, rows, cols);
    transpose(x_imag, rows, cols);
    
    // Apply FFT on columns
    for (int i = 0; i < cols; i++)
        fft_1d_iterative(x_real[i], x_imag[i], twiddle_real, twiddle_imag, rows);
    
    // Transpose back
    transpose(x_real, cols, rows);
    transpose(x_imag, cols, rows);
}

void ifft_2d(float** x_real, float** x_imag, float* twiddle_real, 
             float* twiddle_imag, int rows, int cols, bool test) {
    
    // Call fft_2d with inverse flag
    fft_2d(x_real, x_imag, twiddle_real, twiddle_imag, rows, cols, test, true);
    
    // Normalize by 1/(rows*cols)
    float norm = 1.0f / (rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            x_real[i][j] *= norm;
            x_imag[i][j] *= norm;
        }
    }
}

// ============================================
// FFT SHIFT - Move DC component to center
// ============================================

void fftshift_2d(float** real, float** imag, int rows, int cols) {
    int half_rows = rows / 2;
    int half_cols = cols / 2;
    
    // Swap quadrants:
    // Q1 <-> Q4, Q2 <-> Q3
    for (int i = 0; i < half_rows; i++) {
        for (int j = 0; j < half_cols; j++) {

            // Swap Q1 (top-left) with Q4 (bottom-right)
            swap(&real[i][j], &real[i + half_rows][j + half_cols]);
            swap(&imag[i][j], &imag[i + half_rows][j + half_cols]);
            
            // Swap Q2 (top-right) with Q3 (bottom-left)
            swap(&real[i][j + half_cols], &real[i + half_rows][j]);
            swap(&imag[i][j + half_cols], &imag[i + half_rows][j]);
        }
    }
}

void ifftshift_2d(float** real, float** imag, int rows, int cols) {
    fftshift_2d(real, imag, rows, cols);
}

// ============================================
// HIGH-PASS FILTER (Gaussian)
// ============================================

void create_highpass_filter(float** filter, int rows, int cols, float cutoff) {
    int center_i = rows / 2;
    int center_j = cols / 2;
    float D0_squared = cutoff * cutoff;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            // Calculate distance from center
            float di = (float)(i - center_i);
            float dj = (float)(j - center_j);
            
            float D_squared = di * di + dj * dj;
            
            // Gaussian high-pass
            filter[i][j] = 1.0f - myexp(-D_squared / (2.0f * D0_squared));
        }
    }
}

// ============================================
// EDGE DETECTION WITH THRESHOLDING
// ============================================

void edge_detection(float** x_real, float** x_imag, float* twiddle_real, 
                   float* twiddle_imag, float** filter, int rows, int cols, 
                   float cutoff) {
    
    // Step 1: Apply 2D FFT
    fft_2d(x_real, x_imag, twiddle_real, twiddle_imag, rows, cols, false, false);
    
    // Shift FFT (move DC to center)
    fftshift_2d(x_real, x_imag, rows, cols);
    
    // Create and apply high-pass filter
    create_highpass_filter(filter, rows, cols, cutoff);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            x_real[i][j] *= filter[i][j];
            x_imag[i][j] *= filter[i][j];
        }
    }
    
    // Inverse shift (move DC back to corner)
    ifftshift_2d(x_real, x_imag, rows, cols);
    
    // Apply inverse FFT
    ifft_2d(x_real, x_imag, twiddle_real, twiddle_imag, rows, cols, false);
}

// ============================================
// IMAGE PROCESSING WITH ENHANCED EDGE DETECTION
// ============================================

void process_image(const char* input_path, const char* fft_output_path, 
                  const char* edge_output_path, float cutoff, float threshold_percent) {
    
    printf("\n===========================================\n");
    printf("PROCESSING IMAGE: %s\n", input_path);
    printf("===========================================\n");
    
    // Load image
    int width, height, channels;
    unsigned char* img = stbi_load(input_path, &width, &height, &channels, 0);
    
    if (img == NULL) {
        printf("Failed to load image: %s\n", input_path);
        return;
    }
    
    printf("Loaded %dx%d image with %d channels\n", width, height, channels);
    printf("Cutoff frequency: %.1f\n", cutoff);
    printf("Edge threshold: %.1f%%\n", threshold_percent * 100);
    
    // Allocate 2D arrays
    float** img_real = (float**)malloc(height * sizeof(float*));
    float** img_imag = (float**)malloc(height * sizeof(float*));
    float** filter = (float**)malloc(height * sizeof(float*));
    
    for (int i = 0; i < height; i++) {
        img_real[i] = (float*)malloc(width * sizeof(float));
        img_imag[i] = (float*)malloc(width * sizeof(float));
        filter[i] = (float*)malloc(width * sizeof(float));
    }
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            
            float gray = channels >= 3 
                ? (img[idx] + img[idx+1] + img[idx+2]) / (3.0f * 255.0f)
                : img[idx] / 255.0f;
            img_real[i][j] = gray;
            img_imag[i][j] = 0.0f;
        }
    }
    
    // Allocate twiddle factors
    int max_dim = MAX(width, height);
    float* twiddle_real = (float*)malloc(max_dim/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(max_dim/2 * sizeof(float));
    
    // === FFT MAGNITUDE OUTPUT ===
    printf("\nComputing 2D FFT...\n");
    
    // Make copy for FFT visualization
    float** fft_real_copy = (float**)malloc(height * sizeof(float*));
    float** fft_imag_copy = (float**)malloc(height * sizeof(float*));
    for (int i = 0; i < height; i++) {
        fft_real_copy[i] = (float*)malloc(width * sizeof(float));
        fft_imag_copy[i] = (float*)malloc(width * sizeof(float));
        for (int j = 0; j < width; j++) {
            fft_real_copy[i][j] = img_real[i][j];
            fft_imag_copy[i][j] = img_imag[i][j];
        }
    }
    
    fft_2d(fft_real_copy, fft_imag_copy, twiddle_real, twiddle_imag, 
           height, width, false, false);
    
    // Shift for visualization
    fftshift_2d(fft_real_copy, fft_imag_copy, height, width);
    
    // Compute magnitude and log scale
    unsigned char* fft_output = (unsigned char*)malloc(height * width);
    float max_mag = 0.0f;
    
    float** temp_mag = (float**)malloc(height * sizeof(float*));
    for (int i = 0; i < height; i++) {
        temp_mag[i] = (float*)malloc(width * sizeof(float));
    }
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = sqrtf(fft_real_copy[i][j] * fft_real_copy[i][j] + 
                            fft_imag_copy[i][j] * fft_imag_copy[i][j]);
            float log_mag = log(1.0f + mag);
            temp_mag[i][j] = log_mag;
            if (log_mag > max_mag) max_mag = log_mag;
        }
    }
    
    // Normalize and convert to uint8
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fft_output[i * width + j] = (unsigned char)(255.0f * temp_mag[i][j] / max_mag);
        }
    }
    
    stbi_write_png(fft_output_path, width, height, 1, fft_output, width);
    printf("FFT magnitude saved to: %s\n", fft_output_path);
    
    // === EDGE DETECTION WITH THRESHOLDING ===
    printf("\nPerforming edge detection...\n");
    
    // Reload original image data
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * channels;
            float gray = channels >= 3 
                ? (img[idx] + img[idx+1] + img[idx+2]) / (3.0f * 255.0f)
                : img[idx] / 255.0f;
            img_real[i][j] = gray;
            img_imag[i][j] = 0.0f;
        }
    }
    
    edge_detection(img_real, img_imag, twiddle_real, twiddle_imag, 
                  filter, height, width, cutoff);

    // Compute magnitude of the edge result
    float max_edge = 0.0f;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float mag = sqrtf(img_real[i][j] * img_real[i][j] + 
                            img_imag[i][j] * img_imag[i][j]);
            temp_mag[i][j] = mag;
            if (mag > max_edge) max_edge = mag;
        }
    }
    
    printf("Max edge magnitude: %.4f\n", max_edge);
    
    float threshold = max_edge * threshold_percent;
    printf("Edge threshold value: %.4f\n", threshold);
    
    unsigned char* edge_output = (unsigned char*)malloc(height * width);
    int edge_pixel_count = 0;
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (temp_mag[i][j] > threshold) {
                // Edge detected -> WHITE
                edge_output[i * width + j] = 255;
                edge_pixel_count++;
            } else {
                // No edge detected -> BLACK
                edge_output[i * width + j] = 0;
            }
        }
    }
    
    float edge_percentage = (edge_pixel_count * 100.0f) / (width * height);
    printf("Edge pixels: %d (%.2f%% of image)\n", edge_pixel_count, edge_percentage);
        
    stbi_write_png(edge_output_path, width, height, 1, edge_output, width);
    printf("Edge detection saved to: %s\n", edge_output_path);
    
    // === PRINT 8x8 FFT RESULTS ===
    if (width == 8 && height == 8) {
        printf("\n8x8 FFT Analysis:\n");
        printf("Real part:\n");
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%7.2f ", fft_real_copy[i][j]);
            }
            printf("\n");
        }
        printf("\nImaginary part:\n");
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%7.2f ", fft_imag_copy[i][j]);
            }
            printf("\n");
        }
    }
    
    // Cleanup
    for (int i = 0; i < height; i++) {
        free(img_real[i]);
        free(img_imag[i]);
        free(filter[i]);
        free(fft_real_copy[i]);
        free(fft_imag_copy[i]);
        free(temp_mag[i]);
    }
    free(img_real);
    free(img_imag);
    free(filter);
    free(fft_real_copy);
    free(fft_imag_copy);
    free(temp_mag);
    free(twiddle_real);
    free(twiddle_imag);
    free(fft_output);
    free(edge_output);
    stbi_image_free(img);
    
    printf("===========================================\n\n");
}

// ============================================
// MAIN PROGRAM
// ============================================


void test_1d_fft() {
    printf("\n===========================================\n");
    printf("TEST 2: 1D FFT Validation\n");
    printf("===========================================\n");
    
    int n = 8;  // Test size
    float* x_real_iter = (float*)malloc(n * sizeof(float));
    float* x_imag_iter = (float*)malloc(n * sizeof(float));
    float* x_real_rec = (float*)malloc(n * sizeof(float));
    float* x_imag_rec = (float*)malloc(n * sizeof(float));
    float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(n/2 * sizeof(float));

    generate_twiddle_factors(twiddle_real, twiddle_imag, n);
    
    // Test 2a: Impulse response (delta function)
    printf("\nTest 2a: Impulse Response\n");
    printf("Input: [1, 0, 0, 0, 0, 0, 0, 0]\n");
    
    // Initialize impulse for both versions
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = (i == 0) ? 1.0 : 0.0;
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    // Run iterative FFT
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    
    // Run recursive FFT
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    // Check results (impulse should give all 1+0i)
    printf("Expected: All bins should be 1.0 + 0.0i\n");
    printf("\nIterative FFT Results:\n");
    int iter_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_iter[i], x_imag_iter[i]);
        if (fabs(x_real_iter[i] - 1.0) > EPSILON || fabs(x_imag_iter[i]) > EPSILON) {
            printf(" [FAIL]");
            iter_correct = 0;
        } else {
            printf(" [PASS]");
        }
        printf("\n");
    }
    
    printf("\nRecursive FFT Results:\n");
    int rec_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_rec[i], x_imag_rec[i]);
        if (fabs(x_real_rec[i] - 1.0) > EPSILON || fabs(x_imag_rec[i]) > EPSILON) {
            printf(" [FAIL]");
            rec_correct = 0;
        } else {
            printf(" [PASS]");
        }
        printf("\n");
    }
    
    printf("\nTest 2a Result: Iterative %s, Recursive %s\n", 
           iter_correct ? "PASS" : "FAIL",
           rec_correct ? "PASS" : "FAIL");
    
    // Test 2b: DC signal (all ones)
    printf("\n\nTest 2b: DC Signal\n");
    printf("Input: [1, 1, 1, 1, 1, 1, 1, 1]\n");
    
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = 1.0;
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    printf("Expected: Bin[0] = 8.0, all others = 0.0\n");
    printf("\nIterative FFT Results:\n");
    iter_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_iter[i], x_imag_iter[i]);
        if (i == 0) {
            if (fabs(x_real_iter[i] - 8.0) > EPSILON || fabs(x_imag_iter[i]) > EPSILON) {
                printf(" [FAIL]");
                iter_correct = 0;
            } else {
                printf(" [PASS]");
            }
        } else {
            if (fabs(x_real_iter[i]) > EPSILON || fabs(x_imag_iter[i]) > EPSILON) {
                printf(" [FAIL]");
                iter_correct = 0;
            } else {
                printf(" [PASS]");
            }
        }
        printf("\n");
    }
    
    printf("\nRecursive FFT Results:\n");
    rec_correct = 1;
    for (int i = 0; i < n; i++) {
        printf("  Bin[%d]: %.3f + %.3fi", i, x_real_rec[i], x_imag_rec[i]);
        if (i == 0) {
            if (fabs(x_real_rec[i] - 8.0) > EPSILON || fabs(x_imag_rec[i]) > EPSILON) {
                printf(" [FAIL]");
                rec_correct = 0;
            } else {
                printf(" [PASS]");
            }
        } else {
            if (fabs(x_real_rec[i]) > EPSILON || fabs(x_imag_rec[i]) > EPSILON) {
                printf(" [FAIL]");
                rec_correct = 0;
            } else {
                printf(" [PASS]");
            }
        }
        printf("\n");
    }
    
    printf("\nTest 2b Result: Iterative %s, Recursive %s\n", 
           iter_correct ? "PASS" : "FAIL",
           rec_correct ? "PASS" : "FAIL");

    // Test 2c: Compare iterative vs recursive output
    printf("\n\nTest 2c: Iterative vs Recursive Consistency\n");
    printf("Testing if both methods produce same results...\n");
    
    // Create a test signal
    for (int i = 0; i < n; i++) {
        x_real_iter[i] = (float)(i % 3);  // Arbitrary test signal
        x_imag_iter[i] = 0.0;
        x_real_rec[i] = x_real_iter[i];
        x_imag_rec[i] = x_imag_iter[i];
    }
    
    fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
    fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
    
    int match = 1;
    for (int i = 0; i < n; i++) {
        float diff_real = fabs(x_real_iter[i] - x_real_rec[i]);
        float diff_imag = fabs(x_imag_iter[i] - x_imag_rec[i]);
        
        if (diff_real > EPSILON || diff_imag > EPSILON) {
            printf("  Bin[%d] MISMATCH: Iter=(%.3f+%.3fi), Rec=(%.3f+%.3fi)\n",
                   i, x_real_iter[i], x_imag_iter[i], 
                   x_real_rec[i], x_imag_rec[i]);
            match = 0;
        }
    }
    
    if (match) {
        printf("  All bins match! [PASS]\n");
    } else {
        printf("  Some bins don't match [FAIL]\n");
    }
    
    printf("\nTest 2c Result: %s\n", match ? "PASS" : "FAIL");
    
    free(x_real_iter);
    free(x_imag_iter);
    free(x_real_rec);
    free(x_imag_rec);
}

// ============================================
// Function 2: Benchmark 1D FFT
// ============================================

void benchmark_1d_fft() {
    printf("\n===========================================\n");
    printf("BENCHMARK: 2D FFT Performance\n");
    printf("===========================================\n");
    printf("\nSize\tIterative(s)\tRecursive(s)\tSpeedup\n");
    printf("----\t------------\t------------\t-------\n");
    
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int num_sizes = 8;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        
        // Allocate arrays
        float* x_real_iter = (float*)malloc(n * sizeof(float));
        float* x_imag_iter = (float*)malloc(n * sizeof(float));
        float* x_real_rec = (float*)malloc(n * sizeof(float));
        float* x_imag_rec = (float*)malloc(n * sizeof(float));
        float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
        float* twiddle_imag = (float*)malloc(n/2 * sizeof(float));

        generate_twiddle_factors(twiddle_real, twiddle_imag, n);
        
        // Initialize with test data
        for (int i = 0; i < n; i++) {
            x_real_iter[i] = (float)i;
            x_imag_iter[i] = 0.0;
            x_real_rec[i] = x_real_iter[i];
            x_imag_rec[i] = x_imag_iter[i];
        }
        
        // Benchmark iterative
        clock_t start = clock();
        fft_1d_iterative(x_real_iter, x_imag_iter, twiddle_real, twiddle_imag, n);
        clock_t end = clock();
        double time_iter = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Benchmark recursive
        start = clock();
        fft_1d_recursive(x_real_rec, x_imag_rec, twiddle_real, twiddle_imag, n);
        end = clock();
        double time_rec = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        double speedup = time_rec / time_iter;
        
        printf("%d\t%.6f\t%.6f\t%.2fx\n", n, time_iter, time_rec, speedup);
        
        free(x_real_iter);
        free(x_imag_iter);
        free(x_real_rec);
        free(x_imag_rec);
    }
    
    printf("\n");
}

// ============================================
// Function 3: Test Trigonometric Functions
// ============================================

void test_math_functions() {
    printf("\n===========================================\n");
    printf("TEST 1: Math Functions Validation\n");
    printf("===========================================\n");
    
    // Test points for trig functions
    float test_angles[] = {
        0.0,
        PI / 6.0,      // 30 degrees
        PI / 4.0,      // 45 degrees
        PI / 3.0,      // 60 degrees
        PI / 2.0,      // 90 degrees
        PI,            // 180 degrees
        3.0 * PI / 2.0,  // 270 degrees
        2.0 * PI,      // 360 degrees
        -PI / 4.0,     // -45 degrees
        -PI / 2.0      // -90 degrees
    };
    
    char* angle_names[] = {
        "0pi",
        "30pi",
        "45pi",
        "60pi",
        "90pi",
        "180pi",
        "270pi",
        "360pi",
        "-45pi",
        "-90pi"
    };
    
    // Test points for exponential function
    float test_exponents[] = {
        0.0,
        0.5,
        1.0,
        2.0,
        -1.0,
        -2.0,
        5.0,
        -5.0,
        0.1,
        -0.1
    };
    
    char* exp_names[] = {
        "0.0",
        "0.5",
        "1.0",
        "2.0",
        "-1.0",
        "-2.0",
        "5.0",
        "-5.0",
        "0.1",
        "-0.1"
    };
    
    int num_trig_tests = 10;
    int num_exp_tests = 10;
    int sin_passed = 0;
    int cos_passed = 0;
    int exp_passed = 0;
    
    printf("\nTesting mysin():\n");
    printf("Angle\tYour Result\tExpected\tError\t\tStatus\n");
    printf("-----\t-----------\t--------\t-----\t\t------\n");
    
    for (int i = 0; i < num_trig_tests; i++) {
        float your_sin = mysin(test_angles[i]);
        float expected_sin = sin(test_angles[i]);
        float error = fabs(your_sin - expected_sin);
        
        printf("%s\t%.6f\t%.6f\t%.6f\t", 
               angle_names[i], your_sin, expected_sin, error);
        
        if (error < EPSILON) {
            printf("[PASS]\n");
            sin_passed++;
        } else {
            printf("[FAIL]\n");
        }
    }
    
    printf("\nTesting mycos():\n");
    printf("Angle\tYour Result\tExpected\tError\t\tStatus\n");
    printf("-----\t-----------\t--------\t-----\t\t------\n");
    
    for (int i = 0; i < num_trig_tests; i++) {
        float your_cos = mycos(test_angles[i]);
        float expected_cos = cos(test_angles[i]);
        float error = fabs(your_cos - expected_cos);
        
        printf("%s\t%.6f\t%.6f\t%.6f\t", 
               angle_names[i], your_cos, expected_cos, error);
        
        if (error < EPSILON) {
            printf("[PASS]\n");
            cos_passed++;
        } else {
            printf("[FAIL]\n");
        }
    }
    
    printf("\nTesting myexp():\n");
    printf("x\tYour Result\tExpected\tError\t\tStatus\n");
    printf("-----\t-----------\t--------\t-----\t\t------\n");
    
    for (int i = 0; i < num_exp_tests; i++) {
        float your_exp = myexp(test_exponents[i]);
        float expected_exp = exp(test_exponents[i]);
        float error = fabs(your_exp - expected_exp);
        
        printf("%s\t%.6f\t%.6f\t%.6f\t", 
               exp_names[i], your_exp, expected_exp, error);
        
        if (error < EPSILON) {
            printf("[PASS]\n");
            exp_passed++;
        } else {
            printf("[FAIL]\n");
        }
    }
    
    printf("\n===========================================\n");
    printf("Summary: sin() %d/%d passed, cos() %d/%d passed, exp() %d/%d passed\n", 
           sin_passed, num_trig_tests, cos_passed, num_trig_tests, exp_passed, num_exp_tests);
    printf("===========================================\n");
}

// ============================================
// Function 4: Test 2D FFT
// ============================================

void test_2d_fft() {
    printf("\n===========================================\n");
    printf("TEST 3: 2D FFT Validation\n");
    printf("===========================================\n");
    
    int rows = 4;
    int cols = 4;
    int size = rows * cols;
    
    float** matrix_real = (float**)malloc(rows * sizeof(float*));
    float** matrix_imag = (float**)malloc(rows * sizeof(float*));
    float* twiddle_real = (float*)malloc(rows/2 * sizeof(float));
    float* twiddle_imag = (float*)malloc(rows/2 * sizeof(float));

    // Test 3a: 2D Impulse (delta at [0,0])
    printf("\nTest 3a: 2D Impulse Response\n");
    printf("Input: Delta function at position [0,0]\n");
    
    for (int i = 0; i < rows; i++) {
        matrix_real[i] = (float*)malloc(cols * sizeof(float));
        matrix_imag[i] = (float*)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            if (i == 0 && j == 0) {
                matrix_real[i][j] = 1.0;
                matrix_imag[i][j] = 0.0;
            } else {
                matrix_real[i][j] = 0.0;
                matrix_imag[i][j] = 0.0;
            }
        }
    }
    
    printf("Input matrix (real part):\n");
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", matrix_real[i][j]);
        }
        printf("\n");
    }
    
    fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, rows, cols, true, false);
    
    printf("\nExpected: All frequency bins should be 1.0 + 0.0i\n");
    printf("Output (real + imag):\n");
    
    int test_passed = 1;
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("(%.2f+%.2fi) ", matrix_real[i][j], matrix_imag[i][j]);
            
            if (fabs(matrix_real[i][j] - 1.0) > EPSILON || 
                fabs(matrix_imag[i][j]) > EPSILON) {
                test_passed = 0;
            }
        }
        printf("\n");
    }
    
    printf("\nTest 3a Result: %s\n", test_passed ? "PASS" : "FAIL");
    
    // Test 3b: 2D DC signal (all ones)
    printf("\n\nTest 3b: 2D DC Signal\n");
    printf("Input: All ones\n");

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix_real[i][j] = 1.0;
            matrix_imag[i][j] = 0.0;
        }
    }
    
    printf("Input matrix (real part):\n");
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", matrix_real[i][j]);
        }
        printf("\n");
    }
    
    fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, rows, cols, true, false);
    
    printf("\nExpected: DC bin [0,0] = 16.0, all others = 0.0\n");
    printf("Output (real + imag):\n");
    
    test_passed = 1;
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("(%.2f+%.2fi) ", matrix_real[i][j], matrix_imag[i][j]);
            
            if (i == 0 && j == 0) {
                // DC component should be 16
                if (fabs(matrix_real[i][j] - 16.0) > EPSILON || 
                    fabs(matrix_imag[i][j]) > EPSILON) {
                    test_passed = 0;
                }
            } else {
                // All others should be ~0
                if (fabs(matrix_real[i][j]) > EPSILON || 
                    fabs(matrix_imag[i][j]) > EPSILON) {
                    test_passed = 0;
                }
            }
        }
        printf("\n");
    }
    
    printf("\nTest 3b Result: %s\n", test_passed ? "PASS" : "FAIL");
    
    free(twiddle_real);
    free(twiddle_imag);

    for (int i = 0; i < rows; i++) {
        free(matrix_real[i]);
        free(matrix_imag[i]);
    }
    free(matrix_real);
    free(matrix_imag);
}

// ============================================
// Function 5: Benchmark 2D FFT
// ============================================

void benchmark_2d_fft() {
    printf("\n===========================================\n");
    printf("BENCHMARK: 2D FFT Performance\n");
    printf("===========================================\n");
    printf("\nSize\t\tTime(s)\t\tOperations\n");
    printf("----\t\t-------\t\t----------\n");
    
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = 6;
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        int total_size = n * n;
        
        float** matrix_real = (float**)malloc(n * sizeof(float*));
        float** matrix_imag = (float**)malloc(n * sizeof(float*));
        float* twiddle_real = (float*)malloc(n/2 * sizeof(float));
        float* twiddle_imag = (float*)malloc(n/2 * sizeof(float)); 
        
        // Initialize with test data
        for (int i = 0; i < n; i++) {
            matrix_real[i] = (float*)malloc(n * sizeof(float));
            matrix_imag[i] = (float*)malloc(n * sizeof(float));
            for (int j = 0; j < n; j++) {
                matrix_real[i][j] = (float)((i * n + j) % 10);
                matrix_imag[i][j] = 0.0;
            }
        }
        
        // Benchmark
        clock_t start = clock();
        fft_2d(matrix_real, matrix_imag, twiddle_real, twiddle_imag, n, n, false, false);
        clock_t end = clock();
        
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        
        // Theoretical operations: 2 * n * n * log2(n)
        // (n row FFTs of size n, n column FFTs of size n)
        int log_n = 0;
        int temp = n;
        while (temp > 1) {
            log_n++;
            temp >>= 1;
        }
        long operations = 2L * n * n * log_n;
        
        printf("%dx%d\t\t%.6f\t%ld\n", n, n, time_taken, operations);
        
        for (int i = 0; i < n; i++) {
            free(matrix_real[i]);
            free(matrix_imag[i]);
        }
        free(matrix_real);
        free(matrix_imag);
        free(twiddle_real);
        free(twiddle_imag);
    }
    
    printf("\n");
}

// ============================================
// Function to Run All Tests
// ============================================

void run_all_tests() {
    printf("===========================================\n");
    printf("FFT TEST SUITE - MILESTONE 1\n");
    printf("===========================================\n");
    
    // Run all tests
    test_math_functions();
    test_1d_fft();
    benchmark_1d_fft();
    test_2d_fft();
    benchmark_2d_fft();
    
    printf("\n===========================================\n");
    printf("ALL TESTS COMPLETE\n");
    printf("===========================================\n");
 
}

int main() {
    run_all_tests();

    process_image("./IMAGES/8x8.png", "./FFT/8x8.png", "./EDGES/8x8.png", 2.0f, 0.15f);
    process_image("./IMAGES/8x8-2.png", "./FFT/8x8-2.png", "./EDGES/8x8-2.png", 2.0f, 0.15f);
    process_image("./IMAGES/16x16.png", "./FFT/16x16.png", "./EDGES/16x16.png", 3.0f, 0.15f);
    process_image("./IMAGES/OIP.png", "./FFT/OIP.png", "./EDGES/OIP.png", 100.0f, 0.20f);
    process_image("./IMAGES/R.png", "./FFT/R.png", "./EDGES/R.png", 30.0f, 0.15f);

    return 0;
}
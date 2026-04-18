/*
 * Utility to generate simple test images with geometric shapes.
 * Creates images with known content for testing vision inference.
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

void create_circle_image(const char* filename, int width = 256, int height = 256) {
    std::vector<unsigned char> image(width * height * 3);

    int center_x = width / 2;
    int center_y = height / 2;
    int radius = width / 3;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;

            // Calculate distance from center
            int dx = x - center_x;
            int dy = y - center_y;
            float dist = std::sqrt(dx * dx + dy * dy);

            // Draw white circle on black background
            if (dist <= radius) {
                image[idx + 0] = 255;  // R
                image[idx + 1] = 255;  // G
                image[idx + 2] = 255;  // B
            } else {
                image[idx + 0] = 0;
                image[idx + 1] = 0;
                image[idx + 2] = 0;
            }
        }
    }

    stbi_write_png(filename, width, height, 3, image.data(), width * 3);
    std::cout << "Created circle image: " << filename << std::endl;
}

void create_square_image(const char* filename, int width = 256, int height = 256) {
    std::vector<unsigned char> image(width * height * 3);

    int center_x = width / 2;
    int center_y = height / 2;
    int half_size = width / 3;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;

            // Draw white square on black background
            if (x >= center_x - half_size && x <= center_x + half_size &&
                y >= center_y - half_size && y <= center_y + half_size) {
                image[idx + 0] = 255;  // R
                image[idx + 1] = 255;  // G
                image[idx + 2] = 255;  // B
            } else {
                image[idx + 0] = 0;
                image[idx + 1] = 0;
                image[idx + 2] = 0;
            }
        }
    }

    stbi_write_png(filename, width, height, 3, image.data(), width * 3);
    std::cout << "Created square image: " << filename << std::endl;
}

void create_triangle_image(const char* filename, int width = 256, int height = 256) {
    std::vector<unsigned char> image(width * height * 3);

    int center_x = width / 2;
    int size = width / 3;

    // Triangle vertices (equilateral triangle pointing up)
    int top_x = center_x;
    int top_y = height / 2 - size;
    int bottom_y = height / 2 + size;
    int left_x = center_x - size;
    int right_x = center_x + size;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;

            // Point-in-triangle test using barycentric coordinates
            bool inside = false;

            if (y >= top_y && y <= bottom_y) {
                // Calculate x bounds at this y
                float t = (float)(y - top_y) / (bottom_y - top_y);
                int left_bound = top_x - (int)(t * (top_x - left_x));
                int right_bound = top_x + (int)(t * (right_x - top_x));

                if (x >= left_bound && x <= right_bound) {
                    inside = true;
                }
            }

            // Draw white triangle on black background
            if (inside) {
                image[idx + 0] = 255;  // R
                image[idx + 1] = 255;  // G
                image[idx + 2] = 255;  // B
            } else {
                image[idx + 0] = 0;
                image[idx + 1] = 0;
                image[idx + 2] = 0;
            }
        }
    }

    stbi_write_png(filename, width, height, 3, image.data(), width * 3);
    std::cout << "Created triangle image: " << filename << std::endl;
}

int main() {
    std::cout << "Generating test shape images..." << std::endl;

    create_circle_image("test_data/circle.png");
    create_square_image("test_data/square.png");
    create_triangle_image("test_data/triangle.png");

    std::cout << "Done! Generated 3 test images in test_data/" << std::endl;

    return 0;
}

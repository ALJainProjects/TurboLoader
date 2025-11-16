#include "turboloader/transforms/image_transform.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace turboloader;

// Create a simple test image (solid color)
std::vector<uint8_t> create_test_image(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    std::vector<uint8_t> img(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        img[i * 3 + 0] = r;
        img[i * 3 + 1] = g;
        img[i * 3 + 2] = b;
    }
    return img;
}

int main() {
    std::cout << "Testing Image Transforms...\n\n";

    // Test 1: Center Crop
    std::cout << "Test 1: Center Crop Transform...\n";
    {
        auto img = create_test_image(100, 100, 255, 0, 0);
        CenterCropTransform crop(50, 50);
        auto result = crop.apply(img, 100, 100);

        assert(result.width == 50);
        assert(result.height == 50);
        assert(result.data.size() == 50 * 50 * 3);
        std::cout << "  ✓ PASS: Cropped to 50x50\n\n";
    }

    // Test 2: Horizontal Flip
    std::cout << "Test 2: Horizontal Flip Transform...\n";
    {
        std::vector<uint8_t> img(10 * 10 * 3);
        // Set left half to red, right half to blue
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                int idx = (y * 10 + x) * 3;
                if (x < 5) {
                    img[idx + 0] = 255;  // Red
                    img[idx + 1] = 0;
                    img[idx + 2] = 0;
                } else {
                    img[idx + 0] = 0;    // Blue
                    img[idx + 1] = 0;
                    img[idx + 2] = 255;
                }
            }
        }

        HorizontalFlipTransform flip;
        auto result = flip.apply(img, 10, 10);

        // After flip, left half should be blue, right half should be red
        int left_idx = (5 * 10 + 2) * 3;  // Left half
        assert(result.data[left_idx + 0] == 0);
        assert(result.data[left_idx + 2] == 255);

        int right_idx = (5 * 10 + 7) * 3;  // Right half
        assert(result.data[right_idx + 0] == 255);
        assert(result.data[right_idx + 2] == 0);

        std::cout << "  ✓ PASS: Correctly flipped horizontally\n\n";
    }

    // Test 3: Vertical Flip
    std::cout << "Test 3: Vertical Flip Transform...\n";
    {
        std::vector<uint8_t> img(10 * 10 * 3);
        // Set top half to green, bottom half to red
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 10; ++x) {
                int idx = (y * 10 + x) * 3;
                if (y < 5) {
                    img[idx + 0] = 0;    // Green
                    img[idx + 1] = 255;
                    img[idx + 2] = 0;
                } else {
                    img[idx + 0] = 255;  // Red
                    img[idx + 1] = 0;
                    img[idx + 2] = 0;
                }
            }
        }

        VerticalFlipTransform flip;
        auto result = flip.apply(img, 10, 10);

        // After flip, top half should be red, bottom half should be green
        int top_idx = (2 * 10 + 5) * 3;  // Top half
        assert(result.data[top_idx + 0] == 255);
        assert(result.data[top_idx + 1] == 0);

        int bottom_idx = (7 * 10 + 5) * 3;  // Bottom half
        assert(result.data[bottom_idx + 0] == 0);
        assert(result.data[bottom_idx + 1] == 255);

        std::cout << "  ✓ PASS: Correctly flipped vertically\n\n";
    }

    // Test 4: Composed Transform
    std::cout << "Test 4: Composed Transform (Resize + Center Crop)...\n";
    {
        auto img = create_test_image(256, 256, 128, 128, 128);

        ComposedTransform composed;
        composed.add(std::make_unique<ResizeTransform>(224, 224));
        composed.add(std::make_unique<CenterCropTransform>(192, 192));

        auto result = composed.apply(img, 256, 256);

        assert(result.width == 192);
        assert(result.height == 192);
        std::cout << "  ✓ PASS: Composed transform works (256 -> resize(224) -> crop(192))\n\n";
    }

    // Test 5: Random Crop (just verify it runs and produces valid output)
    std::cout << "Test 5: Random Crop Transform...\n";
    {
        auto img = create_test_image(100, 100, 0, 255, 0);
        RandomCropTransform crop(50, 50);

        // Run multiple times to test randomness
        for (int i = 0; i < 5; ++i) {
            auto result = crop.apply(img, 100, 100);
            assert(result.width == 50);
            assert(result.height == 50);
        }

        std::cout << "  ✓ PASS: Random crop produces valid outputs\n\n";
    }

    std::cout << "All transform tests passed!\n";
    return 0;
}

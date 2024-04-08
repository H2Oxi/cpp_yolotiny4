#include <iostream>  

#include <vector>  

  

// 创建一个填充后的二维数组  

std::vector<std::vector<int>> padArray(const std::vector<std::vector<int>>& original, int padSize, int valueToPad) {  

    int originalRows = original.size();  

    if (originalRows == 0) return {}; // 如果原始数组为空，直接返回空数组  

    int originalCols = original[0].size();  

  

    // 创建新的填充后的数组  

    std::vector<std::vector<int>> padded(originalRows + 2 * padSize, std::vector<int>(originalCols + 2 * padSize, valueToPad));  

  

    // 将原始数组复制到新数组的中央位置  

    for (int i = 0; i < originalRows; ++i) {  

        for (int j = 0; j < originalCols; ++j) {  

            padded[i + padSize][j + padSize] = original[i][j];  

        }  

    }  

  

    return padded;  

}  

  

int main() {  

    // 示例原始二维数组  

    std::vector<std::vector<int>> original = {  

        {1, 2, 3},  

        {4, 5, 6},  

        {7, 8, 9}  

    };  

  

    // 填充大小  

    int padSize = 1;  

    // 要填充的值  

    int valueToPad = 0;  

  

    // 调用填充函数  

    std::vector<std::vector<int>> paddedArray = padArray(original, padSize, valueToPad);  

  

    // 打印填充后的数组  

    for (const auto& row : paddedArray) {  

        for (int value : row) {  

            std::cout << value << " ";  

        }  

        std::cout << std::endl;  

    }  

  

    return 0;  

}
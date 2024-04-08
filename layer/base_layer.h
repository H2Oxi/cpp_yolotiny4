#include <iostream>
#include <vector>   
#include <cstring>
using namespace std;

#define data_t int8_t 
data_t*** padArray(const data_t * originalArray, int originalRows, int originalCols,int chi);
data_t * flatten_3d(data_t *** arr,int size2,int size1,int size0);

class conv_2d
{
    private:
        unsigned int in_shape[3]; //in_channels,r,c
        unsigned int out_shape[3];
        unsigned int output_channels;
        unsigned int in_channels;
        unsigned int ksize;
        unsigned int stride;

        data_t * weights;
        int32_t * bias;
    public:

        conv_2d(unsigned int in_shape[3],unsigned int output_channels,unsigned int ksize,unsigned int stride );
        void data_update(data_t * weights,int32_t * bias);
        void forward(data_t *x,data_t *out);
        void padding_forward(data_t *x ,int32_t *out);
      
};
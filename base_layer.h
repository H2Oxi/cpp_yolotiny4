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
        int16_t m0;
    public:

        conv_2d(unsigned int in_shape[3],unsigned int output_channels,unsigned int ksize,unsigned int stride );
        void data_update(data_t * weights,int32_t * bias,int16_t m0);
        void forward(data_t *x,data_t *out);
        void padding_forward(data_t *x ,int32_t *out);
      
};

class leakyrelu
{
    private:
        unsigned int in_shape[3];
        unsigned short int S_relu;
        unsigned int S_bitshift;

    public:
        leakyrelu(unsigned int in_shape_i[3])  ;
        void forward(data_t *x,int32_t *out);
        void S_update(unsigned short int S_relu,unsigned int S_bitshift) ;
        

};

class maxpooling
{
    private:
        unsigned int in_shape[3];
        unsigned int out_shape[3];
        unsigned int ksize;
        unsigned int stride;
        data_t find_max(data_t x0,data_t x1,data_t x2,data_t x3);
    public:
        maxpooling(unsigned int in_shape[3],unsigned int ksize,unsigned int stride );
        void forward(data_t *x,data_t *out);
};

class basic_conv
{
    private:

    public:
        conv_2d conv;
        leakyrelu activation;

        
};
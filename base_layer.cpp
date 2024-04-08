
#include "base_layer.h"

conv_2d::conv_2d(unsigned int in_shape_i[3],unsigned int output_channels_i,unsigned int ksize_i,unsigned int stride_i )
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape_i[i];
    }
    this->output_channels=output_channels_i;
    this->ksize=ksize_i;
    this->stride=stride_i;
    this->in_channels=this->in_shape[0];
    this->out_shape[0]=this->output_channels;
    this->out_shape[1]=this->in_shape[1]/this->stride;
    this->out_shape[2]=this->in_shape[2]/this->stride;

}

void conv_2d::data_update(data_t * weights,int32_t * bias)
{
    this->weights=weights;
    this->bias = bias;

}


void conv_2d::forward(data_t *x ,data_t *out)
{

//multi_loop
// multi(data_t In[Pif][S*Pr+K-S][S*Pc+K-S],data_t W[Pof][Pif][K][K],data_t Out[Pof][Pr][Pc],data_t W_bn[Pof][Pr][Pc])
    int out_0_count=0;
    int full_count=0;
	//Kernel_Row:
	for(int kr=0;kr<ksize;kr++)
	{
		//Kernel_Col:
		for(int kc=0;kc<ksize;kc++)
		{
			//Row:
			for(int r=0;r<out_shape[1];r++)
			{
				//Column:
				for(int c=0;c<out_shape[2];c++)
				{
					//Out_channel:
					for(int o=0;o<output_channels;o++)
					{
                        int out_offset= (o*out_shape[1]+r)*out_shape[2]+c  ;//((out_shape[1]*c)+r)*out_shape[2]+o;
						//in_channel:
						for(int in=0;in<in_channels;in++)
						{
                            //out[o][r][c]+=x[in][stride*r+kr][stride*c+kc]*this->weights[o][in][kr][kc]+this->bias[o]
                            int x_offset=(in*(in_shape[1]+1)+stride*r+kr)*(in_shape[1]+1)+stride*c+kc;//((stride*c+kc)*(this->in_shape[2])+stride*r+kr)*(this->in_shape[1])+in;
                            int w_offset=((o*in_channels+in)*ksize+kr)*ksize+kc;//((kc*ksize+kr)*ksize+in)*in_channels+o;
                     
                            *(out+out_offset)+= (* (x +x_offset))* (*(weights+w_offset));
                            if( out_offset ==0)
                            {   
                                out_0_count++;
                                cout << "out[" << out_offset << "]="<< *(out+out_offset) <<endl;
                                cout << "x[" << x_offset << "]="<< (* (x +x_offset)) << ","<< in <<  ","<<stride*r+kr << ","<< stride*c+kc <<endl;
                                cout << "w[" << w_offset << "]="<< (*(weights+w_offset)) <<endl;
                            }
                            full_count++;

                            


                            
						}
                        if((kc==ksize-1)&&(kr==ksize-1))
                        {
                            *(out+out_offset) += bias[o];
                        }
						
					}

				}
			}
		}
	}
    cout << " out0_count:" << out_0_count <<endl;
    cout << " full_count:" << full_count <<endl;

}

void conv_2d::padding_forward(data_t *x ,int32_t *out)
{


	//Kernel_Row:
	for(int kr=0;kr<ksize;kr++)
	{
		//Kernel_Col:
		for(int kc=0;kc<ksize;kc++)
		{
			//Row:
			for(int r=0;r<out_shape[1];r++)
			{
				//Column:
				for(int c=0;c<out_shape[2];c++)
				{
					//Out_channel:
					for(int o=0;o<output_channels;o++)
					{
                        int out_offset= (o*out_shape[1]+r)*out_shape[2]+c  ;//((out_shape[1]*c)+r)*out_shape[2]+o;
						//in_channel:
						for(int in=0;in<in_channels;in++)
						{
                            if( ((stride*r+kr)==0)||((stride*c+kc)==0))
                            {

                            }
                            else{

                            //out[o][r][c]+=x[in][stride*r+kr][stride*c+kc]*this->weights[o][in][kr][kc]+this->bias[o]
                            int x_offset=(in*(in_shape[1]+1)+stride*r+kr-1)*(in_shape[1]+1)+stride*c+kc-1;//((stride*c+kc)*(this->in_shape[2])+stride*r+kr)*(this->in_shape[1])+in;
                            int w_offset=((o*in_channels+in)*ksize+kr)*ksize+kc;//((kc*ksize+kr)*ksize+in)*in_channels+o;
                     
                            *(out+out_offset)+= (* (x +x_offset))* (*(weights+w_offset));
                            }
                            
						}
                        if((kc==ksize-1)&&(kr==ksize-1))
                        {
                            *(out+out_offset) += bias[o];
                        }
						
					}

				}
			}
		}
	}

}


data_t*** padArray(const data_t* originalArray,int chi, int originalRows, int originalCols) {  
    int newRows = originalRows + 1;  
    int newCols = originalCols + 1;  
  
    // 分配指向指针数组的指针数组  
    data_t*** paddedArray = new data_t**[chi];  
    for (int k = 0; k < chi; ++k) {  
        // 分配指向行数组的指针数组  
        paddedArray[k] = new data_t*[newRows];  
        for (int i = 0; i < newRows; ++i) {  
            // 分配每行的数据数组  
            paddedArray[k][i] = new data_t[newCols];  
        }  
    }  
  
    // 初始化填充区域  
    /*for (int k = 0; k < chi; ++k) {  
        for (int i = 0; i < newRows; ++i) {  
            for (int j = 0; j < newCols; ++j) {  
                if (i == 0 || i == newRows - 1 || j == 0 || j == newCols - 1) {  
                    paddedArray[k][i][j] = 0; // 填充边界  
                } else {  
                    paddedArray[k][i][j] = data_t(); // 初始化内部元素，假设data_t有默认构造函数  
                }  
            }  
        }  
    }  */
    for (int k = 0; k < chi; ++k) {  
        for (int i = 0; i < newRows; ++i) {  
            for (int j = 0; j < newCols; ++j) {  
                if (i == 0 ||  j == 0 ) {  
                    paddedArray[k][i][j] = 0; // 填充边界  
                } else {  
                    paddedArray[k][i][j] = data_t(); // 初始化内部元素，假设data_t有默认构造函数  
                }  
            }  
        }  
    }
    // 复制原始数据到填充数组  
    int index = 0;  
    for (int k = 0; k < chi; ++k) {  
        for (int i = 1; i < newRows ; ++i) {  
            for (int j = 1; j < newCols ; ++j) {  
                paddedArray[k][i][j] = originalArray[index++];  
            }  
        }  
    }  
  
    return paddedArray;  
}

data_t * flatten_3d(data_t *** arr,int size2,int size1,int size0)
{
    data_t * arr_1d;
    arr_1d= new  data_t[size2*size1*size0];
    int index = 0;  
    for (int k = 0; k < size2; k++) {  
        for (int i = 0; i < size1 ; i++) {  
            for (int j = 0; j < size0 ; j++) {  
                arr_1d[index++] = arr[k][i][j];  
                std::cout << arr_1d[index - 1] << " "; 
            } 
            std::cout << std::endl;  
        }  
        std::cout << std::endl; 
    }  
    return arr_1d;
}
/*
int main()
{
    cout << "start!" <<endl;
    unsigned int in_shape_i[3]={3,4,4};

    conv_2d conv1_tst(&in_shape_i[0],5,3,2);

    data_t in [3][4][4]; 
    data_t w [5][3][3][3];
    data_t b [5];
    data_t Out [5][2][2];

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            for(int k=0;k<4;k++)
            {
                in[i][j][k]=1;
            }
        }
    }
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<3;j++)
        {
            for(int k=0;k<3;k++)
            {
                for(int l=0;l<3;l++)
                {
                    w[i][j][k][l]=1;
                }

            }
        }
    }
    
    for(int i=0;i<5;i++)
    {
        b[i]=1;
    }
    conv1_tst.data_update(&w[0][0][0][0],&b[0]);
    

    data_t*** paddedArray= padArray( &in[0][0][0],3,4, 4);
    data_t * in_padded_array=flatten_3d(paddedArray,3,5,5);

    conv1_tst.padding_forward(&in[0][0][0],&Out[0][0][0]);

    for (int i = 0; i < 4 + 1 ; ++i) {  

        for (int j = 0; j < 4 + 1 ; ++j) {  

            std::cout << paddedArray[0][i][j] << " ";  

        }  

        std::cout << std::endl;  

    }  

    for(int i=0;i<5;i++)
    {
        for(int j=0;j<2;j++)
        {
            for(int k=0;k<2;k++)
            {
                cout << "out[" << ((i*2+j)*2+k) << "]="<< Out[i][j][k] <<endl;
            }
        }
    }

    return 0;
    
}
*/
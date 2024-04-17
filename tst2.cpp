#include <iostream>  
#include <fstream>  
#include <vector>  
#include <string>  
#include "/home/derui/work/C_proj/yolo_v4_tiny_baseline/layer/base_layer.h"

using namespace std; 

//////////////////
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
                            int x_offset=(in*(in_shape[1])+stride*r+kr-1)*(in_shape[1])+stride*c+kc-1;//((stride*c+kc)*(this->in_shape[2])+stride*r+kr)*(this->in_shape[1])+in;
                            int w_offset=((o*in_channels+in)*ksize+kr)*ksize+kc; //cho,chi,kr,kc       
                            
                     
                            *(out+out_offset)+= (* (x +x_offset))* (*(weights+w_offset));
                            if(out_offset==0)
                            {
                                std::cout<<"x "<< in << " "<< (stride*r+kr-1)<<" "<< (stride*c+kc-1) << "="<< static_cast<int>(* (x +x_offset))<<"(x_offset:"<<x_offset<<")";
                                std::cout<<"w "<< o << " "<< in << " "<< kr << " "<< kc << "="<<static_cast<int>(*(weights+w_offset))<<"(w_offset:"<<w_offset<<")";
                                std::cout<<"out[0]="<<*(out+0);
                                std::cout<<endl;
                            }
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


///////////////////////





const string syn_data_dir = "/home/derui/work/Py_proj/yolo_tiny_v4_baseline/hw_tst/conv1_relu0125/int8_m16/";
const string data_dir = "data_tst/conv1_again/int8_m15/";

//void load_int_data(int8_t *buffer , dir);
int load_int8_data(int8_t *buffer , const string dir,streamsize size);
int load_int32_data(int32_t *buffer , const string dir,streamsize size);
int save_int32_data(int32_t *buffer ,const string dir ,int size);
void print_out_tensor(int32_t * out ,int cho,int width);
void tensor_init(int32_t * tensor,int size );
  
int main() {  
    
    int8_t in[3*416*416]; 
    int32_t bias[32];
    int8_t weights[32*3*3*3];
    int32_t Out[32*208*208];
    tensor_init(Out,32*208*208);

    std::cout<<"w_size:"<<sizeof(weights)<<endl;
    load_int8_data(in ,  syn_data_dir +"in_q.bin", sizeof(in)); 
    load_int8_data(weights ,  syn_data_dir +"w_q.bin" , sizeof(weights)); 
    

    for (int i =0;i<32;i++)
    {
        for (int j=0;j<3;j++)
        {
            for (int k=0;k<3;k++)
            {
                for (int l=0;l<3;l++)
                {
                    std::cout<<static_cast<int>(weights[((i*3+j)*3+k)*3+l])<<" ";
                }
                std::cout<<endl;
            }
        std::cout<<endl;
        }
    std::cout<<endl;
    }

    load_int32_data(bias ,  syn_data_dir +"b_q.bin" , sizeof(bias));   

    cout << "start!" <<endl;
    unsigned int in_shape_i[3]={3,416,416};

    conv_2d conv1_tst(&in_shape_i[0],32,3,2);
    conv1_tst.data_update(&weights[0],&bias[0]);

    conv1_tst.padding_forward(&in[0],&Out[0]);
    
    
    save_int32_data(Out ,syn_data_dir +"out_q.bin" ,32*208*208);
      
    return 0;  
}

void tensor_init(int32_t * tensor,int size )
{
    for (int i=0;i<size;i++)
    {
        tensor[i]=0;
    }
}

void print_out_tensor(int32_t * out ,int cho,int width)
{
    for (int i =0;i<cho;i++)
    {
        for (int j=0;j<width;j++)
        {
            for (int k=0;k<width;k++)
            {
                std::cout<<out[(i*width+j)*width+k]<<" ";
            }
        std::cout<<endl;
        }
    std::cout<<endl;
    }

}


int load_int8_data(int8_t *buffer , const string dir,streamsize size)
{
    std::ifstream file(dir, std::ios::binary | std::ios::in);  
      
    if (!file) {  
        std::cerr << "无法打开文件" << std::endl;  
        return 1;  
    }  

      
    if (!file.read(reinterpret_cast<char*>(buffer), size)) {  

        std::cerr << "读取文件时出错" << std::endl;  
        return 1;  
    } 
    // 关闭文件  
    file.close(); 
    return 0;
}

int load_int32_data(int32_t *buffer , const string dir,streamsize size)
{
    std::ifstream file(dir, std::ios::binary | std::ios::in);  
      
    if (!file) {  
        std::cerr << "无法打开文件" << std::endl;  
        return 1;  
    }  

      
    if (!file.read(reinterpret_cast<char*>(buffer), size)) {  

        std::cerr << "读取文件时出错" << std::endl;  
        return 1;  
    } 
    // 关闭文件  
    file.close(); 
    return 0;
}

int save_int32_data(int32_t *buffer ,const string dir ,int size)
{
    // 打开一个二进制文件进行写入  
    std::ofstream binFile(dir, std::ios::out | std::ios::binary);  
  
    // 检查文件是否成功打开  
    if (!binFile) {  
        std::cerr << "无法打开文件\n";  
        return 1;  
    }  
  
    // 将数组写入文件  
    for (int i = 0; i < size; ++i) {  
        binFile.write(reinterpret_cast<const char*>(&buffer[i]), sizeof(buffer[i]));  
    } 
  
    // 关闭文件  
    binFile.close();  
  
    std::cout << "数组已成功保存到二进制文件。\n"; 
    return 0; 
}
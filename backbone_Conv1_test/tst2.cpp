#include <iostream>  
#include <fstream>  
#include <vector>  
#include <string>  

#include "/home/derui/work/C_proj/yolo_v4_tiny_baseline/backbone_Conv1_test/base_layer.h"

using namespace std; 




const string syn_data_dir = "/home/derui/work/Py_proj/yolo_tiny_v4_baseline/hw_tst/conv1_relu0125/int8_m16/";
const string data_dir = "data_tst/conv1_again/int8_m15/";

//void load_int_data(int8_t *buffer , dir);
int load_int8_data(int8_t *buffer , const string dir,streamsize size);
int load_int16_data(int16_t *buffer , const string dir,streamsize size);
int load_int32_data(int32_t *buffer , const string dir,streamsize size);
int save_int32_data(int32_t *buffer ,const string dir ,int size);
int save_int8_data(int8_t *buffer ,const string dir ,int size);
void print_out_tensor(int32_t * out ,int cho,int width);
void tensor_int32_init(int32_t * tensor,int size );
void tensor_int8_init(int8_t * tensor,int size );
void tensor2int8(int32_t * in,int8_t * out,int size0,int size1,int size2);



//////////////////
void conv_2d::init(unsigned int in_shape_i[3],unsigned int output_channels_i,unsigned int ksize_i,unsigned int stride_i )
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

void conv_2d::data_update(data_t * weights,int32_t * bias , int16_t m0)
{
    this->weights=weights;
    this->bias = bias;
    this-> m0=m0;

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
                            if(out_offset==0)
                            {
                                std::cout<<"out["<< out_offset <<"]="<<*(out+out_offset);
                            }
                            *(out+out_offset) += bias[o];
                            if(out_offset==0)
                                std::cout<<"out["<< out_offset <<"]="<<*(out+out_offset);
                            *(out+out_offset) = ((*(out+out_offset)) * this->m0 )>>15;
                            if(out_offset==0)
                            {
                                std::cout<<"out["<< out_offset <<"]="<<*(out+out_offset);
                                std::cout<<endl;
                            }
                        }
						
					}

				}
			}
		}
	}

}

//////////////////////////////
void leakyrelu::init(unsigned int in_shape_i[3]) 
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape_i[i];
    }
    

}
void leakyrelu::S_update(unsigned short int S_relu,unsigned int S_bitshift) 
{
    this->S_relu=S_relu;
    this->S_bitshift=S_bitshift;
}



void leakyrelu::forward(data_t *x,int32_t *out)
{
	//Row:
	for(int r=0;r<in_shape[1];r++)
	{
		//Column:
		for(int c=0;c<in_shape[2];c++)
		{
		    //Out_channel:
			for(int o=0;o<in_shape[0];o++)
			{
                int x_offset=(o*in_shape[1]+r)*in_shape[2]+c;
                if(* (x +x_offset)>=0)
                {
                    *(out+x_offset) = (* (x +x_offset)) * S_relu ;
                    *(out+x_offset) = *(out+x_offset) >> (S_bitshift-1);
                }
                else
                {
                    *(out+x_offset) = (* (x +x_offset)) * S_relu ;
                    *(out+x_offset) = *(out+x_offset) >> (S_bitshift-1+3);
                }

			}
		}
	}
}





void maxpooling::init(unsigned int in_shape[3],unsigned int ksize,unsigned int stride )
{
    for(int i=0;i<3;i++)
    {
        this->in_shape[i]=in_shape[i];
    }
    this->out_shape[0]=this->in_shape[0];
    this->ksize=ksize;
    this->stride=stride;
    this->out_shape[1]=this->in_shape[1]/this->stride;
    this->out_shape[2]=this->in_shape[2]/this->stride;

}

void maxpooling::forward(data_t *x,data_t *out)
{
	//Row:
	for(int r=0;r<out_shape[1];r++)
	{
		//Column:
		for(int c=0;c<out_shape[2];c++)
		{
		    //Out_channel:
			for(int o=0;o<out_shape[0];o++)
			{
                int x_offset0=(o*in_shape[1]+2*r)*in_shape[2]+2*c;
                int x_offset1=(o*in_shape[1]+2*r)*in_shape[2]+2*c+1;
                int x_offset2=(o*in_shape[1]+2*r+1)*in_shape[2]+2*c;
                int x_offset3=(o*in_shape[1]+2*r+1)*in_shape[2]+2*c+1;

                int out_offset=(o*out_shape[1]+r)*out_shape[2]+c;
              
                *(out+out_offset) = maxpooling::find_max(*(x+x_offset0) ,*(x+x_offset1),*(x+x_offset2),*(x+x_offset3)) ;
			}
		}
	}
}

data_t maxpooling::find_max(data_t x0,data_t x1,data_t x2,data_t x3)
{
    data_t max=x0;
    if(max<x1)
        max=x1;
    else
        max=max;
    if(max<x2)
        max=x2;
    else
        max=max;
    if(max<x3)
        max=x3;
    else
        max=max;  
    return max;  
}
////////////////////////////////

void basic_conv::init(unsigned int in_shape[3],unsigned int output_channels,unsigned int ksize,unsigned int stride)
{
    this->conv.init(in_shape ,output_channels ,ksize ,stride);
    this->activation.init(this->conv.out_shape);
    
}

void basic_conv::data_update(const string file_dir)
{
    //int32_t *bias;
    conv.bias = new int32_t[this->conv.output_channels];
    //int8_t *weights;

    conv.weights = new int8_t[conv.output_channels*conv.in_channels*conv.ksize*conv.ksize]; 
    
    int8_t m0_temp[1];
    int16_t m0 [1];
    int16_t s0 [1];

    load_int8_data(conv.weights ,  file_dir +"w_q.bin" , conv.output_channels*conv.in_channels*conv.ksize*conv.ksize); 
    load_int8_data(m0_temp , file_dir +"m_0.bin" , 1);



    * m0 = static_cast<int16_t> (* m0_temp);
    cout<< * m0 <<endl;
    cout<<conv.output_channels<<endl;
    load_int32_data(conv.bias ,  file_dir +"b_q.bin" , streamsize (4*conv.output_channels)); 

    for (int i=0;i<conv.output_channels;i++)
    {
        cout<<conv.bias[i]<<" ";
        cout<<endl;
    }

    conv.data_update(&conv.weights[0],&conv.bias[0],* m0);

    load_int16_data(s0 , file_dir +"S_relu_q.bin" , 2);
    activation.S_update(* s0,15);

}

void basic_conv::forward(data_t *x,data_t *out)
{
    int32_t *buffer1;
    buffer1 = new int32_t[conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]];
    tensor_int32_init(buffer1,conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]);

    conv.padding_forward(x,buffer1);

    save_int32_data(buffer1, syn_data_dir + "conv_Out_q32.bin" , 32*208*208);
    delete [] conv.bias,conv.weights,x;

    int8_t *conv_out;
    conv_out = new int8_t[conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]];
    tensor2int8(buffer1,conv_out,conv.out_shape[0],conv.out_shape[1],conv.out_shape[2]);
    save_int8_data(conv_out, syn_data_dir + "conv_Out_q8.bin" , 32*208*208);
    

    delete [] buffer1;

    int32_t *buffer2;
    buffer2 = new int32_t[conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]];
    
    tensor_int32_init(buffer2,conv.out_shape[0]*conv.out_shape[1]*conv.out_shape[2]);

    activation.forward(conv_out,buffer2);
    delete [] conv_out;

    tensor2int8(buffer2,out,conv.out_shape[0],conv.out_shape[1],conv.out_shape[2]);
    delete [] buffer2;

}



//////////////////////////////////


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

void tensor2int8(int32_t * in,int8_t * out,int size0,int size1,int size2)
{
    for (int i=0;i<size0;i++)
        for ( int j = 0;j<size1;j++)
            for ( int k = 0; k < size2 ; k ++ )
                {
                    //in[i][j][k]
                    if ( in[(i*size1+j)*size2+k] >127)
                        out[(i*size1+j)*size2+k] = 127 ;
                    else if ( in[(i*size1+j)*size2+k] <-128 )
                        out[(i*size1+j)*size2+k] = -128 ;
                    else 
                        out[(i*size1+j)*size2+k] = static_cast<int8_t>(in[(i*size1+j)*size2+k]);

                }
}

///////////////////////






  
int main() {  
    
    int8_t *in;
    in=new int8_t[3*416*416]; 
    load_int8_data(in ,  syn_data_dir +"in_q.bin", 3*416*416); 
    cout << "start!" <<endl;
    unsigned int in_shape_i[3]={3,416,416};
    basic_conv conv1_tst;
    conv1_tst.init(&in_shape_i[0],32,3,2);
    conv1_tst.data_update(syn_data_dir);
    int8_t *Out;
    Out = new int8_t[32*208*208];
    conv1_tst.forward(&in[0],Out);
    
    save_int8_data(Out, syn_data_dir + "acc_Out_q.bin" , 32*208*208);
    delete [] Out;
    
      
    return 0;  
}

void tensor_int32_init(int32_t * tensor,int size )
{
    for (int i=0;i<size;i++)
    {
        tensor[i]=0;
    }
}
void tensor_int8_init(int8_t * tensor,int size )
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

int load_int16_data(int16_t *buffer , const string dir,streamsize size)
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

int save_int8_data(int8_t *buffer ,const string dir ,int size)
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


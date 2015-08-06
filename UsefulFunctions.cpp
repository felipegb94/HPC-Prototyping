void printDeviceInfo()
{
    int device;
    struct cudaDeviceProp props;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    std::cout << "Device info: " <<std::endl;
    std::cout << "Name: " << props.name <<std::endl;
    std::cout << "version: " << props.major << "," <<  props.minor <<std::endl;
}

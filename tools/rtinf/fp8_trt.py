import tensorrt as trt

def build_engine(onnx_file_path, fp16=False, fp8_layers=None):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 指定FP16精度
    if fp16:
        builder.fp16_mode = True
    
    # 指定FP8的层
    if fp8_layers:
        for layer_name in fp8_layers:
            layer = network.get_layer(network.get_layer_index(layer_name))
            layer.precision = trt.DataType.FLOAT8
    
    engine = builder.build_cuda_engine(network)
    return engine
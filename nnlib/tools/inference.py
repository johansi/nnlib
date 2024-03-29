import os
import numpy  as np
from .image_tools import *
import pdb
import time
import PIL
import cv2
from .helper import *
try:
    import pycuda.autoinit
    import tensorrt as trt
    import pycuda.driver as cuda
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except:    
    printing("UNALBE TO IMPORT TENSORRT", print_types.WARNING)
    
    
try:
    import torch
    from torchvision import transforms
except:
    printing("UNALBE TO IMPORT PYTORCH", print_types.WARNING)
  
    
def predict_classification(runtime, image, size, class_types):
    times= {}
    if type(image) == PIL.Image.Image:
        image = np.asarray(image)    
    t_resize = time.time()    
    image = cv2.resize(image, (size, size))
    times["resize"] = round((time.time() - t_resize)*1000)

    t_prediction = time.time()
    prediction = runtime.inference(image)
    times["prediction"] = round((time.time() - t_prediction)*1000)

    #t_img_points = time.time()
    #img_points = get_image_points(prediction, heatmap_types)
    #times["classification"] = round((time.time() - t_img_points)*1000)

    return image, prediction, times
    
    
def predict_heatmap_image(runtime, image, size, heatmap_types, enhance=False):#dist_image,y_pos_dist
    times = {}
    if type(image) == PIL.Image.Image:
        image = np.asarray(image)    
    t_resize = time.time()    
    image = cv2.resize(image, (size, size))
    times["resize"] = round((time.time() - t_resize)*1000)
    if enhance:
        t_enhance = time.time() 
        image = image_enhance(image)
        times["enhance"] = round((time.time() - t_enhance)*1000)

    t_prediction = time.time()
    prediction = np.squeeze(runtime.inference(image))
    times["prediction"] = round((time.time() - t_prediction)*1000)

    t_img_points = time.time()
    img_points = get_image_points(prediction, heatmap_types)
    times["img_points"] = round((time.time() - t_img_points)*1000)

    return image, prediction, img_points, times

class PYTORCH_CONTEXT:
    def __init__(self, model, state_dict_file, norm_stats_mean=None, norm_stats_std=None, batch_norm=False, device="cpu", model_key=None, convert_out_to_np=False):
        self.device = device
        self.model = model.to(self.device)
        checkpoint = torch.load(state_dict_file, map_location=torch.device(self.device))
        checkpoint = checkpoint if model_key is None else checkpoint["model"]
        self.model.load_state_dict(checkpoint)
        self.model = self.model.eval()
        self.batch_norm = batch_norm
        self.convert_out_to_np = convert_out_to_np
        if batch_norm:
            self.normalize = transforms.Normalize(norm_stats_mean,norm_stats_std)
            self.to_tensor = transforms.ToTensor() 
    
    def inference(self, inputs):
        inputs = [inputs] if type(inputs) != list else inputs
                        
        for idx in range(len(inputs)):
            if (type(inputs[idx]) == np.ndarray) and self.batch_norm:
                inputs[idx] = PIL.Image.fromarray(inputs[idx])
            
        if self.batch_norm:
            for idx in range(len(inputs)):
                inputs[idx] = self.normalize(self.to_tensor(inputs[idx]))[None].to(self.device)
        else:
            for idx in range(len(inputs)): 
                inputs[idx] = inputs[idx].to(self.device)

        with torch.no_grad():           
            out = self.model(*inputs)
            if self.convert_out_to_np:
                return np.array(out.cpu())
            else:
                return out
        
class TENSOR_RT_CONTEXT:
    def __init__(self, onnxfile, input_names, output_names, norm_stats_mean=None, norm_stats_std=None, 
                 fp16=True, max_workspace_size=3<<28, batch_norm=True):        
        #print("CREATE TENSOR_RT_CONTEXT")    
        trt.init_libnvinfer_plugins(None, "")    
        self._fp16 = fp16
        self.norm_stats_mean = norm_stats_mean
        self.norm_stats_std = norm_stats_std
        self._max_workspace_size = max_workspace_size
        self.input_names = input_names
        self.output_names = output_names
        self.engine_path = onnxfile.parent/(onnxfile.stem+".engine")        
        self.batch_norm = batch_norm        
        if os.path.exists(self.engine_path):
            #print("ENINGE EXISTS")
            self.load_tensorrt_engine()
        else:
            if trt.__version__[0] == "8":                
                self.onnx_to_tensorrt_8(onnxfile)
            else:
                self.onnx_to_tensorrt_7(onnxfile)
            
        self.create_execution_context()

    def create_execution_context(self):
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []          
        for name in self.input_names:
            idx = self.engine.get_binding_index(name)
            host_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(idx)), dtype=np.float32)
            device_input = cuda.mem_alloc(host_input.nbytes)            
            self.bindings.append(int(device_input))
            self.host_inputs.append(host_input)
            self.device_inputs.append(device_input)
                                
        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            host_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(idx)), dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)            
            self.bindings.append(int(device_output))
            self.host_outputs.append(host_output)
            self.device_outputs.append(device_output)
                        
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()
        # create execution context
        self.context = self.engine.create_execution_context()                

    def inference(self, inputs):        
        
        inputs = [inputs] if type(inputs) != list else inputs        
        for idx in range(len(inputs)):
            
            if self.batch_norm:
                inputs[idx] = batch_and_normalize(inputs[idx], mean=self.norm_stats_mean, std=self.norm_stats_std)
            np.copyto(self.host_inputs[idx], inputs[idx].ravel()) 
            cuda.memcpy_htod_async(self.device_inputs[idx], self.host_inputs[idx], self.stream)            
       
        self.context.execute_async(batch_size= 1, bindings=self.bindings, stream_handle=self.stream.handle)
        for idx in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[idx], self.device_outputs[idx], self.stream)
        self.stream.synchronize()
        outs = []
        for idx in range(len(self.output_names)):
            binding_idx = self.engine.get_binding_index(self.output_names[idx])
            outs.append(np.reshape(self.host_outputs[idx],self.engine.get_binding_shape(binding_idx)))
        return tuple(outs) if len(outs) > 1 else outs

    def load_tensorrt_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def onnx_to_tensorrt_8(self,onnx_file):

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        config = builder.create_builder_config()
        success = parser.parse_from_file(str(onnx_file))
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        config = builder.create_builder_config()
        if self._fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        builder.max_batch_size = 1
        serialized_engine = builder.build_serialized_network(network, config)
        
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
            
        with trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    def onnx_to_tensorrt_7(self,onnx_file):        

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network,TRT_LOGGER)
        model = open(onnx_file, 'rb')
        parser.parse(model.read())
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        builder.max_batch_size = 1
        builder.fp16_mode = self._fp16
        builder.max_workspace_size = self._max_workspace_size
        self.engine = builder.build_cuda_engine(network)
        with open(self.engine_path, "wb") as f:
            f.write(self.engine.serialize())



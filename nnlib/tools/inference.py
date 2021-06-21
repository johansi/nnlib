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
    
    
def predict_heatmap_image(runtime, image, size, heatmap_types, normalize_stats, enhance=False):#dist_image,y_pos_dist
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
    prediction = np.squeeze(runtime.inference(batch))
    times["prediction"] = round((time.time() - t_prediction)*1000)

    t_img_points = time.time()
    img_points = get_image_points(prediction, heatmap_types)
    times["img_points"] = round((time.time() - t_img_points)*1000)

    return image, prediction, img_points, times

class PYTORCH_CONTEXT:
    def __init__(self, model, state_dict_file, norm_stats_mean, norm_stats_std, device="cpu"):
        self.model = model
        self.model.load_state_dict(torch.load(state_dict_file, map_location=torch.device(device))["model"])
        self.model = self.model.eval()
        self.normalize = transforms.Normalize(norm_stats_mean,norm_stats_std)
        self.to_tensor = transforms.ToTensor() 
    
    def inference(self, inp):
                
        if type(inp) == np.ndarray:
            inp = PIL.Image.fromarray(inp)
            
        net_inp = self.normalize(self.to_tensor(inp))[None]
        with torch.no_grad():
            out = self.model(net_inp)
        return np.array(out)
        
class TENSOR_RT_CONTEXT:
    def __init__(self, onnxfile,norm_stats_mean, norm_stats_std, fp16=True, max_workspace_size=3<<28):        
        self._fp16 = fp16
        self.norm_stats_mean = norm_stats_mean
        self.norm_stats_std = norm_stats_std
        self._max_workspace_size = max_workspace_size
        self.engine_path = onnxfile.parent/(onnxfile.stem+".engine")
        if os.path.exists(self.engine_path):
            self.load_tensorrt_engine()
        else:
            self.onnx_to_tensorrt(onnxfile)

        self.create_execution_context()

    def create_execution_context(self):        
        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
        
        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()
        # create execution context
        self.context = self.engine.create_execution_context()
        self.bindings = [int(self.d_input), int(self.d_output)]

    def inference(self, input_data):
        input_data = batch_and_normalize(input_data, mean=self.norm_stats_mean, std=self.norm_stats_std)
        np.copyto(self.h_input, input_data.ravel())
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async(batch_size= 1, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return np.reshape(self.h_output,self.engine.get_binding_shape(1))

    def load_tensorrt_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def onnx_to_tensorrt(self,onnx_file):

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


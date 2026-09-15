#!/usr/bin/env python3
"""
Single-Threaded Optimized Webcam Inference Server
Avoids CUDA context issues by handling clients sequentially.

Usage:
    python optimized_server.py --engine model.engine --port 65432
"""

import os
import sys
import argparse
import time
import socket
import struct
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import traceback

class TensorRTEngine:
    """TensorRT engine wrapper - single threaded"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        
        # Memory management
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        self._load_engine()
        self._allocate_memory()
    
    def _load_engine(self):
        """Load TensorRT engine"""
        print(f"🔧 Loading TensorRT engine: {self.engine_path}")
        
        self.runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, 'rb') as f:
            engine_bytes = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if not self.engine:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context")
        
        self.stream = cuda.Stream()
        print(f"✅ Engine loaded successfully")
    
    def _allocate_memory(self):
        """Pre-allocate host and device memory"""
        print(f"🧠 Allocating memory buffers...")
        
        # Check TensorRT version
        use_new_api = hasattr(self.engine, 'num_io_tensors')
        
        if use_new_api:
            # TensorRT 10.x+ API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(tensor_name)
                dtype = self.engine.get_tensor_dtype(tensor_name)
                is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
                
                # Handle dynamic shapes - set to batch size 1
                if -1 in shape:
                    actual_shape = tuple(1 if dim == -1 else dim for dim in shape)
                    self.context.set_input_shape(tensor_name, actual_shape)
                    shape = actual_shape
                
                # Convert dtype and calculate size
                if dtype == trt.DataType.FLOAT:
                    np_dtype, element_size = np.float32, 4
                elif dtype == trt.DataType.HALF:
                    np_dtype, element_size = np.float16, 2
                elif dtype == trt.DataType.INT32:
                    np_dtype, element_size = np.int32, 4
                elif dtype == trt.DataType.INT64:
                    np_dtype, element_size = np.int64, 8
                else:
                    np_dtype, element_size = np.float32, 4
                
                size = trt.volume(shape)
                
                # Allocate memory
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(size * element_size)
                
                if is_input:
                    self.host_inputs.append(host_mem)
                    self.device_inputs.append(device_mem)
                    self.input_shapes[tensor_name] = shape
                else:
                    self.host_outputs.append(host_mem)
                    self.device_outputs.append(device_mem)
                    self.output_shapes[tensor_name] = shape
                
                # Set tensor address
                self.context.set_tensor_address(tensor_name, int(device_mem))
                
                print(f"   {'📥' if is_input else '📤'} {tensor_name}: {shape} ({size * element_size / 1024 / 1024:.1f} MB)")
        
        else:
            # Legacy TensorRT API
            self.bindings = []
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
                
                if -1 in shape:
                    actual_shape = tuple(1 if dim == -1 else dim for dim in shape)
                    self.context.set_binding_shape(i, actual_shape)
                    shape = actual_shape
                
                if dtype == trt.DataType.FLOAT:
                    np_dtype, element_size = np.float32, 4
                elif dtype == trt.DataType.INT64:
                    np_dtype, element_size = np.int64, 8
                else:
                    np_dtype, element_size = np.float32, 4
                
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, np_dtype)
                device_mem = cuda.mem_alloc(size * element_size)
                
                if is_input:
                    self.host_inputs.append(host_mem)
                    self.device_inputs.append(device_mem)
                    self.input_shapes[binding_name] = shape
                else:
                    self.host_outputs.append(host_mem)
                    self.device_outputs.append(device_mem)
                    self.output_shapes[binding_name] = shape
                
                self.bindings.append(int(device_mem))
                
                print(f"   {'📥' if is_input else '📤'} {binding_name}: {shape} ({size * element_size / 1024 / 1024:.1f} MB)")
    
    def preprocess_frame(self, frame: np.ndarray) -> tuple:
        """Fast preprocessing"""
        # Resize frame
        resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB and normalize (simplified)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (keep this if your model needs it)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        # Target size tensor
        target_size_tensor = np.array([[640, 640]], dtype=np.int64)
        
        return input_tensor, target_size_tensor
    
    def infer(self, inputs: list) -> list:
        """Perform inference"""
        
        # Copy inputs to device
        for i, input_data in enumerate(inputs):
            np.copyto(self.host_inputs[i][:input_data.size], input_data.ravel())
            cuda.memcpy_htod_async(self.device_inputs[i], self.host_inputs[i], self.stream)
        
        # Execute inference
        if hasattr(self.context, 'execute_async_v3'):
            success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        elif hasattr(self.context, 'execute_async_v2'):
            success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            success = self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Copy outputs from device
        outputs = []
        for i, (host_output, device_output, shape) in enumerate(
            zip(self.host_outputs, self.device_outputs, self.output_shapes.values())
        ):
            cuda.memcpy_dtoh_async(host_output, device_output, self.stream)
            self.stream.synchronize()
            output = host_output[:np.prod(shape)].reshape(shape).copy()
            outputs.append(output)
        
        return outputs

def create_segmentation_overlay(frame: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Create segmentation overlay on frame"""
    
    # Pascal Person Parts color map (7 classes)
    colors = [
        [0, 0, 0],         # 0: background - black
        [128, 0, 0],       # 1: head - dark red
        [255, 0, 0],       # 2: torso - red  
        [0, 128, 0],       # 3: upper arms - dark green
        [0, 255, 0],       # 4: lower arms - green
        [0, 0, 128],       # 5: upper legs - dark blue
        [0, 0, 255],       # 6: lower legs - blue
    ]
    
    height, width = frame.shape[:2]
    
    # Resize segmentation mask to frame size
    seg_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask = (seg_resized == class_id)
        overlay[mask] = color
    
    # Blend with original frame
    result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
    
    return result

def draw_detections(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, 
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """Draw detection boxes on frame"""
    
    height, width = frame.shape[:2]
    result = frame.copy()
    
    # COCO class names (simplified)
    class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light'
    }
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
              (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0)]
    
    for i in range(len(scores)):
        if scores[i] < confidence_threshold:
            continue
            
        # Convert normalized coordinates to pixel coordinates
        x1 = int(boxes[i][0] * width)
        y1 = int(boxes[i][1] * height)
        x2 = int(boxes[i][2] * width)
        y2 = int(boxes[i][3] * height)
        
        label = int(labels[i])
        color = colors[label % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        class_name = class_names.get(label, f'Class {label}')
        label_text = f'{class_name}: {scores[i]:.2f}'
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(result, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result

class SingleThreadedWebcamServer:
    """Single-threaded webcam inference server (avoids CUDA context issues)"""
    
    def __init__(self, engine_path: str, port: int = 65432):
        self.engine_path = engine_path
        self.port = port
        
        # Server components
        self.server_socket = None
        self.running = False
        
        # Inference engine (single instance, single thread)
        self.engine = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.perf_counter()
        
    def initialize_engine(self):
        """Initialize the inference engine"""
        print("🚀 Initializing TensorRT engine (single-threaded)...")
        try:
            # Initialize CUDA context in main thread
            cuda.init()
            
            self.engine = TensorRTEngine(self.engine_path)
            
            # Warmup
            print("🔥 Warming up...")
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(3):
                input_tensor, target_size_tensor = self.engine.preprocess_frame(dummy_frame)
                _ = self.engine.infer([input_tensor, target_size_tensor])
            
            print("✅ Engine ready!")
            return True
            
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def handle_client_session(self, client_socket, client_address):
        """Handle a complete client session"""
        print(f"🔗 Client connected: {client_address}")
        
        try:
            # Set socket timeouts to avoid hanging
            client_socket.settimeout(30.0)
            
            while self.running:
                try:
                    # Receive frame size
                    size_data = self._recv_exact(client_socket, 4)
                    if not size_data:
                        print("❌ Client disconnected (no size data)")
                        break
                    
                    frame_size = struct.unpack("!I", size_data)[0]
                    
                    # Sanity check
                    if frame_size <= 0 or frame_size > 2 * 1024 * 1024:  # Max 2MB
                        print(f"⚠️ Invalid frame size: {frame_size}")
                        continue
                    
                    # Receive compressed frame
                    compressed_data = self._recv_exact(client_socket, frame_size)
                    if not compressed_data:
                        print("❌ Client disconnected (no frame data)")
                        break
                    
                    # Decompress frame
                    frame_array = np.frombuffer(compressed_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("⚠️ Failed to decode frame")
                        # Send back error frame
                        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(error_frame, "Decode Error", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self._send_frame(client_socket, error_frame)
                        continue
                    
                except socket.timeout:
                    print("⚠️ Client timeout - closing connection")
                    break
                except Exception as e:
                    print(f"❌ Frame receive error: {e}")
                    break
                
                # Process frame (single-threaded, no CUDA context issues)
                try:
                    process_start = time.perf_counter()
                    
                    # Preprocess and infer
                    input_tensor, target_size_tensor = self.engine.preprocess_frame(frame)
                    outputs = self.engine.infer([input_tensor, target_size_tensor])
                    
                    # Extract outputs and create visualization
                    if len(outputs) >= 5:
                        labels = outputs[0].flatten()      # Detection labels
                        boxes = outputs[1][0]              # Detection boxes
                        scores = outputs[2].flatten()      # Detection scores  
                        seg_probs = outputs[3][0]          # Segmentation probabilities
                        seg_preds = outputs[4][0]          # Segmentation predictions
                        
                        # Create processed frame
                        result_frame = frame.copy()
                        
                        # Add segmentation overlay
                        result_frame = create_segmentation_overlay(result_frame, seg_preds.astype(np.uint8), alpha=0.6)
                        
                        # Add detection boxes
                        result_frame = draw_detections(result_frame, boxes, scores, labels, confidence_threshold=0.5)
                        
                    else:
                        print("⚠️ Unexpected number of outputs, using original frame")
                        result_frame = frame.copy()
                    
                    # Add performance info
                    process_time = time.perf_counter() - process_start
                    self.frame_count += 1
                    
                    elapsed = time.perf_counter() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    
                    cv2.putText(result_frame, f"Server FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Process: {process_time*1000:.1f}ms", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Send result back to client
                    if not self._send_frame(client_socket, result_frame):
                        print("❌ Failed to send result")
                        break
                    
                except Exception as e:
                    print(f"❌ Processing error: {e}")
                    traceback.print_exc()
                    
                    # Send original frame back as fallback
                    try:
                        error_frame = frame.copy()
                        cv2.putText(error_frame, f"Process Error: {str(e)[:30]}", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self._send_frame(client_socket, error_frame)
                    except:
                        break
                
        except Exception as e:
            print(f"❌ Client session error: {e}")
            traceback.print_exc()
        finally:
            try:
                client_socket.close()
            except:
                pass
            print(f"👋 Client disconnected: {client_address}")
    
    def _recv_exact(self, sock, size: int) -> bytes:
        """Receive exactly 'size' bytes"""
        data = b''
        while len(data) < size:
            try:
                chunk = sock.recv(size - len(data))
                if not chunk:
                    return b''
                data += chunk
            except Exception as e:
                return b''
        return data
    
    def _send_frame(self, sock, frame: np.ndarray) -> bool:
        """Send frame as JPEG to client"""
        try:
            # Compress frame
            _, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            # Send size and data
            size_data = struct.pack("!I", len(compressed))
            sock.sendall(size_data)
            sock.sendall(compressed.tobytes())
            return True
            
        except Exception as e:
            print(f"❌ Send error: {e}")
            return False
    
    def start_server(self):
        """Start the server"""
        try:
            print(f"🚀 Starting single-threaded server on port {self.port}...")
            
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(1)  # Only accept 1 client at a time
            
            self.running = True
            self.start_time = time.perf_counter()
            
            print(f"✅ Server listening on port {self.port}")
            print("📝 Note: Server handles one client at a time (avoids CUDA issues)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def run(self):
        """Main server loop"""
        print("🏃 Server running (single-threaded)... Press Ctrl+C to stop")
        
        try:
            while self.running:
                try:
                    print("⏳ Waiting for client connection...")
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Handle this client completely before accepting another
                    self.handle_client_session(client_socket, client_address)
                    
                except Exception as e:
                    if self.running:
                        print(f"❌ Accept error: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Server shutdown requested")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server"""
        print("🛑 Stopping server...")
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("✅ Server stopped")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single-Threaded Optimized Webcam Inference Server')
    parser.add_argument('-e', '--engine', type=str, required=True,
                        help='Path to TensorRT engine file')
    parser.add_argument('-p', '--port', type=int, default=65432,
                        help='Server port (default: 65432)')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    if not os.path.isfile(args.engine):
        print(f"❌ Engine file not found: {args.engine}")
        sys.exit(1)
    
    print("=" * 70)
    print("🚀 SINGLE-THREADED OPTIMIZED WEBCAM INFERENCE SERVER")
    print("=" * 70)
    print(f"📁 Engine: {args.engine}")
    print(f"🌐 Port: {args.port}")
    print("🎯 Focus: No CUDA context issues + fast inference")
    print("⚠️  Note: Handles one client at a time")
    print("=" * 70)
    
    # System info
    if cuda.Device.count() > 0:
        device = cuda.Device(0)
        print(f"🎮 GPU: {device.name()} ({device.total_memory() / 1024**3:.1f} GB)")
    
    server = SingleThreadedWebcamServer(args.engine, args.port)
    
    try:
        if not server.initialize_engine():
            print("❌ Failed to initialize engine")
            sys.exit(1)
        
        if not server.start_server():
            print("❌ Failed to start server")
            sys.exit(1)
        
        server.run()
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
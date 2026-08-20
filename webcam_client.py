#!/usr/bin/env python3
"""
Real-time Webcam Client for TensorRT Segmentation
Runs on your local computer to capture webcam and display results.

Usage:
    python webcam_client.py --server-ip YOUR_SERVER_IP --server-port 65432
"""

import cv2
import numpy as np
import socket
import pickle
import struct
import time
import argparse
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import sys

@dataclass
class FrameData:
    """Container for frame data"""
    frame: np.ndarray
    timestamp: float
    frame_id: int

class WebcamClient:
    """Real-time webcam client"""
    
    def __init__(self, server_ip: str, server_port: int, webcam_id: int = 0):
        self.server_ip = server_ip
        self.server_port = server_port
        self.webcam_id = webcam_id
        
        # Network
        self.socket = None
        self.connected = False
        
        # Webcam
        self.cap = None
        
        # Threading
        self.running = False
        self.send_thread = None
        self.receive_thread = None
        
        # Frame management
        self.frame_queue = deque(maxlen=5)  # Smaller buffer to reduce latency
        self.result_queue = deque(maxlen=5)  # Smaller buffer to reduce latency
        self.frame_counter = 0
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.latency_history = deque(maxlen=30)
        
        # Display settings
        self.show_fps = True
        self.show_latency = True
        self.display_scale = 1.0
        
    def connect_to_server(self) -> bool:
        """Connect to the inference server"""
        try:
            print(f"🔗 Connecting to server {self.server_ip}:{self.server_port}...")
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set timeouts
            self.socket.settimeout(10.0)
            
            self.socket.connect((self.server_ip, self.server_port))
            self.connected = True
            
            print("✅ Connected to server successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def initialize_webcam(self) -> bool:
        """Initialize webcam capture"""
        try:
            print(f"📷 Initializing webcam {self.webcam_id}...")
            
            # Try different backends for better compatibility
            backends = [
                cv2.CAP_DSHOW,   # DirectShow (Windows)
                cv2.CAP_V4L2,    # Video4Linux (Linux)
                cv2.CAP_AVFOUNDATION,  # AVFoundation (macOS)
                cv2.CAP_ANY      # Auto-detect
            ]
            
            self.cap = None
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.webcam_id, backend)
                    if self.cap.isOpened():
                        print(f"✅ Using backend: {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                except:
                    continue
            
            if self.cap is None or not self.cap.isOpened():
                # Try without specifying backend
                self.cap = cv2.VideoCapture(self.webcam_id)
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open webcam {self.webcam_id}")
            
            # Set webcam properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise ValueError("Failed to capture test frame")
            
            # Get actual properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Webcam initialized: {width}x{height} @ {fps:.1f} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize webcam: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def send_frame_worker(self):
        """Worker thread for sending frames to server"""
        print("🔄 Send worker thread started")
        
        while self.running and self.connected:
            try:
                if len(self.frame_queue) == 0:
                    time.sleep(0.001)  # Short sleep if no frames
                    continue
                
                frame_data = self.frame_queue.popleft()
                
                # Serialize frame data
                try:
                    serialized_data = pickle.dumps(frame_data, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"❌ Serialization error: {e}")
                    continue
                
                # Send data size first
                data_size = len(serialized_data)
                size_data = struct.pack("!I", data_size)
                
                try:
                    self.socket.sendall(size_data)
                    self.socket.sendall(serialized_data)
                except socket.error as e:
                    print(f"❌ Socket send error: {e}")
                    self.connected = False
                    break
                
            except Exception as e:
                if self.running:  # Only print error if we're supposed to be running
                    print(f"❌ Send worker error: {e}")
                    self.connected = False
                break
        
        print("🔄 Send worker thread stopped")
    
    def receive_result_worker(self):
        """Worker thread for receiving processed frames from server"""
        print("🔄 Receive worker thread started")
        
        while self.running and self.connected:
            try:
                # Receive data size
                size_data = self._recv_all(4)
                if not size_data:
                    print("❌ Failed to receive size data")
                    break
                
                data_size = struct.unpack("!I", size_data)[0]
                
                # Sanity check on data size
                if data_size <= 0 or data_size > 100 * 1024 * 1024:  # Max 100MB
                    print(f"❌ Invalid data size: {data_size}")
                    break
                
                # Receive actual data
                serialized_data = self._recv_all(data_size)
                if not serialized_data:
                    print("❌ Failed to receive frame data")
                    break
                
                # Deserialize result
                try:
                    result_data = pickle.loads(serialized_data)
                except Exception as e:
                    print(f"❌ Deserialization error: {e}")
                    continue
                
                # Add to result queue
                if len(self.result_queue) >= self.result_queue.maxlen:
                    # Drop oldest if queue is full
                    self.result_queue.popleft()
                
                self.result_queue.append(result_data)
                
            except Exception as e:
                if self.running:  # Only print error if we're supposed to be running
                    print(f"❌ Receive worker error: {e}")
                    self.connected = False
                break
        
        print("🔄 Receive worker thread stopped")
    
    def _recv_all(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes from socket"""
        data = b''
        while len(data) < size:
            try:
                chunk = self.socket.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                print("⚠️ Socket timeout during receive")
                return None
            except socket.error as e:
                print(f"❌ Socket receive error: {e}")
                return None
            except Exception as e:
                print(f"❌ Unexpected receive error: {e}")
                return None
        return data
    
    def capture_and_display_loop(self):
        """Main loop for capturing frames and displaying results"""
        print("🎬 Starting capture and display loop...")
        print("Press 'q' to quit, 's' to toggle stats, 'f' to toggle fullscreen")
        
        last_capture_time = time.perf_counter()
        fullscreen = False
        window_name = "Real-time Segmentation"
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_skip_counter = 0
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("❌ Failed to capture frame")
                    time.sleep(0.1)  # Wait a bit before retrying
                    continue
                
                current_time = time.perf_counter()
                
                # Skip frames to reduce queue buildup (send every 2nd frame)
                frame_skip_counter += 1
                if frame_skip_counter % 2 == 0 and self.connected:
                    # Add frame to send queue (skip if queue is full to maintain real-time)
                    if len(self.frame_queue) < self.frame_queue.maxlen:
                        frame_data = FrameData(
                            frame=frame.copy(),
                            timestamp=current_time,
                            frame_id=self.frame_counter
                        )
                        self.frame_queue.append(frame_data)
                        self.frame_counter += 1
                
                # Calculate capture FPS
                capture_fps = 1.0 / (current_time - last_capture_time) if current_time > last_capture_time else 0
                self.fps_history.append(capture_fps)
                last_capture_time = current_time
                
                # Display frame (either processed or original)
                display_frame = frame.copy()
                
                # Try to get processed result
                if len(self.result_queue) > 0:
                    result_data = self.result_queue.popleft()
                    
                    # Calculate latency
                    if 'client_timestamp' in result_data:
                        latency = current_time - result_data['client_timestamp']
                        self.latency_history.append(latency * 1000)  # Convert to ms
                    
                    # Use processed frame
                    if 'processed_frame' in result_data and result_data['processed_frame'] is not None:
                        display_frame = result_data['processed_frame']
                
                # Add performance overlay
                if self.show_fps or self.show_latency:
                    self._add_performance_overlay(display_frame)
                
                # Scale frame if needed
                if self.display_scale != 1.0:
                    new_width = int(display_frame.shape[1] * self.display_scale)
                    new_height = int(display_frame.shape[0] * self.display_scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # Display frame
                cv2.imshow(window_name, display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Quit requested by user")
                    break
                elif key == ord('s'):
                    self.show_fps = not self.show_fps
                    self.show_latency = not self.show_latency
                    print(f"📊 Stats display: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key == ord('+') or key == ord('='):
                    self.display_scale = min(2.0, self.display_scale + 0.1)
                    print(f"🔍 Display scale: {self.display_scale:.1f}x")
                elif key == ord('-'):
                    self.display_scale = max(0.5, self.display_scale - 0.1)
                    print(f"🔍 Display scale: {self.display_scale:.1f}x")
                elif key == ord('r'):  # Reconnect
                    if not self.connected:
                        print("🔄 Attempting to reconnect...")
                        if self.connect_to_server():
                            self._restart_worker_threads()
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Display loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Brief pause before continuing
        
        cv2.destroyAllWindows()
    
    def _restart_worker_threads(self):
        """Restart worker threads after reconnection"""
        if self.send_thread and self.send_thread.is_alive():
            return  # Already running
        
        self.send_thread = threading.Thread(target=self.send_frame_worker, daemon=True)
        self.receive_thread = threading.Thread(target=self.receive_result_worker, daemon=True)
        
        self.send_thread.start()
        self.receive_thread.start()
    
    def _add_performance_overlay(self, frame: np.ndarray):
        """Add performance information overlay to frame"""
        overlay_y = 30
        
        if self.show_fps and len(self.fps_history) > 0:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            fps_text = f"Capture FPS: {avg_fps:.1f}"
            cv2.putText(frame, fps_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            overlay_y += 30
        
        if self.show_latency and len(self.latency_history) > 0:
            avg_latency = sum(self.latency_history) / len(self.latency_history)
            latency_text = f"Latency: {avg_latency:.1f}ms"
            cv2.putText(frame, latency_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            overlay_y += 30
        
        # Connection status
        status_text = f"Server: {'Connected' if self.connected else 'Disconnected'}"
        status_color = (0, 255, 0) if self.connected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        overlay_y += 30
        
        # Queue status
        send_queue_size = len(self.frame_queue)
        result_queue_size = len(self.result_queue)
        queue_text = f"Queues: S:{send_queue_size} R:{result_queue_size}"
        cv2.putText(frame, queue_text, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def start(self):
        """Start the client"""
        print("🚀 Starting webcam client...")
        
        # Initialize webcam first
        if not self.initialize_webcam():
            print("❌ Cannot start without webcam")
            return False
        
        # Connect to server
        if not self.connect_to_server():
            print("⚠️ Starting without server connection (you can press 'r' to reconnect)")
        
        # Start worker threads
        self.running = True
        
        if self.connected:
            self.send_thread = threading.Thread(target=self.send_frame_worker, daemon=True)
            self.receive_thread = threading.Thread(target=self.receive_result_worker, daemon=True)
            
            self.send_thread.start()
            self.receive_thread.start()
        
        # Start main loop
        try:
            self.capture_and_display_loop()
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the client"""
        print("🛑 Stopping webcam client...")
        
        self.running = False
        
        # Wait for threads to finish
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=2.0)
        
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        # Close connections
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.connected = False
        print("✅ Client stopped")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-time Webcam Client for TensorRT Segmentation')
    parser.add_argument('--server-ip', type=str, required=True,
                        help='IP address of the inference server')
    parser.add_argument('--server-port', type=int, default=65432,
                        help='Port of the inference server (default: 65432)')
    parser.add_argument('--webcam-id', type=int, default=0,
                        help='Webcam device ID (default: 0)')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    print("=" * 60)
    print("📷 Real-time Webcam Segmentation Client")
    print("=" * 60)
    print(f"🖥️  Server: {args.server_ip}:{args.server_port}")
    print(f"📷 Webcam: {args.webcam_id}")
    print("=" * 60)
    
    client = WebcamClient(args.server_ip, args.server_port, args.webcam_id)
    
    try:
        success = client.start()
        if not success:
            print("❌ Failed to start client")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Client error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("👋 Client finished")

if __name__ == "__main__":
    main()

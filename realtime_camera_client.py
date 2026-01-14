"""
Real-time Camera Client for Face Detection & Recognition
Connects to WebSocket API and displays results
"""
import cv2
import asyncio
import websockets
import json
import base64
import numpy as np
from datetime import datetime

class RealtimeFaceRecognition:
    def __init__(self, api_url="ws://localhost:8000/ws/realtime"):
        self.api_url = api_url
        self.cap = None
        self.websocket = None
        
    async def connect(self):
        """Connect to WebSocket API"""
        try:
            self.websocket = await websockets.connect(self.api_url)
            print(f"Connected to {self.api_url}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
    
    def frame_to_base64(self, frame):
        """Convert OpenCV frame to base64"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    async def send_frame(self, frame):
        """Send frame to API for processing"""
        if not self.websocket:
            return None
        
        try:
            image_base64 = self.frame_to_base64(frame)
            message = {
                "type": "frame",
                "image": image_base64,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.websocket.send(json.dumps(message))
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                return json.loads(response)
            except asyncio.TimeoutError:
                return None
        
        except Exception as e:
            print(f"Error sending frame: {e}")
            return None
    
    def draw_results(self, frame, results):
        """Draw detection/recognition results on frame"""
        for face in results.get("faces", []):
            bbox = face.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                
                # Choose color based on recognition
                if face.get("matched", False):
                    color = (0, 255, 0)  # Green for recognized
                    label = f"Visitor: {face.get('visitor_id', 'Unknown')[:8]}"
                    if face.get("confidence"):
                        label += f" ({face['confidence']:.3f})"
                else:
                    color = (0, 165, 255)  # Orange for unknown
                    label = "Unknown"
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    async def run(self, camera_id=0, process_interval=3):
        """Run real-time face detection and recognition"""
        if not await self.connect():
            return
        
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            print("Cannot open camera")
            await self.disconnect()
            return
        
        print("Starting real-time face detection & recognition")
        print("Press 'q' to quit")
        
        frame_count = 0
        last_results = {"faces": []}
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_count % process_interval == 0:
                    results = await self.send_frame(frame)
                    if results and results.get("type") == "results":
                        last_results = results
                
                # Draw results
                frame = self.draw_results(frame, last_results)
                
                # Display info
                info_text = f"Faces: {last_results.get('count', 0)} | Process: {frame_count % process_interval == 0}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Real-time Face Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            await self.disconnect()

async def main():
    client = RealtimeFaceRecognition()
    await client.run(camera_id=0, process_interval=3)

if __name__ == "__main__":
    asyncio.run(main())

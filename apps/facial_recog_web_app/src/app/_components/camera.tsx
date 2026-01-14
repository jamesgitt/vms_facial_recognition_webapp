"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface Face {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface DetectionResponse {
  faces: number[][]; // Array of [x, y, w, h] arrays
  count: number;
  error?: string;
}

interface ComparisonResponse {
  similarity_score: number;
  is_match: boolean;
  features1?: number[];
  features2?: number[];
}

export function FaceRecognitionCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [isRunning, setIsRunning] = useState(false);
  const [faceCount, setFaceCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [processedFrames, setProcessedFrames] = useState(0);
  const [status, setStatus] = useState("Ready to start");
  const [isLoading, setIsLoading] = useState(false);
  
  const [detectionThreshold, setDetectionThreshold] = useState(0.6);
  const [processInterval, setProcessInterval] = useState(3);
  const [apiUrl, setApiUrl] = useState(
    typeof window !== "undefined" 
      ? (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
      : "http://localhost:8000"
  );
  const [comparisonThreshold, setComparisonThreshold] = useState(0.363);
  const [referenceFace, setReferenceFace] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResponse | null>(null);
  const [enableComparison, setEnableComparison] = useState(false);
  
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const animationFrameRef = useRef<number | undefined>(undefined);

  // Start camera
  const startCamera = useCallback(async () => {
    // Check if getUserMedia is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Error: Camera API not available. Please use a modern browser with HTTPS or localhost.");
      return;
    }

    // Check if we're on a secure context (HTTPS or localhost)
    if (!window.isSecureContext && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
      setStatus("Error: Camera access requires HTTPS. Please use HTTPS or localhost.");
      return;
    }

    try {
      setStatus("Requesting camera permission...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        await videoRef.current.play();

        // Set canvas size to match video
        if (canvasRef.current && videoRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
        }

        setIsRunning(true);
        setStatus("Camera active - Processing frames...");
        // Processing will start automatically via useEffect when isRunning changes
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      
      let errorMessage = "Could not access camera. ";
      
      if (error instanceof DOMException) {
        switch (error.name) {
          case "NotAllowedError":
            errorMessage += "Permission denied. Please allow camera access in your browser settings and try again.";
            break;
          case "NotFoundError":
            errorMessage += "No camera found. Please connect a camera and try again.";
            break;
          case "NotReadableError":
            errorMessage += "Camera is already in use by another application.";
            break;
          case "OverconstrainedError":
            errorMessage += "Camera doesn't support the requested settings.";
            break;
          case "SecurityError":
            errorMessage += "Camera access blocked for security reasons. Please use HTTPS or localhost.";
            break;
          default:
            errorMessage += error.message || "Unknown error occurred.";
        }
      } else if (error instanceof Error) {
        errorMessage += error.message;
      } else {
        errorMessage += "Unknown error occurred.";
      }
      
      setStatus(errorMessage);
    }
  }, []);

  // Stop camera
  const stopCamera = useCallback(() => {
    // Cancel any pending animation frames first
    if (animationFrameRef.current !== undefined) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = undefined;
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsRunning(false);
    setStatus("Camera stopped");
    setIsLoading(false);
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    setFaceCount(0);
    setFps(0);
    frameCountRef.current = 0;
  }, []);

  // Convert frame to base64
  const frameToBase64 = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    return canvasRef.current.toDataURL("image/jpeg", 0.8).split(",")[1] ?? null;
  }, []);

  // Detect faces via API
  const detectFaces = useCallback(
    async (imageBase64: string): Promise<DetectionResponse> => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/detect`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: imageBase64,
            score_threshold: detectionThreshold,
            return_landmarks: false,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = (await response.json()) as DetectionResponse;
        return data;
      } catch (error) {
        console.error("Detection error:", error);
        return {
          faces: [],
          count: 0,
          error: error instanceof Error ? error.message : "Unknown error",
        };
      }
    },
    [apiUrl, detectionThreshold]
  );

  // Compare faces via API
  const compareFaces = useCallback(
    async (image1Base64: string, image2Base64: string): Promise<ComparisonResponse | null> => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/compare`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image1: image1Base64,
            image2: image2Base64,
            threshold: comparisonThreshold,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = (await response.json()) as ComparisonResponse;
        return data;
      } catch (error) {
        console.error("Comparison error:", error);
        return null;
      }
    },
    [apiUrl, comparisonThreshold]
  );

  // Capture reference face
  const captureReferenceFace = useCallback(() => {
    const imageBase64 = frameToBase64();
    if (imageBase64) {
      setReferenceFace(imageBase64);
      setStatus("Reference face captured! Comparison enabled.");
      setEnableComparison(true);
    }
  }, [frameToBase64]);

  // Clear reference face
  const clearReferenceFace = useCallback(() => {
    setReferenceFace(null);
    setEnableComparison(false);
    setComparisonResult(null);
    setStatus("Reference face cleared.");
  }, []);

  // Draw bounding boxes
  const drawResults = useCallback((faces: number[][]) => {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    if (!faces || faces.length === 0) return;

    faces.forEach((face, index) => {
      // Handle array format [x, y, w, h]
      const x = face[0] || 0;
      const y = face[1] || 0;
      const w = face[2] || 0;
      const h = face[3] || 0;

      // Choose color based on comparison result
      let color = "#4CAF50"; // Green for detected
      let label = `Face ${index + 1}`;
      
      if (enableComparison && comparisonResult) {
        if (comparisonResult.is_match) {
          color = "#00FF00"; // Bright green for match
          label = `Match (${(comparisonResult.similarity_score * 100).toFixed(1)}%)`;
        } else {
          color = "#FFA500"; // Orange for no match
          label = `No Match (${(comparisonResult.similarity_score * 100).toFixed(1)}%)`;
        }
      }

      // Draw rectangle with thicker line for visibility
      ctx.strokeStyle = color;
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      // Draw label background
      ctx.fillStyle = color;
      ctx.fillRect(x, y - 25, Math.max(120, label.length * 8), 25);

      // Draw label text
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 14px Arial";
      ctx.fillText(label, x + 5, y - 8);
    });
  }, [enableComparison, comparisonResult]);

  // Process frame
  const processFrame = useCallback(async () => {
    // Check if still running before processing - exit immediately if stopped
    if (!isRunning) {
      animationFrameRef.current = undefined;
      return;
    }

    frameCountRef.current++;

    // Process every Nth frame
    if (frameCountRef.current % processInterval === 0) {
      // Check again before starting async operation
      if (!isRunning) {
        animationFrameRef.current = undefined;
        return;
      }
      
      setIsLoading(true);
      const imageBase64 = frameToBase64();
      
      if (imageBase64 && isRunning) {
        const result = await detectFaces(imageBase64);
        
        // Check again if still running after async operation
        if (isRunning) {
          if (result.faces && result.faces.length > 0) {
            // If comparison is enabled and we have a reference face, compare
            if (enableComparison && referenceFace) {
              const comparison = await compareFaces(referenceFace, imageBase64);
              if (comparison) {
                setComparisonResult(comparison);
              }
            }
            
            drawResults(result.faces);
            setFaceCount(result.count ?? result.faces.length);
            setProcessedFrames((prev) => prev + 1);
          } else if (result.error) {
            setStatus(`Error: ${result.error}`);
          } else {
            // No faces detected, clear comparison result
            setComparisonResult(null);
            drawResults([]);
          }
        }
      }
      
      setIsLoading(false);
    }

    // Check again before calculating FPS and scheduling next frame
    if (!isRunning) {
      animationFrameRef.current = undefined;
      return;
    }

    // Calculate FPS
    const now = Date.now();
    if (now - lastTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    }

    // Only continue loop if still running
    if (isRunning) {
      animationFrameRef.current = requestAnimationFrame(processFrame);
    } else {
      animationFrameRef.current = undefined;
    }
  }, [isRunning, processInterval, frameToBase64, detectFaces, drawResults]);

  // Start processing when camera is running
  useEffect(() => {
    if (isRunning) {
      // Start the processing loop
      processFrame();
    } else {
      // Cancel any pending animation frames immediately when stopped
      if (animationFrameRef.current !== undefined) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = undefined;
      }
      // Reset loading state
      setIsLoading(false);
    }
    
    // Cleanup function to cancel frames when component unmounts or isRunning changes
    return () => {
      if (animationFrameRef.current !== undefined) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = undefined;
      }
    };
  }, [isRunning, processFrame]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div className="container mx-auto max-w-6xl p-6">
      <div className="rounded-2xl bg-white p-8 shadow-2xl">
        <h1 className="mb-2 text-center text-4xl font-bold text-gray-800">
          🎥 Real-time Face Detection & Recognition
        </h1>
        <p className="mb-8 text-center text-gray-600">
          Using YuNet & Sface Models
        </p>

        {/* Video Container */}
        <div className="relative mb-6 overflow-hidden rounded-lg bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full"
          />
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute left-0 top-0 h-full w-full"
          />
        </div>

        {/* Controls */}
        <div className="mb-6 flex flex-wrap justify-center gap-4">
          <button
            onClick={startCamera}
            disabled={isRunning}
            className="rounded-lg bg-green-500 px-8 py-3 font-semibold text-white transition hover:bg-green-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Start Camera
          </button>
          <button
            onClick={stopCamera}
            disabled={!isRunning}
            className="rounded-lg bg-red-500 px-8 py-3 font-semibold text-white transition hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Stop Camera
          </button>
          {isRunning && (
            <>
              <button
                onClick={captureReferenceFace}
                className="rounded-lg bg-blue-500 px-8 py-3 font-semibold text-white transition hover:bg-blue-600"
              >
                Capture Reference Face
              </button>
              {referenceFace && (
                <button
                  onClick={clearReferenceFace}
                  className="rounded-lg bg-orange-500 px-8 py-3 font-semibold text-white transition hover:bg-orange-600"
                >
                  Clear Reference
                </button>
              )}
            </>
          )}
        </div>

        {/* Status */}
        <div
          className={`mb-6 rounded-lg p-4 text-center ${
            isRunning
              ? "bg-green-50 text-green-800"
              : status.includes("Error")
              ? "bg-red-50 text-red-800"
              : "bg-gray-50 text-gray-800"
          }`}
        >
          {status}
        </div>

        <div className="mb-6 h-6 text-center">
          {isLoading && (
            <span className="text-gray-600">Processing frame...</span>
          )}
        </div>

        {/* Stats */}
        <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="rounded-lg bg-gray-50 p-6 text-center">
            <div className="text-4xl font-bold text-purple-600">{faceCount}</div>
            <div className="mt-2 text-gray-600">Faces Detected</div>
          </div>
          <div className="rounded-lg bg-gray-50 p-6 text-center">
            <div className="text-4xl font-bold text-purple-600">{fps}</div>
            <div className="mt-2 text-gray-600">FPS</div>
          </div>
          <div className="rounded-lg bg-gray-50 p-6 text-center">
            <div className="text-4xl font-bold text-purple-600">
              {processedFrames}
            </div>
            <div className="mt-2 text-gray-600">Frames Processed</div>
          </div>
        </div>

        {/* Settings */}
        <div className="rounded-lg bg-gray-50 p-6">
          <h3 className="mb-4 text-xl font-semibold text-gray-800">Settings</h3>

          <div className="mb-4">
            <label className="mb-2 block text-gray-700">
              Detection Threshold:{" "}
              <span className="font-bold text-purple-600">
                {detectionThreshold.toFixed(1)}
              </span>
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={detectionThreshold}
              onChange={(e) => setDetectionThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="mb-4">
            <label className="mb-2 block text-gray-700">
              Processing Interval (frames):{" "}
              <span className="font-bold text-purple-600">{processInterval}</span>
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={processInterval}
              onChange={(e) => setProcessInterval(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="mb-4">
            <label className="mb-2 block text-gray-700">
              API URL:{" "}
              <span className="font-bold text-purple-600">{apiUrl}</span>
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full rounded border border-gray-300 px-4 py-2"
            />
          </div>

          {enableComparison && (
            <div className="mb-4 rounded-lg border-2 border-blue-300 bg-blue-50 p-4">
              <h4 className="mb-2 font-semibold text-blue-800">Face Comparison Active</h4>
              <div className="mb-2">
                <label className="mb-2 block text-gray-700">
                  Comparison Threshold:{" "}
                  <span className="font-bold text-purple-600">
                    {comparisonThreshold.toFixed(3)}
                  </span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.01"
                  value={comparisonThreshold}
                  onChange={(e) => setComparisonThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              {comparisonResult && (
                <div className="mt-2">
                  <div className={`rounded p-2 ${comparisonResult.is_match ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                    <strong>Similarity:</strong> {(comparisonResult.similarity_score * 100).toFixed(2)}% |{" "}
                    <strong>Match:</strong> {comparisonResult.is_match ? "Yes ✓" : "No ✗"}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { env } from "~/env";

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

interface RecognitionResponse {
  visitor_id: string | null;
  confidence: number | null;
  matched: boolean;
  firstName?: string | null;
  lastName?: string | null;
  visitor?: string | null;
  match_score?: number | null;
  matches?: Array<{
    visitor_id?: string;
    visitor?: string;
    match_score?: number;
    is_match?: boolean;
    firstName?: string | null;
    lastName?: string | null;
  }>;
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
    env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
  );
  const [comparisonThreshold, setComparisonThreshold] = useState(0.363);
  const [referenceFace, setReferenceFace] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResponse | null>(null);
  const [enableComparison, setEnableComparison] = useState(false);
  
  // Mode selection: 'detect' | 'compare' | 'recognize'
  const [mode, setMode] = useState<"detect" | "compare" | "recognize">("detect");
  const [recognitionResult, setRecognitionResult] = useState<RecognitionResponse | null>(null);
  const [recognitionThreshold, setRecognitionThreshold] = useState(0.363);
  
  // Recognition trigger and countdown state
  const [recognitionTriggered, setRecognitionTriggered] = useState(false);
  const [recognitionStartTime, setRecognitionStartTime] = useState<number | null>(null);
  const RECOGNITION_DELAY = 5000; // 5 seconds in milliseconds

  // HNSW status check
  interface HNSWStatus {
    available: boolean;
    initialized: boolean;
    total_vectors: number;
    dimension: number;
    index_type: string;
    m?: number;
    ef_construction?: number;
    ef_search?: number;
    visitors_indexed: number;
    details?: Record<string, unknown>;
  }
  const [hnswStatus, setHnswStatus] = useState<HNSWStatus | null>(null);
  const [hnswStatusLoading, setHnswStatusLoading] = useState(false);
  
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const animationFrameRef = useRef<number | undefined>(undefined);

  // Check HNSW status
  const checkHNSWStatus = useCallback(async () => {
    setHnswStatusLoading(true);
    try {
      const response = await fetch(`${apiUrl}/api/v1/hnsw/status`);
      if (response.ok) {
        const data = (await response.json()) as HNSWStatus;
        setHnswStatus(data);
      } else {
        setHnswStatus({
          available: false,
          initialized: false,
          total_vectors: 0,
          dimension: 512,
          index_type: "HNSW",
          visitors_indexed: 0,
          details: { error: `HTTP ${response.status}` }
        });
      }
    } catch (error) {
      console.error("Error checking HNSW status:", error);
      setHnswStatus({
        available: false,
        initialized: false,
        total_vectors: 0,
        dimension: 512,
        index_type: "HNSW",
        visitors_indexed: 0,
        details: { error: String(error) }
      });
    } finally {
      setHnswStatusLoading(false);
    }
  }, [apiUrl]);

  // Check HNSW status on mount and when API URL changes
  useEffect(() => {
    void checkHNSWStatus();
  }, [checkHNSWStatus]);

  // Start camera
  const startCamera = useCallback(async () => {
    // Check if getUserMedia is available
    if (!navigator.mediaDevices?.getUserMedia) {
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

  // Recognize face from database via API
  const recognizeFace = useCallback(
    async (imageBase64: string): Promise<RecognitionResponse | null> => {
      try {
        const formData = new FormData();
        formData.append("image_base64", imageBase64);
        formData.append("threshold", recognitionThreshold.toString());

        const response = await fetch(`${apiUrl}/api/v1/recognize`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = (await response.json()) as RecognitionResponse;
        return data;
      } catch (error) {
        console.error("Recognition error:", error);
        return null;
      }
    },
    [apiUrl, recognitionThreshold]
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

  // Handle mode change
  const handleModeChange = useCallback((newMode: "detect" | "compare" | "recognize") => {
    setMode(newMode);
    setComparisonResult(null);
    setRecognitionResult(null);
    setRecognitionTriggered(false);
    setRecognitionStartTime(null);
    if (newMode !== "compare") {
      setEnableComparison(false);
    }
    if (newMode === "detect") {
      setStatus("Face detection mode active");
    } else if (newMode === "compare") {
      setStatus("Face comparison mode - Capture a reference face to start");
    } else if (newMode === "recognize") {
      setStatus("Database recognition mode - Press 'Start Recognition' button to begin");
    }
  }, []);

  // Start recognition process
  const startRecognition = useCallback(() => {
    if (mode === "recognize" && isRunning) {
      setRecognitionTriggered(true);
      setRecognitionStartTime(Date.now());
      setRecognitionResult(null);
      setStatus("Detecting face... (5 seconds)");
    }
  }, [mode, isRunning]);

  // Cancel recognition process
  const cancelRecognition = useCallback(() => {
    setRecognitionTriggered(false);
    setRecognitionStartTime(null);
    setRecognitionResult(null);
    if (mode === "recognize") {
      setStatus("Recognition cancelled. Press 'Start Recognition' to try again.");
    }
  }, [mode]);

  // Draw bounding boxes
  const drawResults = useCallback((faces: number[][]) => {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    if (!faces || faces.length === 0) return;

    faces.forEach((face, index) => {
      // Handle array format [x, y, w, h]
      const x = face[0] ?? 0;
      const y = face[1] ?? 0;
      const w = face[2] ?? 0;
      const h = face[3] ?? 0;

      // Choose color based on mode and results
      let color = "#4CAF50"; // Green for detected
      let label = `Face ${index + 1}`;
      
      if (mode === "compare" && enableComparison && comparisonResult) {
        if (comparisonResult.is_match) {
          color = "#00FF00"; // Bright green for match
          label = `Match (${(comparisonResult.similarity_score * 100).toFixed(1)}%)`;
        } else {
          color = "#FFA500"; // Orange for no match
          label = `No Match (${(comparisonResult.similarity_score * 100).toFixed(1)}%)`;
        }
      } else if (mode === "recognize" && recognitionResult) {
        if (recognitionResult.matched) {
          color = "#00FF00"; // Bright green for recognized
          const confidence = recognitionResult.confidence ?? 0;
          const visitorId = recognitionResult.visitor_id ?? recognitionResult.visitor ?? "Unknown";
          const firstName = recognitionResult.firstName ?? "";
          const lastName = recognitionResult.lastName ?? "";
          const name = firstName || lastName ? `${firstName} ${lastName}`.trim() : visitorId;
          label = `${name} (${(confidence * 100).toFixed(1)}%)`;
        } else {
          color = "#FF6B6B"; // Red for not recognized
          label = "Not Recognized";
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
  }, [mode, enableComparison, comparisonResult, recognitionResult]);

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
            // Handle different modes
            if (mode === "compare" && enableComparison && referenceFace) {
              // Compare mode: compare reference face with current frame
              const comparison = await compareFaces(referenceFace, imageBase64);
              if (comparison) {
                setComparisonResult(comparison);
              }
            } else if (mode === "recognize" && recognitionTriggered && recognitionStartTime !== null) {
              // Recognize mode: check if 5 seconds have passed for face detection
              const currentTime = Date.now();
              const elapsedTime = currentTime - recognitionStartTime;
              const remainingTime = RECOGNITION_DELAY - elapsedTime;
              
              if (remainingTime > 0) {
                // Still in detection phase - show countdown
                const secondsRemaining = Math.ceil(remainingTime / 1000);
                setStatus(`Detecting face... (${secondsRemaining} second${secondsRemaining !== 1 ? 's' : ''} remaining)`);
              } else {
                // 5 seconds have passed - perform recognition
                setStatus("Recognizing face from database...");
                const recognition = await recognizeFace(imageBase64);
                if (recognition) {
                  setRecognitionResult(recognition);
                  setRecognitionTriggered(false);
                  setRecognitionStartTime(null);
                  if (recognition.matched) {
                    const recognizedName = (recognition.firstName || recognition.lastName)
                      ? `${recognition.firstName ?? ""} ${recognition.lastName ?? ""}`.trim()
                      : (recognition.visitor_id ?? recognition.visitor ?? 'Unknown');
                    setStatus(`Recognized: ${recognizedName} (${((recognition.confidence ?? 0) * 100).toFixed(1)}%)`);
                  } else {
                    setStatus("Face not recognized in database. Press 'Start Recognition' to try again.");
                  }
                } else {
                  setRecognitionTriggered(false);
                  setRecognitionStartTime(null);
                  setStatus("Recognition failed. Press 'Start Recognition' to try again.");
                }
              }
            }
            
            drawResults(result.faces);
            setFaceCount(result.count ?? result.faces.length);
            setProcessedFrames((prev) => prev + 1);
          } else if (result.error) {
            setStatus(`Error: ${result.error}`);
          } else {
            // No faces detected, clear results
            setComparisonResult(null);
            if (mode !== "recognize" || !recognitionTriggered) {
              setRecognitionResult(null);
            }
            drawResults([]);
            // If recognition is triggered but no face detected, show message
            if (mode === "recognize" && recognitionTriggered && recognitionStartTime !== null) {
              const currentTime = Date.now();
              const elapsedTime = currentTime - recognitionStartTime;
              if (elapsedTime < RECOGNITION_DELAY) {
                const secondsRemaining = Math.ceil((RECOGNITION_DELAY - elapsedTime) / 1000);
                setStatus(`No face detected. Keep your face in view... (${secondsRemaining}s remaining)`);
              }
            }
            // If recognition is triggered but no face detected, show message
            if (mode === "recognize" && recognitionTriggered && recognitionStartTime !== null) {
              const currentTime = Date.now();
              const elapsedTime = currentTime - recognitionStartTime;
              if (elapsedTime < RECOGNITION_DELAY) {
                const secondsRemaining = Math.ceil((RECOGNITION_DELAY - elapsedTime) / 1000);
                setStatus(`No face detected. Keep your face in view... (${secondsRemaining}s remaining)`);
              }
            }
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
      animationFrameRef.current = requestAnimationFrame(() => {
        void processFrame();
      });
    } else {
      animationFrameRef.current = undefined;
    }
    }, [isRunning, processInterval, frameToBase64, detectFaces, drawResults, compareFaces, enableComparison, referenceFace, mode, recognizeFace, recognitionTriggered, recognitionStartTime, RECOGNITION_DELAY]);

  // Start processing when camera is running
  useEffect(() => {
    if (isRunning) {
      // Start the processing loop
      void processFrame();
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

        {/* Mode Selection */}
        <div className="mb-6 flex flex-wrap justify-center gap-4">
          <button
            onClick={() => handleModeChange("detect")}
            className={`rounded-lg px-6 py-2 font-semibold text-white transition ${
              mode === "detect"
                ? "bg-purple-600 hover:bg-purple-700"
                : "bg-gray-400 hover:bg-gray-500"
            }`}
          >
            🔍 Detect Faces
          </button>
          <button
            onClick={() => handleModeChange("compare")}
            className={`rounded-lg px-6 py-2 font-semibold text-white transition ${
              mode === "compare"
                ? "bg-blue-600 hover:bg-blue-700"
                : "bg-gray-400 hover:bg-gray-500"
            }`}
          >
            ⚖️ Compare Faces
          </button>
          <button
            onClick={() => handleModeChange("recognize")}
            className={`rounded-lg px-6 py-2 font-semibold text-white transition ${
              mode === "recognize"
                ? "bg-green-600 hover:bg-green-700"
                : "bg-gray-400 hover:bg-gray-500"
            }`}
          >
            🎯 Recognize from Database
          </button>
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
          {isRunning && mode === "compare" && (
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

          {/* HNSW Status Check */}
          <div className="mb-4 rounded-lg border-2 border-purple-300 bg-purple-50 p-4">
            <div className="mb-2 flex items-center justify-between">
              <h4 className="font-semibold text-purple-800">HNSW Index Status</h4>
              <button
                onClick={checkHNSWStatus}
                disabled={hnswStatusLoading}
                className="rounded bg-purple-600 px-3 py-1 text-sm text-white transition hover:bg-purple-700 disabled:bg-gray-400"
              >
                {hnswStatusLoading ? "Checking..." : "🔄 Refresh"}
              </button>
            </div>
            {hnswStatus ? (
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-semibold">Status:</span>
                  {hnswStatus.available && hnswStatus.initialized ? (
                    <span className="rounded bg-green-100 px-2 py-1 text-green-800">
                      ✓ Active ({hnswStatus.total_vectors} vectors)
                    </span>
                  ) : hnswStatus.available ? (
                    <span className="rounded bg-yellow-100 px-2 py-1 text-yellow-800">
                      ⚠ Available but not initialized
                    </span>
                  ) : (
                    <span className="rounded bg-red-100 px-2 py-1 text-red-800">
                      ✗ Not Available
                    </span>
                  )}
                </div>
                {hnswStatus.initialized && (
                  <>
                    <div>
                      <span className="font-semibold">Visitors Indexed:</span>{" "}
                      <span className="text-purple-600">{hnswStatus.visitors_indexed}</span>
                    </div>
                    <div>
                      <span className="font-semibold">Total Vectors:</span>{" "}
                      <span className="text-purple-600">{hnswStatus.total_vectors}</span>
                    </div>
                    {hnswStatus.m && (
                      <div>
                        <span className="font-semibold">HNSW Parameters:</span>{" "}
                        <span className="text-purple-600">
                          M={hnswStatus.m}, ef_construction={hnswStatus.ef_construction}, ef_search={hnswStatus.ef_search}
                        </span>
                      </div>
                    )}
                  </>
                )}
                {"error" in (hnswStatus.details ?? {}) && (
                  <div className="mt-2 rounded bg-red-100 p-2 text-xs text-red-800">
                    <strong>Error:</strong> {typeof hnswStatus.details?.error === "string" 
                      ? hnswStatus.details.error 
                      : JSON.stringify(hnswStatus.details?.error ?? "")}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-sm text-gray-600">Loading status...</div>
            )}
          </div>

          {/* Comparison Mode Settings */}
          {mode === "compare" && enableComparison && (
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

          {/* Recognition Mode Settings */}
          {mode === "recognize" && (
            <div className="mb-4 rounded-lg border-2 border-green-300 bg-green-50 p-4">
              <h4 className="mb-2 font-semibold text-green-800">Database Recognition Active</h4>
              
              {/* Start Recognition Button */}
              <div className="mb-4 flex gap-2">
                {!recognitionTriggered ? (
                  <button
                    onClick={startRecognition}
                    disabled={!isRunning}
                    className="flex-1 rounded-lg bg-green-600 px-6 py-3 font-semibold text-white transition hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    🎯 Start Recognition
                  </button>
                ) : (
                  <button
                    onClick={cancelRecognition}
                    className="flex-1 rounded-lg bg-red-600 px-6 py-3 font-semibold text-white transition hover:bg-red-700"
                  >
                    ❌ Cancel Recognition
                  </button>
                )}
              </div>
              
              <div className="mb-2">
                <label className="mb-2 block text-gray-700">
                  Recognition Threshold:{" "}
                  <span className="font-bold text-purple-600">
                    {recognitionThreshold.toFixed(3)}
                  </span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.01"
                  value={recognitionThreshold}
                  onChange={(e) => setRecognitionThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              {recognitionResult && (
                <div className="mt-2 space-y-2">
                  <div className={`rounded p-3 ${recognitionResult.matched ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}>
                    <div className="font-semibold">
                      {recognitionResult.matched ? "✓ Recognized" : "✗ Not Recognized"}
                    </div>
                    {recognitionResult.matched && (
                      <div className="mt-1">
                        <div><strong>Visitor ID:</strong> {recognitionResult.visitor_id ?? recognitionResult.visitor ?? "Unknown"}</div>
                        {(recognitionResult.firstName || recognitionResult.lastName) && (
                          <div><strong>Name:</strong> {`${recognitionResult.firstName ?? ""} ${recognitionResult.lastName ?? ""}`.trim()}</div>
                        )}
                        <div><strong>Confidence:</strong> {((recognitionResult.confidence ?? 0) * 100).toFixed(2)}%</div>
                      </div>
                    )}
                  </div>
                  {recognitionResult.matches && recognitionResult.matches.length > 0 && (
                    <div className="rounded bg-gray-100 p-2">
                      <div className="text-sm font-semibold text-gray-700">Top Matches:</div>
                      <div className="mt-1 space-y-1">
                        {recognitionResult.matches.slice(0, 5).map((match, idx) => {
                          const matchName = (match.firstName || match.lastName) 
                            ? `${match.firstName ?? ""} ${match.lastName ?? ""}`.trim()
                            : (match.visitor_id ?? match.visitor ?? "Unknown");
                          return (
                            <div key={idx} className="text-xs text-gray-600">
                              {idx + 1}. {matchName} - 
                              {match.match_score ? ` ${(match.match_score * 100).toFixed(1)}%` : " N/A"}
                              {match.is_match && " ✓"}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

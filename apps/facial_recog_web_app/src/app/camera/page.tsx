import { FaceRecognitionCamera } from "~/app/_components/camera";

export default function CameraPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-600 via-purple-700 to-indigo-800">
      <div className="py-8">
        <FaceRecognitionCamera />
      </div>
    </main>
  );
}

# Real-Time Face Detection & Recognition Web App

This project enables real-time face detection and recognition through your web browser using a webcam, backed by a FastAPI service with YuNet and SFace models.

## 🚀 How to Use the Real-Time Face Recognition Demo (Vercel Deployment)

1. **Access the Web App**  
   Visit your deployed site on Vercel (e.g. `https://your-vercel-app-url.vercel.app`).  
   The FastAPI backend must also be running and accessible by the frontend (see below).

2. **Configure the Backend API URL**  
   - The web app requires the FastAPI backend for detection and recognition.  
   - If your backend is deployed (or running elsewhere), make sure to provide its public URL in the "API URL" field on the web interface's settings panel.  
   - Example: `https://your-fastapi-app-url/api`

3. **Using the Camera:**
   - **Start Camera:** Click "Start Camera" and grant browser permission to use your device's webcam.
   - **Detect Faces:** Detected faces will be highlighted live in the video feed.
   - **Capture Reference Face:** Click this to save a reference for live face comparison.
   - **Live Comparison:** With a reference set, new detected faces are compared, with matching and similarity shown in real time.
   - **Stop Camera:** Click "Stop Camera" to turn off your webcam stream.

4. **Settings:**  
   - Adjust **Detection Threshold** for sensitivity of face detection.
   - Change **Processing Interval** to control how many frames are skipped between detections.
   - Update **API URL** anytime to point to your running FastAPI backend.
   - Fine-tune **Comparison Threshold** for stricter or looser face matching.
   - See live stats: Detected faces, FPS, and frames processed.

5. **Troubleshooting:**  
   - Ensure the FastAPI backend is accessible on a public URL (CORS enabled).
   - If the API is unreachable, check the "API URL" value on the settings panel.
   - Make sure your browser has permission to access the camera.

---

# Create T3 App

This is a [T3 Stack](https://create.t3.gg/) project bootstrapped with `create-t3-app`.

## What's next? How do I make an app with this?

We try to keep this project as simple as possible, so you can start with just the scaffolding we set up for you, and add additional things later when they become necessary.

If you are not familiar with the different technologies used in this project, please refer to the respective docs. If you still are in the wind, please join our [Discord](https://t3.gg/discord) and ask for help.

- [Next.js](https://nextjs.org)
- [NextAuth.js](https://next-auth.js.org)
- [Prisma](https://prisma.io)
- [Drizzle](https://orm.drizzle.team)
- [Tailwind CSS](https://tailwindcss.com)
- [tRPC](https://trpc.io)

## Learn More

To learn more about the [T3 Stack](https://create.t3.gg/), take a look at the following resources:

- [Documentation](https://create.t3.gg/)
- [Learn the T3 Stack](https://create.t3.gg/en/faq#what-learning-resources-are-currently-available) — Check out these awesome tutorials

You can check out the [create-t3-app GitHub repository](https://github.com/t3-oss/create-t3-app) — your feedback and contributions are welcome!

## How do I deploy this?

Follow our deployment guides for [Vercel](https://create.t3.gg/en/deployment/vercel), [Netlify](https://create.t3.gg/en/deployment/netlify) and [Docker](https://create.t3.gg/en/deployment/docker) for more information.

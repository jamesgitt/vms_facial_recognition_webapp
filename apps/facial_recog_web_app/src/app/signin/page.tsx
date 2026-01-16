"use client";

import { signIn } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import { useState, Suspense } from "react";

function SignInForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get("callbackUrl") ?? "/";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // Normalize email (lowercase and trim) to match auth config
      const normalizedEmail = email.toLowerCase().trim();
      console.log("[SignIn] Attempting to sign in with email:", normalizedEmail);
      
      const result = await signIn("credentials", {
        email: normalizedEmail,
        password,
        redirect: false,
      });

      console.log("[SignIn] Sign in result:", result);

      if (result?.error) {
        console.error("[SignIn] Sign in error:", result.error);
        setError(result.error === "CredentialsSignin" 
          ? "Invalid email or password" 
          : `Sign in failed: ${result.error}`);
        setLoading(false);
      } else if (result?.ok) {
        console.log("[SignIn] Sign in successful, redirecting...");
        router.push(callbackUrl);
        router.refresh();
      } else {
        console.warn("[SignIn] Unexpected result:", result);
        setError("Unexpected response. Please try again.");
        setLoading(false);
      }
    } catch (err) {
      console.error("[SignIn] Exception during sign in:", err);
      setError("Something went wrong. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c]">
      <div className="w-full max-w-md rounded-lg bg-white/10 p-8 shadow-lg">
        <h1 className="mb-6 text-center text-3xl font-bold text-white">
          Sign In
        </h1>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-white">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-purple-500 focus:outline-none focus:ring-purple-500"
              placeholder="your@email.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-white">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-purple-500 focus:outline-none focus:ring-purple-500"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <div className="rounded-md bg-red-500/20 p-3 text-sm text-red-200">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-md bg-purple-600 px-4 py-2 font-semibold text-white hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50"
          >
            {loading ? "Signing in..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default function SignInPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c]">
        <div className="w-full max-w-md rounded-lg bg-white/10 p-8 shadow-lg">
          <h1 className="mb-6 text-center text-3xl font-bold text-white">
            Sign In
          </h1>
          <p className="text-center text-white/70">Loading...</p>
        </div>
      </div>
    }>
      <SignInForm />
    </Suspense>
  );
}

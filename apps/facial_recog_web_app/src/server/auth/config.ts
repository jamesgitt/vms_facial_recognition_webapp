import { type DefaultSession, type NextAuthConfig } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import { compare } from "bcryptjs";

import { db } from "~/server/db";
import { env } from "~/env";

/**
 * Module augmentation for `next-auth` types. Allows us to add custom properties to the `session`
 * object and keep type safety.
 *
 * @see https://next-auth.js.org/getting-started/typescript#module-augmentation
 */
declare module "next-auth" {
  interface Session extends DefaultSession {
    user: {
      id: string;
      // ...other properties
      // role: UserRole;
    } & DefaultSession["user"];
  }

  // interface User {
  //   // ...other properties
  //   // role: UserRole;
  // }
}

/**
 * Options for NextAuth.js used to configure adapters, providers, callbacks, etc.
 *
 * @see https://next-auth.js.org/configuration/options
 */
export const authConfig = {
  providers: [
    // Email/Password authentication
    CredentialsProvider({
      id: "credentials",
      name: "Email",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          console.log("[Auth] Missing credentials");
          return null;
        }

        try {
          // Normalize email (lowercase and trim)
          // Type guard to ensure email and password are strings
          if (typeof credentials.email !== "string" || typeof credentials.password !== "string") {
            console.log("[Auth] Invalid credential types");
            return null;
          }

          const email = credentials.email.toLowerCase().trim();
          const password = credentials.password;
          
          // Find user by email
          const user = await db.user.findUnique({
            where: { email },
          });

          if (!user) {
            console.log(`[Auth] User not found: ${email}`);
            return null;
          }

          if (!user.password) {
            console.log(`[Auth] User has no password set: ${email}`);
            return null;
          }

          // Verify password
          const isValid = await compare(password, user.password);
          
          if (!isValid) {
            console.log(`[Auth] Invalid password for: ${email}`);
            return null;
          }

          console.log(`[Auth] Successfully authenticated: ${email}`);
          return {
            id: user.id,
            email: user.email,
            name: user.name,
            image: user.image,
          };
        } catch (error) {
          console.error("[Auth] Error during authentication:", error);
          return null;
        }
      },
    }),
  ],
  // JWT session strategy (required for CredentialsProvider)
  session: {
    strategy: "jwt",
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  secret: env.AUTH_SECRET,
  callbacks: {
    session: ({ session, token }) => ({
      ...session,
      user: {
        ...session.user,
        id: token.sub ?? "",
      },
    }),
    jwt: ({ token, user }) => {
      if (user) {
        token.sub = user.id;
        token.email = user.email ?? undefined;
        token.name = user.name ?? undefined;
      }
      return token;
    },
  },
  pages: {
    signIn: "/signin",
  },
  debug: process.env.NODE_ENV === "development",
} satisfies NextAuthConfig;

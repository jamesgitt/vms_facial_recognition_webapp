# Email/Password Authentication Setup

## Overview

The app now uses simple **Email/Password authentication**. No external OAuth providers needed!

---

## Quick Setup

### Step 1: Install Dependencies

```bash
cd apps/facial_recog_web_app
npm install
```

This will install `bcryptjs` for password hashing (already added to package.json).

---

### Step 2: Update Database Schema

The Prisma schema has been updated to include a `password` field. Run the migration:

```bash
# Option 1: Create migration (recommended for production)
npx prisma migrate dev --name add_password_field

# Option 2: Push schema directly (for development)
npx prisma db push
```

This adds the `password` field to the User model.

---

### Step 3: Set Environment Variables

**Local Development (.env.local):**

Create `apps/facial_recog_web_app/.env.local`:

```env
AUTH_SECRET=generate-with-openssl-rand-base64-32
DATABASE_URL=postgresql://user:password@localhost:5432/visitors_db
```

**Generate AUTH_SECRET:**
```bash
openssl rand -base64 32
```

**Vercel (Production):**

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add:
   - `AUTH_SECRET` = your generated secret
   - `DATABASE_URL` = your database connection string
3. Make sure to select all environments (Production, Preview, Development)
4. Click "Save"

---

### Step 4: Test Registration

**Option 1: Use the API directly**

```bash
curl -X POST http://localhost:3000/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123",
    "name": "Test User"
  }'
```

**Option 2: Create a registration page (optional)**

You can create a registration form in your frontend. The API endpoint is:
- **POST** `/api/register`
- **Body**: `{ email: string, password: string, name?: string }`

---

### Step 5: Test Sign In

1. Start your dev server:
   ```bash
   npm run dev
   ```x

2. Go to `http://localhost:3000`
3. Click "Sign in"
4. Enter the email and password you registered
5. You should be logged in!

---

## API Endpoints

### Register User

**POST** `/api/register`

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "password123",
  "name": "User Name" // optional
}
```

**Success Response (201):**
```json
{
  "message": "User created successfully",
  "user": {
    "id": "...",
    "email": "user@example.com",
    "name": "User Name",
    "image": null
  }
}
```

**Error Response (400):**
```json
{
  "error": "User with this email already exists"
}
```

### Sign In

Use NextAuth's built-in sign-in page:
- **GET** `/api/auth/signin` - Sign in page
- **POST** `/api/auth/signin` - Sign in endpoint

Or use the `signIn` function from `next-auth`:
```typescript
import { signIn } from "next-auth/react";

await signIn("credentials", {
  email: "user@example.com",
  password: "password123",
  redirect: true,
});
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AUTH_SECRET` | ✅ Yes | Secret for JWT signing (generate with `openssl rand -base64 32`) |
| `DATABASE_URL` | ✅ Yes | PostgreSQL connection string |

**No Google OAuth credentials needed!**

---

## Security Notes

1. **Passwords are hashed**: All passwords are hashed with bcryptjs before storing
2. **Minimum password length**: 6 characters (enforced in registration API)
3. **Email validation**: Email format is validated
4. **Unique emails**: Each email can only be used once
5. **JWT sessions**: Uses JWT tokens for sessions (no database sessions needed)

---

## Troubleshooting

### "User with this email already exists"

**Problem**: Trying to register with an email that's already in use.

**Solution**: Use a different email or sign in with existing account.

---

### "Invalid email address"

**Problem**: Email format is invalid.

**Solution**: Use a valid email format (e.g., `user@example.com`).

---

### "Password must be at least 6 characters"

**Problem**: Password is too short.

**Solution**: Use a password with at least 6 characters.

---

### Can't sign in after registration

**Problem**: Password might not be hashed correctly or user not created.

**Solution**:
1. Check database - verify user was created
2. Check password hash - should be a long string starting with `$2a$` or `$2b$`
3. Make sure you're using the same password you registered with

---

### Database migration fails

**Problem**: Schema changes not applied.

**Solution**:
```bash
# Reset database (⚠️ deletes all data)
npx prisma migrate reset

# Or push schema directly
npx prisma db push
```

---

## Next Steps

1. ✅ Run database migration
2. ✅ Set environment variables
3. ✅ Test registration
4. ✅ Test sign in
5. ✅ Deploy to production

---

## Optional: Create Registration Page

You can create a registration page in your frontend:

**Example: `apps/facial_recog_web_app/src/app/register/page.tsx`**

```typescript
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, name }),
      });

      if (!res.ok) {
        const data = await res.json();
        setError(data.error || "Registration failed");
        return;
      }

      // Redirect to sign in
      router.push("/api/auth/signin");
    } catch (err) {
      setError("Something went wrong");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
        minLength={6}
      />
      <input
        type="text"
        placeholder="Name (optional)"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />
      {error && <p>{error}</p>}
      <button type="submit">Register</button>
    </form>
  );
}
```

---

## Summary

✅ **Simple**: Just email and password, no external services  
✅ **Secure**: Passwords are hashed with bcryptjs  
✅ **Easy**: One API endpoint for registration  
✅ **No dependencies**: No Google OAuth or other external services needed

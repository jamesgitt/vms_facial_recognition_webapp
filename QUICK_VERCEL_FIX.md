# Quick Fix for Vercel 404 Error

## ⚡ Most Likely Causes (Check These First!)

### 1. **Root Directory is Wrong** (90% of 404 errors)

**Fix:**
1. Vercel Dashboard → Your Project → Settings → General
2. Find "Root Directory"
3. Set to: `apps/facial_recog_web_app`
4. Click "Save"
5. **Redeploy**

---

### 2. **Missing SKIP_ENV_VALIDATION** (80% of build failures)

**Fix:**
1. Vercel Dashboard → Settings → Environment Variables
2. Add new variable:
   - Name: `SKIP_ENV_VALIDATION`
   - Value: `true`
   - Environments: ✅ Production ✅ Preview ✅ Development
3. Click "Save"
4. **Redeploy**

---

### 3. **Build is Failing Silently**

**Check Build Logs:**
1. Vercel Dashboard → Deployments
2. Click on the failed deployment
3. Click "Build Logs" tab
4. **Scroll to the bottom** - look for red errors
5. **Share the error message** if you see one

---

## 🔍 Diagnostic Steps

### Step 1: Verify Root Directory

**In Vercel Dashboard:**
- Settings → General → Root Directory
- **Must be**: `apps/facial_recog_web_app`
- **NOT**: `.` or empty or `./apps/facial_recog_web_app`

---

### Step 2: Check Environment Variables

**Required variables:**
```
SKIP_ENV_VALIDATION=true
DATABASE_URL=postgresql://...
AUTH_SECRET=...
NEXT_PUBLIC_API_URL=https://...
```

**Verify:**
- All are set
- No typos
- Set for all environments (Production, Preview, Development)

---

### Step 3: Check Build Logs

**This is critical!**

1. Go to Deployments tab
2. Click on the deployment with 404
3. Click "Build Logs"
4. Look for:
   - ❌ Red error messages
   - ⚠️ Warnings
   - "Build failed"
   - "Error:"

**Common errors:**
- `Invalid environment variables` → Add `SKIP_ENV_VALIDATION=true`
- `Module not found` → Check dependencies
- `Prisma Client not generated` → Check postinstall script
- `Type error` → Check TypeScript errors

---

### Step 4: Test Build Locally

**Before deploying, test locally:**

```bash
cd apps/facial_recog_web_app

# Create .env.local
cat > .env.local << EOF
SKIP_ENV_VALIDATION=true
DATABASE_URL=postgresql://test:test@localhost:5432/test
AUTH_SECRET=test-secret
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

# Install and build
npm install
npm run build
```

**If local build fails**, fix those errors first!

---

## 🎯 Quick Fix Checklist

Do these in order:

1. [ ] **Root Directory** = `apps/facial_recog_web_app` ✅
2. [ ] **SKIP_ENV_VALIDATION** = `true` ✅
3. [ ] **All environment variables** set ✅
4. [ ] **Build logs checked** for errors ✅
5. [ ] **Redeployed** after fixes ✅

---

## 📋 What to Share for Help

If still not working, share:

1. **Build Logs** (from Vercel)
   - Copy the error message from the bottom of build logs

2. **Root Directory Setting**
   - Screenshot or tell me what it's set to

3. **Environment Variables**
   - List which ones you have set (don't share values!)

4. **Local Build Result**
   - Output of `npm run build` locally

---

## 🚨 Most Common Issue

**Root Directory is wrong or missing `SKIP_ENV_VALIDATION`**

Fix both of these and redeploy - this fixes 90% of 404 errors!

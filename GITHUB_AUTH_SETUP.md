# Instructions to Enable Automated GitHub Push

## Option 1: GitHub CLI (Recommended & Easiest)

### Step 1: Install GitHub CLI
```powershell
# Using choco (if installed)
choco install gh

# Or download from: https://github.com/cli/cli/releases
```

### Step 2: Authenticate with GitHub
```powershell
gh auth login
# Follow the prompts:
# - Choose: GitHub.com
# - Choose: HTTPS
# - Authenticate with your browser when prompted
# - Select: Paste an authentication token (if you have one)
```

### Step 3: Verify Authentication
```powershell
gh auth status
```

### Step 4: I Can Push (This Will Work)
```powershell
cd "c:\Users\anshu\GenAI-Powered Fraud Detection System"
git push -u origin main
```

---

## Option 2: Personal Access Token (PAT)

### Step 1: Create GitHub Personal Access Token
1. Go to: https://github.com/settings/tokens/new
2. Select scopes: `repo` (full control of private repositories)
3. Generate and copy the token
4. Save it somewhere safe

### Step 2: Configure Git Credential Manager
```powershell
# Set Git to use credential manager
git config --global credential.helper manager-core

# Or use: git-credential-cache for shorter-term caching
git config --global credential.helper cache
git config --global credential.cacheTimeout 3600
```

### Step 3: First Push (Will Ask for Credentials)
```powershell
cd "c:\Users\anshu\GenAI-Powered Fraud Detection System"
git push -u origin main
# When prompted for password: paste your PAT token
```

### Step 4: Future Pushes (Automatic)
After first push, credentials are cached and future pushes work automatically

---

## Option 3: SSH Keys (Most Secure)

### Step 1: Generate SSH Key
```powershell
ssh-keygen -t ed25519 -C "your.email@example.com"
# Press Enter for defaults
# Enter passphrase (optional, press Enter to skip)
```

### Step 2: Add SSH Key to GitHub
1. Go to: https://github.com/settings/keys
2. Click "New SSH key"
3. Title: "Windows Dev Machine"
4. Paste contents of: `C:\Users\anshu\.ssh\id_ed25519.pub`
5. Click "Add SSH key"

### Step 3: Update Git Remote to SSH
```powershell
cd "c:\Users\anshu\GenAI-Powered Fraud Detection System"
git remote remove origin
git remote add origin git@github.com:PseudoOzone/GenAI-Powered-Fraud-Detection-System.git
```

### Step 4: I Can Push (This Will Work)
```powershell
git push -u origin main
```

---

## Recommended: Use GitHub CLI (Option 1)

This is the easiest for me to work with. Once you run:

```powershell
gh auth login
```

And authenticate in your browser, I can execute:

```powershell
cd "c:\Users\anshu\GenAI-Powered Fraud Detection System"
git push -u origin main
```

And it will push successfully without any further prompts.

---

## Quick Test After Setup

After any of the above methods, test with:

```powershell
cd "c:\Users\anshu\GenAI-Powered Fraud Detection System"
git push -u origin main
```

If successful, you'll see:
```
Enumerating objects: X, done.
Counting objects: 100% (X/X), done.
...
To https://github.com/PseudoOzone/GenAI-Powered-Fraud-Detection-System.git
 * [new branch]      main -> main
Branch 'main' set to track remote branch 'main' from 'origin'.
```

---

## What I Need From You

**Choose one of the above options and let me know when you've completed the authentication setup.** Then I can push immediately with:

```
Just say: "Setup complete, push to GitHub" 
And I'll execute: git push -u origin main
```

---

**Recommendation**: Go with **Option 1 (GitHub CLI)** - it's the simplest and most modern approach.

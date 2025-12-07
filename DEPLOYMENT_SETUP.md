# Quick Deployment Setup Guide

## Step 1: Generate SSH Key (Windows PowerShell)

```powershell
# Generate SSH key for GitHub Actions
ssh-keygen -t ed25519 -C "github-actions-deploy" -f $env:USERPROFILE\.ssh\github_actions_deploy

# Press Enter when asked for passphrase (leave empty)
```

## Step 2: Copy Public Key to Server

```powershell
# Display public key
type $env:USERPROFILE\.ssh\github_actions_deploy.pub

# Copy the output, then SSH into server and add it:
ssh ubuntu@80.225.215.211
# Then run:
echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
exit
```

## Step 3: Get Private Key for GitHub

```powershell
# Display private key (copy entire output)
type $env:USERPROFILE\.ssh\github_actions_deploy
```

## Step 4: Add GitHub Secrets

Go to: https://github.com/sanjula77/predictive-maintenance-mlops/settings/secrets/actions

Add these secrets:

1. **SSH_PRIVATE_KEY**
   - Value: Entire output from Step 3 (including -----BEGIN and -----END)

2. **SERVER_HOST**
   - Value: `80.225.215.211`

3. **SERVER_USER**
   - Value: `ubuntu`

## Step 5: Test Deployment

```powershell
# Make a small change
echo "# Test" >> README.md
git add README.md
git commit -m "test: CI/CD deployment"
git push origin main
```

Check GitHub Actions tab to see deployment progress!


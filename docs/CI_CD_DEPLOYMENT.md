# CI/CD Deployment Setup Guide

Complete guide to set up automated CI/CD deployment to your Ubuntu server using GitHub Actions.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
- [GitHub Secrets Configuration](#github-secrets-configuration)
- [SSH Key Setup](#ssh-key-setup)
- [Testing the Pipeline](#testing-the-pipeline)
- [Troubleshooting](#troubleshooting)

## Overview

This CI/CD pipeline automatically:
1. ✅ Runs tests and linting on every push
2. ✅ Builds Docker containers
3. ✅ Deploys to your Ubuntu server via SSH
4. ✅ Verifies deployment with health checks
5. ✅ Provides deployment status

## Prerequisites

- GitHub repository with code
- Ubuntu server with SSH access
- Docker and Docker Compose installed on server
- Git installed on server

## Step-by-Step Setup

### Step 1: Generate SSH Key Pair

**On your local Windows machine:**

```powershell
# Open PowerShell
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "github-actions-deploy" -f $env:USERPROFILE\.ssh\github_actions_deploy

# This creates:
# - github_actions_deploy (private key) - for GitHub Secrets
# - github_actions_deploy.pub (public key) - for server
```

**Or use existing SSH key:**
```powershell
# If you already have SSH key for the server
# Use that instead
```

### Step 2: Add Public Key to Ubuntu Server

**Option A: Using existing SSH access**

```powershell
# From your Windows machine
# Copy public key to server
type $env:USERPROFILE\.ssh\github_actions_deploy.pub | ssh ubuntu@80.225.215.211 "cat >> ~/.ssh/authorized_keys"
```

**Option B: Manual copy**

1. Display public key:
   ```powershell
   type $env:USERPROFILE\.ssh\github_actions_deploy.pub
   ```

2. SSH into server:
   ```bash
   ssh ubuntu@80.225.215.211
   ```

3. Add to authorized_keys:
   ```bash
   echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

### Step 3: Test SSH Connection

**From your local machine:**

```powershell
# Test SSH connection
ssh -i $env:USERPROFILE\.ssh\github_actions_deploy ubuntu@80.225.215.211

# Should connect without password
```

### Step 4: Configure GitHub Secrets

**In your GitHub repository:**

1. Go to: **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add these secrets:

#### Required Secrets

**1. SSH_PRIVATE_KEY**
- Name: `SSH_PRIVATE_KEY`
- Value: Content of your **private** key file
  ```powershell
  # Get private key content
  type $env:USERPROFILE\.ssh\github_actions_deploy
  # Copy entire output (including -----BEGIN and -----END lines)
  ```

**2. SERVER_HOST**
- Name: `SERVER_HOST`
- Value: `80.225.215.211`

**3. SERVER_USER**
- Name: `SERVER_USER`
- Value: `ubuntu`

**4. SERVER_PORT** (Optional)
- Name: `SERVER_PORT`
- Value: `22` (default SSH port)

### Step 5: Verify Server Setup

**On your Ubuntu server:**

```bash
# SSH into server
ssh ubuntu@80.225.215.211

# Check Docker
docker --version
docker-compose --version

# Check Git
cd ~/predictive-maintenance-mlops
git remote -v

# Ensure git is set up correctly
git config --global user.name "Deploy Bot"
git config --global user.email "deploy@example.com"
```

### Step 6: Test the Pipeline

**Make a test change:**

```powershell
# On your local machine
cd C:\Users\ASUS\Desktop\predictive-maintenance-mlops

# Make a small change
echo "# CI/CD Test" >> README.md

# Commit and push
git add README.md
git commit -m "test: CI/CD deployment pipeline"
git push origin main
```

**Watch deployment:**

1. Go to GitHub → Your repository
2. Click **Actions** tab
3. Watch the workflow run
4. Check deployment logs

## GitHub Secrets Configuration

### Secret Values

| Secret Name | Value | Example |
|------------|-------|---------|
| `SSH_PRIVATE_KEY` | Private SSH key content | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `SERVER_HOST` | Server IP or domain | `80.225.215.211` |
| `SERVER_USER` | SSH username | `ubuntu` |
| `SERVER_PORT` | SSH port (optional) | `22` |

### How to Get Private Key

**Windows PowerShell:**
```powershell
# Display private key
Get-Content $env:USERPROFILE\.ssh\github_actions_deploy

# Or
type $env:USERPROFILE\.ssh\github_actions_deploy
```

**Copy the entire output**, including:
- `-----BEGIN OPENSSH PRIVATE KEY-----`
- All the key content
- `-----END OPENSSH PRIVATE KEY-----`

## SSH Key Setup

### Generate New Key (Recommended)

```powershell
# Generate dedicated key for CI/CD
ssh-keygen -t ed25519 -C "github-actions" -f $env:USERPROFILE\.ssh\github_actions_deploy

# No passphrase (or GitHub Actions won't work)
# Just press Enter when asked for passphrase
```

### Use Existing Key

If you already have SSH access to the server:

1. Find your existing private key
2. Use that for `SSH_PRIVATE_KEY` secret
3. Make sure public key is in server's `~/.ssh/authorized_keys`

### Verify Key Works

```powershell
# Test connection
ssh -i $env:USERPROFILE\.ssh\github_actions_deploy ubuntu@80.225.215.211 "echo 'SSH connection successful!'"
```

## Testing the Pipeline

### Manual Trigger

1. Go to GitHub → Actions
2. Select "Deploy to Production" workflow
3. Click "Run workflow"
4. Select branch: `main`
5. Click "Run workflow"

### Automatic Trigger

Push to `main` branch:
```powershell
git push origin main
```

### Check Deployment Status

**In GitHub Actions:**
- Green checkmark = Success
- Red X = Failed (check logs)

**On Server:**
```bash
# SSH into server
ssh ubuntu@80.225.215.211

# Check containers
cd ~/predictive-maintenance-mlops
docker-compose ps

# Check logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/health
```

## Workflow Details

### What Happens on Deploy

1. **Pull Latest Code**
   ```bash
   git fetch origin
   git reset --hard origin/main
   ```

2. **Stop Containers**
   ```bash
   docker-compose down
   ```

3. **Build Containers**
   ```bash
   docker-compose build --no-cache
   ```

4. **Start Containers**
   ```bash
   docker-compose up -d
   ```

5. **Health Check**
   ```bash
   curl http://localhost:8000/health
   ```

6. **Verify Deployment**
   - Tests API from GitHub Actions
   - Checks container status
   - Shows recent logs

## Troubleshooting

### SSH Connection Fails

**Error:** `Permission denied (publickey)`

**Solutions:**
1. Verify public key is in `~/.ssh/authorized_keys` on server
2. Check private key in GitHub Secrets (must include BEGIN/END lines)
3. Verify username is correct (`ubuntu`)
4. Check SSH port (default 22)

```bash
# On server, check authorized_keys
cat ~/.ssh/authorized_keys

# Check permissions
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### Deployment Fails

**Error:** `docker-compose: command not found`

**Solution:**
```bash
# Install Docker Compose on server
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Health Check Fails

**Error:** API not responding

**Solutions:**
1. Check container logs:
   ```bash
   docker-compose logs api
   ```

2. Check if port 8000 is accessible:
   ```bash
   curl http://localhost:8000/health
   ```

3. Check firewall:
   ```bash
   sudo ufw status
   sudo ufw allow 8000/tcp
   ```

### Git Pull Fails

**Error:** `Permission denied` or `fatal: not a git repository`

**Solutions:**
```bash
# On server
cd ~/predictive-maintenance-mlops
git remote -v  # Check remote URL
git config --global --add safe.directory ~/predictive-maintenance-mlops
```

## Security Best Practices

1. **Use Dedicated SSH Key**
   - Don't use your personal SSH key
   - Generate key specifically for CI/CD

2. **Limit Key Permissions**
   - Key should only access deployment directory
   - Use restricted user if possible

3. **Rotate Keys Regularly**
   - Change SSH keys periodically
   - Update GitHub Secrets when rotated

4. **Monitor Deployments**
   - Review deployment logs
   - Set up alerts for failures

## Advanced Configuration

### Deploy Only on Tags

Modify workflow trigger:
```yaml
on:
  push:
    tags:
      - 'v*'
```

### Deploy to Staging First

Add staging job:
```yaml
deploy-staging:
  # Deploy to staging server
  # Run tests
  # Then deploy to production
```

### Rollback on Failure

Add rollback step:
```yaml
- name: Rollback on failure
  if: failure()
  run: |
    ssh ... "cd ~/predictive-maintenance-mlops && git checkout HEAD~1 && docker-compose up -d"
```

## Monitoring

### GitHub Actions Status

- View in Actions tab
- Check deployment logs
- See deployment summary

### Server Monitoring

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Check API
curl http://localhost:8000/health
```

## Next Steps

1. ✅ Set up SSH keys
2. ✅ Configure GitHub Secrets
3. ✅ Test deployment
4. ✅ Monitor first few deployments
5. ✅ Set up notifications (optional)

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Check server logs: `docker-compose logs`
3. Verify SSH connection manually
4. Review this guide's troubleshooting section


# SSH Key Setup for CI/CD Deployment

## Common Error: "error in libcrypto"

This error occurs when the SSH private key format is incorrect or corrupted.

## Solution: Generate Correct Key Format

### Option 1: Generate New Key (Recommended)

**On Windows PowerShell:**

```powershell
# Generate OpenSSH format key (ed25519 is recommended)
ssh-keygen -t ed25519 -C "github-actions-deploy" -f $env:USERPROFILE\.ssh\github_actions_deploy

# When prompted for passphrase, press Enter (leave empty)
# CI/CD cannot use keys with passphrases
```

**Verify key format:**
```powershell
# Check first line of private key
Get-Content $env:USERPROFILE\.ssh\github_actions_deploy | Select-Object -First 1

# Should be: -----BEGIN OPENSSH PRIVATE KEY-----
# OR: -----BEGIN RSA PRIVATE KEY-----
```

### Option 2: Convert Existing Key

**If you have a PuTTY key (.ppk):**

1. Install PuTTY tools: https://www.putty.org/
2. Convert to OpenSSH format:
   ```powershell
   puttygen github_actions_deploy.ppk -O private-openssh -o github_actions_deploy
   ```

**If you have an old RSA key:**

```powershell
# Convert to OpenSSH format
ssh-keygen -p -f $env:USERPROFILE\.ssh\github_actions_deploy -m pem
```

## Adding Key to GitHub Secrets

### Step 1: Get Private Key Content

```powershell
# Display entire private key
type $env:USERPROFILE\.ssh\github_actions_deploy
```

### Step 2: Copy to GitHub

1. Go to: https://github.com/sanjula77/predictive-maintenance-mlops/settings/secrets/actions
2. Click "New repository secret"
3. Name: `SSH_PRIVATE_KEY`
4. Value: Paste the **entire** key content including:
   - `-----BEGIN OPENSSH PRIVATE KEY-----`
   - All lines in between
   - `-----END OPENSSH PRIVATE KEY-----`

**Important:**
- ✅ Include BEGIN and END lines
- ✅ Include all lines (even if they look long)
- ✅ Don't add extra spaces or line breaks
- ❌ Don't use PuTTY format (.ppk)
- ❌ Don't use keys with passphrases

### Step 3: Verify Key Format

The key should look like this:

```
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAACFwAAAAdzc2gtcn
NhAAAAAwEAAQAAAgEAy... (many more lines) ...
-----END OPENSSH PRIVATE KEY-----
```

## Add Public Key to Server

```powershell
# Display public key
type $env:USERPROFILE\.ssh\github_actions_deploy.pub

# Copy output, then SSH to server:
ssh ubuntu@80.225.215.211

# On server:
echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

## Test SSH Connection

```powershell
# Test connection
ssh -i $env:USERPROFILE\.ssh\github_actions_deploy ubuntu@80.225.215.211 "echo 'SSH connection successful!'"
```

## Troubleshooting

### Error: "error in libcrypto"

**Causes:**
1. Key format is wrong (PuTTY format instead of OpenSSH)
2. Key has passphrase
3. Key was copied incorrectly (missing lines, extra spaces)

**Solutions:**
1. Generate new key without passphrase
2. Ensure key is in OpenSSH format
3. Copy entire key including BEGIN/END lines
4. Check for hidden characters or encoding issues

### Error: "Permission denied (publickey)"

**Causes:**
1. Public key not in server's authorized_keys
2. Wrong username
3. Wrong server IP

**Solutions:**
1. Verify public key is in `~/.ssh/authorized_keys` on server
2. Check file permissions: `chmod 600 ~/.ssh/authorized_keys`
3. Verify username is `ubuntu`
4. Verify server IP is correct

### Error: "Host key verification failed"

**Solution:**
The workflow automatically adds the server to known_hosts, but if you test manually:
```bash
ssh-keyscan -H 80.225.215.211 >> ~/.ssh/known_hosts
```

## Best Practices

1. **Use dedicated key for CI/CD**
   - Don't use your personal SSH key
   - Generate separate key for deployments

2. **No passphrase**
   - CI/CD cannot handle interactive passphrase prompts
   - Generate key without passphrase

3. **Use ed25519 keys**
   - More secure and smaller than RSA
   - Faster and recommended by OpenSSH

4. **Rotate keys regularly**
   - Change SSH keys periodically
   - Update GitHub Secrets when rotated

5. **Limit key access**
   - Key should only access deployment directory
   - Use restricted user if possible


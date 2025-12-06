# GitHub Actions Workflows

This directory contains CI/CD workflows for the Predictive Maintenance MLOps project.

## Available Workflows

### 1. CI - Lint & Test (`ci.yml`)
**Triggers**: Push to `main`/`develop`, Pull Requests  
**What it does**:
- Runs code linting (flake8, black, isort)
- Performs type checking (mypy)
- Runs unit tests with coverage
- Uploads coverage reports

**Runs on**: Every push and PR

### 2. Build Docker Image (`docker-build.yml`)
**Triggers**: Push to `main` when Docker-related files change  
**What it does**:
- Builds Docker image using Dockerfile
- Tests that the image builds successfully
- Uses Docker Buildx for better caching

**Runs on**: Changes to `src/`, `requirements.txt`, or `Dockerfile`

### 3. Push to Docker Hub (`docker-push.yml`)
**Triggers**: Push to `main` or version tags (`v*`)  
**What it does**:
- Builds and pushes Docker image to Docker Hub
- Tags images with version, branch, and SHA
- Requires Docker Hub credentials in GitHub Secrets

**Required Secrets**:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub access token (not password!)

**Setup Instructions**:
1. Go to GitHub repository Settings → Secrets → Actions
2. Add `DOCKER_USERNAME` secret
3. Add `DOCKER_PASSWORD` secret (create access token at https://hub.docker.com/settings/security)

### 4. Test API (`test-api.yml`)
**Triggers**: Push/PR when API code changes  
**What it does**:
- Runs API unit tests
- Starts the API server
- Tests API endpoints
- Verifies health check

**Runs on**: Changes to `src/api/` or `test_api.py`

### 5. Security Checks (`security.yml`)
**Triggers**: Push, PR, Weekly schedule  
**What it does**:
- Reviews dependencies in PRs
- Scans for security vulnerabilities
- Uses Safety to check Python packages

**Runs on**: Every push, PR, and weekly (Mondays)

## Setting Up Secrets

To enable Docker Hub pushes, add these secrets in GitHub:

1. Navigate to: Repository → Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub access token (create at https://hub.docker.com/settings/security)

## Customization

### Update Badge URLs
In `README.md`, replace `yourusername` with your GitHub username/organization:

```markdown
[![CI](https://github.com/yourusername/predictive-maintenance-mlops/workflows/...)]
```

### Modify Workflows
- Edit workflow files in `.github/workflows/`
- Workflows use YAML syntax
- Test changes in a branch before merging

## Workflow Status

Check workflow status:
- GitHub Actions tab in your repository
- Status badges in README.md
- Individual workflow pages for detailed logs

## Troubleshooting

### Workflow Fails
1. Check the Actions tab for error messages
2. Verify all secrets are set correctly
3. Ensure Python version matches your code
4. Check that test files exist and are valid

### Docker Build Fails
1. Verify Dockerfile syntax
2. Check that all dependencies are in requirements.txt
3. Ensure `.dockerignore` is configured correctly

### Tests Fail
1. Run tests locally: `pytest tests/ -v`
2. Check for missing dependencies
3. Verify test data files exist if needed


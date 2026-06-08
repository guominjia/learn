# GitHub Actions & GitHub Pages: A Practical Deep Dive

## How GitHub Pages Deployment Actually Works (Jekyll + GitHub Actions)

When you use Jekyll with GitHub Actions to build and deploy a site, **the generated HTML is never committed back to any branch in your repository**. This surprises many developers who expect to find a `gh-pages` branch or a `_site` folder in their repo.

The full pipeline looks like this:

```
+--------------------------------------------------+
|  Your Repository (main branch)                   |
|  - README.md, *.md source files                  |
|  - _config.yml, assets, layouts                  |
+--------------------+-----------------------------+
                     |
                     | push triggers GitHub Actions
                     v
+--------------------------------------------------+
|  GitHub Actions (CI/CD pipeline)                 |
|  - Jekyll converts Markdown -> HTML              |
|  - Output lands in ./_site (ephemeral)           |
|  - Packaged as a Pages artifact                  |
+--------------------+-----------------------------+
                     |
                     | deploy-pages action
                     v
+--------------------------------------------------+
|  GitHub Pages Hosting Infrastructure (CDN)       |
|  - Stores final HTML/CSS/JS                      |
|  - Served via *.github.io                        |
|  - Physical location undisclosed by GitHub       |
+--------------------------------------------------+
```

### Workflow Example

```yaml
# .github/workflows/jekyll-gh-pages.yml
- name: Build with Jekyll
  uses: actions/jekyll-build-pages@v1
  with:
    source: ./
    destination: ./_site

- name: Upload artifact
  uses: actions/upload-pages-artifact@v3

- name: Deploy to GitHub Pages
  uses: actions/deploy-pages@v4
```

The HTML only ever exists in two transient places: the runner's `_site` directory during the build, and GitHub's internal Pages CDN after deployment. If you want to inspect the output, download the artifact from the **Actions** tab of a successful run.

### Personal vs. Organization Repositories

GitHub Pages behaves identically for both. The only difference is the URL structure:

| Repository Type | URL |
|---|---|
| User / Org site (`<owner>.github.io`) | `https://<owner>.github.io` |
| Project site (any repo name) | `https://<owner>.github.io/<repo>` |

### Resource Limits to Know

| Limit | Value |
|---|---|
| Published site size | 1 GB |
| Bandwidth (soft) | 100 GB / month |
| Build frequency (soft) | 10 builds / hour (custom Actions workflows are exempt) |

---

## The "Pages build and deployment" Workflow You Did Not Create

If you see a workflow called **Pages build and deployment** running automatically without you defining it, here is why: as long as **Settings > Pages > Source** is set to *Deploy from a branch*, GitHub silently injects this system workflow on every push.

**Fix:** Go to **Settings > Pages** and switch **Source** to *GitHub Actions*. The system workflow disappears and your custom workflow takes full control.

---

## Self-Hosted Runners: How They Actually Work

A self-hosted runner is **a long-lived process on your local machine that long-polls GitHub for jobs**. It is not a webhook receiver — it reaches out to GitHub, not the other way around.

### The Mechanics

1. **Long polling over HTTPS** — After startup, the runner opens a persistent HTTPS connection to GitHub and waits for job assignments. This is an efficient server-push pattern, not a rapid short-poll loop.
2. **Job execution** — When a job arrives, the runner fetches the workflow definition, clones the repository, executes each step, and streams logs back to GitHub in real time.
3. **Idle waiting** — When no jobs are queued, the process stays alive and keeps its connection open.

### Why Jobs Get Stuck in "Queued"

If a workflow uses `runs-on: self-hosted` and the job never starts, the runner process is almost certainly not running or has lost its connection. Check with:

```bash
ps aux | grep Runner.Worker
```

### Managing the Runner as a System Service

Running `./run.sh` directly keeps the runner alive only in your terminal session. For production use, install it as a service:

```bash
# Install and start as a background service
sudo ./svc.sh install
sudo ./svc.sh start

# Lifecycle management
sudo ./svc.sh status
sudo ./svc.sh stop
sudo ./svc.sh restart

# Remove the service entirely
sudo ./svc.sh uninstall
```

To check which user account the service runs under:

```bash
systemctl show actions.runner.* -p User
```

### Alternative: Use GitHub-Hosted Runners

If you do not need local hardware, swap `runs-on: self-hosted` for `runs-on: ubuntu-latest` (or `windows-latest` / `macos-latest`). GitHub provisions and tears down the VM automatically — no local process to maintain.

---

## Fixing the EACCES Permission Error with actions/jekyll-build-pages

`actions/jekyll-build-pages@v1` runs Jekyll inside a Docker container **as root**. Files written to `_site` end up owned by `root`, which causes the subsequent `upload-pages-artifact` step to fail with `EACCES` when it tries to clean up temp files.

**Fix:** Add a step immediately after the Jekyll build to correct ownership:

```yaml
- name: Build with Jekyll
  uses: actions/jekyll-build-pages@v1
  with:
    source: ./
    destination: ./_site

- name: Fix _site ownership
  run: docker run --rm -v "${{ github.workspace }}":/workspace alpine chown -R $(id -u):$(id -g) /workspace/_site

- name: Upload artifact
  uses: actions/upload-pages-artifact@v3
```

This spins up a minimal Alpine container, remounts the workspace, and `chown`s the `_site` directory to the runner UID/GID before the upload step runs.
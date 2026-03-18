## Handling Pages That Require Login with Playwright

When using Playwright to convert web pages to PDF, `wait_until="networkidle"` waits for all network requests to settle. However, if the target page requires authentication, the browser may hang indefinitely or get redirected to a login page.

This post covers two practical approaches to handle login-protected pages.

---

### Approach 1: Save Session After Manual Login (Recommended)

Run a script once to log in manually, save the cookies and storage state, then reuse it for subsequent requests:

````python
from playwright.sync_api import sync_playwright
import json, os

STORAGE_FILE = "auth_state.json"

def save_login_state(login_url: str):
    """Run once: log in manually and save the session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Launch with UI for manual login
        context = browser.new_context()
        page = context.new_page()
        page.goto(login_url)
        
        # Wait for manual login (press Enter in terminal when done)
        print("Please complete the login in the browser, then press Enter to continue...")
        input()
        
        # Save authentication state (cookies + localStorage)
        context.storage_state(path=STORAGE_FILE)
        print(f"Login state saved to {STORAGE_FILE}")
        browser.close()

def webpage_to_pdf(url: str, output_path: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Load saved login state if available
        if os.path.exists(STORAGE_FILE):
            context = browser.new_context(storage_state=STORAGE_FILE)
        else:
            context = browser.new_context()
        
        page = context.new_page()

        # Set a timeout to avoid waiting forever
        page.goto(url, wait_until="networkidle", timeout=30000)
        
        # Check if we were redirected to a login page
        if "login" in page.url or "signin" in page.url:
            print("Authentication required. Please run save_login_state() first.")
            browser.close()
            return

        page.pdf(
            path=output_path,
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"}
        )

        browser.close()
        print(f"PDF saved to {output_path}")

# First time: save login state
# save_login_state("https://example.com/login")

# Subsequent runs: convert page to PDF
webpage_to_pdf(
    "https://example.com",
    "output.pdf"
)
````

**How it works:**

1. `save_login_state()` launches a visible browser where you log in manually.
2. After login, `context.storage_state()` persists all cookies and localStorage to a JSON file.
3. `webpage_to_pdf()` loads that saved state into a new browser context, so the server sees you as already authenticated.

---

### Approach 2: Programmatic Login via Form Submission

If the site uses a standard username/password form, you can automate the login entirely:

````python
def login_and_save_pdf(login_url: str, target_url: str, output_path: str,
                       username: str, password: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to login page
        page.goto(login_url)
        page.fill('input[name="username"]', username)  # Adjust selector as needed
        page.fill('input[name="password"]', password)
        page.click('button[type="submit"]')
        
        # Wait for login to complete (wait for a post-login URL or element)
        page.wait_for_url("**/dashboard**")  # Or use wait_for_selector()
        
        # Navigate to the target page
        page.goto(target_url, wait_until="networkidle", timeout=30000)
        page.pdf(path=output_path, format="A4", print_background=True)
        
        browser.close()

# Example usage
login_and_save_pdf(
    login_url="https://example.com/login",
    target_url="https://example.com",
    output_path="output.pdf",
    username="user@example.com",
    password="your-password"
)
````

**When to use this:** Works well for simple form-based logins. Not suitable for sites that use OAuth, SSO, CAPTCHA, or multi-factor authentication — use Approach 1 for those.

---

### Recommended Workflow

```
First run  → save_login_state()  → Log in manually  → Saves auth_state.json
Later runs → Auto-loads auth_state.json → Directly accesses the target page
```

> **Warning:** `auth_state.json` contains sensitive cookies and tokens. Add it to `.gitignore` and never commit it to version control.
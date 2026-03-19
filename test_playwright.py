from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import os

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
        PREFIX = urlparse(login_url).netloc + "_"
        context.storage_state(path=PREFIX + STORAGE_FILE)
        print(f"Login state saved to {PREFIX + STORAGE_FILE}")
        browser.close()

def webpage_to_pdf(url: str, output_path: str):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Load saved login state if available
        PREFIX = urlparse(url).netloc + "_"
        if os.path.exists(PREFIX + STORAGE_FILE):
            context = browser.new_context(storage_state=PREFIX + STORAGE_FILE)
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

print("Run `playwright install chromium` to install necessary drivers if you haven't already.\n",
      "Then unmask save_login_state() to log in and save the session, followed by webpage_to_pdf() to convert the page to PDF.")

# First time: save login state
# save_login_state("https://example.com/login")

# Subsequent runs: convert page to PDF
# webpage_to_pdf("https://example.com", "output.pdf")
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _find_chrome_binary() -> Optional[Path]:
    env_path = os.getenv("CHROME_BINARY")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Google Chrome 2.app/Contents/MacOS/Google Chrome"),
        Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _get_chromedriver_service() -> Service:
    local_driver = os.getenv("CHROMEDRIVER")
    if local_driver and Path(local_driver).exists():
        return Service(local_driver)
    driver_version = os.getenv("CHROMEDRIVER_VERSION")
    manager = ChromeDriverManager(driver_version=driver_version) if driver_version else ChromeDriverManager()
    return Service(manager.install())


def build_chrome(headless: bool = False, user_agent: Optional[str] = None) -> webdriver.Chrome:
    """Configure and return a Chrome WebDriver."""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={user_agent or DEFAULT_UA}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_binary = _find_chrome_binary()
    if chrome_binary:
        options.binary_location = str(chrome_binary)
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(
        service=_get_chromedriver_service(),
        options=options,
    )
    driver.set_window_size(1280, 900)
    return driver


def human_delay(min_s: float = 1.0, max_s: float = 3.5) -> None:
    time.sleep(random.uniform(min_s, max_s))


def scroll_incremental(
    driver: webdriver.Chrome,
    steps: int = 8,
    step_px: int = 1400,
    pause_range: Tuple[float, float] = (1.0, 3.0),
) -> None:
    """Scrolls down in increments with human-like pauses."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(steps):
        driver.execute_script(f"window.scrollBy(0, {step_px});")
        human_delay(*pause_range)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

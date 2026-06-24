#!/usr/bin/env python3
"""
check_worldbank.py
──────────────────
Run this script on its own to diagnose World Bank API connectivity:

    python check_worldbank.py

It tests the API step by step and prints exactly what is failing,
so you can fix the issue before running the full pipeline.
"""

import sys
import socket

print("=" * 60)
print("  World Bank API Connectivity Diagnostic")
print("=" * 60)

# ── 1. Basic DNS resolution ───────────────────────────────────────
print("\n[1] DNS resolution for api.worldbank.org ...")
try:
    ip = socket.gethostbyname("api.worldbank.org")
    print(f"    OK  -> resolved to {ip}")
except socket.gaierror as e:
    print(f"    FAIL -> DNS error: {e}")
    print("\n    FIX: You have no internet access or DNS is blocked.")
    print("    The pipeline will use built-in fallback data — results")
    print("    will still be valid, just not using live WB figures.")
    sys.exit(1)

# ── 2. TCP connection ─────────────────────────────────────────────
print("\n[2] TCP connection to api.worldbank.org:443 ...")
try:
    s = socket.create_connection(("api.worldbank.org", 443), timeout=8)
    s.close()
    print("    OK  -> TCP connection established")
except OSError as e:
    print(f"    FAIL -> {e}")
    print("\n    FIX: Port 443 (HTTPS) is blocked by a firewall or proxy.")
    print("    Ask your IT team to allow outbound HTTPS to api.worldbank.org,")
    print("    or run the pipeline from a machine without the restriction.")
    sys.exit(1)

# ── 3. HTTP GET with requests ─────────────────────────────────────
print("\n[3] HTTP GET (simple country metadata) ...")
try:
    import requests
except ImportError:
    print("    FAIL -> 'requests' library not installed.")
    print("    Run:  pip install requests")
    sys.exit(1)

try:
    url  = "https://api.worldbank.org/v2/country/US?format=json"
    resp = requests.get(url, timeout=10)
    print(f"    Status code: {resp.status_code}")
    if resp.status_code == 200:
        print("    OK  -> API reachable")
    elif resp.status_code == 403:
        print("    FAIL -> 403 Forbidden")
        print("    CAUSE: Your IP or network is blocked by the World Bank API.")
        print("    This sometimes happens on corporate/VPN networks.")
        print("    FIX options:")
        print("      a) Disable VPN and retry")
        print("      b) Use a different network (e.g. mobile hotspot)")
        print("      c) The pipeline will use fallback data automatically.")
        sys.exit(1)
    elif resp.status_code == 429:
        print("    FAIL -> 429 Too Many Requests (rate limited)")
        print("    FIX: Wait a few minutes and retry.")
        sys.exit(1)
    else:
        print(f"    Unexpected status. Response: {resp.text[:300]}")
        sys.exit(1)
except requests.exceptions.ProxyError as e:
    print(f"    FAIL -> Proxy error: {e}")
    print("\n    FIX: Configure your proxy settings:")
    print("      Option A — set environment variables before running:")
    print('        set HTTPS_PROXY=http://your-proxy:port')
    print('        set HTTP_PROXY=http://your-proxy:port')
    print("      Option B — add proxies= to requests.get() in data_loader.py:")
    print('        PROXIES = {"https": "http://your-proxy:port"}')
    print('        resp = requests.get(url, timeout=20, proxies=PROXIES)')
    sys.exit(1)
except requests.exceptions.SSLError as e:
    print(f"    FAIL -> SSL/TLS error: {e}")
    print("\n    FIX: Your network intercepts HTTPS (common on corporate networks).")
    print("    Ask IT for the corporate CA certificate, or try:")
    print('        resp = requests.get(url, timeout=20, verify=False)')
    print("    (add verify=False to _wb_fetch_indicator in data/data_loader.py)")
    sys.exit(1)
except requests.exceptions.ConnectionError as e:
    print(f"    FAIL -> Connection error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"    FAIL -> {type(e).__name__}: {e}")
    sys.exit(1)

# ── 4. Fetch a real indicator ─────────────────────────────────────
print("\n[4] Fetching GDP per capita for India + Germany (2020-2022) ...")
url2 = ("https://api.worldbank.org/v2/country/IN;DE"
        "/indicator/NY.GDP.PCAP.CD"
        "?date=2020:2022&format=json&per_page=20")
try:
    resp2   = requests.get(url2, timeout=15)
    payload = resp2.json()
    if len(payload) >= 2 and payload[1]:
        for r in payload[1]:
            if r.get("value"):
                print(f"    {r['country']['value']:20s}  {r['date']}  "
                      f"GDP/cap = ${r['value']:,.0f}")
        print("    OK  -> Real data fetched successfully!")
    else:
        print("    WARNING: API responded but returned no data.")
        print(f"    Raw response: {str(payload)[:300]}")
except Exception as e:
    print(f"    FAIL -> {type(e).__name__}: {e}")
    sys.exit(1)

# ── 5. Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  All checks passed. World Bank API is working correctly.")
print("  The pipeline will use live WB socioeconomic data.")
print("=" * 60 + "\n")

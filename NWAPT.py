import os
import sys
import json
import time
import threading
import logging
import argparse
import asyncio
import aiohttp
import random
import socket
import ssl
import re
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.table import Table as RichTable
from rich.progress import Progress
from websocket import create_connection, WebSocketException
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import ParagraphStyle
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

# Initialize Rich Console and Logging
console = Console()
logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("DigitalGraffitiHacker")

# Digital Graffiti Hacker Logo
LOGO = """
[bold cyan]
    ███╗   ██╗    ██╗    ██║  █████╗  ██████╗ ████████╗
    ████╗  ██║    ██║    ██║██╔══██╗ ██╔══██╗  ══██╔══╝
    ██╔██╗ ██║    ██║ █╗ ██║███████║ ██████╔╝    ██║   
    ██║╚██╗██║    ██║███╗██║██╔══██ ║██╔═══╝     ██║   
    ██║ ╚████║    ╚███╔███╔╝██║  ██ ║██║         ██║   
    ╚═╝  ╚═══╝        ╚══╝╚══╝ ╚═╝   ╚═╝╚═╝        ╚═╝   
[/bold cyan]
[green]       [ DIGITAL GRAFFITI HACKER SCANNER v2.0 ]
[/green]
"""

# Payload Manager
class PayloadManager:
    def __init__(self, payloads_file: str = "payloads.json"):
        self.payloads_file = payloads_file
        self.payloads = self.load_payloads()

    def load_payloads(self) -> Dict[str, List[Any]]:
        try:
            with open(self.payloads_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Payloads file {self.payloads_file} not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing payloads file: {e}")
            sys.exit(1)

    def get_payloads(self, vuln_type: str) -> List[Any]:
        return self.payloads.get(vuln_type, [])

# Vulnerability Scanner Core
class VulnerabilityScanner:
    def __init__(self, target_url: str, timeout: int = 10, threads: int = 10, proxy: Optional[str] = None):
        self.target_url = target_url
        self.timeout = timeout
        self.proxy = proxy
        self.max_threads = threads
        self.payload_manager = PayloadManager()
        self.session = None  # Initialize session as None
        self.report = {
            'url': self.target_url,
            'vulnerabilities': [],
            'scan_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_tests': 0,
            'vulnerabilities_found': 0
        }
        self.lock = threading.Lock()
        self.rate_limit = asyncio.Semaphore(self.max_threads)

    async def initialize_session(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(ssl=False)
        )
        if self.proxy:
            self.session._default_headers.update({'Proxy': self.proxy})

    async def close(self):
        if self.session:
            await self.session.close()

    def log_vulnerability(self, vuln_type: str, details: str, severity: str = "High"):
        with self.lock:
            self.report['vulnerabilities'].append({
                'type': vuln_type,
                'details': details,
                'severity': severity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self.report['vulnerabilities_found'] += 1
            logger.info(f"[!] {vuln_type} Found: {details} (Severity: {severity})")

    def save_report(self, output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(output_dir, f"scan_report_{timestamp}.json")
        html_file = os.path.join(output_dir, f"scan_report_{timestamp}.html")
        pdf_file = os.path.join(output_dir, f"scan_report_{timestamp}.pdf")

        # Save JSON report
        with open(json_file, 'w') as f:
            json.dump(self.report, f, indent=4)

        # Generate HTML report
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('report_template.html')
        html_content = template.render(report=self.report)
        with open(html_file, 'w') as f:
            f.write(html_content)

        # Generate PDF report
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        title_style = ParagraphStyle(name='Title', fontSize=18, alignment=1)  # Center alignment
        elements = [Paragraph("Digital Graffiti Hacker Scanner Report", title_style)]
        elements.append(Spacer(1, 12))
        data = [['Vulnerability', 'Details', 'Severity', 'Timestamp']]
        for vuln in self.report['vulnerabilities']:
            data.append([vuln['type'], vuln['details'], vuln['severity'], vuln['timestamp']])
        table = Table(data)
        table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12)
        ])
        elements.append(table)
        doc.build(elements)

        logger.info(f"[+] Reports saved: {json_file}, {html_file}, {pdf_file}")

    async def fetch(self, url: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Tuple[int, str]:
        async with self.rate_limit:
            try:
                if method == "GET":
                    async with self.session.get(url, headers=headers, ssl=False) as response:
                        return response.status, await response.text()
                elif method == "POST":
                    async with self.session.post(url, json=data, headers=headers, ssl=False) as response:
                        return response.status, await response.text()
            except aiohttp.ClientError as e:
                logger.warning(f"Request failed for {url}: {e}")
                return 500, ""

    async def check_sql_injection(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("sql_injection")
        for payload in payloads:
            test_url = f"{self.target_url}?id={urllib.parse.quote(payload)}"
            status, response_text = await self.fetch(test_url)
            if any(error in response_text.lower() for error in ["mysql", "sql syntax", "unclosed quotation", "error in your sql"]):
                self.log_vulnerability("SQL Injection", f"Potential SQL Injection with payload: {payload}")
            progress.advance(task_id)

    async def check_xss(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("xss")
        for payload in payloads:
            test_url = f"{self.target_url}?q={urllib.parse.quote(payload)}"
            status, response_text = await self.fetch(test_url)
            if payload in response_text or "script" in response_text.lower():
                self.log_vulnerability("XSS", f"Potential XSS vulnerability with payload: {payload}")
            progress.advance(task_id)

    async def check_csrf(self, progress: Progress, task_id: int):
        status, response_text = await self.fetch(self.target_url)
        soup = BeautifulSoup(response_text, 'html.parser')
        forms = soup.find_all('form')
        for form in forms:
            if not form.find('input', {'name': re.compile('csrf|token', re.I)}):
                self.log_vulnerability("CSRF", "Form without CSRF token detected", "Medium")
        progress.advance(task_id)

    async def check_cors(self, progress: Progress, task_id: int):
        headers = {'Origin': 'http://malicious.com'}
        async with self.session.get(self.target_url, headers=headers, ssl=False) as response:
            acao = response.headers.get('Access-Control-Allow-Origin', '')
            if acao == '*' or 'malicious.com' in acao:
                self.log_vulnerability("CORS", f"Misconfigured CORS policy: {acao}")
        progress.advance(task_id)

    async def check_auth_bypass(self, progress: Progress, task_id: int):
        endpoints = ["/admin", "/login", "/dashboard"]
        for endpoint in endpoints:
            status, response_text = await self.fetch(f"{self.target_url}{endpoint}")
            if status == 200 and "login" not in response_text.lower():
                self.log_vulnerability("Authentication Bypass", f"Access to {endpoint} without authentication")
        progress.advance(task_id)

    async def check_request_smuggling(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("request_smuggling")
        for payload in payloads:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                host = urllib.parse.urlparse(self.target_url).netloc
                sock.connect((host, 80))
                sock.send(payload.encode())
                response = sock.recv(4096).decode()
                if "admin" in response.lower() or "error" in response.lower():
                    self.log_vulnerability("Request Smuggling", f"Potential HTTP Request Smuggling with payload: {payload}")
                sock.close()
            except (socket.timeout, socket.error) as e:
                logger.warning(f"Error during Request Smuggling test: {e}")
        progress.advance(task_id)

    async def check_deserialization(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("deserialization")
        for payload in payloads:
            status, response_text = await self.fetch(self.target_url, method="POST", data={'data': payload})
            if "stdClass" in response_text or status == 500:
                self.log_vulnerability("Deserialization", f"Potential insecure deserialization with payload: {payload}")
        progress.advance(task_id)

    async def check_web_cache_poisoning(self, progress: Progress, task_id: int):
        headers = {'X-Forwarded-Host': 'malicious.com', 'X-Custom-Header': 'test'}
        status, response_text = await self.fetch(self.target_url, headers=headers)
        if 'malicious.com' in response_text or 'test' in response_text:
            self.log_vulnerability("Web Cache Poisoning", "Potential cache poisoning vulnerability")
        progress.advance(task_id)

    async def check_web_socket(self, progress: Progress, task_id: int):
        try:
            ws_url = self.target_url.replace("http", "ws")
            ws = create_connection(ws_url, timeout=self.timeout)
            payloads = self.payload_manager.get_payloads("web_socket")
            for payload in payloads:
                ws.send(payload)
                response = ws.recv()
                if response and "error" not in response.lower():
                    self.log_vulnerability("Web Socket", f"WebSocket endpoint exposed with payload: {payload}", "Medium")
            ws.close()
        except WebSocketException as e:
            logger.warning(f"Error during Web Socket test: {e}")
        progress.advance(task_id)

    async def check_business_logic(self, progress: Progress, task_id: int):
        endpoints = ["/cart/add", "/order/place"]
        for endpoint in endpoints:
            status, response_text = await self.fetch(f"{self.target_url}{endpoint}?item=1&qty=1")
            status_2, response_text_2 = await self.fetch(f"{self.target_url}{endpoint}?item=1&qty=-1")
            if status == 200 and status_2 == 200:
                self.log_vulnerability("Business Logic", f"Negative quantity allowed in {endpoint}", "Medium")
        progress.advance(task_id)

    async def check_ssrf(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("ssrf")
        for payload in payloads:
            test_url = f"{self.target_url}?url={urllib.parse.quote(payload)}"
            status, response_text = await self.fetch(test_url)
            if "iam" in response_text.lower() or "instance-id" in response_text.lower():
                self.log_vulnerability("SSRF", f"Potential SSRF vulnerability with payload: {payload}")
        progress.advance(task_id)

    async def check_pci(self, progress: Progress, task_id: int):
        try:
            status, response_text = await self.fetch(self.target_url)
            if status == 200:
                context = ssl.create_default_context()
                host = urllib.parse.urlparse(self.target_url).netloc
                with socket.create_connection((host, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        if 'TLSv1' in ssock.version() or 'SSL' in ssock.version():
                            self.log_vulnerability("PCI Compliance", f"Weak TLS/SSL version detected: {ssock.version()}")
        except Exception as e:
            logger.warning(f"Error during PCI test: {e}")
        progress.advance(task_id)

    async def check_waf(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("waf_bypass")
        for payload in payloads:
            headers = {'User-Agent': payload}
            status, response_text = await self.fetch(self.target_url, headers=headers)
            if not re.search(r'fortiweb|waf|blocked', response_text.lower()):
                self.log_vulnerability("WAF (Fortiweb)", f"No WAF (Fortiweb) detected or bypassed with: {payload}", "Medium")
        progress.advance(task_id)

    async def check_idor(self, progress: Progress, task_id: int):
        test_url = f"{self.target_url}/profile?id=1"
        status, response_text = await self.fetch(test_url)
        test_url_2 = f"{self.target_url}/profile?id=2"
        status_2, response_text_2 = await self.fetch(test_url_2)
        if status == 200 and status_2 == 200 and "user" in response_text_2.lower():
            self.log_vulnerability("IDOR", "Potential IDOR vulnerability: Access to other user's data")
        progress.advance(task_id)

    async def check_xxe(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("xxe")
        for payload in payloads:
            status, response_text = await self.fetch(self.target_url, method="POST", data=payload, headers={'Content-Type': 'application/xml'})
            if "root:" in response_text or "error" in response_text.lower():
                self.log_vulnerability("XXE", f"Potential XXE vulnerability with payload: {payload}")
        progress.advance(task_id)

    async def check_ssti(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("ssti")
        for payload in payloads:
            test_url = f"{self.target_url}?name={urllib.parse.quote(payload)}"
            status, response_text = await self.fetch(test_url)
            if "49" in response_text or "error" in response_text.lower():
                self.log_vulnerability("SSTI", f"Potential SSTI with payload: {payload}")
        progress.advance(task_id)

    async def check_lfi_rfi(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("lfi_rfi")
        for payload in payloads:
            test_url = f"{self.target_url}?file={urllib.parse.quote(payload)}"
            status, response_text = await self.fetch(test_url)
            if "root:" in response_text or "malicious" in response_text:
                vuln_type = "LFI" if "etc/passwd" in payload else "RFI"
                self.log_vulnerability(vuln_type, f"Potential {vuln_type} with payload: {payload}")
        progress.advance(task_id)

    async def check_open_redirect(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("open_redirect")
        for payload in payloads:
            test_url = f"{self.target_url}?redirect={urllib.parse.quote(payload)}"
            async with self.session.get(test_url, allow_redirects=False, ssl=False) as response:
                if response.status in [301, 302] and payload in response.headers.get('Location', ''):
                    self.log_vulnerability("Open Redirect", f"Potential Open Redirect with payload: {payload}")
        progress.advance(task_id)

    async def check_rce(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("rce")
        for payload in payloads:
            test_url = f"{self.target_url}?{payload}"
            status, response_text = await self.fetch(test_url)
            if re.search(r"root|www-data", response_text):
                self.log_vulnerability("RCE", f"Potential RCE with payload: {payload}")
        progress.advance(task_id)

    async def check_graphql_injection(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("graphql_injection")
        for payload in payloads:
            status, response_text = await self.fetch(f"{self.target_url}/graphql", method="POST", data=payload)
            if "user" in response_text and "name" in response_text:
                self.log_vulnerability("GraphQL Injection", f"Potential GraphQL Injection with payload: {payload}")
        progress.advance(task_id)

    async def check_race_condition(self, progress: Progress, task_id: int):
        async def make_request():
            status, response_text = await self.fetch(f"{self.target_url}/update?value={random.randint(1, 1000)}")
            return response_text

        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        status, response_text = await self.fetch(f"{self.target_url}/check")
        if "inconsistent" in response_text.lower() or "error" in response_text.lower():
            self.log_vulnerability("Race Condition", "Potential race condition detected")
        progress.advance(task_id)

    async def check_jwt_issues(self, progress: Progress, task_id: int):
        status, response_text = await self.fetch(self.target_url)
        jwt_pattern = re.compile(r'eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+')
        jwt = jwt_pattern.search(response_text)
        if jwt:
            token = jwt.group(0)
            try:
                header = json.loads(urllib.parse.unquote(token.split('.')[0].replace('-', '+').replace('_', '/')))
                if header.get('alg') == 'none':
                    self.log_vulnerability("JWT Issues", "JWT with 'none' algorithm detected")
            except json.JSONDecodeError:
                logger.warning("Error decoding JWT header")
        progress.advance(task_id)

    async def check_prototype_pollution(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("prototype_pollution")
        for payload in payloads:
            status, response_text = await self.fetch(self.target_url, method="POST", data=payload)
            if "polluted" in response_text.lower():
                self.log_vulnerability("Prototype Pollution", f"Potential Prototype Pollution with payload: {payload}")
        progress.advance(task_id)

    async def check_web_llm_attack(self, progress: Progress, task_id: int):
        payloads = self.payload_manager.get_payloads("web_llm_attack")
        for payload in payloads:
            test_url = f"{self.target_url}?prompt={urllib.parse.quote(payload)}"
            status, response_text = await self.fetch(test_url)
            if "secret" in response_text.lower() or "ignore" in response_text.lower():
                self.log_vulnerability("Web LLM Attack", f"Potential LLM prompt injection with payload: {payload}")
        progress.advance(task_id)

    async def run(self):
        await self.initialize_session()  # Initialize session in async context
        console.print(Text(LOGO, style="bold"))
        logger.info(f"[*] Starting scan on {self.target_url}")
        start_time = time.time()

        tests = [
            ("SQL Injection", self.check_sql_injection),
            ("XSS", self.check_xss),
            ("CSRF", self.check_csrf),
            ("CORS", self.check_cors),
            ("Authentication Bypass", self.check_auth_bypass),
            ("Request Smuggling", self.check_request_smuggling),
            ("Deserialization", self.check_deserialization),
            ("Web Cache Poisoning", self.check_web_cache_poisoning),
            ("Web Socket", self.check_web_socket),
            ("Business Logic", self.check_business_logic),
            ("SSRF", self.check_ssrf),
            ("PCI Compliance", self.check_pci),
            ("WAF (Fortiweb)", self.check_waf),
            ("IDOR", self.check_idor),
            ("XXE", self.check_xxe),
            ("SSTI", self.check_ssti),
            ("LFI/RFI", self.check_lfi_rfi),
            ("Open Redirect", self.check_open_redirect),
            ("RCE", self.check_rce),
            ("GraphQL Injection", self.check_graphql_injection),
            ("Race Condition", self.check_race_condition),
            ("JWT Issues", self.check_jwt_issues),
            ("Prototype Pollution", self.check_prototype_pollution),
            ("Web LLM Attack", self.check_web_llm_attack)
        ]

        with Progress() as progress:
            tasks = {}
            total_tests = sum(len(self.payload_manager.get_payloads(vuln[0].lower().replace(" ", "_"))) for vuln in tests) + len(tests)
            self.report['total_tests'] = total_tests
            task_group = progress.add_task("[cyan]Scanning...", total=total_tests)

            for test_name, test_func in tests:
                tasks[test_name] = progress.add_task(f"[yellow]{test_name}", total=len(self.payload_manager.get_payloads(test_name.lower().replace(" ", "_"))) + 1)

            await asyncio.gather(*(test_func(progress, tasks[test_name]) for test_name, test_func in tests))

        await self.close()
        self.save_report()
        duration = time.time() - start_time
        logger.info(f"[*] Scan completed in {duration:.2f} seconds")
        logger.info(f"[*] Total Tests: {self.report['total_tests']}, Vulnerabilities Found: {self.report['vulnerabilities_found']}")
        sys.exit(0)

# CLI Interface
def parse_args():
    parser = argparse.ArgumentParser(description="Digital Graffiti Hacker Scanner v2.0")
    parser.add_argument("url", help="Target URL to scan (e.g., https://example.com)")
    return parser.parse_args()

# HTML Report Template
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Digital Graffiti Hacker Scanner Report</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; color: #333; }
        h1 { text-align: center; color: #2e7d32; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #2e7d32; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary { margin-top: 20px; font-size: 16px; }
    </style>
</head>
<body>
    <h1>Digital Graffiti Hacker Scanner Report</h1>
    <p><strong>Target URL:</strong> {{ report.url }}</p>
    <p><strong>Scan Time:</strong> {{ report.scan_time }}</p>
    <p><strong>Total Tests:</strong> {{ report.total_tests }}</p>
    <p><strong>Vulnerabilities Found:</strong> {{ report.vulnerabilities_found }}</p>
    <table>
        <tr>
            <th>Vulnerability</th>
            <th>Details</th>
            <th>Severity</th>
            <th>Timestamp</th>
        </tr>
        {% for vuln in report.vulnerabilities %}
        <tr>
            <td>{{ vuln.type }}</td>
            <td>{{ vuln.details }}</td>
            <td>{{ vuln.severity }}</td>
            <td>{{ vuln.timestamp }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

# Main Execution
if __name__ == "__main__":
    args = parse_args()
    if not args.url.startswith(("http://", "https://")):
        logger.error("Target URL must start with http:// or https://")
        sys.exit(1)

    # Save HTML template
    with open("report_template.html", "w") as f:
        f.write(REPORT_TEMPLATE)

    scanner = VulnerabilityScanner(target_url=args.url)
    asyncio.run(scanner.run())

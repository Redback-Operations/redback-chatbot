import os
import re
import ast
import logging
import traceback
from typing import List, Dict, Any
import bandit
from bandit.core import manager as bandit_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedVulnerabilityScanner:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vulnerabilities: List[Dict[str, Any]] = []
        self.code_lines: List[str] = []
        self.ast_tree: ast.AST = None
        self.vulnerability_db = self.load_vulnerability_db()
        self.whitelist = self.load_whitelist()

    def load_vulnerability_db(self):
        # This could be expanded to load from an actual database
        return {
            'requests': {'2.25.0': ['CVE-2021-12345']},
            'django': {'2.2.0': ['CVE-2021-67890']}
        }

    def load_whitelist(self):
        # This could be expanded to load from a configuration file
        return [
            r'print\(.+\)',  # Whitelist print statements
            r'logging\..+',  # Whitelist logging statements
        ]

    def parse_file(self):
        logging.info(f"Parsing file: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.code_lines = file.readlines()
                self.ast_tree = ast.parse(''.join(self.code_lines))
            logging.info(f"File parsed. Total lines: {len(self.code_lines)}")
        except Exception as e:
            logging.error(f"Error parsing file {self.file_path}: {str(e)}")
            raise

    def run_bandit(self):
        try:
            b_mgr = bandit_manager.BanditManager(bandit.config.BanditConfig(), agg_type='file')
            b_mgr.discover_files([self.file_path])
            b_mgr.run_tests()
            return b_mgr.get_issue_list()
        except Exception as e:
            logging.error(f"Error running Bandit: {str(e)}")
            return []

    def add_vulnerability(self, category: str, description: str, line_number: int, severity: str, confidence: str):
        if confidence != 'HIGH':
            return  # Only add high confidence vulnerabilities
        if line_number > 0 and line_number <= len(self.code_lines):
            for pattern in self.whitelist:
                if re.search(pattern, self.code_lines[line_number-1]):
                    return  # Skip whitelisted patterns
        self.vulnerabilities.append({
            'category': category,
            'description': description,
            'line_number': line_number,
            'severity': severity,
            'confidence': confidence
        })
        logging.info(f"Vulnerability added: {category} at line {line_number}")

    def check_hardcoded_secrets(self):
        pattern = re.compile(r'(?i)(password|secret|key|token)\s*=\s*["\'][^"\']+["\']')
        for i, line in enumerate(self.code_lines):
            if match := pattern.search(line):
                # Check if it's not just a variable name containing these words
                if not re.search(r'[A-Z_]+_PASSWORD|[A-Z_]+_SECRET|[A-Z_]+_KEY|[A-Z_]+_TOKEN', match.group(0)):
                    self.add_vulnerability('Hardcoded Secret', f"Potential hardcoded secret: {match.group(0)}", i+1, 'HIGH', 'HIGH')

    def check_sql_injection(self):
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ['execute', 'raw']:
                    for arg in node.args:
                        if isinstance(arg, ast.JoinedStr):
                            self.add_vulnerability('SQL Injection', f"Potential SQL injection: {ast.get_source_segment(self.code_lines, node)}", node.lineno, 'HIGH', 'HIGH')

    def check_xss_vulnerabilities(self):
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ['render', 'render_template_string']:
                    for arg in node.args:
                        if isinstance(arg, (ast.Name, ast.Attribute)):
                            self.add_vulnerability('Cross-Site Scripting (XSS)', f"Potential XSS vulnerability: {ast.get_source_segment(self.code_lines, node)}", node.lineno, 'HIGH', 'HIGH')

    def check_vulnerable_components(self):
        import_pattern = r'(?:from|import)\s+([\w\.]*)(?:\s+import)?'
        for i, line in enumerate(self.code_lines):
            if match := re.search(import_pattern, line):
                lib = match.group(1).split('.')[0]
                if lib in self.vulnerability_db:
                    self.add_vulnerability('Vulnerable Component', f"Potentially vulnerable library: {lib}", i+1, 'HIGH', 'HIGH')

    def perform_taint_analysis(self):
        tainted_vars = set()
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in ['input', 'request.form.get']:
                            tainted_vars.add(target.id)
            elif isinstance(node, ast.Call) and any(isinstance(arg, ast.Name) and arg.id in tainted_vars for arg in node.args):
                self.add_vulnerability('Tainted Variable Usage', f"Potentially tainted variable used: {ast.get_source_segment(self.code_lines, node)}", node.lineno, 'HIGH', 'HIGH')

    def check_ssrf_vulnerabilities(self):
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ['get', 'post', 'put', 'delete', 'head', 'options', 'patch']:
                    for arg in node.args:
                        if isinstance(arg, (ast.Name, ast.Attribute)):
                            try:
                                source_segment = ast.get_source_segment(self.code_lines, node)
                                self.add_vulnerability('SSRF', f"Potential SSRF vulnerability: {source_segment}", node.lineno, 'HIGH', 'HIGH')
                            except Exception as e:
                                logging.warning(f"Error in SSRF check for line {node.lineno}: {str(e)}")

    def analyze(self):
        try:
            self.parse_file()
            self.check_sql_injection()
            self.check_xss_vulnerabilities()
            self.check_hardcoded_secrets()
            self.check_vulnerable_components()
            self.perform_taint_analysis()
            self.check_ssrf_vulnerabilities()

            bandit_issues = self.run_bandit()
            for issue in bandit_issues:
                if issue.confidence == 'HIGH':
                    self.add_vulnerability(f"Bandit: {issue.test_id}", issue.text, issue.lineno, issue.severity, issue.confidence)

            logging.info("Analysis completed successfully")
        except Exception as e:
            logging.error(f"An error occurred during analysis: {str(e)}")
            logging.error(traceback.format_exc())

    def generate_report(self):
        report = f"Advanced Vulnerability Scan Results for {self.file_path}:\n"
        report += f"Total lines of code: {len(self.code_lines)}\n\n"
        report += "Detected Vulnerabilities:\n"
        if not self.vulnerabilities:
            report += "No high-confidence vulnerabilities detected.\n"
        else:
            for vuln in sorted(self.vulnerabilities, key=lambda x: x['severity'], reverse=True):
                report += f"- {vuln['category']}: {vuln['description']}\n"
                report += f"  Severity: {vuln['severity']}, Confidence: {vuln['confidence']}\n"
                if vuln['line_number'] > 0 and vuln['line_number'] <= len(self.code_lines):
                    report += f"  Location: Line {vuln['line_number']}\n"
                    report += f"  Code: {self.code_lines[vuln['line_number']-1].strip()}\n"
                report += "\n"
        return report

def scan_file_or_directory(path):
    if os.path.isfile(path):
        scanner = AdvancedVulnerabilityScanner(path)
        scanner.analyze()
        return scanner.generate_report()
    elif os.path.isdir(path):
        full_report = ""
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        scanner = AdvancedVulnerabilityScanner(file_path)
                        scanner.analyze()
                        full_report += scanner.generate_report() + "\n\n"
                    except Exception as e:
                        logging.error(f"Error scanning file {file_path}: {str(e)}")
        return full_report
    else:
        return f"Error: {path} is not a valid file or directory."

def main():
    try:
        path = "."  # Scan the entire repository
        report = scan_file_or_directory(path)
        with open('security-scan-results.txt', 'w') as f:
            f.write(report)
        print("Security scan completed successfully")
    except Exception as e:
        error_message = f"Error during security scan: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_message)
        with open('security-scan-results.txt', 'w') as f:
            f.write(error_message)

if __name__ == "__main__":
    main()
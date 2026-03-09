#!/usr/bin/env python3
"""
Trivy SAST Report Normalizer

This module transforms raw Trivy scan results into a normalized format
for SecuBot processing. Follows SOLID principles and Clean Code practices.

Author: SecuBot Team
License: MIT
"""

import json
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class SeverityLevel(Enum):
    """Enumeration of vulnerability severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class VulnerabilityType(Enum):
    """Enumeration of vulnerability types."""
    CODE_VULNERABILITY = "code_vulnerability"
    SECRET = "secret"
    IAC_MISCONFIGURATION = "iac_misconfiguration"


class ScanStatus(Enum):
    """Enumeration of scan statuses."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ScanMetadata:
    """Metadata information about the security scan."""
    scan_id: str
    service: str = "trivy-sast"
    repository: str = ""
    branch: str = ""
    commit_sha: str = ""
    scan_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    trivy_version: str = ""
    scan_duration_seconds: float = 0.0


@dataclass
class ScanConfiguration:
    """Configuration used for the security scan."""
    scanners: List[str] = field(default_factory=lambda: ["vuln", "secret", "config"])
    severity_threshold: List[str] = field(
        default_factory=lambda: ["UNKNOWN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
    ignore_unfixed: bool = False
    scan_targets: Dict[str, bool] = field(
        default_factory=lambda: {
            "filesystem": True,
            "secrets": True,
            "iac": True
        }
    )


@dataclass
class ScanSummary:
    """Summary statistics of the security scan."""
    total_vulnerabilities: int = 0
    by_severity: Dict[str, int] = field(
        default_factory=lambda: {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "UNKNOWN": 0
        }
    )
    by_type: Dict[str, int] = field(
        default_factory=lambda: {
            "code_vulnerabilities": 0,
            "secrets": 0,
            "iac_misconfigurations": 0
        }
    )
    scan_status: str = ScanStatus.COMPLETED.value


# Single Responsibility Principle: Each parser handles one type of data
class IParser(ABC):
    """Interface for parsers following Interface Segregation Principle."""

    @abstractmethod
    def parse(self, data: Any) -> Any:
        """Parse raw data into normalized format."""
        pass


class MetadataParser(IParser):
    """Parses scan metadata from Trivy report."""

    def __init__(self, repository: str, branch: str, commit_sha: str):
        self.repository = repository
        self.branch = branch
        self.commit_sha = commit_sha

    def parse(self, trivy_data: Dict[str, Any]) -> ScanMetadata:
        """Parse metadata from Trivy report."""
        metadata = trivy_data.get("Metadata", {})

        return ScanMetadata(
            scan_id=str(uuid.uuid4()),
            repository=self.repository,
            branch=self.branch,
            commit_sha=self.commit_sha,
            trivy_version=metadata.get("Version", "unknown"),
            scan_duration_seconds=self._calculate_duration(metadata)
        )

    def _calculate_duration(self, metadata: Dict[str, Any]) -> float:
        """Calculate scan duration if available."""
        # Trivy doesn't provide duration by default
        # This could be enhanced by tracking workflow execution time
        return 0.0


class VulnerabilityParser(IParser):
    """Parses vulnerability data from Trivy report."""

    def parse(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse vulnerabilities from Trivy results."""
        vulnerabilities = []

        for result in results:
            vulns = result.get("Vulnerabilities", [])
            if not vulns:
                continue

            for vuln in vulns:
                normalized_vuln = self._normalize_vulnerability(vuln, result)
                vulnerabilities.append(normalized_vuln)

        return vulnerabilities

    def _normalize_vulnerability(
        self,
        vuln: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize a single vulnerability."""
        return {
            "id": vuln.get("VulnerabilityID", "UNKNOWN"),
            "type": VulnerabilityType.CODE_VULNERABILITY.value,
            "severity": vuln.get("Severity", SeverityLevel.UNKNOWN.value),
            "title": vuln.get("Title", "No title provided"),
            "description": vuln.get("Description", "No description available"),
            "package": {
                "name": vuln.get("PkgName", "unknown"),
                "version": vuln.get("InstalledVersion", "unknown"),
                "ecosystem": self._determine_ecosystem(result)
            },
            "fix": {
                "available": bool(vuln.get("FixedVersion")),
                "fixed_version": vuln.get("FixedVersion", ""),
                "recommendation": self._generate_fix_recommendation(vuln)
            },
            "location": {
                "file": result.get("Target", "unknown"),
                "line_start": 0,
                "line_end": 0
            },
            "references": vuln.get("References", []),
            "cvss_score": self._extract_cvss_score(vuln),
            "published_date": vuln.get("PublishedDate", "")
        }

    def _determine_ecosystem(self, result: Dict[str, Any]) -> str:
        """Determine package ecosystem from result type."""
        target = result.get("Target", "").lower()
        result_type = result.get("Type", "").lower()

        ecosystem_map = {
            "package.json": "npm",
            "package-lock.json": "npm",
            "requirements.txt": "pip",
            "pipfile": "pip",
            "go.mod": "go",
            "pom.xml": "maven",
            "build.gradle": "gradle",
            "composer.json": "composer",
            "gemfile": "rubygems"
        }

        for key, value in ecosystem_map.items():
            if key in target:
                return value

        return result_type if result_type else "unknown"

    def _generate_fix_recommendation(self, vuln: Dict[str, Any]) -> str:
        """Generate fix recommendation based on vulnerability data."""
        fixed_version = vuln.get("FixedVersion")
        pkg_name = vuln.get("PkgName", "the package")

        if fixed_version:
            return f"Upgrade {pkg_name} to version {fixed_version}"
        return f"No fix available yet for {pkg_name}. Monitor for updates."

    def _extract_cvss_score(self, vuln: Dict[str, Any]) -> float:
        """Extract CVSS score from vulnerability data."""
        cvss = vuln.get("CVSS", {})

        # Try different CVSS versions
        for version in ["nvd", "redhat", "V3Score"]:
            if version in cvss:
                score_data = cvss[version]
                if isinstance(score_data, dict):
                    return float(score_data.get("V3Score", 0.0))
                return float(score_data) if score_data else 0.0

        return 0.0


class SecretParser(IParser):
    """Parses secret detection results from Trivy report."""

    def parse(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse secrets from Trivy results."""
        secrets = []

        for result in results:
            secret_findings = result.get("Secrets", [])
            if not secret_findings:
                continue

            for secret in secret_findings:
                normalized_secret = self._normalize_secret(secret, result)
                secrets.append(normalized_secret)

        return secrets

    def _normalize_secret(
        self,
        secret: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize a single secret finding."""
        rule_id = secret.get("RuleID", "unknown")

        return {
            "id": f"secret-{rule_id}",
            "type": VulnerabilityType.SECRET.value,
            "severity": secret.get("Severity", SeverityLevel.HIGH.value),
            "title": secret.get("Title", f"Secret detected: {rule_id}"),
            "description": self._generate_secret_description(secret),
            "secret_type": rule_id,
            "location": {
                "file": result.get("Target", "unknown"),
                "line_start": secret.get("StartLine", 0),
                "line_end": secret.get("EndLine", 0)
            },
            "match": secret.get("Match", "***REDACTED***"),
            "recommendation": self._generate_secret_recommendation(rule_id)
        }

    def _generate_secret_description(self, secret: Dict[str, Any]) -> str:
        """Generate description for secret finding."""
        category = secret.get("Category", "credential")
        rule_id = secret.get("RuleID", "unknown")
        return f"{category.capitalize()} detected: {rule_id}"

    def _generate_secret_recommendation(self, rule_id: str) -> str:
        """Generate remediation recommendation for secret."""
        base_recommendation = (
            "Remove hardcoded credentials and use environment variables "
            "or secret management systems like AWS Secrets Manager, "
            "HashiCorp Vault, or GitHub Secrets."
        )

        if "api" in rule_id.lower() or "key" in rule_id.lower():
            return f"API key detected. {base_recommendation}"
        elif "password" in rule_id.lower():
            return f"Password detected. {base_recommendation}"
        elif "token" in rule_id.lower():
            return f"Token detected. {base_recommendation}"

        return base_recommendation


class IaCParser(IParser):
    """Parses Infrastructure as Code misconfiguration results."""

    def parse(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse IaC misconfigurations from Trivy results."""
        misconfigurations = []

        for result in results:
            misconfig_findings = result.get("Misconfigurations", [])
            if not misconfig_findings:
                continue

            for misconfig in misconfig_findings:
                normalized_misconfig = self._normalize_misconfiguration(
                    misconfig, result
                )
                misconfigurations.append(normalized_misconfig)

        return misconfigurations

    def _normalize_misconfiguration(
        self,
        misconfig: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize a single IaC misconfiguration."""
        return {
            "id": misconfig.get("ID", "unknown"),
            "type": VulnerabilityType.IAC_MISCONFIGURATION.value,
            "severity": misconfig.get("Severity", SeverityLevel.MEDIUM.value),
            "title": misconfig.get("Title", "Configuration issue detected"),
            "description": misconfig.get("Description", "No description available"),
            "location": {
                "file": result.get("Target", "unknown"),
                "line_start": self._extract_line_number(misconfig, "start"),
                "line_end": self._extract_line_number(misconfig, "end")
            },
            "iac_type": self._determine_iac_type(result),
            "recommendation": misconfig.get("Resolution", "Review and fix the configuration"),
            "references": misconfig.get("References", [])
        }

    def _extract_line_number(self, misconfig: Dict[str, Any], position: str) -> int:
        """Extract line number from misconfiguration."""
        cause_metadata = misconfig.get("CauseMetadata", {})

        if position == "start":
            return cause_metadata.get("StartLine", 0)
        return cause_metadata.get("EndLine", 0)

    def _determine_iac_type(self, result: Dict[str, Any]) -> str:
        """Determine IaC type from result."""
        target = result.get("Target", "").lower()

        if "dockerfile" in target:
            return "dockerfile"
        elif any(k8s in target for k8s in [".yaml", ".yml", "k8s", "kubernetes"]):
            return "kubernetes"
        elif ".tf" in target or "terraform" in target:
            return "terraform"
        elif "cloudformation" in target:
            return "cloudformation"

        return "unknown"


class SummaryCalculator:
    """Calculates summary statistics from parsed vulnerabilities."""

    def calculate(
        self,
        vulnerabilities: List[Dict[str, Any]],
        secrets: List[Dict[str, Any]],
        iac_misconfigs: List[Dict[str, Any]]
    ) -> ScanSummary:
        """Calculate summary statistics."""
        summary = ScanSummary()

        all_findings = vulnerabilities + secrets + iac_misconfigs
        summary.total_vulnerabilities = len(all_findings)

        # Calculate by severity
        for finding in all_findings:
            severity = finding.get("severity", SeverityLevel.UNKNOWN.value)
            if severity in summary.by_severity:
                summary.by_severity[severity] += 1

        # Calculate by type
        summary.by_type["code_vulnerabilities"] = len(vulnerabilities)
        summary.by_type["secrets"] = len(secrets)
        summary.by_type["iac_misconfigurations"] = len(iac_misconfigs)

        return summary


# Open/Closed Principle: Easy to extend with new normalizers
class TrivyReportNormalizer:
    """
    Main normalizer class that orchestrates the normalization process.
    Follows Dependency Inversion Principle by depending on abstractions (IParser).
    """

    def __init__(
        self,
        repository: str,
        branch: str,
        commit_sha: str
    ):
        self.metadata_parser = MetadataParser(repository, branch, commit_sha)
        self.vulnerability_parser = VulnerabilityParser()
        self.secret_parser = SecretParser()
        self.iac_parser = IaCParser()
        self.summary_calculator = SummaryCalculator()

    def normalize(self, trivy_report_path: Path) -> Dict[str, Any]:
        """Normalize Trivy report to SecuBot format."""
        trivy_data = self._load_report(trivy_report_path)
        results = trivy_data.get("Results", [])

        metadata = self.metadata_parser.parse(trivy_data)
        vulnerabilities = self.vulnerability_parser.parse(results)
        secrets = self.secret_parser.parse(results)
        iac_misconfigurations = self.iac_parser.parse(results)
        summary = self.summary_calculator.calculate(
            vulnerabilities, secrets, iac_misconfigurations
        )

        return self._build_normalized_report(
            metadata,
            ScanConfiguration(),
            summary,
            vulnerabilities,
            secrets,
            iac_misconfigurations
        )

    def _load_report(self, report_path: Path) -> Dict[str, Any]:
        """Load Trivy JSON report from file."""
        if not report_path.exists():
            raise FileNotFoundError(f"Report file not found: {report_path}")

        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_normalized_report(
        self,
        metadata: ScanMetadata,
        configuration: ScanConfiguration,
        summary: ScanSummary,
        vulnerabilities: List[Dict[str, Any]],
        secrets: List[Dict[str, Any]],
        iac_misconfigurations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build the final normalized report structure."""
        return {
            "scan_metadata": {
                "scan_id": metadata.scan_id,
                "service": metadata.service,
                "repository": metadata.repository,
                "branch": metadata.branch,
                "commit_sha": metadata.commit_sha,
                "scan_timestamp": metadata.scan_timestamp,
                "trivy_version": metadata.trivy_version,
                "scan_duration_seconds": metadata.scan_duration_seconds
            },
            "scan_configuration": {
                "scanners": configuration.scanners,
                "severity_threshold": configuration.severity_threshold,
                "ignore_unfixed": configuration.ignore_unfixed,
                "scan_targets": configuration.scan_targets
            },
            "summary": {
                "total_vulnerabilities": summary.total_vulnerabilities,
                "by_severity": summary.by_severity,
                "by_type": summary.by_type,
                "scan_status": summary.scan_status
            },
            "vulnerabilities": vulnerabilities,
            "secrets": secrets,
            "iac_misconfigurations": iac_misconfigurations,
            "artifacts": {
                "raw_report": "trivy_report.json",
                "sarif_report": "trivy_results.sarif",
                "normalized_report": "trivy_normalized.json"
            }
        }


class ReportWriter:
    """Handles writing normalized reports to disk."""

    @staticmethod
    def write(report: Dict[str, Any], output_path: Path) -> None:
        """Write normalized report to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Normalized report written to: {output_path}")


def main():
    """Main entry point for the normalization script."""
    if len(sys.argv) < 5:
        print("Usage: normalize_trivy.py <report_path> <repository> <branch> <commit_sha>")
        sys.exit(1)

    report_path = Path(sys.argv[1])
    repository = sys.argv[2]
    branch = sys.argv[3]
    commit_sha = sys.argv[4]

    try:
        normalizer = TrivyReportNormalizer(repository, branch, commit_sha)
        normalized_report = normalizer.normalize(report_path)

        output_path = report_path.parent / "trivy_normalized.json"
        ReportWriter.write(normalized_report, output_path)

        # Print summary
        summary = normalized_report["summary"]
        print(f"\nScan Summary:")
        print(f"  Total Vulnerabilities: {summary['total_vulnerabilities']}")
        print(f"  Critical: {summary['by_severity']['CRITICAL']}")
        print(f"  High: {summary['by_severity']['HIGH']}")
        print(f"  Medium: {summary['by_severity']['MEDIUM']}")
        print(f"  Low: {summary['by_severity']['LOW']}")

    except Exception as e:
        print(f"Error normalizing report: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

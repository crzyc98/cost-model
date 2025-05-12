# Phase 10 – Data Governance & Security

Ensure the platform meets enterprise standards for data protection & compliance.

---
## 1  PII Handling
- Use **postgres column-level encryption** for SSN, DOB.
- Field-level masking when serving API (last4 SSN only).

## 2  Audit Trails
- All API mutations write to `audit_log` table with actor, before/after diff.

## 3  RBAC & Least Privilege
- Roles: Viewer, Analyst, Admin.
- AWS IAM policies for S3 artifact buckets; principle of least privilege.

## 4  Compliance Automation
- Integrate **Terraform + Open Policy Agent** to enforce tagging, encryption.
- Use **AWS Config rules** to flag drift.

## 5  Disaster Recovery
- Nightly RDS snapshots, weekly cross-region copy.
- Run `chaos engineering` game days.

## 6  Definition of Done
- SOC2 control mapping doc complete.
- Automated compliance checks pass.

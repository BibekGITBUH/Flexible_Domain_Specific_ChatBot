"""
domain/registry.py
--------------------
Maps a domain key -> DomainConfig. To add a new domain:
  1. Create domain/your_domain.py exporting a `CONFIG = DomainConfig(...)`
  2. Register it below.
Nothing in core/ needs to change.
"""

from domain.college_regulations import CONFIG as COLLEGE_REGULATIONS

DOMAIN_REGISTRY = {
    "college_regulations": COLLEGE_REGULATIONS,
    # "hr_policy": HR_POLICY_CONFIG,
    # "gov_services": GOV_SERVICES_CONFIG,
}


def get_domain(key: str):
    if key not in DOMAIN_REGISTRY:
        raise KeyError(
            f"Unknown domain '{key}'. Available: {list(DOMAIN_REGISTRY)}"
        )
    return DOMAIN_REGISTRY[key]

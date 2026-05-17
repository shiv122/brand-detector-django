"""Detector Django project package.

We suppress the RequestsDependencyWarning here because `requests`'s
check_compatibility() asserts urllib3.minor >= 21 (designed for the urllib3
1.21+ era), and modern urllib3 2.x reports minor < 21 (e.g. 2.5 → 5). Same
for chardet 7.x. The runtime is fine — only the cosmetic warning fires.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"^urllib3 .* doesn't match a supported version!.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*chardet .* doesn't match a supported version!.*",
)

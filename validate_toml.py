import sys

import tomli

try:
    with open("pyproject.toml", "rb") as f:
        tomli.load(f)
    print("TOML is valid.")
    sys.exit(0)
except tomli.TOMLDecodeError as e:
    print(f"TOMLDecodeError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

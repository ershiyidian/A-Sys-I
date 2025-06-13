import os
import subprocess
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """
        在构建开始时执行。
        """
        print("--- Running A-Sys-I pre-build hook: Generating C++ headers ---")
        
        project_root = self.root
        script_path = os.path.join(project_root, "scripts", "generate_cpp_header.py")
        
        if not os.path.exists(script_path):
            print(f"Error: Header generation script not found at {script_path}", file=sys.stderr)
            sys.exit(1)
            
        try:
            # 使用与当前构建环境相同的 Python 解释器执行脚本
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            print(result.stdout)
            print("--- C++ header generation successful ---")
        except subprocess.CalledProcessError as e:
            print("!!! FATAL: Failed to generate C++ headers !!!", file=sys.stderr)
            print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
            print(f"Stdout:\n{e.stdout}", file=sys.stderr)
            print(f"Stderr:\n{e.stderr}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"!!! FATAL: An unexpected error occurred during header generation: {e} !!!", file=sys.stderr)
            sys.exit(1)


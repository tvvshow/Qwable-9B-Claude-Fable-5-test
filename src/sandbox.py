"""
CodeGym: Code interaction sandbox for StraTA training.
Provides a Docker-based isolated environment for code tasks.
"""
import os
import json
import subprocess
import tempfile
import shutil
import re
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field


class CodeGym:
    """
    Code interaction environment.
    Interface: reset(task) -> obs, step(action) -> (obs, reward, done), get_total_reward()
    """

    def __init__(self, task: Dict, max_steps: int = 20, sandbox_dir: str = "/tmp/sandbox"):
        self.task = task
        self.max_steps = max_steps
        self.sandbox_dir = sandbox_dir
        self.step_count = 0
        self.last_actions = []
        self.last_observations = []
        self._total_reward = 0.0
        self._done = False
        self._success = False

        # Setup project directory
        self.project_dir = os.path.join(sandbox_dir, f"task_{hash(task['description']) % 10000}")

    def reset(self) -> str:
        """Initialize project from task template, return initial observation"""
        # Clean up any previous run
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)
        os.makedirs(self.project_dir, exist_ok=True)

        # Write project files from template
        if "files" in self.task:
            for filepath, content in self.task["files"].items():
                full_path = os.path.join(self.project_dir, filepath)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)

        self.step_count = 0
        self.last_actions = []
        self.last_observations = []
        self._total_reward = 0.0
        self._done = False
        self._success = False

        return self._get_observation("Project initialized. Ready to start.")

    def step(self, action: str) -> Tuple[str, float, bool]:
        """Execute an action, return (observation, reward, done)"""
        self.step_count += 1
        result = self._execute_action(action)
        observation = self._get_observation(result["output"])
        self.last_actions.append(action)
        self.last_observations.append(result["output"])

        # Check completion
        self._done = self.step_count >= self.max_steps or self._check_success()
        if self._done:
            self._success = self._check_success()
            self._total_reward = self._compute_reward()

        reward = 0.0  # Intermediate steps get 0 reward (sparse terminal reward)
        return observation, reward, self._done

    def get_total_reward(self) -> float:
        """Get the terminal reward for this episode"""
        return self._total_reward

    def _execute_action(self, action: str) -> Dict:
        """Execute a single action in the sandbox"""
        action = action.strip()

        # Parse action type
        if action.startswith("bash:") or action.startswith("$ "):
            # Shell command
            cmd = action.replace("bash:", "").replace("$ ", "").strip()
            return self._run_shell(cmd)
        elif action.startswith("write:") or action.startswith("edit:"):
            # File write/edit
            return self._write_file(action)
        elif action.startswith("read:"):
            # File read
            return self._read_file(action)
        elif action.startswith("create:"):
            # Create file
            return self._create_file(action)
        elif action.startswith("delete:"):
            # Delete file
            return self._delete_file(action)
        elif action.startswith("test:") or action.startswith("pytest"):
            # Run tests
            return self._run_tests(action)
        else:
            # Try as shell command by default
            return self._run_shell(action)

    def _run_shell(self, cmd: str, timeout: int = 15) -> Dict:
        """Run a shell command in the project directory"""
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=self.project_dir,
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "HOME": self.project_dir,
                     "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
            )
            output = ""
            if result.stdout:
                output += result.stdout[-2000:]  # Truncate long output
            if result.stderr:
                output += "\n[stderr] " + result.stderr[-1000:]
            if not output:
                output = "(no output)"
            return {"output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"output": f"[TIMEOUT] Command exceeded {timeout}s limit", "returncode": -1}
        except Exception as e:
            return {"output": f"[ERROR] {str(e)}", "returncode": -1}

    def _write_file(self, action: str) -> Dict:
        """Write content to a file: write:path/to/file\n<content>"""
        parts = action.split("\n", 1)
        filepath = parts[0].replace("write:", "").replace("edit:", "").strip()
        content = parts[1] if len(parts) > 1 else ""
        full_path = os.path.join(self.project_dir, filepath)

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            return {"output": f"File written: {filepath} ({len(content)} bytes)", "returncode": 0}
        except Exception as e:
            return {"output": f"[ERROR] Failed to write {filepath}: {e}", "returncode": -1}

    def _read_file(self, action: str) -> Dict:
        """Read a file: read:path/to/file"""
        filepath = action.replace("read:", "").strip()
        full_path = os.path.join(self.project_dir, filepath)

        try:
            with open(full_path) as f:
                content = f.read()
            if len(content) > 3000:
                content = content[:3000] + "\n... [truncated]"
            return {"output": content, "returncode": 0}
        except FileNotFoundError:
            return {"output": f"[ERROR] File not found: {filepath}", "returncode": -1}

    def _create_file(self, action: str) -> Dict:
        """Create a new file: create:path/to/file\n<content>"""
        return self._write_file(action.replace("create:", "write:"))

    def _delete_file(self, action: str) -> Dict:
        """Delete a file: delete:path/to/file"""
        filepath = action.replace("delete:", "").strip()
        full_path = os.path.join(self.project_dir, filepath)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                return {"output": f"Deleted: {filepath}", "returncode": 0}
            return {"output": f"[WARN] File not found: {filepath}", "returncode": 0}
        except Exception as e:
            return {"output": f"[ERROR] {e}", "returncode": -1}

    def _run_tests(self, action: str) -> Dict:
        """Run tests: test: or pytest [args]"""
        cmd = action.replace("test:", "").strip()
        if not cmd:
            cmd = "python3 -m pytest -v --tb=short 2>&1"
        elif not cmd.startswith("python") and not cmd.startswith("pytest"):
            cmd = f"python3 -m pytest {cmd} -v --tb=short 2>&1"
        return self._run_shell(cmd, timeout=30)

    def _check_success(self) -> bool:
        """Check if task tests pass"""
        if "test_command" in self.task:
            result = self._run_shell(self.task["test_command"], timeout=30)
            return result["returncode"] == 0
        elif "test_files" in self.task:
            # Run pytest on specified test files
            test_files = " ".join(self.task["test_files"])
            result = self._run_shell(f"python3 -m pytest {test_files} -v --tb=short 2>&1", timeout=30)
            return result["returncode"] == 0
        return False

    def _compute_reward(self) -> float:
        """Compute terminal reward with step decay"""
        if self._success:
            decay = max(0.1, 1.0 - 0.03 * self.step_count)
            return decay
        return 0.0

    def _get_observation(self, result_text: str) -> str:
        """Build observation string from current state"""
        # List project files
        file_list = self._list_files()
        obs = f"[Step {self.step_count}/{self.max_steps}]\n"
        obs += f"Project files:\n{file_list}\n\n"
        obs += f"Last action result:\n{result_text}"
        return obs

    def _list_files(self) -> str:
        """List files in project directory"""
        result = self._run_shell("find . -type f -not -path './.git/*' | head -30")
        return result["output"]

    def cleanup(self):
        """Clean up sandbox"""
        if os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir, ignore_errors=True)

"""
Executor Agent

Safely executes FFmpeg commands with timeout and error handling.
"""

import subprocess
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import config


@dataclass
class ExecutionResult:
    """Result of FFmpeg execution."""
    success: bool
    output_path: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


class ExecutorAgent:
    """Safely executes FFmpeg commands."""
    
    def __init__(self, timeout: int = None):
        self.timeout = timeout or config.FFMPEG_TIMEOUT
    
    def execute(self, command: str) -> ExecutionResult:
        """
        Execute an FFmpeg command.
        
        Args:
            command: FFmpeg command to run
            
        Returns:
            ExecutionResult with status and output details
        """
        # Validate command
        if not self._validate_command(command):
            return ExecutionResult(
                success=False,
                error="Invalid command: must be an FFmpeg command"
            )
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                output_path = self._extract_output_path(command)
                
                # Verify file was created
                if output_path and Path(output_path).exists():
                    return ExecutionResult(
                        success=True,
                        output_path=output_path,
                        stdout=result.stdout,
                        stderr=result.stderr
                    )
                else:
                    return ExecutionResult(
                        success=False,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        error="Output file was not created"
                    )
            else:
                return ExecutionResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=f"FFmpeg error (code {result.returncode}): {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Command timed out after {self.timeout}s"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    def _validate_command(self, command: str) -> bool:
        """Check if command is a valid FFmpeg command."""
        cmd_lower = command.lower().strip()
        return cmd_lower.startswith("ffmpeg") and "-i" in command
    
    def _extract_output_path(self, command: str) -> Optional[str]:
        """Extract output path from FFmpeg command."""
        # Output is typically the last argument
        parts = command.split()
        if parts:
            path = parts[-1].strip('"\'')
            return path
        return None

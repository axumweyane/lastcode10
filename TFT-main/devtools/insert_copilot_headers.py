#!/usr/bin/env python3
"""
Automatic Copilot Header Insertion Script
Scans Python files and inserts appropriate Copilot prompts as headers
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Mapping of file patterns to their corresponding Copilot prompts
PROMPT_MAPPINGS = {
    "polygon_data_loader.py": {
        "prompt": '''"""
# COPILOT PROMPT: Implement batch OHLCV fetcher for 100+ symbols with Polygon.io
# Features: auto-throttling (5 req/min), retry on 429, and PostgreSQL caching
# Return: Dict of DataFrames with {symbol: df}
# EXPECTED OUTPUT: Production-ready data loader with rate limiting and caching
# POLYGON INTEGRATION: Aggregates API, fundamental data, error handling
"""''',
        "imports": [
            "import requests",
            "import pandas as pd",
            "from typing import Dict, List",
        ],
    },
    "data_preprocessing.py": {
        "prompt": '''"""
# COPILOT PROMPT: Calculate technical indicators from Polygon's vwap:
# - Volume-weighted RSI(14) using vwap instead of close
# - MACD(12,26,9) using vwap for signal line calculation
# - Bollinger %B(20,2) with vwap-based standard deviation
# Handle corporate actions via Polygon's 'adjusted' flag
# EXPECTED OUTPUT: Enhanced technical indicator functions with VWAP integration
# POLYGON INTEGRATION: VWAP data, adjusted prices, volume-weighted calculations
"""''',
        "imports": [
            "import pandas as pd",
            "import numpy as np",
            "from typing import Tuple",
        ],
    },
    "api_postgres.py": {
        "prompt": '''"""
# COPILOT PROMPT: Create FastAPI endpoint: /polygon/realtime-predict
# Input: List of Polygon-formatted symbols (e.g., 'O:SPY230818C00325000')
# Output: Predictions with Polygon's native symbol format
# Use Polygon's WebSocket client for live data streaming
# EXPECTED OUTPUT: WebSocket-enabled FastAPI endpoint with real-time processing
# POLYGON INTEGRATION: WebSocket streaming, options symbols, real-time predictions
"""''',
        "imports": [
            "from fastapi import FastAPI",
            "import asyncio",
            "from typing import List, Dict",
        ],
    },
    "enhanced_data_pipeline.py": {
        "prompt": '''"""
# COPILOT PROMPT: Process Polygon news into trading features:
# 1. Calculate sentiment polarity score per article using TextBlob
# 2. Compute daily sentiment momentum (3-day rolling change)
# 3. Merge with OHLCV using Polygon's news timestamp alignment
# EXPECTED OUTPUT: Comprehensive news sentiment processing pipeline
# POLYGON INTEGRATION: News API, sentiment analysis, temporal alignment
"""''',
        "imports": [
            "import pandas as pd",
            "from textblob import TextBlob",
            "from datetime import datetime",
        ],
    },
    "realtime_handler.py": {
        "prompt": '''"""
# COPILOT PROMPT: Implement Polygon WebSocket client that:
# 1. Subscribes to specified symbols via Polygon's streaming API
# 2. Updates PostgreSQL every 15 seconds with batched inserts
# 3. Triggers predictions on volume spikes (3x average volume)
# 4. Handles WebSocket reconnection and error recovery
# EXPECTED OUTPUT: Production-ready WebSocket client with health monitoring
# POLYGON INTEGRATION: WebSocket streaming, real-time data, automatic reconnection
"""''',
        "imports": [
            "import asyncio",
            "import websockets",
            "import json",
            "from typing import Set, Dict",
        ],
    },
    "scheduler.py": {
        "prompt": '''"""
# COPILOT PROMPT: Create daily job to:
# 1. Fetch all S&P 500 symbols from Polygon reference API
# 2. Update OHLCV data in parallel threads (10 concurrent)
# 3. Validate corporate action adjustments using splits/dividends API
# EXPECTED OUTPUT: Robust batch processing system with parallel execution
# POLYGON INTEGRATION: Reference API, parallel processing, corporate actions
"""''',
        "imports": [
            "import asyncio",
            "import concurrent.futures",
            "from datetime import datetime",
        ],
    },
    "tft_postgres_model.py": {
        "prompt": '''"""
# COPILOT PROMPT: Add Polygon-specific features to TFT:
# - vwap_ratio: vwap relative to close price
# - news_sentiment_momentum: 3-day sentiment change
# - fundamental_zscore: sector-adjusted fundamental metrics
# Quantize model weights for faster Polygon real-time predictions
# EXPECTED OUTPUT: Enhanced TFT model with Polygon-optimized features
# POLYGON INTEGRATION: VWAP features, sentiment data, model optimization
"""''',
        "imports": [
            "import torch",
            "import pandas as pd",
            "from pytorch_forecasting import TemporalFusionTransformer",
        ],
    },
}

# Generic prompts for files not in specific mappings
GENERIC_PROMPTS = {
    "test_": '''"""
# COPILOT PROMPT: Create comprehensive test suite with:
# - Unit tests for all major functions
# - Mock Polygon.io API responses
# - Validation of data quality and format
# EXPECTED OUTPUT: Production-ready test coverage
"""''',
    "config": '''"""
# COPILOT PROMPT: Configuration management with:
# - Environment variable handling
# - Polygon.io API key management
# - Database connection settings
# EXPECTED OUTPUT: Secure, flexible configuration system
"""''',
    "utils": '''"""
# COPILOT PROMPT: Utility functions for:
# - Polygon symbol format validation
# - Date/time handling for market hours
# - Error handling and logging
# EXPECTED OUTPUT: Reusable utility functions
"""''',
}


class CopilotHeaderInserter:
    """Manages insertion of Copilot prompts into Python files"""

    def __init__(self, root_dir: str, dry_run: bool = False):
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.processed_files = []
        self.skipped_files = []
        self.errors = []

    def scan_python_files(self) -> List[Path]:
        """Scan for Python files in the project"""
        python_files = []

        # Scan main directory
        for file_path in self.root_dir.glob("*.py"):
            if not file_path.name.startswith("."):
                python_files.append(file_path)

        # Scan subdirectories (excluding common ignore patterns)
        ignore_dirs = {".git", "__pycache__", ".pytest_cache", "venv", "env", ".env"}

        for subdir in self.root_dir.iterdir():
            if subdir.is_dir() and subdir.name not in ignore_dirs:
                for file_path in subdir.glob("**/*.py"):
                    if not any(part.startswith(".") for part in file_path.parts):
                        python_files.append(file_path)

        return sorted(python_files)

    def get_prompt_for_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """Get appropriate Copilot prompt for a file"""
        filename = file_path.name

        # Check specific mappings first
        if filename in PROMPT_MAPPINGS:
            return PROMPT_MAPPINGS[filename]

        # Check generic patterns
        for pattern, prompt in GENERIC_PROMPTS.items():
            if pattern in filename:
                return {"prompt": prompt, "imports": []}

        return None

    def has_copilot_header(self, content: str) -> bool:
        """Check if file already has a Copilot prompt header"""
        patterns = [
            r"# COPILOT PROMPT:",
            r"# COPILOT:",
            r'""".*COPILOT.*"""',
            r"'''.*COPILOT.*'''",
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                return True

        return False

    def extract_existing_imports(self, content: str) -> List[str]:
        """Extract existing import statements from file"""
        import_pattern = r"^(import\s+\w+|from\s+\w+\s+import\s+.*?)$"
        imports = re.findall(import_pattern, content, re.MULTILINE)
        return imports

    def insert_prompt_header(
        self, file_path: Path, prompt_data: Dict[str, str]
    ) -> bool:
        """Insert Copilot prompt header into file"""
        try:
            # Read current content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if already has header
            if self.has_copilot_header(content):
                self.skipped_files.append((file_path, "Already has Copilot header"))
                return False

            # Build new content
            new_content = self.build_file_with_header(content, prompt_data)

            if self.dry_run:
                print(f"[DRY RUN] Would update: {file_path}")
                print(f"Header to insert:\n{prompt_data['prompt']}\n")
                return True

            # Write updated content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            self.processed_files.append(file_path)
            return True

        except Exception as e:
            self.errors.append((file_path, str(e)))
            return False

    def build_file_with_header(self, content: str, prompt_data: Dict[str, str]) -> str:
        """Build file content with Copilot header inserted"""
        lines = content.split("\n")

        # Find insertion point (after shebang and initial docstring if present)
        insert_index = 0

        # Skip shebang
        if lines and lines[0].startswith("#!"):
            insert_index = 1

        # Skip file-level docstring
        if insert_index < len(lines):
            if lines[insert_index].strip().startswith('"""') or lines[
                insert_index
            ].strip().startswith("'''"):
                quote = '"""' if '"""' in lines[insert_index] else "'''"

                # Find end of docstring
                if lines[insert_index].count(quote) >= 2:
                    # Single-line docstring
                    insert_index += 1
                else:
                    # Multi-line docstring
                    for i in range(insert_index + 1, len(lines)):
                        if quote in lines[i]:
                            insert_index = i + 1
                            break

        # Insert header
        header_lines = prompt_data["prompt"].split("\n")

        # Add blank line before header if needed
        if insert_index > 0 and lines[insert_index - 1].strip():
            header_lines.insert(0, "")

        # Add blank line after header
        header_lines.append("")

        # Insert header
        new_lines = lines[:insert_index] + header_lines + lines[insert_index:]

        return "\n".join(new_lines)

    def process_files(self, file_paths: List[Path] = None) -> Dict[str, int]:
        """Process files and insert headers"""
        if file_paths is None:
            file_paths = self.scan_python_files()

        stats = {"processed": 0, "skipped": 0, "errors": 0, "no_prompt": 0}

        for file_path in file_paths:
            prompt_data = self.get_prompt_for_file(file_path)

            if prompt_data is None:
                stats["no_prompt"] += 1
                print(f"⚠️  No prompt mapping for: {file_path.name}")
                continue

            if self.insert_prompt_header(file_path, prompt_data):
                stats["processed"] += 1
                if not self.dry_run:
                    print(f"✅ Updated: {file_path}")
            else:
                if file_path in [f[0] for f in self.skipped_files]:
                    stats["skipped"] += 1
                    print(f"⏭️  Skipped: {file_path.name} (already has header)")
                else:
                    stats["errors"] += 1
                    print(f"❌ Error: {file_path}")

        return stats

    def generate_summary_report(self, stats: Dict[str, int]) -> str:
        """Generate summary report of processing"""
        report = [
            "=" * 50,
            "COPILOT HEADER INSERTION SUMMARY",
            "=" * 50,
            f"Files processed: {stats['processed']}",
            f"Files skipped: {stats['skipped']}",
            f"Errors: {stats['errors']}",
            f"No prompt mapping: {stats['no_prompt']}",
            "",
        ]

        if self.processed_files:
            report.extend(
                [
                    "✅ PROCESSED FILES:",
                    *[f"  - {f.name}" for f in self.processed_files],
                    "",
                ]
            )

        if self.skipped_files:
            report.extend(
                [
                    "⏭️  SKIPPED FILES:",
                    *[f"  - {f[0].name}: {f[1]}" for f in self.skipped_files],
                    "",
                ]
            )

        if self.errors:
            report.extend(
                ["❌ ERRORS:", *[f"  - {f[0].name}: {f[1]}" for f in self.errors], ""]
            )

        return "\n".join(report)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Insert Copilot prompts into Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python insert_copilot_headers.py --dry-run
  
  # Process all files
  python insert_copilot_headers.py
  
  # Process specific files
  python insert_copilot_headers.py --files polygon_data_loader.py api_postgres.py
  
  # Custom root directory
  python insert_copilot_headers.py --root-dir /path/to/project
        """,
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory to scan for Python files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="Specific files to process (relative to root-dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if header already exists",
    )
    parser.add_argument(
        "--list-mappings",
        action="store_true",
        help="List all available prompt mappings and exit",
    )

    args = parser.parse_args()

    if args.list_mappings:
        print("Available Copilot Prompt Mappings:")
        print("=" * 40)
        for filename, data in PROMPT_MAPPINGS.items():
            print(f"\n📄 {filename}")
            prompt_preview = data["prompt"].split("\n")[2][:80] + "..."
            print(f"   {prompt_preview}")

        print(f"\nGeneric Patterns:")
        for pattern, prompt in GENERIC_PROMPTS.items():
            print(f"  {pattern}*: {prompt.split()[3][:50]}...")

        return

    # Initialize inserter
    inserter = CopilotHeaderInserter(args.root_dir, dry_run=args.dry_run)

    # Determine files to process
    if args.files:
        file_paths = [Path(args.root_dir) / filename for filename in args.files]
        # Validate files exist
        missing_files = [f for f in file_paths if not f.exists()]
        if missing_files:
            print(f"❌ Files not found: {[f.name for f in missing_files]}")
            return 1
    else:
        file_paths = None  # Process all files

    # Process files
    print(f"🚀 {'[DRY RUN] ' if args.dry_run else ''}Processing Copilot headers...")
    print(f"📂 Root directory: {Path(args.root_dir).resolve()}")
    print()

    stats = inserter.process_files(file_paths)

    # Generate and display report
    report = inserter.generate_summary_report(stats)
    print(report)

    # Return appropriate exit code
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    exit(main())

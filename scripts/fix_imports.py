#!/usr/bin/env python3
"""
Script to remove sys.path.insert statements from all Python files
and ensure proper imports work with the new package structure.
"""

import os
import re
from pathlib import Path

# Pattern to match sys.path.insert lines
SYSPATH_PATTERN = re.compile(
    r'^.*sys\.path\.insert\(.*\).*$\n?',
    re.MULTILINE
)

# Pattern to match import sys if it's only used for sys.path
IMPORT_SYS_PATTERN = re.compile(
    r'^import sys\s*$\n',
    re.MULTILINE
)

def should_remove_import_sys(content: str) -> bool:
    """Check if 'import sys' should be removed (only used for sys.path)"""
    # Check if sys is used for anything other than sys.path.insert
    sys_usages = re.findall(r'\bsys\.((?!path\.insert)\w+)', content)
    return len(sys_usages) == 0

def fix_file(filepath: Path) -> tuple[bool, str]:
    """
    Fix a single Python file by removing sys.path.insert statements.
    Returns (changed, message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Skip if no sys.path.insert
        if 'sys.path.insert' not in original_content:
            return False, "No sys.path.insert found"
        
        # Remove sys.path.insert lines
        modified_content = SYSPATH_PATTERN.sub('', original_content)
        
        # Also remove the related comment if present
        modified_content = re.sub(
            r'^\s*# Add core modules to path\s*$\n',
            '',
            modified_content,
            flags=re.MULTILINE
        )
        
        # Check if we should remove 'import sys'
        if should_remove_import_sys(modified_content):
            modified_content = IMPORT_SYS_PATTERN.sub('', modified_content)
        
        # Write back only if changed
        if modified_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            return True, "Removed sys.path.insert"
        
        return False, "No changes needed"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main function to process all Python files"""
    root_dir = Path('/home/runner/work/CognitionOS/CognitionOS')
    
    # Find all Python files
    python_files = []
    for pattern in ['**/*.py']:
        for filepath in root_dir.glob(pattern):
            # Skip virtual environments and build directories
            if any(part in filepath.parts for part in ['.git', '__pycache__', 'venv', '.venv', 'build', 'dist', '.eggs']):
                continue
            python_files.append(filepath)
    
    print(f"Found {len(python_files)} Python files to check")
    print("=" * 70)
    
    changed_files = []
    error_files = []
    
    for filepath in sorted(python_files):
        changed, message = fix_file(filepath)
        
        if changed:
            changed_files.append(filepath)
            rel_path = filepath.relative_to(root_dir)
            print(f"✓ {rel_path}: {message}")
        elif "Error" in message:
            error_files.append((filepath, message))
            rel_path = filepath.relative_to(root_dir)
            print(f"✗ {rel_path}: {message}")
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Fixed: {len(changed_files)} files")
    print(f"  Errors: {len(error_files)} files")
    
    if error_files:
        print("\nFiles with errors:")
        for filepath, message in error_files:
            rel_path = filepath.relative_to(root_dir)
            print(f"  - {rel_path}: {message}")
    
    print("\nDone! Package structure is ready.")
    print("Next steps:")
    print("  1. Install package in development mode: pip install -e .")
    print("  2. Run tests to verify imports work correctly")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
clear_python_caches.py

Utility script to purge Python import caches, linecache, __pycache__ directories, sys.modules entries, and LRU caches to ensure debuggers can hit breakpoints reliably.
"""

import os
import shutil
import importlib
import sys
import linecache
import gc
import argparse

def clear_pycache(root_dir='.'):
    """
    Recursively remove all __pycache__ directories under root_dir.
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed pycache: {pycache_path}")
            except Exception as e:
                print(f"Failed to remove {pycache_path}: {e}")

def clear_import_and_linecache():
    """
    Invalidate importlib caches and clear linecache to purge stale source lines.
    """
    importlib.invalidate_caches()
    linecache.clearcache()
    print("Invalidated import caches and cleared linecache")

def clear_sys_modules(prefix=None):
    """
    Remove modules from sys.modules optionally filtered by prefix.
    """
    to_remove = [name for name in sys.modules if prefix is None or name.startswith(prefix)]
    for name in to_remove:
        del sys.modules[name]
        print(f"Cleared module: {name}")

def clear_lru_caches():
    """
    Clear caches of any functions decorated with functools.lru_cache.
    """
    removed = 0
    for obj in gc.get_objects():
        if callable(obj) and hasattr(obj, 'cache_clear'):
            try:
                obj.cache_clear()
                removed += 1
            except Exception:
                pass
    print(f"Cleared {removed} lru_cache(s)")

def clear_macos_dot_underscore_files(root_dir='.'):
    """
    Recursively remove all macOS metadata files starting with '._' under root_dir.
    """

    removed = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith('._'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Removed macOS meta file: {file_path}")
                    removed += 1
                except Exception as e:
                    print(f"Failed to remove {file_path}: {e}")
    print(f"Removed {removed} macOS ._ files")

def main():
    parser = argparse.ArgumentParser(
        description="Efface Python caches to ensure breakpoints are hit reliably."
    )
    parser.add_argument('--root', '-r',
                        default='.', help='Project root directory to purge caches')
    parser.add_argument('--module-prefix', '-p',
                        default=None, help='Prefix of modules to clear from sys.modules')
    args = parser.parse_args()

    clear_pycache(args.root)
    clear_import_and_linecache()
    clear_sys_modules(args.module_prefix)
    clear_lru_caches()
    clear_macos_dot_underscore_files(args.root)
    
if __name__ == '__main__':
    main()

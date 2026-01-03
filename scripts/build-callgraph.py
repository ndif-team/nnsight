#!/usr/bin/env python3
"""
Generates call graph data from git history for visualization.

Usage: python scripts/build-callgraph.py
Output: docs/callgraph-data-*.js, docs/callgraph.html

This script analyzes the Python source files in src/nnsight/intervention/
and src/nnsight/remote.py across git history to build a visualization
of how the codebase evolved.
"""

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configuration
REPO_NAME = 'davidbau/nnsight'
SOURCE_FILES = [
    'src/nnsight/intervention/serialization_source.py',
    'src/nnsight/remote.py',
    'src/nnsight/intervention/restricted_execution.py',
]
# For link generation, use the first file as primary
PRIMARY_SOURCE_FILE = 'src/nnsight/intervention/serialization_source.py'


class CallGraphExtractor(ast.NodeVisitor):
    """Extract call graph from Python AST."""

    def __init__(self, source: str, filename: str):
        self.source = source
        self.filename = filename
        self.lines = source.split('\n')

        # Collected definitions
        self.classes: Dict[str, Dict[str, Any]] = {}  # name -> {bases, methods, line, end_line, decorators}
        self.functions: Dict[str, Dict[str, Any]] = {}  # name -> {calls, line, end_line, type, decorators}

        # Current context
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.current_scope_locals: Set[str] = set()

        # Track all defined names for call resolution
        self.all_defined: Set[str] = set()

    def extract(self) -> Tuple[List[Dict], List[Dict]]:
        """Parse and extract call graph."""
        try:
            tree = ast.parse(self.source)
        except SyntaxError as e:
            print(f"  Parse error in {self.filename}: {e}")
            return [], []

        # First pass: collect all definitions
        self.visit(tree)

        # Build nodes and edges
        nodes = []
        edges = []

        # Add class nodes
        for name, data in self.classes.items():
            node = {
                'id': name,
                'type': 'class',
                'line': data['line'],
                'endLine': data['end_line'],
                'file': self.filename,
            }
            if data.get('decorators'):
                node['decorators'] = data['decorators']
            nodes.append(node)

            # Add inheritance edges
            for base in data.get('bases', []):
                if base in self.classes:
                    edges.append({'source': name, 'target': base, 'type': 'extends'})

        # Add function/method nodes and call edges
        for name, data in self.functions.items():
            node = {
                'id': name,
                'type': data['type'],
                'line': data['line'],
                'endLine': data['end_line'],
                'file': self.filename,
            }
            if data.get('decorators'):
                node['decorators'] = data['decorators']
            nodes.append(node)

            # Add call edges
            for callee in data.get('calls', set()):
                if callee in self.functions or callee in self.classes:
                    edges.append({'source': name, 'target': callee, 'type': 'calls'})

        return nodes, edges

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        class_name = node.name

        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)

        self.classes[class_name] = {
            'bases': bases,
            'methods': set(),
            'line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'decorators': decorators,
        }
        self.all_defined.add(class_name)

        # Visit methods within class context
        old_class = self.current_class
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function/method definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        self._visit_function(node, is_async=True)

    def _visit_function(self, node, is_async=False):
        """Common handler for function definitions."""
        if self.current_class:
            # It's a method
            func_name = f"{self.current_class}.{node.name}"
            func_type = 'method'
            if self.current_class in self.classes:
                self.classes[self.current_class]['methods'].add(node.name)
        else:
            func_name = node.name
            func_type = 'function'

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)

        # Track as special type if decorated
        if 'staticmethod' in decorators:
            func_type = 'staticmethod'
        elif 'classmethod' in decorators:
            func_type = 'classmethod'
        elif 'property' in decorators:
            func_type = 'property'

        self.functions[func_name] = {
            'calls': set(),
            'line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'type': func_type,
            'decorators': decorators,
        }
        self.all_defined.add(func_name)

        # Collect local variables in this function
        old_locals = self.current_scope_locals
        self.current_scope_locals = set()
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            self.current_scope_locals.add(arg.arg)
        if node.args.vararg:
            self.current_scope_locals.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.current_scope_locals.add(node.args.kwarg.arg)

        # Visit body to find calls
        old_func = self.current_function
        self.current_function = func_name
        self.generic_visit(node)
        self.current_function = old_func
        self.current_scope_locals = old_locals

    def visit_Call(self, node: ast.Call):
        """Visit function call."""
        if self.current_function is None:
            self.generic_visit(node)
            return

        callee = self._resolve_callee(node.func)
        if callee and self.current_function in self.functions:
            self.functions[self.current_function]['calls'].add(callee)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Track local variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.current_scope_locals.add(target.id)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        """Track walrus operator assignments."""
        if isinstance(node.target, ast.Name):
            self.current_scope_locals.add(node.target.id)
        self.generic_visit(node)

    def _resolve_callee(self, node) -> Optional[str]:
        """Resolve a call target to a function name."""
        if isinstance(node, ast.Name):
            name = node.id
            # Skip if it's a local variable
            if name in self.current_scope_locals:
                return None
            # Check if it's a known function
            if name in self.functions:
                return name
            # Check if it's a class (constructor call)
            if name in self.classes:
                constructor = f"{name}.__init__"
                if constructor in self.functions:
                    return constructor
                return name
            return name  # Return anyway for potential future resolution

        elif isinstance(node, ast.Attribute):
            # Handle self.method() or Class.method() or obj.method()
            attr = node.attr

            if isinstance(node.value, ast.Name):
                obj_name = node.value.id

                # self.method()
                if obj_name == 'self' and self.current_class:
                    method_name = f"{self.current_class}.{attr}"
                    if method_name in self.functions:
                        return method_name
                    # Check parent classes
                    return method_name

                # cls.method() for classmethod
                if obj_name == 'cls' and self.current_class:
                    method_name = f"{self.current_class}.{attr}"
                    return method_name

                # ClassName.method()
                if obj_name in self.classes:
                    method_name = f"{obj_name}.{attr}"
                    return method_name

                # module.function() - just return the attribute
                return attr

            elif isinstance(node.value, ast.Call):
                # chained call like foo().bar()
                return attr

        return None


def extract_callgraph_from_source(source: str, filename: str) -> Tuple[List[Dict], List[Dict]]:
    """Extract call graph from Python source code."""
    extractor = CallGraphExtractor(source, filename)
    return extractor.extract()


def merge_callgraphs(all_nodes: List[Dict], all_edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Merge callgraphs from multiple files, deduplicating nodes."""
    seen_nodes = {}
    merged_edges = []

    for node in all_nodes:
        node_id = node['id']
        if node_id not in seen_nodes:
            seen_nodes[node_id] = node
        else:
            # Keep the one with more info
            existing = seen_nodes[node_id]
            if node.get('decorators') and not existing.get('decorators'):
                seen_nodes[node_id] = node

    # Deduplicate edges
    seen_edges = set()
    for edge in all_edges:
        key = (edge['source'], edge['target'], edge['type'])
        if key not in seen_edges:
            seen_edges.add(key)
            merged_edges.append(edge)

    return list(seen_nodes.values()), merged_edges


def get_file_at_commit(commit: str, filepath: str, repo_root: str) -> Optional[str]:
    """Get file content at a specific commit."""
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{filepath}'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def get_commit_history(repo_root: str, source_files: List[str]) -> List[Dict[str, str]]:
    """Get commit history that touches any of the source files from both main and current branch."""
    commits_set = set()
    commits_list = []

    # Get current branch name
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        current_branch = result.stdout.strip()
    except subprocess.CalledProcessError:
        current_branch = 'HEAD'

    # Branches to search: main and current branch
    branches = ['main', current_branch]
    if current_branch == 'main':
        branches = ['main']

    for branch in branches:
        for source_file in source_files:
            try:
                result = subprocess.run(
                    ['git', 'log', branch, '--format=%H|%ad|%s', '--date=short', '--follow', '--', source_file],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    parts = line.split('|', 2)
                    if len(parts) >= 3:
                        hash_val, date, message = parts
                        if hash_val not in commits_set:
                            commits_set.add(hash_val)
                            commits_list.append({
                                'hash': hash_val,
                                'date': date,
                                'message': message
                            })
            except subprocess.CalledProcessError:
                pass

    # Sort by date (primary) and hash (secondary for stable ordering)
    commits_list.sort(key=lambda x: (x['date'], x['hash']))
    return commits_list


def get_coauthor_flags(hash_val: str, repo_root: str) -> Dict[str, bool]:
    """Check if commit was co-authored by AI assistants."""
    try:
        # Get author email
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ae', hash_val],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        author_email = result.stdout.strip().lower()

        # Get commit body
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%b', hash_val],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        body = result.stdout

        def has_coauthor(name: str) -> bool:
            pattern = re.compile(f'Co-Authored-By.*{name}', re.IGNORECASE)
            return bool(pattern.search(body))

        return {
            'claude': has_coauthor('Claude') or 'anthropic' in author_email,
            'gemini': has_coauthor('Gemini') or 'google' in author_email,
            'codex': has_coauthor('Codex') or 'openai' in author_email,
        }
    except subprocess.CalledProcessError:
        return {'claude': False, 'gemini': False, 'codex': False}


def count_tests(hash_val: str, repo_root: str) -> int:
    """Count test functions at a given commit."""
    try:
        # Get list of test files
        result = subprocess.run(
            ['git', 'ls-tree', '-r', '--name-only', hash_val, '--', 'tests/'],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().split('\n')

        test_count = 0
        for file in files:
            if not file.endswith('.py') or not file:
                continue
            try:
                content = get_file_at_commit(hash_val, file, repo_root)
                if content:
                    # Count def test_ functions
                    matches = re.findall(r'^\s*def\s+test_', content, re.MULTILINE)
                    test_count += len(matches)
            except Exception:
                pass

        return test_count
    except subprocess.CalledProcessError:
        return 0


def hash_content(text: str) -> str:
    """Compute SHA1 hash of content."""
    return hashlib.sha1(text.encode()).hexdigest()


def main():
    print('Extracting call graph history for nnsight source serialization...')

    # Get repo root
    repo_root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    # Get commit history
    commits = get_commit_history(repo_root, SOURCE_FILES)
    print(f'Found {len(commits)} commits')

    timeline = []
    graph_cache = {}  # content_hash -> (nodes, edges, line_count)

    for i, commit in enumerate(commits):
        hash_val = commit['hash']
        print(f'\rProcessing {i + 1}/{len(commits)}: {commit["date"]} {hash_val[:7]}', end='', flush=True)

        # Collect source from all files at this commit
        all_sources = []
        total_lines = 0

        for source_file in SOURCE_FILES:
            content = get_file_at_commit(hash_val, source_file, repo_root)
            if content:
                all_sources.append((source_file, content))
                total_lines += len(content.split('\n'))

        if not all_sources:
            continue

        # Create combined hash for caching
        combined_hash = hash_content(''.join(c for _, c in all_sources))

        if combined_hash in graph_cache:
            nodes, edges, line_count = graph_cache[combined_hash]
        else:
            # Extract callgraph from all files
            all_nodes = []
            all_edges = []

            for filename, content in all_sources:
                try:
                    nodes, edges = extract_callgraph_from_source(content, filename)
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                except Exception as e:
                    print(f'\n  Error processing {filename}: {e}')

            nodes, edges = merge_callgraphs(all_nodes, all_edges)
            line_count = total_lines
            graph_cache[combined_hash] = (nodes, edges, line_count)

        # Get coauthor flags
        coauthors = get_coauthor_flags(hash_val, repo_root)

        # Count tests
        test_count = count_tests(hash_val, repo_root)

        timeline.append({
            'hash': hash_val[:7],
            'date': commit['date'],
            'message': commit['message'][:60],
            'claudeCoauthored': coauthors['claude'],
            'geminiCoauthored': coauthors['gemini'],
            'codexCoauthored': coauthors['codex'],
            'lineCount': line_count,
            'testCount': test_count,
            'nodeCount': len(nodes),
            'edgeCount': len(edges),
            'nodes': nodes,
            'edges': edges,
        })

    print('\n')

    if not timeline:
        print('No commits found!')
        return

    # Ensure docs directory exists
    docs_dir = Path(repo_root) / 'docs'
    docs_dir.mkdir(exist_ok=True)

    # Split timeline into chunks targeting ~25MB per file
    TARGET_CHUNK_SIZE = 25 * 1024 * 1024
    chunks = []
    current_chunk = []
    current_size = 0

    for snapshot in timeline:
        snapshot_json = json.dumps(snapshot)
        snapshot_size = len(snapshot_json)

        if current_size + snapshot_size > TARGET_CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append(snapshot)
        current_size += snapshot_size

    if current_chunk:
        chunks.append(current_chunk)

    # Write each chunk as a JSONP file
    chunk_files = []
    for i, chunk in enumerate(chunks):
        filename = f'callgraph-data-{i}.js'
        output_path = docs_dir / filename
        jsonp = f'loadCallgraphChunk({i}, {json.dumps(chunk)});'
        output_path.write_text(jsonp)
        size_mb = len(jsonp) / (1024 * 1024)
        print(f'Written {filename}: {len(chunk)} snapshots, {size_mb:.2f}MB')
        chunk_files.append(filename)

    # Write manifest
    manifest = {'chunkCount': len(chunks), 'files': chunk_files}
    manifest_path = docs_dir / 'callgraph-manifest.js'
    manifest_jsonp = f'loadCallgraphManifest({json.dumps(manifest)});'
    manifest_path.write_text(manifest_jsonp)

    # Copy/create HTML viewer
    html_template_path = Path(__file__).parent / 'callgraph-template.html'
    html_dest = docs_dir / 'callgraph.html'

    if html_template_path.exists():
        shutil.copy(html_template_path, html_dest)
    else:
        print(f'Warning: Template not found at {html_template_path}, HTML not copied')

    print(f'\nTotal: {len(chunks)} chunk files')
    print(f'Timeline: {len(timeline)} snapshots')
    if timeline:
        print(f'Final graph: {timeline[-1]["nodeCount"]} nodes, {timeline[-1]["edgeCount"]} edges')


if __name__ == '__main__':
    main()

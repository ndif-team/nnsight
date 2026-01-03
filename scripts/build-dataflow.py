#!/usr/bin/env python3
"""
Generates data flow graph from git history for visualization.
Shows function <-> state variable relationships (reads/writes).

Usage: python scripts/build-dataflow.py
Output: docs/dataflow.html, docs/dataflow-data.js
"""

import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configuration
REPO_NAME = 'davidbau/nnsight'
SOURCE_FILES = [
    'src/nnsight/__init__.py',
    'src/nnsight/intervention/backends/remote.py',
    'src/nnsight/intervention/envoy.py',
    'src/nnsight/intervention/restricted_execution.py',
    'src/nnsight/intervention/serialization_source.py',
    'src/nnsight/intervention/serialization.py',
    'src/nnsight/modeling/base.py',
    'src/nnsight/modeling/huggingface.py',
    'src/nnsight/modeling/language.py',
    'src/nnsight/modeling/transformers.py',
    'src/nnsight/remote.py',
    'src/nnsight/schema/request.py',
]


class DataFlowExtractor(ast.NodeVisitor):
    """Extract data flow (function <-> state variable relationships) from Python AST."""

    def __init__(self, source: str):
        self.source = source
        self.lines = source.split('\n')

        # Function data: name -> {reads: Set, writes: Set, line: int, type: str}
        self.functions: Dict[str, Dict] = {}

        # State variables: name -> {line: int, readers: Set, writers: Set, is_const: bool}
        self.state_vars: Dict[str, Dict] = {}

        # Module-level variable names (candidates for state)
        self.module_vars: Set[str] = set()

        # Class-level attribute names
        self.class_attrs: Dict[str, Set[str]] = {}  # class_name -> set of attr names

        # Current context
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.current_function_stack: List[str] = []

        # Track self.attr reads/writes
        self.instance_attrs: Set[str] = set()

    def visit_Module(self, node: ast.Module) -> None:
        """First pass: collect module-level variables and class attributes."""
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        self.module_vars.add(target.id)
                        # Skip if it's a type alias or constant (all uppercase)
                        if not target.id.isupper() and not target.id.startswith('_'):
                            self.state_vars[target.id] = {
                                'line': child.lineno,
                                'readers': set(),
                                'writers': set(),
                                'is_const': False,
                                'scope': 'module'
                            }
            elif isinstance(child, ast.AnnAssign) and child.target:
                if isinstance(child.target, ast.Name):
                    name = child.target.id
                    self.module_vars.add(name)
                    if not name.isupper() and not name.startswith('_'):
                        self.state_vars[name] = {
                            'line': child.lineno,
                            'readers': set(),
                            'writers': set(),
                            'is_const': False,
                            'scope': 'module'
                        }

        # Visit all children
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions and their attributes."""
        old_class = self.current_class
        self.current_class = node.name
        self.class_attrs[node.name] = set()

        # Collect class-level attributes
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attr_name = f"{node.name}.{target.id}"
                        self.class_attrs[node.name].add(target.id)
                        self.state_vars[attr_name] = {
                            'line': child.lineno,
                            'readers': set(),
                            'writers': set(),
                            'is_const': target.id.isupper(),
                            'scope': 'class'
                        }
            elif isinstance(child, ast.AnnAssign) and child.target:
                if isinstance(child.target, ast.Name):
                    attr_name = f"{node.name}.{child.target.id}"
                    self.class_attrs[node.name].add(child.target.id)
                    self.state_vars[attr_name] = {
                        'line': child.lineno,
                        'readers': set(),
                        'writers': set(),
                        'is_const': False,
                        'scope': 'class'
                    }

        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function/method definitions."""
        self._handle_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function/method definitions."""
        self._handle_function(node)

    def _handle_function(self, node) -> None:
        """Common handler for function and async function definitions."""
        if self.current_class:
            func_name = f"{self.current_class}.{node.name}"
            func_type = 'method'
        else:
            func_name = node.name
            func_type = 'function'

        # Skip private methods and dunder methods
        if node.name.startswith('_') and not node.name.startswith('__'):
            func_name = func_name  # Keep it, may be important

        self.functions[func_name] = {
            'reads': set(),
            'writes': set(),
            'line': node.lineno,
            'type': func_type
        }

        # Track context
        old_function = self.current_function
        self.current_function = func_name
        self.current_function_stack.append(func_name)

        self.generic_visit(node)

        self.current_function_stack.pop()
        self.current_function = old_function

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track assignments (writes)."""
        if self.current_function:
            func_data = self.functions.get(self.current_function)
            if func_data:
                for target in node.targets:
                    self._track_write(target, func_data)

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Track augmented assignments (+=, etc.) - both read and write."""
        if self.current_function:
            func_data = self.functions.get(self.current_function)
            if func_data:
                self._track_write(node.target, func_data)
                self._track_read(node.target, func_data)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments."""
        if self.current_function and node.value:
            func_data = self.functions.get(self.current_function)
            if func_data and node.target:
                self._track_write(node.target, func_data)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Track name references (potential reads)."""
        if self.current_function and isinstance(node.ctx, ast.Load):
            func_data = self.functions.get(self.current_function)
            if func_data:
                var_name = node.id
                if var_name in self.state_vars:
                    func_data['reads'].add(var_name)
                    self.state_vars[var_name]['readers'].add(self.current_function)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Track attribute access (self.x, cls.x, ClassName.x)."""
        if self.current_function:
            func_data = self.functions.get(self.current_function)
            if func_data:
                # Check for self.attr or cls.attr
                if isinstance(node.value, ast.Name):
                    if node.value.id in ('self', 'cls') and self.current_class:
                        attr_name = f"{self.current_class}.{node.attr}"
                        if isinstance(node.ctx, ast.Load):
                            # Read from instance/class attribute
                            if attr_name not in self.state_vars:
                                # Create state var entry for instance attrs seen first time
                                self.state_vars[attr_name] = {
                                    'line': node.lineno,
                                    'readers': set(),
                                    'writers': set(),
                                    'is_const': False,
                                    'scope': 'instance'
                                }
                            func_data['reads'].add(attr_name)
                            self.state_vars[attr_name]['readers'].add(self.current_function)
                    elif node.value.id in self.class_attrs:
                        # ClassName.attr access
                        attr_name = f"{node.value.id}.{node.attr}"
                        if attr_name in self.state_vars:
                            if isinstance(node.ctx, ast.Load):
                                func_data['reads'].add(attr_name)
                                self.state_vars[attr_name]['readers'].add(self.current_function)

        self.generic_visit(node)

    def _track_write(self, target, func_data: Dict) -> None:
        """Track a write to a target."""
        if isinstance(target, ast.Name):
            var_name = target.id
            if var_name in self.state_vars:
                func_data['writes'].add(var_name)
                self.state_vars[var_name]['writers'].add(self.current_function)
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name):
                if target.value.id in ('self', 'cls') and self.current_class:
                    attr_name = f"{self.current_class}.{target.attr}"
                    if attr_name not in self.state_vars:
                        self.state_vars[attr_name] = {
                            'line': target.lineno,
                            'readers': set(),
                            'writers': set(),
                            'is_const': False,
                            'scope': 'instance'
                        }
                    func_data['writes'].add(attr_name)
                    self.state_vars[attr_name]['writers'].add(self.current_function)

    def _track_read(self, target, func_data: Dict) -> None:
        """Track a read from a target."""
        if isinstance(target, ast.Name):
            var_name = target.id
            if var_name in self.state_vars:
                func_data['reads'].add(var_name)
                self.state_vars[var_name]['readers'].add(self.current_function)
        elif isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name):
                if target.value.id in ('self', 'cls') and self.current_class:
                    attr_name = f"{self.current_class}.{target.attr}"
                    if attr_name in self.state_vars:
                        func_data['reads'].add(attr_name)
                        self.state_vars[attr_name]['readers'].add(self.current_function)

    def get_dataflow(self) -> Dict[str, Any]:
        """Build and return the data flow graph."""
        nodes = []
        edges = []

        # Add function nodes (only those that interact with state)
        for name, data in self.functions.items():
            if data['reads'] or data['writes']:
                nodes.append({
                    'id': name,
                    'type': 'function',
                    'line': data['line'],
                    'readCount': len(data['reads']),
                    'writeCount': len(data['writes']),
                    'funcType': data['type']
                })

        # Add state variable nodes (only those that are accessed)
        for name, data in self.state_vars.items():
            if data['readers'] or data['writers']:
                nodes.append({
                    'id': name,
                    'type': 'state',
                    'line': data['line'],
                    'isConst': data['is_const'],
                    'readerCount': len(data['readers']),
                    'writerCount': len(data['writers']),
                    'scope': data['scope']
                })

        # Add edges
        edge_set = set()
        for func_name, func_data in self.functions.items():
            # Read edges
            for var_name in func_data['reads']:
                if var_name in self.state_vars and (self.state_vars[var_name]['readers'] or self.state_vars[var_name]['writers']):
                    edge_key = (func_name, var_name, 'read')
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append({
                            'source': func_name,
                            'target': var_name,
                            'type': 'read'
                        })
            # Write edges
            for var_name in func_data['writes']:
                if var_name in self.state_vars:
                    edge_key = (func_name, var_name, 'write')
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append({
                            'source': func_name,
                            'target': var_name,
                            'type': 'write'
                        })

        return {
            'nodes': nodes,
            'edges': edges,
            'funcCount': len(self.functions),
            'stateCount': len(self.state_vars)
        }


def extract_dataflow(source: str) -> Dict[str, Any]:
    """Parse Python source and extract data flow graph."""
    try:
        tree = ast.parse(source)
        extractor = DataFlowExtractor(source)
        extractor.visit(tree)
        return extractor.get_dataflow()
    except SyntaxError as e:
        print(f'  Parse error: {e}')
        return {'nodes': [], 'edges': [], 'funcCount': 0, 'stateCount': 0}


def get_repo_root() -> str:
    """Get the repository root directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return os.getcwd()


def get_file_at_commit(commit_hash: str, file_path: str, repo_root: str) -> Optional[str]:
    """Get file content at a specific commit."""
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit_hash}:{file_path}'],
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
    commits_list = []
    commits_set = set()

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

    # Search both main and current branch
    branches = ['main']
    if current_branch != 'main':
        branches.append(current_branch)

    for branch in branches:
        for source_file in source_files:
            try:
                # Use ISO format for proper time ordering
                result = subprocess.run(
                    ['git', 'log', branch, '--format=%H|%aI|%ad|%s', '--date=short', '--follow', '--', source_file],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        hash_val, iso_time, date, message = parts
                        if hash_val not in commits_set:
                            commits_set.add(hash_val)
                            commits_list.append({
                                'hash': hash_val,
                                'iso_time': iso_time,  # For sorting
                                'date': date,  # For display
                                'message': message
                            })
            except subprocess.CalledProcessError:
                pass

    # Sort by ISO timestamp for proper chronological ordering
    commits_list.sort(key=lambda x: x['iso_time'])
    return commits_list


def get_coauthor_flags(commit_hash: str, repo_root: str) -> Dict[str, bool]:
    """Detect AI co-authors from commit message."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%B', commit_hash],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True
        )
        message = result.stdout.lower()
        return {
            'claudeCoauthored': 'claude' in message or 'anthropic' in message,
            'geminiCoauthored': 'gemini' in message or 'google' in message,
            'codexCoauthored': 'codex' in message or 'openai' in message or 'copilot' in message
        }
    except subprocess.CalledProcessError:
        return {'claudeCoauthored': False, 'geminiCoauthored': False, 'codexCoauthored': False}


def main():
    print('Extracting data flow history for nnsight source serialization...')

    repo_root = get_repo_root()
    commits = get_commit_history(repo_root, SOURCE_FILES)
    print(f'Found {len(commits)} commits')

    if not commits:
        print('No commits found')
        return

    snapshots = []

    for i, commit in enumerate(commits):
        print(f'Processing {i + 1}/{len(commits)}: {commit["date"]} {commit["hash"][:7]}', end='')

        # Combine all source files for this commit
        all_nodes = []
        all_edges = []
        total_lines = 0
        total_funcs = 0
        total_state = 0

        for source_file in SOURCE_FILES:
            content = get_file_at_commit(commit['hash'], source_file, repo_root)
            if content:
                # Add file prefix to node names for clarity
                file_base = os.path.basename(source_file).replace('.py', '')
                result = extract_dataflow(content)

                # Prefix nodes with file name
                for node in result['nodes']:
                    node['id'] = f"{file_base}:{node['id']}"
                    node['file'] = source_file
                    all_nodes.append(node)

                # Prefix edges
                for edge in result['edges']:
                    edge['source'] = f"{file_base}:{edge['source']}"
                    edge['target'] = f"{file_base}:{edge['target']}"
                    all_edges.append(edge)

                total_lines += len(content.split('\n'))
                total_funcs += result['funcCount']
                total_state += result['stateCount']

        coauthor_flags = get_coauthor_flags(commit['hash'], repo_root)

        snapshots.append({
            'hash': commit['hash'][:7],
            'date': commit['date'],
            'message': commit['message'][:80],
            **coauthor_flags,
            'lineCount': total_lines,
            'funcCount': total_funcs,
            'stateCount': total_state,
            'nodeCount': len(all_nodes),
            'edgeCount': len(all_edges),
            'nodes': all_nodes,
            'edges': all_edges
        })

        print(f' - {len(all_nodes)} nodes, {len(all_edges)} edges')

    # Write output
    docs_dir = os.path.join(repo_root, 'docs')
    os.makedirs(docs_dir, exist_ok=True)

    # Copy HTML template
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_source = os.path.join(script_dir, 'dataflow-template.html')
    html_dest = os.path.join(docs_dir, 'dataflow.html')
    if os.path.exists(html_source):
        shutil.copy(html_source, html_dest)
        print(f'Copied {html_dest}')

    # Write data as JSONP
    data = {'snapshots': snapshots}
    jsonp = f'window.dataflowData = {json.dumps(data)};'
    output_path = os.path.join(docs_dir, 'dataflow-data.js')
    with open(output_path, 'w') as f:
        f.write(jsonp)

    print(f'\nWritten {output_path}: {len(snapshots)} snapshots')

    if snapshots:
        last = snapshots[-1]
        func_nodes = len([n for n in last['nodes'] if n['type'] == 'function'])
        state_nodes = len([n for n in last['nodes'] if n['type'] == 'state'])
        print(f'Final graph: {last["nodeCount"]} nodes ({func_nodes} functions, {state_nodes} state vars), {last["edgeCount"]} edges')


if __name__ == '__main__':
    main()

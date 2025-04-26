"""
Terraform AST Analyzer module for advanced semantic analysis of Terraform configurations.
"""

import re
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Generator
import os
import json
from dataclasses import dataclass
import hcl2
import networkx as nx
from collections import defaultdict


@dataclass
class TerraformAttribute:
    """
    Represents a single attribute within a Terraform resource or block.
    """
    path: str  # Full path to the attribute (e.g., "aws_instance.web.ebs_block_device[0].volume_size")
    value: Any  # The value of the attribute
    location: Dict[str, int] = None  # Line numbers where the attribute appears


@dataclass
class TerraformReference:
    """
    Represents a reference from one Terraform resource to another.
    """
    source_path: str  # Path of the resource containing the reference
    target_path: str  # Path of the referenced resource
    attribute_path: str  # Full path of the attribute containing the reference
    reference_type: str  # Type of reference (interpolation, depends_on, etc.)


class TerraformAstAnalyzer:
    """
    Advanced Terraform AST analyzer that provides deep semantic analysis
    of Terraform configurations, including hierarchical relationships,
    resource references, and attribute paths.
    
    This class extends beyond basic HCL parsing to provide comprehensive
    understanding of Terraform's hierarchical structure and semantics.
    """
    
    def __init__(self):
        """Initialize the TerraformAstAnalyzer."""
        self.ast: Dict[str, Any] = {}
        self.attributes: List[TerraformAttribute] = []
        self.references: List[TerraformReference] = []
        self.dependency_graph = nx.DiGraph()
        self.interpolation_pattern = re.compile(r'\${([^}]+)}')
        self.resource_ref_pattern = re.compile(r'([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_.-]+)')
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a complete Terraform file into an AST.
        
        Args:
            file_path: Path to the Terraform file.
            
        Returns:
            Dict containing the parsed AST.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self.parse_hcl(content)
    
    def parse_hcl(self, content: str) -> Dict[str, Any]:
        """
        Parse HCL content into an AST.
        
        Args:
            content: HCL content as a string.
            
        Returns:
            Dict containing the parsed AST.
        """
        try:
            self.ast = hcl2.loads(content)
            self._extract_attributes()
            self._extract_references()
            self._build_dependency_graph()
            return self.ast
        except Exception as e:
            # Handle parsing errors for partial snippets
            # Try to recover by wrapping in a dummy block
            try:
                wrapped_content = f'resource "dummy" "dummy" {{\n{content}\n}}'
                parsed = hcl2.loads(wrapped_content)
                # Extract only the inner content
                if parsed and 'resource' in parsed and 'dummy' in parsed['resource']:
                    self.ast = parsed['resource']['dummy']['dummy']
                    self._extract_attributes(parent_path="dummy_wrapper")
                    self._extract_references(parent_path="dummy_wrapper")
                    return self.ast
            except Exception:
                pass
            
            # If both attempts fail, return an empty dict
            self.ast = {}
            return {}
    
    def parse_diff_fragment(self, diff_fragment: str) -> Dict[str, Any]:
        """
        Parse a Terraform diff fragment into an AST.
        
        Args:
            diff_fragment: A Terraform diff fragment from git.
            
        Returns:
            Dict containing the parsed AST for the fragment.
        """
        # Clean up the diff fragment to get just HCL content
        lines = diff_fragment.split('\n')
        hcl_lines = []
        
        for line in lines:
            # Skip diff control lines
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                continue
                
            # Remove diff markers
            if line.startswith('+'):
                hcl_lines.append(line[1:])
            elif line.startswith('-'):
                # Skip removed lines as they're not part of the new AST
                continue
            else:
                hcl_lines.append(line)
        
        cleaned_content = '\n'.join(hcl_lines)
        return self.parse_hcl(cleaned_content)
    
    def _extract_attributes(self, parent_path: str = "") -> None:
        """
        Extract all attributes from the AST into a flat list with full paths.
        
        Args:
            parent_path: Current path in the AST traversal.
        """
        self.attributes = []
        self._traverse_ast(self.ast, parent_path)
    
    def _traverse_ast(self, node: Any, parent_path: str) -> None:
        """
        Recursively traverse the AST to extract attributes.
        
        Args:
            node: Current node in the AST.
            parent_path: Path to the current node.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                # Handle special Terraform block types
                if key in ('resource', 'data', 'module', 'variable', 'output', 'locals'):
                    if isinstance(value, dict):
                        for type_name, instances in value.items():
                            if isinstance(instances, dict):
                                for instance_name, attributes in instances.items():
                                    new_path = f"{key}.{type_name}.{instance_name}"
                                    self._traverse_ast(attributes, new_path)
                else:
                    new_path = f"{parent_path}.{key}" if parent_path else key
                    if isinstance(value, (dict, list)):
                        self._traverse_ast(value, new_path)
                    else:
                        self.attributes.append(TerraformAttribute(
                            path=new_path,
                            value=value
                        ))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                new_path = f"{parent_path}[{i}]"
                if isinstance(item, (dict, list)):
                    self._traverse_ast(item, new_path)
                else:
                    self.attributes.append(TerraformAttribute(
                        path=new_path,
                        value=item
                    ))
    
    def _extract_references(self, parent_path: str = "") -> None:
        """
        Extract all references between resources in the Terraform configuration.
        
        Args:
            parent_path: Current path in the AST traversal.
        """
        self.references = []
        
        # Process all attributes for references
        for attr in self.attributes:
            if isinstance(attr.value, str):
                # Look for interpolation syntax ${...}
                interpolations = self.interpolation_pattern.findall(attr.value)
                for interp in interpolations:
                    # Look for resource references
                    matches = self.resource_ref_pattern.findall(interp)
                    for match in matches:
                        if len(match) >= 3:
                            resource_type, resource_name, attr_path = match
                            target_path = f"{resource_type}.{resource_name}"
                            self.references.append(TerraformReference(
                                source_path=parent_path if parent_path else attr.path.split('.')[0],
                                target_path=target_path,
                                attribute_path=attr.path,
                                reference_type="interpolation"
                            ))
        
        # Process depends_on
        self._process_depends_on_references()
    
    def _process_depends_on_references(self) -> None:
        """Process 'depends_on' references in the configuration."""
        for resource_type, resources in self.ast.get('resource', {}).items():
            for resource_name, attributes in resources.items():
                source_path = f"resource.{resource_type}.{resource_name}"
                depends_on = attributes.get('depends_on', [])
                
                if isinstance(depends_on, list):
                    for dependency in depends_on:
                        if isinstance(dependency, str):
                            target_match = self.resource_ref_pattern.match(dependency)
                            if target_match:
                                target_type, target_name, _ = target_match.groups()
                                target_path = f"{target_type}.{target_name}"
                                self.references.append(TerraformReference(
                                    source_path=source_path,
                                    target_path=target_path,
                                    attribute_path=f"{source_path}.depends_on",
                                    reference_type="depends_on"
                                ))
    
    def _build_dependency_graph(self) -> None:
        """Build a directed graph of resource dependencies based on references."""
        self.dependency_graph = nx.DiGraph()
        
        # Add nodes for all resources
        for resource_type, resources in self.ast.get('resource', {}).items():
            for resource_name in resources:
                self.dependency_graph.add_node(f"{resource_type}.{resource_name}")
        
        # Add edges for all references
        for ref in self.references:
            if ref.source_path in self.dependency_graph and ref.target_path in self.dependency_graph:
                self.dependency_graph.add_edge(ref.source_path, ref.target_path)
    
    def get_resource_dependencies(self, resource_path: str) -> List[str]:
        """
        Get all resources that the specified resource depends on.
        
        Args:
            resource_path: Path of the resource (e.g., "aws_instance.web").
            
        Returns:
            List of resource paths that the specified resource depends on.
        """
        if resource_path not in self.dependency_graph:
            return []
        
        return list(self.dependency_graph.successors(resource_path))
    
    def get_resource_dependents(self, resource_path: str) -> List[str]:
        """
        Get all resources that depend on the specified resource.
        
        Args:
            resource_path: Path of the resource (e.g., "aws_instance.web").
            
        Returns:
            List of resource paths that depend on the specified resource.
        """
        if resource_path not in self.dependency_graph:
            return []
        
        return list(self.dependency_graph.predecessors(resource_path))
    
    def get_attribute_by_path(self, path: str) -> Optional[TerraformAttribute]:
        """
        Get an attribute by its full path.
        
        Args:
            path: Full path to the attribute.
            
        Returns:
            TerraformAttribute if found, None otherwise.
        """
        for attr in self.attributes:
            if attr.path == path:
                return attr
        return None
    
    def analyze_changes(self, old_content: str, new_content: str) -> Dict[str, List[Dict]]:
        """
        Analyze changes between two Terraform configurations.
        
        Args:
            old_content: The old Terraform configuration.
            new_content: The new Terraform configuration.
            
        Returns:
            Dict with added, removed, and modified attributes.
        """
        old_analyzer = TerraformAstAnalyzer()
        old_analyzer.parse_hcl(old_content)
        
        new_analyzer = TerraformAstAnalyzer()
        new_analyzer.parse_hcl(new_content)
        
        # Find added, removed, and modified attributes
        old_paths = {attr.path: attr.value for attr in old_analyzer.attributes}
        new_paths = {attr.path: attr.value for attr in new_analyzer.attributes}
        
        added = [{"path": path, "value": value} 
                for path, value in new_paths.items() 
                if path not in old_paths]
        
        removed = [{"path": path, "value": value} 
                  for path, value in old_paths.items() 
                  if path not in new_paths]
        
        modified = [{"path": path, "old_value": old_paths[path], "new_value": new_paths[path]} 
                   for path in set(old_paths) & set(new_paths)
                   if old_paths[path] != new_paths[path]]
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified
        }
    
    def resolve_dynamic_blocks(self) -> Dict[str, List[str]]:
        """
        Identify and resolve dynamic blocks in the Terraform configuration.
        
        Returns:
            Dict mapping resource paths to lists of dynamic block paths.
        """
        dynamic_blocks = defaultdict(list)
        
        for resource_type, resources in self.ast.get('resource', {}).items():
            for resource_name, attributes in resources.items():
                resource_path = f"resource.{resource_type}.{resource_name}"
                self._find_dynamic_blocks(attributes, resource_path, dynamic_blocks)
        
        return dict(dynamic_blocks)
    
    def _find_dynamic_blocks(self, node: Dict, parent_path: str, result: Dict[str, List[str]]) -> None:
        """
        Recursively find dynamic blocks in the configuration.
        
        Args:
            node: Current node in the AST.
            parent_path: Path to the current node.
            result: Dictionary to store results.
        """
        if not isinstance(node, dict):
            return
            
        for key, value in node.items():
            if key == "dynamic" and isinstance(value, dict):
                for block_name, block_content in value.items():
                    dynamic_path = f"{parent_path}.dynamic.{block_name}"
                    result[parent_path].append(dynamic_path)
                    
                    # Recursively check for nested dynamic blocks
                    if isinstance(block_content, dict) and "content" in block_content:
                        self._find_dynamic_blocks(block_content["content"], dynamic_path, result)
            elif isinstance(value, dict):
                new_path = f"{parent_path}.{key}"
                self._find_dynamic_blocks(value, new_path, result)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        new_path = f"{parent_path}.{key}[{i}]"
                        self._find_dynamic_blocks(item, new_path, result)
    
    def find_resources_by_type(self, resource_type: str) -> List[str]:
        """
        Find all resources of a specific type.
        
        Args:
            resource_type: Type of resource to find (e.g., "aws_instance").
            
        Returns:
            List of resource paths matching the specified type.
        """
        resources = []
        
        for res_type, instances in self.ast.get('resource', {}).items():
            if res_type == resource_type:
                for instance_name in instances:
                    resources.append(f"{res_type}.{instance_name}")
        
        return resources
    
    def get_resource_count(self, resource_path: str) -> Optional[Union[int, str]]:
        """
        Get the 'count' parameter for a resource if it exists.
        
        Args:
            resource_path: Path of the resource (e.g., "aws_instance.web").
            
        Returns:
            The count value (int or expression string) if it exists, None otherwise.
        """
        parts = resource_path.split('.')
        if len(parts) < 2:
            return None
            
        resource_type, resource_name = parts[0], parts[1]
        
        if (resource_type in self.ast.get('resource', {}) and 
            resource_name in self.ast.get('resource', {}).get(resource_type, {})):
            count = self.ast['resource'][resource_type][resource_name].get('count')
            return count
        
        return None
    
    def get_resource_for_each(self, resource_path: str) -> Optional[Dict]:
        """
        Get the 'for_each' parameter for a resource if it exists.
        
        Args:
            resource_path: Path of the resource (e.g., "aws_instance.web").
            
        Returns:
            The for_each value if it exists, None otherwise.
        """
        parts = resource_path.split('.')
        if len(parts) < 2:
            return None
            
        resource_type, resource_name = parts[0], parts[1]
        
        if (resource_type in self.ast.get('resource', {}) and 
            resource_name in self.ast.get('resource', {}).get(resource_type, {})):
            for_each = self.ast['resource'][resource_type][resource_name].get('for_each')
            return for_each
        
        return None
    
    def to_json(self) -> str:
        """
        Serialize the analyzer's state to JSON.
        
        Returns:
            JSON string representing the current state.
        """
        state = {
            "attributes": [
                {"path": attr.path, "value": str(attr.value)}
                for attr in self.attributes
            ],
            "references": [
                {
                    "source": ref.source_path,
                    "target": ref.target_path,
                    "attribute": ref.attribute_path,
                    "type": ref.reference_type
                }
                for ref in self.references
            ],
            # Convert dependency graph to adjacency list
            "dependencies": {
                node: list(self.dependency_graph.successors(node))
                for node in self.dependency_graph.nodes
            }
        }
        
        return json.dumps(state, indent=2)
    
    def analyze_attributes(self, resource_type: str = None) -> Dict[str, Dict[str, int]]:
        """
        Analyze attribute usage patterns across resources.
        
        Args:
            resource_type: Optional filter to only analyze certain resource types.
            
        Returns:
            Dict with statistics about attribute usage.
        """
        attribute_stats = defaultdict(lambda: defaultdict(int))
        
        for attr in self.attributes:
            parts = attr.path.split('.')
            if len(parts) < 3:
                continue
                
            current_type = parts[0] if parts[0] != 'resource' else parts[1]
            
            if resource_type and current_type != resource_type:
                continue
                
            # Extract the attribute name (last part of the path)
            attr_name = parts[-1]
            attribute_stats[current_type][attr_name] += 1
        
        return dict(attribute_stats) 
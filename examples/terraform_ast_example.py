#!/usr/bin/env python
"""
Example script demonstrating usage of the TerraformAstAnalyzer.

This script shows how to:
1. Parse Terraform configurations
2. Analyze resource dependencies
3. Extract attribute paths
4. Compare changes between versions
5. Generate visualization of resource dependencies
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Ensure the parent directory is in the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.terraform_ast import TerraformAstAnalyzer
from src.analysis.diff_utils import analyze_terraform_changes


def visualize_dependencies(graph, output_file=None):
    """
    Visualize a resource dependency graph.
    
    Args:
        graph: NetworkX graph of resource dependencies
        output_file: Optional file path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Use a layout that spreads nodes out
    pos = nx.spring_layout(graph, seed=42)
    
    # Draw nodes with different colors based on resource type
    node_colors = []
    resource_types = {}
    
    for node in graph.nodes():
        resource_type = node.split('.')[0]
        if resource_type not in resource_types:
            resource_types[resource_type] = len(resource_types)
        node_colors.append(resource_types[resource_type])
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.tab10, 
                           node_size=700, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrowsize=15, width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    plt.title("Terraform Resource Dependencies")
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Dependency graph visualization saved to {output_file}")
    else:
        plt.show()


def analyze_single_file(file_path):
    """
    Demonstrate analyzing a single Terraform file.
    
    Args:
        file_path: Path to the Terraform file
    """
    analyzer = TerraformAstAnalyzer()
    
    # Parse the file
    print(f"Analyzing Terraform file: {file_path}")
    ast = analyzer.parse_file(file_path)
    
    # Print summary of resources found
    resource_count = {}
    for res_type, instances in ast.get('resource', {}).items():
        resource_count[res_type] = len(instances)
    
    print("\nResources found:")
    for res_type, count in resource_count.items():
        print(f"  {res_type}: {count} instance(s)")
    
    # Print attributes with their paths
    print("\nAttribute paths:")
    for attr in analyzer.attributes[:10]:  # Limit to 10 for brevity
        print(f"  {attr.path} = {attr.value}")
    
    if len(analyzer.attributes) > 10:
        print(f"  ... and {len(analyzer.attributes) - 10} more attributes")
    
    # Print references between resources
    print("\nResource references:")
    for ref in analyzer.references:
        print(f"  {ref.source_path} -> {ref.target_path} (via {ref.attribute_path})")
    
    # Visualize the dependency graph
    if analyzer.dependency_graph.nodes:
        print("\nGenerating dependency graph visualization...")
        output_file = os.path.join(os.path.dirname(file_path), 
                                 f"{os.path.basename(file_path)}.dependencies.png")
        visualize_dependencies(analyzer.dependency_graph, output_file)


def compare_files(old_file, new_file):
    """
    Demonstrate comparing two versions of a Terraform file.
    
    Args:
        old_file: Path to the old version of the file
        new_file: Path to the new version of the file
    """
    # Read file contents
    with open(old_file, 'r') as f:
        old_content = f.read()
    
    with open(new_file, 'r') as f:
        new_content = f.read()
    
    # Analyze changes
    print(f"Comparing changes between {old_file} and {new_file}")
    changes = analyze_terraform_changes(old_content, new_content)
    
    # Print summary of changes
    print("\nChange summary:")
    print(f"  Added attributes: {len(changes['added'])}")
    print(f"  Removed attributes: {len(changes['removed'])}")
    print(f"  Modified attributes: {len(changes['modified'])}")
    print(f"  Resources with dependency changes: {len(changes['dependency_changes'])}")
    
    # Print added attributes
    if changes['added']:
        print("\nAdded attributes:")
        for item in changes['added']:
            print(f"  + {item['path']} = {item['value']}")
    
    # Print removed attributes
    if changes['removed']:
        print("\nRemoved attributes:")
        for item in changes['removed']:
            print(f"  - {item['path']} = {item['value']}")
    
    # Print modified attributes
    if changes['modified']:
        print("\nModified attributes:")
        for item in changes['modified']:
            print(f"  ~ {item['path']}: {item['old_value']} -> {item['new_value']}")
    
    # Print dependency changes
    if changes['dependency_changes']:
        print("\nDependency changes:")
        for item in changes['dependency_changes']:
            resource = item['resource']
            if item['added_dependencies']:
                for dep in item['added_dependencies']:
                    print(f"  {resource} now depends on {dep}")
            if item['removed_dependencies']:
                for dep in item['removed_dependencies']:
                    print(f"  {resource} no longer depends on {dep}")


def find_resource_references(file_path, resource_type, resource_name):
    """
    Find all references to a specific resource.
    
    Args:
        file_path: Path to the Terraform file
        resource_type: Type of the resource to find references for
        resource_name: Name of the resource to find references for
    """
    analyzer = TerraformAstAnalyzer()
    analyzer.parse_file(file_path)
    
    resource_path = f"{resource_type}.{resource_name}"
    dependents = analyzer.get_resource_dependents(resource_path)
    
    print(f"\nResources that reference {resource_path}:")
    if dependents:
        for dependent in dependents:
            print(f"  {dependent}")
            
            # Find the specific attributes that reference this resource
            references = [ref for ref in analyzer.references 
                          if ref.source_path == dependent and ref.target_path == resource_path]
            
            for ref in references:
                print(f"    via {ref.attribute_path}")
    else:
        print("  No references found")


def main():
    parser = argparse.ArgumentParser(description="Terraform AST Analyzer example")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single Terraform file")
    analyze_parser.add_argument("file", help="Path to the Terraform file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two Terraform files")
    compare_parser.add_argument("old_file", help="Path to the old version of the file")
    compare_parser.add_argument("new_file", help="Path to the new version of the file")
    
    # Find references command
    references_parser = subparsers.add_parser("references", help="Find references to a resource")
    references_parser.add_argument("file", help="Path to the Terraform file")
    references_parser.add_argument("resource_type", help="Type of the resource")
    references_parser.add_argument("resource_name", help="Name of the resource")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_single_file(args.file)
    elif args.command == "compare":
        compare_files(args.old_file, args.new_file)
    elif args.command == "references":
        find_resource_references(args.file, args.resource_type, args.resource_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
"""
Tests for the TerraformAstAnalyzer class.
"""

import unittest
import os
import tempfile
from analysis.terraform_ast import TerraformAstAnalyzer, TerraformAttribute, TerraformReference


class TestTerraformAstAnalyzer(unittest.TestCase):
    """Test cases for the TerraformAstAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TerraformAstAnalyzer()
        
        # Sample Terraform content
        self.valid_hcl = """
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "allow_ssh"
  }
}

resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]
  
  depends_on = [aws_vpc.main]
  
  ebs_block_device {
    device_name = "/dev/sdg"
    volume_size = 10
    volume_type = "gp2"
  }
  
  dynamic "network_interface" {
    for_each = var.network_interfaces
    content {
      network_interface_id = network_interface.value.id
      device_index         = network_interface.value.index
    }
  }
  
  count = 2
}

resource "aws_s3_bucket" "logs" {
  bucket = "my-logs-bucket"
  acl    = "private"
  
  lifecycle_rule {
    enabled = true
    expiration {
      days = 90
    }
  }
  
  for_each = {
    log1 = "logs/log1"
    log2 = "logs/log2"
  }
}
"""

        # Sample diff fragment
        self.diff_fragment = """
@@ -10,7 +10,7 @@ resource "aws_security_group" "allow_ssh" {
   ingress {
     from_port   = 22
     to_port     = 22
-    protocol    = "tcp"
+    protocol    = "udp"
     cidr_blocks = ["0.0.0.0/0"]
   }
"""

    def test_parse_hcl(self):
        """Test parsing a full HCL configuration."""
        result = self.analyzer.parse_hcl(self.valid_hcl)
        
        # Check if resources are parsed correctly
        self.assertIn('resource', result)
        self.assertIn('aws_vpc', result['resource'])
        self.assertIn('main', result['resource']['aws_vpc'])
        
        # Check if a specific attribute is parsed correctly
        self.assertEqual(result['resource']['aws_vpc']['main']['cidr_block'], "10.0.0.0/16")
        
        # Check if attributes are extracted
        self.assertTrue(len(self.analyzer.attributes) > 0)
        
        # Check if references are extracted
        self.assertTrue(len(self.analyzer.references) > 0)
        
        # Check if dependency graph is built
        self.assertTrue(len(self.analyzer.dependency_graph.nodes) > 0)
        self.assertTrue(len(self.analyzer.dependency_graph.edges) > 0)
        
    def test_parse_file(self):
        """Test parsing a Terraform file."""
        # Create a temporary file with Terraform content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tf', delete=False) as temp:
            temp.write(self.valid_hcl)
            temp_path = temp.name
        
        try:
            result = self.analyzer.parse_file(temp_path)
            
            # Check if the file is parsed correctly
            self.assertIn('resource', result)
            self.assertIn('aws_vpc', result['resource'])
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
    
    def test_parse_diff_fragment(self):
        """Test parsing a diff fragment."""
        result = self.analyzer.parse_diff_fragment(self.diff_fragment)
        
        # Check if the fragment is parsed
        self.assertIsInstance(result, dict)
        
        # The protocol should be "udp" as per the diff
        self.assertTrue(any(attr.value == "udp" for attr in self.analyzer.attributes 
                          if attr.path.endswith("protocol")))
    
    def test_get_resource_dependencies(self):
        """Test getting resource dependencies."""
        self.analyzer.parse_hcl(self.valid_hcl)
        
        # aws_instance.web should depend on aws_security_group.allow_ssh and aws_vpc.main
        dependencies = self.analyzer.get_resource_dependencies("aws_instance.web")
        self.assertIn("aws_security_group.allow_ssh", dependencies)
        self.assertIn("aws_vpc.main", dependencies)
    
    def test_get_resource_dependents(self):
        """Test getting resource dependents."""
        self.analyzer.parse_hcl(self.valid_hcl)
        
        # aws_vpc.main should be depended on by aws_security_group.allow_ssh and aws_instance.web
        dependents = self.analyzer.get_resource_dependents("aws_vpc.main")
        self.assertIn("aws_security_group.allow_ssh", dependents)
        self.assertIn("aws_instance.web", dependents)
    
    def test_get_attribute_by_path(self):
        """Test getting an attribute by path."""
        self.analyzer.parse_hcl(self.valid_hcl)
        
        # Test getting a specific attribute
        attr = self.analyzer.get_attribute_by_path("resource.aws_vpc.main.cidr_block")
        self.assertIsNotNone(attr)
        self.assertEqual(attr.value, "10.0.0.0/16")
    
    def test_analyze_changes(self):
        """Test analyzing changes between two configurations."""
        old_content = """
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}
"""
        new_content = """
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/8"
}
"""
        changes = self.analyzer.analyze_changes(old_content, new_content)
        
        # There should be a modified attribute for cidr_block
        self.assertEqual(len(changes["modified"]), 1)
        self.assertEqual(changes["modified"][0]["path"], "resource.aws_vpc.main.cidr_block")
        self.assertEqual(changes["modified"][0]["old_value"], "10.0.0.0/16")
        self.assertEqual(changes["modified"][0]["new_value"], "10.0.0.0/8")
    
    def test_resolve_dynamic_blocks(self):
        """Test resolving dynamic blocks."""
        self.analyzer.parse_hcl(self.valid_hcl)
        dynamic_blocks = self.analyzer.resolve_dynamic_blocks()
        
        # There should be a dynamic block for network_interface in aws_instance.web
        self.assertIn("resource.aws_instance.web", dynamic_blocks)
        self.assertIn("resource.aws_instance.web.dynamic.network_interface", 
                     dynamic_blocks["resource.aws_instance.web"])
    
    def test_find_resources_by_type(self):
        """Test finding resources by type."""
        self.analyzer.parse_hcl(self.valid_hcl)
        resources = self.analyzer.find_resources_by_type("aws_instance")
        
        # There should be one aws_instance resource
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0], "aws_instance.web")
    
    def test_get_resource_count(self):
        """Test getting the count parameter for a resource."""
        self.analyzer.parse_hcl(self.valid_hcl)
        count = self.analyzer.get_resource_count("aws_instance.web")
        
        # aws_instance.web has count = 2
        self.assertEqual(count, 2)
    
    def test_get_resource_for_each(self):
        """Test getting the for_each parameter for a resource."""
        self.analyzer.parse_hcl(self.valid_hcl)
        for_each = self.analyzer.get_resource_for_each("aws_s3_bucket.logs")
        
        # aws_s3_bucket.logs has a for_each block
        self.assertIsNotNone(for_each)
        self.assertIsInstance(for_each, dict)
        self.assertEqual(len(for_each), 2)
    
    def test_to_json(self):
        """Test serializing the analyzer state to JSON."""
        self.analyzer.parse_hcl(self.valid_hcl)
        json_data = self.analyzer.to_json()
        
        # The JSON should be a non-empty string
        self.assertIsInstance(json_data, str)
        self.assertTrue(len(json_data) > 0)
    
    def test_analyze_attributes(self):
        """Test analyzing attribute usage patterns."""
        self.analyzer.parse_hcl(self.valid_hcl)
        stats = self.analyzer.analyze_attributes()
        
        # There should be stats for aws_vpc
        self.assertIn("aws_vpc", stats)
        # There should be stats for aws_instance
        self.assertIn("aws_instance", stats)
        
        # Test filtering by resource type
        vpc_stats = self.analyzer.analyze_attributes("aws_vpc")
        self.assertIn("aws_vpc", vpc_stats)
        self.assertNotIn("aws_instance", vpc_stats)


if __name__ == '__main__':
    unittest.main() 
# test_diff_utils.py
import unittest
import pandas as pd
from diff_utils import (
    parse_patch_to_dataframe,
    is_bugfix_commit,
    parse_hcl_snippet,
    enrich_dataframe_with_terraform_semantics
)

class TestDiffUtils(unittest.TestCase):
    
    def test_parse_patch_to_dataframe(self):
        """Test parsing a patch into a dataframe."""
        sample_patch = """--- old/test.tf
+++ new/test.tf
@@ -1,3 +1,4 @@
 resource "aws_instance" "example" {
-  ami = "ami-12345"
+  ami = "ami-67890"
+  instance_type = "t2.micro"
 }"""
        
        df = parse_patch_to_dataframe(sample_patch)
        
        # Check basic structure
        self.assertEqual(len(df), 8)  # Should have 8 rows
        self.assertEqual(df['change'].value_counts()['added'], 2)
        self.assertEqual(df['change'].value_counts()['removed'], 1)
        
        # Check that line numbers are parsed
        self.assertIsNotNone(df[df['change'] == 'removed']['old_lineno'].iloc[0])
        self.assertIsNotNone(df[df['change'] == 'added']['new_lineno'].iloc[0])
        
    def test_is_bugfix_commit(self):
        """Test identifying bugfix commits."""
        bugfix_messages = [
            "Fix invalid ami reference",
            "Bug in security group configuration",
            "Fixing the crash in the provisioner",
            "Resolved issue with incorrect attribute"
        ]
        
        non_bugfix_messages = [
            "Add new EC2 instance",
            "Update documentation",
            "Initial commit"
        ]
        
        for msg in bugfix_messages:
            self.assertTrue(is_bugfix_commit(msg), f"Failed on: {msg}")
            
        for msg in non_bugfix_messages:
            self.assertFalse(is_bugfix_commit(msg), f"Failed on: {msg}")
    
    def test_parse_hcl_snippet(self):
        """Test parsing HCL snippets."""
        valid_hcl = """resource "aws_instance" "example" {
  ami = "ami-12345"
  instance_type = "t2.micro"
}"""
        
        invalid_hcl = "ami = "
        
        # Valid HCL should parse to a non-empty dict
        result = parse_hcl_snippet(valid_hcl)
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)
        
        # Invalid HCL should return empty dict
        result = parse_hcl_snippet(invalid_hcl)
        self.assertEqual(result, {})
    
    def test_enrich_dataframe_with_terraform_semantics(self):
        """Test enriching dataframe with Terraform semantics."""
        # Create a sample dataframe
        data = [{
            'change': 'added',
            'content': 'resource "aws_instance" "example" {'
        }, {
            'change': 'added',
            'content': '  ami = "ami-12345"'
        }]
        df = pd.DataFrame(data)
        
        # Enrich the dataframe
        enriched = enrich_dataframe_with_terraform_semantics(df)
        
        # Check that resource_type is identified correctly
        self.assertIn('resource_type', enriched.columns)
        self.assertEqual(enriched.iloc[0].get('resource_type'), 'aws_instance')
        
        # Check that attr_name is identified correctly
        self.assertIn('attr_name', enriched.columns)
        self.assertEqual(enriched.iloc[1].get('attr_name'), 'ami')

if __name__ == '__main__':
    unittest.main()
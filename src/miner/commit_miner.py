import os
import subprocess
from git import Repo, GitCommandError

class CommitMiner:
    """
    Mine commits containing keywords and Terraform file changes.
    Includes enhanced error handling and fallback mechanisms.
    """
    def __init__(self, keywords=None):
        self.keywords = [k.lower() for k in (keywords or [])]
        
    def _safe_get_diff(self, repo, commit, parent=None):
        """
        Safely get diff between commit and parent using multiple methods.
        Provides fallback mechanism when standard diff fails.
        
        Args:
            repo (Repo): Git repository object
            commit: Commit object to get diff for
            parent: Parent commit or None for initial commit
            
        Returns:
            list: Diff items or empty list on failure
        """
        import logging
        
        # Method 1: Try standard GitPython diff
        try:
            return commit.diff(parent, create_patch=True)
        except GitCommandError as e:
            error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
            logging.warning(f"Standard diff failed for {commit.hexsha[:8]}: {error_msg}")
            
            # Extract detailed error message for better diagnostics
            if "exit code(128)" in str(e):
                logging.warning(f"Exit code 128 indicates a serious Git error, possibly related to repository state")
                if "bad revision" in error_msg:
                    logging.warning(f"Bad revision error: One of the commits may not exist in this shallow clone")
                elif "ambiguous argument" in error_msg:
                    logging.warning(f"Ambiguous argument error: Git couldn't resolve commit references")
                
        # Method 2: Try using git directly through subprocess with alternative syntax
        try:
            logging.info(f"Trying fallback diff method for {commit.hexsha[:8]}")
            parent_hash = parent.hexsha if parent else "HEAD"
            
            # Alternative approach - use a different revision specification format
            if parent:
                cmd = ["git", "-C", repo.working_dir, "diff", f"{parent.hexsha}", f"{commit.hexsha}", "--name-only"]
            else:
                # For commits without parents (initial commits)
                cmd = ["git", "-C", repo.working_dir, "show", "--name-only", "--format=", f"{commit.hexsha}"]
                
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Get changed files
            changed_files = [f for f in result.stdout.split('\n') if f and f.endswith('.tf')]
            
            # If we found Terraform files, get their content
            if changed_files:
                logging.info(f"Fallback found {len(changed_files)} Terraform files for {commit.hexsha[:8]}")
                
                # Create synthetic diff objects with minimal info
                diffs = []
                for file_path in changed_files:
                    # Get file content at this commit
                    try:
                        # Use git show to get file content
                        content_cmd = ["git", "-C", repo.working_dir, "show", f"{commit.hexsha}:{file_path}"]
                        content_result = subprocess.run(content_cmd, capture_output=True, text=True, check=False)
                        
                        # Create a minimal diff object (with limited information)
                        class MinimalDiff:
                            def __init__(self, path, content):
                                self.b_path = path
                                self._diff = content.encode('utf-8', errors='ignore')
                                
                            @property
                            def diff(self):
                                return self._diff
                        
                        diffs.append(MinimalDiff(file_path, content_result.stdout))
                    except Exception as inner_e:
                        logging.warning(f"Could not get content for {file_path}: {inner_e}")
                
                return diffs
            
            return []
        except Exception as e:
            logging.warning(f"Fallback diff method also failed for {commit.hexsha[:8]}: {e}")
            
            # Method 3: One more fallback using a different Git command approach
            try:
                logging.info(f"Trying second fallback method for {commit.hexsha[:8]}")
                
                # Try direct git command without using diff-tree
                cmd = ["git", "-C", repo.working_dir, "ls-tree", "-r", "--name-only", commit.hexsha]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Get Terraform files in this commit
                tf_files = [f for f in result.stdout.split('\n') if f and f.endswith('.tf')]
                
                if tf_files:
                    logging.info(f"Second fallback found {len(tf_files)} Terraform files for {commit.hexsha[:8]}")
                    
                    # Create minimal diff objects
                    diffs = []
                    for file_path in tf_files:
                        try:
                            content_cmd = ["git", "-C", repo.working_dir, "show", f"{commit.hexsha}:{file_path}"]
                            content_result = subprocess.run(content_cmd, capture_output=True, text=True, check=False)
                            
                            class MinimalDiff:
                                def __init__(self, path, content):
                                    self.b_path = path
                                    self._diff = content.encode('utf-8', errors='ignore')
                                    
                                @property
                                def diff(self):
                                    return self._diff
                            
                            diffs.append(MinimalDiff(file_path, content_result.stdout))
                        except Exception as inner_e:
                            logging.warning(f"Could not get content for {file_path}: {inner_e}")
                    
                    return diffs
            except Exception as e2:
                logging.warning(f"Second fallback method also failed for {commit.hexsha[:8]}: {e2}")
            
            return []

    def mine(self, repo_path, seen_hashes=None, last_head=None):
        """
        Iterate commits, filter by message and .tf changes.
        Supports incremental mining by skipping already processed commits.
        Enhanced with improved error handling and detailed logging.
        
        Args:
            repo_path (str): Path to the cloned repository
            seen_hashes (set): Set of commit hashes that have already been processed
            last_head (str): Hash of the last HEAD commit processed
            
        Returns:
            list: Entries with commit information
        """
        import logging
        repo = Repo(repo_path)
        entries = []
        seen_hashes = set(seen_hashes or [])
        
        # Update the repo to get latest commits
        try:
            origin = repo.remotes.origin
            origin.fetch()
        except Exception as e:
            logging.warning(f"Failed to fetch latest commits: {e}")
        
        # Get the current HEAD hash for future reference
        current_head = None
        try:
            current_head = repo.head.commit.hexsha
        except Exception as e:
            logging.warning(f"Failed to get current HEAD: {e}")
        
        # Determine which commits to iterate through
        if last_head:
            try:
                # Get only commits since the last processed HEAD
                commit_range = f"{last_head}..HEAD"
                commits_to_process = list(repo.iter_commits(commit_range))
                logging.info(f"Mining {len(commits_to_process)} new commits since {last_head[:8]}")
            except Exception as e:
                logging.warning(f"Failed to use commit range, falling back to all commits: {e}")
                commits_to_process = repo.iter_commits()
        else:
            # No last_head, process all commits but skip those in seen_hashes
            commits_to_process = repo.iter_commits()
        
        # Process commits with improved error handling
        for commit in commits_to_process:
            # Skip already processed commits
            if commit.hexsha in seen_hashes:
                continue
                
            msg = commit.message.lower()
            if any(k in msg for k in self.keywords):
                try:
                    # Get parent commit or None for initial commit
                    parent = commit.parents[0] if commit.parents else None
                    
                    # Use enhanced diff method with fallback handling
                    diffs = self._safe_get_diff(repo, commit, parent)
                    
                    # Filter for Terraform files
                    tf_diffs = [d for d in diffs if d.b_path and d.b_path.endswith('.tf')]
                    
                    if tf_diffs:
                        for d in tf_diffs:
                            try:
                                # Handle potential encoding issues with patch data
                                patch_data = d.diff.decode('utf-8', errors='ignore')
                                
                                entries.append({
                                    'repo': os.path.basename(repo_path),
                                    'commit_hash': commit.hexsha,
                                    'author': commit.author.name,
                                    'date': commit.committed_datetime.isoformat(),
                                    'message': commit.message.strip(),
                                    'file': d.b_path,
                                    'patch': patch_data,
                                    'head_hash': current_head  # Store the HEAD at time of mining
                                })
                            except Exception as inner_e:
                                logging.warning(f"Error extracting diff data for file {d.b_path}: {inner_e}")
                                
                except GitCommandError as e:
                    # Enhanced error handling for Git errors
                    error_msg = e.stderr.strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                    logging.warning(f"Git error processing commit {commit.hexsha[:8]}: {error_msg}")
                    
                    # Extract command details for better diagnostics
                    if hasattr(e, 'command'):
                        logging.warning(f"Failed command: {e.command}")
                    
                    continue
                except Exception as e:
                    logging.warning(f"Error processing commit {commit.hexsha[:8]}: {e}")
                    continue
                    
        return entries
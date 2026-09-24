import os
import subprocess
from git import Repo, GitCommandError, NULL_TREE

class _FallbackDiff:
    """Minimal stand-in for git.Diff with the attributes CommitMiner uses."""
    def __init__(self, a_path, b_path, diff):
        self.a_path = a_path
        self.b_path = b_path
        self.diff = diff


def _strip_patch_header(patch):
    """Drop the 'diff --git'/'index'/'---'/'+++' header, keeping the patch from the first hunk."""
    for marker in (b"@@", b"Binary files"):
        if patch.startswith(marker):
            return patch
        pos = patch.find(b"\n" + marker)
        if pos != -1:
            return patch[pos + 1:]
    return b""


class DiffUnavailableError(RuntimeError):
    """Both diff methods failed for a commit (e.g. a partial clone can't fetch blobs while offline)."""


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
            list: Diff items

        Raises:
            DiffUnavailableError: if both methods fail. Skipping the commit instead would silently
            drop it from the dataset, and the repo's checkpoint would keep it from being mined again.
        """
        import logging
        
        # Method 1: Try standard GitPython diff.
        # a.diff(b) describes a -> b, so the diff must start at the parent. A root commit is
        # compared with NULL_TREE (GitPython already orients that one as empty tree -> commit);
        # commit.diff(None) would compare against the working tree instead.
        try:
            if parent is None:
                return commit.diff(NULL_TREE, create_patch=True)
            return parent.diff(commit, create_patch=True)
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
                
        # Method 2: git diff-tree through subprocess. Produces the same per-file patches as
        # Method 1, limited to the .tf files the commit actually changed.
        try:
            logging.info(f"Trying fallback diff method for {commit.hexsha[:8]}")
            return self._fallback_diff(repo, commit, parent)
        except Exception as e:
            logging.warning(f"Fallback diff method also failed for {commit.hexsha[:8]}: {e}")
            raise DiffUnavailableError(f"No diff for commit {commit.hexsha[:8]}: {e}") from e

    @staticmethod
    def _git(repo, *args):
        cmd = ["git", "-C", repo.working_dir, *args]
        return subprocess.run(cmd, capture_output=True, check=True).stdout

    def _fallback_diff(self, repo, commit, parent):
        """
        Diff parent -> commit (or empty tree -> commit for a root commit) with git diff-tree.

        Returns:
            list: _FallbackDiff items (a_path, b_path, diff) for changed .tf files, where
            diff is the patch body starting at the first hunk, as GitPython returns it.
        """
        revs = [parent.hexsha, commit.hexsha] if parent else ["--root", commit.hexsha]
        base = ["-c", "core.quotepath=false", "diff-tree", "-r", "-M", "--no-commit-id",
                "--no-color", "--no-ext-diff", *revs]

        # -z output: status\0path\0 (renames/copies: status\0old\0new\0)
        tokens = self._git(repo, *base, "--name-status", "-z").decode("utf-8", errors="surrogateescape").split("\0")
        diffs = []
        i = 0
        while i < len(tokens) and tokens[i]:
            status = tokens[i][0]
            if status in "RC":
                a_path, b_path = tokens[i + 1], tokens[i + 2]
                i += 3
            else:
                path = tokens[i + 1]
                a_path = None if status == "A" else path
                b_path = None if status == "D" else path
                i += 2

            paths = [p for p in (a_path, b_path) if p]
            if not any(p.endswith(".tf") for p in paths):
                continue
            patch = self._git(repo, *base, "-p", "--", *dict.fromkeys(paths))
            diffs.append(_FallbackDiff(a_path, b_path, _strip_patch_header(patch)))
        return diffs

    def mine(self, repo_path, seen_hashes=None, last_head=None, until_rev=None):
        """
        Iterate commits, filter by message and .tf changes.
        Supports incremental mining by skipping already processed commits.
        Enhanced with improved error handling and detailed logging.

        Args:
            repo_path (str): Path to the cloned repository
            seen_hashes (set): Set of commit hashes that have already been processed
            last_head (str): Hash of the last HEAD commit processed
            until_rev (str): Mine only the history reachable from this revision instead of HEAD
                (used by --replay-from to freeze the history at the original mining point).
                It becomes the head_hash of the entries. Raises ValueError if it can't be resolved.

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
        if until_rev:
            try:
                # repo.commit() doesn't validate a full-length hash, rev-parse --verify does
                current_head = repo.git.rev_parse('--verify', f'{until_rev}^{{commit}}')
            except Exception as e:
                raise ValueError(f"Revision {until_rev} not found in {repo_path}: {e}") from e
        else:
            try:
                current_head = repo.head.commit.hexsha
            except Exception as e:
                logging.warning(f"Failed to get current HEAD: {e}")
        tip = current_head if until_rev else "HEAD"

        # Determine which commits to iterate through
        if last_head:
            try:
                # Get only commits since the last processed HEAD
                commit_range = f"{last_head}..{tip}"
                commits_to_process = list(repo.iter_commits(commit_range))
                logging.info(f"Mining {len(commits_to_process)} new commits since {last_head[:8]}")
            except Exception as e:
                logging.warning(f"Failed to use commit range, falling back to all commits: {e}")
                commits_to_process = repo.iter_commits(tip)
        else:
            # No last_head, process all commits but skip those in seen_hashes
            commits_to_process = repo.iter_commits(tip)
        
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
                    
                    # Filter for Terraform files. Deleted files only have a_path.
                    tf_diffs = []
                    for d in diffs:
                        file_path = d.b_path or getattr(d, 'a_path', None)
                        if file_path and file_path.endswith('.tf'):
                            tf_diffs.append((file_path, d))
                    
                    if tf_diffs:
                        for file_path, d in tf_diffs:
                            try:
                                # Handle potential encoding issues with patch data
                                patch_data = d.diff.decode('utf-8', errors='ignore')
                                
                                entries.append({
                                    'repo': os.path.basename(repo_path),
                                    'commit_hash': commit.hexsha,
                                    'author': commit.author.name,
                                    'date': commit.committed_datetime.isoformat(),
                                    'message': commit.message.strip(),
                                    'file': file_path,
                                    'patch': patch_data,
                                    'head_hash': current_head  # Store the HEAD at time of mining
                                })
                            except Exception as inner_e:
                                logging.warning(f"Error extracting diff data for file {file_path}: {inner_e}")
                                
                except DiffUnavailableError:
                    raise
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
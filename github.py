from typing import Dict, Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta

import requests
import yaml
import os

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass

class GitHubAPIError(Exception):
    """Raised when GitHub API returns an error."""
    pass

class GitHubClient:
    """Base class for GitHub API interactions with common functionality."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.github_token = self._get_github_token()
        self.session = self._create_session()

    def _load_config(self, config_path: str) -> Dict:
        config_file = Path(config_path)
        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}. "
                f"Please copy config.example.yaml to config.yaml"
            )

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")

    def _get_github_token(self) -> str:
        token = self.config['github'].get('token', '')

        # Handle environment variable substitution
        if token.startswith('${') and token.endswith('}'):
            env_var = token[2:-1]
            token = os.environ.get(env_var, '')

        if not token:
            raise ConfigurationError(
                "GitHub token not found. Set GITHUB_TOKEN environment variable "
                "or configure it in config.yaml"
            )

        return token

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': f'{self.__class__.__name__}'
        })
        return session

    def _parse_repo_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub repository URL to extract owner and repo name.

        Supports various URL formats:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - git@github.com:owner/repo.git

        Returns:
            Tuple of (owner, repo_name)

        Raises:
            ConfigurationError: If URL format is invalid
        """
        url = url.rstrip('/')
        if url.endswith('.git'):
            url = url[:-4]

        if 'github.com' in url:
            parts = url.split('github.com')[-1].strip('/:').split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]

        raise ConfigurationError(f"Invalid GitHub repository URL: {url}")

    def _make_request(self, method: str, url: str, error_msg: str, **kwargs) -> Dict:
        """Make an API request with standardized error handling.

        Args:
            method: HTTP method ('get', 'post', 'put', 'delete', etc.)
            url: API endpoint URL
            error_msg: Base error message for exceptions
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response JSON data

        Raises:
            GitHubAPIError: If request fails
        """
        try:
            response = getattr(self.session, method.lower())(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            full_error_msg = f"{error_msg}: {e}"
            if hasattr(e, 'response') and e.response is not None:
                full_error_msg += f"\nResponse: {e.response.text}"
            raise GitHubAPIError(full_error_msg)

    def get_issues(self, owner: str, repo: str, state: str = 'open', per_page: int = 30) -> List[Dict]:
        """Get list of issues for a repository.

        Args:
            state: Issue state ('open', 'closed', or 'all')
        """

        params = {'state': state, 'sort': 'updated', 'direction': 'desc', 'per_page': per_page, 'page': 1}
        if labels := self.config['github'].get('issue_labels', []):
            params['labels'] = ','.join(labels)

        url = f'https://api.github.com/repos/{owner}/{repo}/issues'
        return self._make_request('get', url, "Failed to list issues", params=params)

    def get_issue(self, owner: str, repo: str, issue_number: int) -> Dict:
        """Get issue details by number."""

        url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}'
        return self._make_request('get', url, f"Failed to get issue #{issue_number}")

    def get_issue_comments(self, owner: str, repo: str, issue_number: int, per_page: int = 30) -> Dict:
        """Get comments for an issue."""

        url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments'
        params = {"per_page": per_page, "page": 1, "sort": "updated", "direction": "desc"}
        return self._make_request('get', url, f"Failed to get comments for issue #{issue_number}", params=params)

    def add_issue_comment(self, owner: str, repo: str, issue_number: int, comment: str) -> Dict:
        """Add a comment to an issue."""

        url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments'
        payload = {'body': comment}
        return self._make_request('post', url, f"Failed to add comment to issue #{issue_number}", json=payload)

    def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: List[str]) -> Dict:
        """Add labels to an issue."""

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/labels"
        self._make_request('post', url, f"Failed to add labels to issue #{issue_number}", json={"labels": labels})

    def get_pr(self, owner: str, repo: str, pr_number: int) -> Dict:
        """Get PR details by number."""

        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
        return self._make_request('get', url, f"Failed to get PR #{pr_number}")

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list:
        """Get list of files changed in a PR."""

        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files'
        return self._make_request('get', url, "Failed to fetch PR files")

    def get_pr_file_content(self, owner: str, repo: str, pr_number: int,
                            file_path: str, status: str, head: bool = True) -> Optional[str]:
        """Get file content from PR.

        Args:
            status: File status (added, modified, removed, renamed)
            head: Whether to get content from PR head (True) or base (False)

        Returns:
            File content as string, or None if file was removed
        """
        if status == 'removed':
            return None

        # Get the file content from the PR head
        pr_data = self.get_pr(owner, repo, pr_number)
        ref_sha = pr_data['head']['sha'] if head else pr_data['base']['sha']

        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}'
        params = {'ref': ref_sha}
        file_data = self._make_request('get', url, f"Failed to get content for {file_path} in PR #{pr_number}", params=params)

        # GitHub returns base64 encoded content
        import base64
        content = base64.b64decode(file_data['content']).decode('utf-8')
        return content

    def get_branch_file_content(self, owner: str, repo: str, branch: str, file_path: str) -> Optional[str]:
        """Get file content from a branch."""

        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}'
        params = {'ref': branch}
        file_data = self._make_request('get', url, f"Failed to get content for {file_path} in branch {branch}", params=params)

        import base64
        content = base64.b64decode(file_data['content']).decode('utf-8')
        return content

    def create_branch(self, owner: str, repo: str, base_branch: str, branch_name: str) -> Dict:
        """Create a new branch in repository."""

        url = f'https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{base_branch}'
        res = self._make_request('get', url, f"Failed to get base branch {base_branch} for creating new branch {branch_name}")

        payload = {
            'ref': f'refs/heads/{branch_name}',
            'sha': res['object']['sha']
        }            
        create_url = f'https://api.github.com/repos/{owner}/{repo}/git/refs'
        create_response = self._make_request('post', create_url, f"Failed to create branch {branch_name}", json=payload)

        # If branch already exists, try to update it
        if create_response.status_code == 422:
            update_payload = {
                'sha': res['object']['sha'],
                'force': True
            }
            update_url = f'https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch_name}'
            create_response = self._make_request('patch', update_url, f"Failed to update branch {branch_name}", json=update_payload)

        return create_response.json()

    def get_commits(self, owner: str, repo: str, pr_number: int) -> Dict:
        """Get commits for a PR."""

        url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits'
        return self._make_request('get', url, f"Failed to get commits for PR #{pr_number}")

    def search_issues(self, owner: str, repo: str, branch:str, since_hours: int = 24) -> List[Dict]:
        """Fetch recent pull requests from repository."""

        url = 'https://api.github.com/search/issues'

        cutoff_time = datetime.now() - timedelta(hours=since_hours)
        cutoff_str = cutoff_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        query = (
            f'repo:{owner}/{repo} '
            f'is:pr '
            f'is:merged '
            f'updated:>={cutoff_str} '
            f'base:{branch}'
        )

        params = {
            'q': query,
            'sort': 'updated',
            'order': 'desc',
            'per_page': 100,
            'page': 1
        }

        try:
            recent_prs = []

            # Paginate through all results
            while True:
                data = self._make_request('get', url, "Failed to search issues", params=params)

                items = data.get('items', [])

                if not items:
                    break

                # Convert search to PR
                for item in items:
                    pr_number = item['number']
                    pr_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'
                    resp = self._make_request('get', pr_url, f"Failed to get PR #{pr_number}")
                    recent_prs.append(resp)

                # Check if there are more pages
                if len(items) < params['per_page']:
                    break

                params['page'] += 1

            return recent_prs

        except (requests.RequestException, Exception) as e:
            raise GitHubAPIError(f"Failed to fetch PRs: {e}")
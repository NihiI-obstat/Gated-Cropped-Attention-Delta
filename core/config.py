import os
from pathlib import Path

def load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from a .env file if it exists."""
    env_file = Path(env_path)
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

class Config:
    """Minimal credential handling for evaluation scripts."""
    
    def __init__(self):
        # Load environment variables from .env file if it exists
        # Try multiple locations: cwd, core/, and repo root
        core_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(core_dir)
        for env_path in [".env", os.path.join(core_dir, ".env"), os.path.join(repo_dir, ".env")]:
            load_env_file(env_path)
        
        self._openai_api_key = None
        
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment variables."""
        if self._openai_api_key is None:
            self._openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not self._openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it in your .env file or environment."
                )
        return self._openai_api_key

    def setup_environment(self) -> None:
        """Set up environment variables used by the OpenAI client."""
        os.environ['OPENAI_API_KEY'] = self.openai_api_key

# Global config instance
config = Config()

def setup_credentials() -> Config:
    """Set up credentials and return the global config instance."""
    config.setup_environment()
    return config 

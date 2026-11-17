#!/usr/bin/env python3
"""
Setup script for the Pine Script Expert Agent.
This script helps set up the environment and database for the agent.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def create_virtualenv():
    """Create a virtual environment and install dependencies"""
    print("Creating virtual environment...")
    venv_path = Path("venv")

    if venv_path.exists():
        print("Virtual environment already exists")
        return True

    try:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)

        # Get the pip path
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"

        # Upgrade pip
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)

        # Install dependencies
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)

        print("Virtual environment created and dependencies installed")
        return True
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        return False


def create_env_file():
    """Create a .env file if it doesn't exist"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if env_path.exists():
        print(".env file already exists")
        return

    if not env_example_path.exists():
        print("Error: .env.example file not found")
        return

    shutil.copy(env_example_path, env_path)
    print(".env file created from .env.example")
    print("Please edit the .env file to add your API keys and database URL")


def main():
    """Main entry point"""
    print("Setting up Pine Script Expert Agent...")

    # Create virtual environment
    create_virtualenv()

    # Create .env file
    create_env_file()

    activate = r"venv\Scripts\activate" if sys.platform == "win32" else "source venv/bin/activate"
    print("\nSetup complete!")
    print("\nNext steps:")
    print(f"1. Activate the virtual environment: {activate}")
    print("2. Edit the .env file to add your API keys and database URL")
    print("3. Initialize the database: python init_db.py")
    print("4. Populate the database: python pinescript_crawler.py")
    print("5. Start the interactive shell: python run.py interactive")


if __name__ == "__main__":
    main()

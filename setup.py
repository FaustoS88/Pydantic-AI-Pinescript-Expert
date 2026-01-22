#!/usr/bin/env python3
"""
Setup script for the Pine Script Expert Agent.
This script helps set up the environment and database for the agent.
"""

import os
import sys
import subprocess
import shutil
import getpass
from pathlib import Path

def print_banner():
    """Print a welcome banner"""
    print("=" * 60)
    print("  Pine Script Expert Agent Setup")
    print("=" * 60)
    print()

def get_user_confirmation(message):
    """Get user confirmation for an action"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def create_virtualenv():
    """Create a virtual environment and install dependencies"""
    print("🔧 Setting up virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        if not get_user_confirmation("Do you want to recreate it?"):
            return True
        else:
            print("🗑️  Removing existing virtual environment...")
            shutil.rmtree(venv_path)
    
    try:
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Get the pip path
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
            python_path = venv_path / "Scripts" / "python"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"
        
        print("⬆️  Upgrading pip...")
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        print("📚 Installing dependencies from requirements.txt...")
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        
        print("✅ Virtual environment created and dependencies installed")
        print(f"📍 Virtual environment location: {venv_path.absolute()}")
        
        # Show activation instructions
        if sys.platform == "win32":
            print(f"💡 To activate: venv\\Scripts\\activate")
        else:
            print(f"💡 To activate: source venv/bin/activate")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def setup_environment_file():
    """Interactive setup of .env file"""
    print("\n🔐 Setting up environment configuration...")
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✅ .env file already exists")
        if not get_user_confirmation("Do you want to reconfigure it?"):
            return
        else:
            # Backup existing .env
            backup_path = env_path.with_suffix('.env.backup')
            shutil.copy(env_path, backup_path)
            print(f"📋 Backed up existing .env to {backup_path}")
    
    if not env_example_path.exists():
        print("❌ Error: .env.example file not found")
        return
    
    # Copy template
    shutil.copy(env_example_path, env_path)
    print("📄 Created .env file from template")
    
    # Interactive configuration
    print("\n🔧 Let's configure your environment variables...")
    print("(Press Enter to skip and configure manually later)")
    
    # OpenAI API Key
    openai_key = getpass.getpass("Enter your OpenAI API key (hidden): ").strip()
    if openai_key:
        update_env_var(env_path, "OPENAI_API_KEY", openai_key)
        print("✅ OpenAI API key configured")
    
    # OpenAI Base URL
    base_url = input("Enter OpenAI Base URL (default: https://api.openai.com/v1): ").strip()
    if base_url:
        update_env_var(env_path, "OPENAI_BASE_URL", base_url)
        print("✅ OpenAI Base URL configured")
    
    # OpenAI Model
    model = input("Enter OpenAI Model (default: gpt-4o-mini): ").strip()
    if model:
        update_env_var(env_path, "OPENAI_MODEL", model)
        print("✅ OpenAI Model configured")
    
    # OpenAI Embedding Model
    embedding_model = input("Enter OpenAI Embedding Model (default: text-embedding-nomic-embed-text-v1.5@f16): ").strip()
    if embedding_model:
        update_env_var(env_path, "OPENAI_EMBEDDING_MODEL", embedding_model)
        print("✅ OpenAI Embedding Model configured")
    
    # OpenRouter API Key (optional)
    openrouter_key = getpass.getpass("Enter your OpenRouter API key (optional, hidden): ").strip()
    if openrouter_key:
        update_env_var(env_path, "OPENROUTER_API_KEY", openrouter_key)
        print("✅ OpenRouter API key configured")
    
    # OpenRouter Model
    if openrouter_key:
        openrouter_model = input("Enter OpenRouter Model (default: openai/gpt-4.1-mini): ").strip()
        if openrouter_model:
            update_env_var(env_path, "OPENROUTER_MODEL", openrouter_model)
            print("✅ OpenRouter Model configured")
    
    # Database URL
    db_url = input("Enter Database URL (default: postgresql://postgres:postgres@localhost:5432/pydantic): ").strip()
    if db_url:
        update_env_var(env_path, "DATABASE_URL", db_url)
        print("✅ Database URL configured")
    
    print(f"\n✅ Environment configuration saved to {env_path}")
    print("💡 You can edit this file manually anytime to update your settings")

def update_env_var(env_path, key, value):
    """Update a specific environment variable in the .env file"""
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key}={value}\n")
    
    with open(env_path, 'w') as f:
        f.writelines(lines)

def make_script_executable():
    """Make the shell script executable"""
    script_path = Path("pinescript-agent.sh")
    
    if not script_path.exists():
        print("⚠️  pinescript-agent.sh not found, skipping...")
        return
    
    try:
        script_path.chmod(0o755)
        print("✅ Made pinescript-agent.sh executable")
    except Exception as e:
        print(f"⚠️  Error making script executable: {e}")
        print("💡 You may need to run: chmod +x pinescript-agent.sh")

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    else:
        print("✅ requirements.txt found")
    
    return True

def show_next_steps():
    """Show next steps after setup"""
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("- Activate your virtual environment before running commands")
    if sys.platform == "win32":
        print("  Windows: venv\\Scripts\\activate")
    else:
        print("  macOS/Linux: source venv/bin/activate")
    print("- Check the README.md for detailed usage instructions")
    print("1. 🔐 Verify your .env file has the correct API keys")
    print("2. 🗄️  Set up your database: python run.py check")
    print("3. 📚 Populate the database: python run.py populate")
    print("4. 🚀 Start the interactive shell: python run.py interactive")
    print("5. 🌐 Or run the Streamlit UI: streamlit run streamlit_ui.py")
    


def main():
    """Main entry point"""
    print_banner()
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🚀 Starting setup process...")
    
    # Create virtual environment
    if not create_virtualenv():
        print("\n❌ Failed to create virtual environment. Setup aborted.")
        sys.exit(1)
    
    # Setup environment file
    setup_environment_file()
    
    # Make script executable (if exists)
    make_script_executable()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()

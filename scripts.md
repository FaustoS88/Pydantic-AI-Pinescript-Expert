# PineScript Expert Agent: Scripts Overview

This document provides a comprehensive overview of all the scripts in the PineScript Expert Agent project, explaining their purpose, functionality, and usage order.

## Core Components

### 1. `agent.py`
**Purpose**: Main implementation of the PineScript Expert Agent using Pydantic AI.
**Features**:
- Defines the agent with OpenAI/OpenRouter model configuration
- Implements the retrieval tool for searching Pine Script documentation
- Contains the dependency injection mechanism for database and API connections
- Handles API key management and validation

### 2. `db_schema.py`
**Purpose**: Defines the database schema for storing Pine Script documentation.
**Features**:
- Creates tables with pgvector support for vector embeddings
- Includes validation functions to verify the database setup
- Defines the schema for the `pinescript_docs` table

## Setup and Initialization Scripts

### 3. `setup.py`
**Purpose**: Initial project setup script to prepare the environment.
**Features**:
- Creates a virtual environment
- Installs dependencies from requirements.txt
- Sets up the .env file from the template
- Makes the shell script executable
**Usage**: Run this first when setting up a new installation.

### 4. `init_db.py`
**Purpose**: Initializes the database with the required schema.
**Features**:
- Creates necessary tables and extensions
- Verifies that pgvector is installed
- Checks and reports on database connection status
**Usage**: Run after `setup.py` to prepare the database.

## Data Management Scripts

### 5. `clear_database.py`
**Purpose**: Utility to clear the database for a fresh start.
**Features**:
- Removes all data from the pinescript_docs table
- Keeps schema intact
- Provides confirmation prompt for safety

### 6. `pinescript_crawler.py`
**Purpose**: Crawls the TradingView Pine Script documentation.
**Features**:
- Uses crawl4ai to extract documentation from the TradingView website
- Processes and splits documentation into sections
- Generates embeddings and stores in the database
**Usage**: Run after `init_db.py` to populate the database with documentation.

### 7. `db_inspect.py`
**Purpose**: Utility to inspect and query the database.
**Features**:
- Counts entries in the database
- Lists document titles and URLs
- Tests search functionality
- Verifies vector quality

## User Interface Scripts

### 8. `run.py`
**Purpose**: Main entry point that provides multiple command options.
**Features**:
- `interactive`: Launches interactive shell
- `query`: Processes a single query
- `check`: Verifies database setup
**Usage**: The recommended way to interact with the agent.

### 9. `interactive.py`
**Purpose**: Provides an interactive command-line interface.
**Features**:
- Command-line conversation with the agent
- History management
- Example queries
**Usage**: Run directly or through `run.py interactive`.

### 10. `streamlit_ui.py`
**Purpose**: Streamlit UI with persistent conversation history.
**Features**:
- Saves chat history to disk
- Maintains conversation context between sessions
- Same features as the regular Streamlit UI
- Example queries
- Status monitoring for database and API keys

**Usage**: Run with `streamlit run streamlit_ui.py`

## Debugging and Testing Scripts

### 11. `api_debug.py`
**Purpose**: Debugging tool for API connections.
**Features**:
- Tests OpenAI API connection
- Diagnoses API key issues
- Tests various client configurations
**Usage**: Use when troubleshooting API connection problems.

## Setup Process Workflow

For a new installation, follow these steps in order:

1. **Initial Setup**: 
   ```bash
   python setup.py
   ```
   This prepares your environment, installing dependencies and setting up configuration files.

2. **Database Initialization**:
   ```bash
   python init_db.py
   ```
   This creates the necessary database schema with pgvector support.

3. **Populate the Database** (Crawl documentation):
   ```bash
   python pinescript_crawler.py
   ```

4. **Verify Setup**:
   ```bash
   python run.py check
   ```
   This confirms that everything is set up correctly.

5. **Run the Agent** (choose one interface):
   ```bash
   # Command-line interface
   python run.py interactive
   
   # Web interface
   streamlit run streamlit_ui.py 
   ```

## Maintenance Tasks

- **Clear Database**: If you need to start fresh with documentation:
  ```bash
  python clear_database.py
  ```

- **Inspect Database**: To check the contents of the database:
  ```bash
  python db_inspect.py count
  python db_inspect.py list
  python db_inspect.py search "your query"
  ```

- **Debug API Issues**: If you encounter API problems:
  ```bash
  python api_debug.py
  ```
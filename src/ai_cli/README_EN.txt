# AI CLI Assistant

**Version:** 1.4.0

A production-ready command-line interface (CLI) wrapper for interacting with OpenAI and OpenRouter-like API endpoints. This tool allows for interactive AI-powered conversations and file operations within a secure working directory.

---

## Features

- **Interactive Chat**: Have a continuous conversation with an AI model directly in your terminal.
- **File Operations**:
  - `attach` - Include content from a file in your prompt.
  - `create` - Generate a new file using AI based on an instruction.
  - `edit` - Modify an existing file using AI instructions.
  - `save` - Save the AI's last response to a file.
- **Workspace Security**: All file operations are confined to a specified working directory to prevent accidental path traversal.
- **Rich Display**: Uses the `rich` library for beautiful Markdown rendering and formatted panels.
- **Session History**: Maintains a history of the conversation and supports command history.
- **Multi-Provider Support**: Configured to work with any OpenAI-compatible API (OpenAI, OpenRouter, local hosts).

---

## Installation

1.  **Ensure you have Python 3.8+ installed.**
2.  **Install the package using pip:**
    ```bash
    pip install ai-cli-assistant
    ```
    *Note: The package name above is a placeholder; use the actual name if published on PyPI. Alternatively, you can run the script directly by downloading the source code.*

3.  **Install required dependencies:**
    The script requires several libraries. Install them via pip:
    ```bash
    pip install requests python-dotenv rich prompt_toolkit yaspin tqdm
    ```

---

## Configuration

### Environment Variables

Set the following environment variables in your shell or a `.env` file:

- **API Key**: Set one of these variables with your API key.
   - `AI_CLI_API_KEY`
  - `OPENROUTER_API_KEY`

- **Model & Endpoint (Optional)**:
  - `AI_CLI_MODEL`: The model to use (default: `moonshotai/kimi-k2`).
  - `AI_CLI_API_BASE`: The API base URL (default: `https://openrouter.ai/api/v1`).

- **Logging (Optional)**:
  - `AI_CLI_LOG_LEVEL`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default is `WARNING`.

### Command-Line Arguments

You can also configure the tool at runtime:

```bash
python ai_cli.py --model "gpt-4" --base "https://api.openai.com/v1" --work-dir "./my_project"


help			Show this help message.	
exit			Exit the program.	
attach <path>	Read a file and attach its content to the next prompt.	attach script.py
create <path>	 <instruction>	Generate a new file using AI.	create hello_world.py a simple python script that prints hello world
edit   <path> 	<instruction>	Edit an existing file using AI.	edit README.md fix the grammar mistakes
save   <path>	Save the AI's last response to a file.	save idea.txt
history			Show the recent conversation history.	history
clear			Clear the current conversation history.	clear
pwd				Print the current secure working directory.	pwd
cd 	   <path>	Change the current working directory (must be a subdirectory).	cd src
ls				List files in the current working directory.	ls
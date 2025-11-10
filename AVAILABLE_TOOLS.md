# Available Tool Calls in Claude Code

This document lists all tool calls available in Claude Code with short descriptions.

## Core Development Tools

### Task
Launch specialized agents for complex, multi-step tasks autonomously. Available agent types:
- **general-purpose**: Research complex questions, search code, execute multi-step tasks
- **Explore**: Fast agent for exploring codebases, finding files by patterns
- **Plan**: Fast agent for planning implementations

### Bash
Execute bash commands in a persistent shell session for terminal operations like git, npm, docker, etc.
**Note**: Do NOT use for file operations - use specialized tools instead.

### BashOutput
Retrieve output from a running or completed background bash shell.

### KillShell
Terminate a running background bash shell by its ID.

## File Operations

### Read
Read files from the local filesystem. Supports text files, images, PDFs, and Jupyter notebooks.
Returns content with line numbers in cat -n format.

### Write
Write or overwrite files to the local filesystem.
**Important**: Must read existing files first before overwriting.

### Edit
Perform exact string replacements in files.
**Important**: Must read file first before editing. Preserves exact indentation.

### NotebookEdit
Replace, insert, or delete cells in Jupyter notebooks (.ipynb files).

## Search & Discovery

### Glob
Fast file pattern matching tool using glob patterns (e.g., "**/*.js", "src/**/*.ts").
Returns matching file paths sorted by modification time.

### Grep
Powerful search tool built on ripgrep for searching file contents.
- Supports full regex syntax
- Filter by file type or glob patterns
- Multiple output modes: content, files_with_matches, count
- Context lines support (-A, -B, -C)

## Web & External Resources

### WebFetch
Fetch content from a URL and process it with AI.
Converts HTML to markdown and analyzes content based on provided prompt.

### WebSearch
Search the web and get up-to-date information.
Supports domain filtering (allowed/blocked domains).

## Task Management

### TodoWrite
Create and manage a structured task list for tracking progress.
- Task states: pending, in_progress, completed
- Helps break down complex tasks into manageable steps
- Provides visibility into progress

### ExitPlanMode
Use when finished planning and ready to code.
Prompts user to exit plan mode after presenting implementation plan.

## Custom Commands & Skills

### SlashCommand
Execute custom slash commands defined in .claude/commands/*.md.
Routes user intentions to specialized workflows.

### Skill
Execute specialized skills for domain-specific tasks.
Available skills are listed in the system prompt.

## Notes

- **Parallel Execution**: Multiple independent tool calls can be made in parallel for efficiency
- **Sequential Dependencies**: Tools with dependencies must be called sequentially
- **File Operations**: Always use specialized tools (Read, Edit, Write) instead of bash commands (cat, sed, echo)
- **Security**: Be careful not to introduce vulnerabilities (XSS, SQL injection, command injection, etc.)

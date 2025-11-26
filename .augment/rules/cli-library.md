---
type: agent_requested
description: "cli python library"
keywords: ["cli", "typer", "python", "command-line", "terminal"]
---

# Python CLI Framework: Typer

## Framework Choice

**Always use Typer** for building Python CLI applications unless there's a specific reason to use another framework.

Typer is the modern standard for Python CLIs, built on top of Click with automatic help generation from type hints.

## Installation

```bash
pip install "typer[all]"
```

The `[all]` extra includes:
- `rich` for beautiful terminal output
- `shellingham` for shell detection
- All recommended dependencies

## Best Practices

### 1. Use Type Hints

Always use type hints for automatic validation and help generation:

```python
import typer
from typing import Optional
from pathlib import Path

app = typer.Typer()

@app.command()
def process(
    input_file: Path,
    output_file: Optional[Path] = None,
    verbose: bool = False,
    count: int = typer.Option(1, "--count", "-c", help="Number of iterations")
):
    """Process INPUT_FILE and optionally write to OUTPUT_FILE."""
    if verbose:
        typer.echo(f"Processing {input_file}")
```

### 2. Use Typer.Option and Typer.Argument

Be explicit about options vs arguments:

```python
@app.command()
def deploy(
    environment: str = typer.Argument(..., help="Target environment (dev/staging/prod)"),
    config: Path = typer.Option("config.yml", "--config", "-c", help="Config file path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without making changes"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deployment"),
):
    """Deploy to ENVIRONMENT."""
    pass
```

### 3. Organize Commands with Typer Apps

For complex CLIs, use sub-commands:

```python
import typer

app = typer.Typer()
db_app = typer.Typer()
app.add_typer(db_app, name="db", help="Database commands")

@db_app.command("migrate")
def db_migrate():
    """Run database migrations."""
    pass

@db_app.command("seed")
def db_seed():
    """Seed database with test data."""
    pass

@app.command()
def serve():
    """Start the server."""
    pass
```

### 4. Use Rich for Beautiful Output

Typer integrates seamlessly with Rich:

```python
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

@app.command()
def list_items():
    """List all items with beautiful formatting."""
    table = Table(title="Items")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")

    for item in items:
        table.add_row(str(item.id), item.name, item.status)

    console.print(table)

@app.command()
def process_batch():
    """Process items with progress bar."""
    for item in track(items, description="Processing..."):
        process_item(item)
```

### 5. Handle Errors Gracefully

Use Typer's built-in error handling:

```python
@app.command()
def delete(item_id: int):
    """Delete an item by ID."""
    try:
        delete_item(item_id)
        typer.secho(f"✓ Deleted item {item_id}", fg=typer.colors.GREEN)
    except ItemNotFound:
        typer.secho(f"✗ Item {item_id} not found", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
```

### 6. Use Callbacks for Shared Options

For options that apply to all commands:

```python
app = typer.Typer()

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config: Path = typer.Option("config.yml", "--config", help="Config file"),
):
    """
    My CLI Application.

    Use --verbose for detailed output.
    """
    if verbose:
        typer.echo("Verbose mode enabled")
    # Load config and store in context if needed
```

### 7. Use Enums for Choices

Instead of string choices, use Enums:

```python
from enum import Enum

class Environment(str, Enum):
    dev = "dev"
    staging = "staging"
    production = "production"

@app.command()
def deploy(env: Environment):
    """Deploy to environment."""
    typer.echo(f"Deploying to {env.value}")
```



### 8. Add Confirmation Prompts for Dangerous Operations

```python
@app.command()
def delete_all():
    """Delete all data (dangerous!)."""
    typer.confirm("Are you sure you want to delete all data?", abort=True)
    # Proceed with deletion
    typer.echo("All data deleted")
```

### 9. Use Context for Shared State

```python
app = typer.Typer()

@app.callback()
def main(ctx: typer.Context, config: Path = typer.Option("config.yml")):
    """Load configuration."""
    ctx.obj = load_config(config)

@app.command()
def process(ctx: typer.Context):
    """Process using loaded config."""
    config = ctx.obj
    # Use config
```

### 10. Structure Your CLI Module

Recommended project structure:

```
my_project/
├── my_project/
│   ├── __init__.py
│   ├── cli.py          # Main CLI entry point
│   ├── commands/       # Command modules
│   │   ├── __init__.py
│   │   ├── db.py
│   │   └── server.py
│   └── core/           # Business logic
└── pyproject.toml      # With [project.scripts] entry point
```

Entry point in `pyproject.toml`:

```toml
[project.scripts]
myapp = "my_project.cli:app"
```

## Common Patterns

### Interactive Prompts

```python
name = typer.prompt("What's your name?")
password = typer.prompt("Password", hide_input=True)
```

### File Operations

```python
@app.command()
def process(
    input_file: typer.FileText = typer.Argument(...),
    output_file: typer.FileTextWrite = typer.Option(None),
):
    """Process text files."""
    content = input_file.read()
    result = process_content(content)
    if output_file:
        output_file.write(result)
    else:
        typer.echo(result)
```

### Version Option

```python
def version_callback(value: bool):
    if value:
        typer.echo("My App v1.0.0")
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)
):
    pass
```

## Testing

Use Typer's testing utilities:

```python
from typer.testing import CliRunner

runner = CliRunner()

def test_command():
    result = runner.invoke(app, ["command", "--option", "value"])
    assert result.exit_code == 0
    assert "expected output" in result.stdout
```

## Summary

- **Always use type hints** for automatic validation
- **Use Rich** for beautiful output (tables, progress bars, colors)
- **Organize with sub-commands** for complex CLIs
- **Handle errors gracefully** with proper exit codes
- **Add confirmation prompts** for dangerous operations
- **Use Enums** instead of string choices
- **Test with CliRunner** for reliable CLI testing

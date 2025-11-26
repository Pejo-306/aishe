# Development Environment Configuration

## Nix-Based Development Environment

All development environments and system dependencies MUST be managed using **Nix flakes**.

## Core Principles

1. **Use Nix for all system dependencies** - Never ask users to install system packages manually
2. **Use flake.nix** - All development shells must be defined in `flake.nix`
3. **Reproducible environments** - Ensure consistent development environments across all machines
4. **No global installations** - All dependencies should be project-scoped via Nix

## File Structure

Development environment configuration should be in:
- `flake.nix` - Main Nix flake configuration (REQUIRED)
- `flake.lock` - Lock file for reproducible builds (auto-generated)

## Basic flake.nix Template

When creating a new project or adding Nix support, use this structure:

```nix
{
  description = "Project development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Add system dependencies here
            # Example: python311, nodejs, go, etc.
          ];

          shellHook = ''
            echo "Development environment loaded"
            # Add any initialization commands here
          '';
        };
      }
    );
}
```

## Language-Specific Examples

### Python Projects

```nix
buildInputs = with pkgs; [
  python311
  python311Packages.pip
  python311Packages.virtualenv
];

shellHook = ''
  # Create virtual environment if it doesn't exist
  if [ ! -d .venv ]; then
    python -m venv .venv
  fi
  source .venv/bin/activate
  echo "Python development environment loaded"
'';
```

### Node.js Projects

```nix
buildInputs = with pkgs; [
  nodejs_20
  nodePackages.npm
  nodePackages.pnpm
];
```

### Go Projects

```nix
buildInputs = with pkgs; [
  go_1_21
  gotools
  gopls
];
```

### Rust Projects

```nix
buildInputs = with pkgs; [
  rustc
  cargo
  rustfmt
  clippy
];
```

## Common System Dependencies

Include these common tools as needed:

```nix
buildInputs = with pkgs; [
  # Version control
  git

  # Build tools
  gnumake
  gcc

  # Utilities
  curl
  wget
  jq

  # Database clients (if needed)
  postgresql
  redis
  
  # Other tools
  docker
  docker-compose
];
```

## Entering the Development Shell

Users should enter the development environment using:

```bash
# Enter the dev shell
nix develop

# Or use direnv for automatic activation (recommended)
echo "use flake" > .envrc
direnv allow
```

## direnv Integration (Recommended)

For automatic environment activation, include `.envrc`:

```bash
use flake
```

And add to `.gitignore`:
```
.direnv/
```

## Best Practices

1. **Pin nixpkgs version** - Use specific nixpkgs commits for stability when needed
2. **Document dependencies** - Add comments explaining why each dependency is needed
3. **Minimal dependencies** - Only include what's actually required
4. **Use overlays** - For custom package versions or modifications
5. **Test the shell** - Always test `nix develop` works on a clean checkout

## What NOT to Do

❌ **Don't** ask users to install system packages with `apt`, `brew`, `yum`, etc.
❌ **Don't** use global Python/Node/etc. installations
❌ **Don't** commit `flake.lock` to `.gitignore` (it should be committed)
❌ **Don't** mix Nix with other environment managers (like conda, nvm) unless necessary

## Updating Dependencies

To update all flake inputs:
```bash
nix flake update
```

To update a specific input:
```bash
nix flake lock --update-input nixpkgs
```

## Troubleshooting

If users encounter issues:

1. **Ensure Nix is installed** with flakes enabled:
   ```bash
   nix --version
   # Should be 2.4 or higher
   ```

2. **Enable flakes** in `~/.config/nix/nix.conf`:
   ```
   experimental-features = nix-command flakes
   ```

3. **Clear the cache** if needed:
   ```bash
   nix-collect-garbage
   ```

## Example: Complete Python + Ollama Project

```nix
{
  description = "RAG application with Ollama";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            python311Packages.pip
            python311Packages.virtualenv
            ollama
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              python -m venv .venv
            fi
            source .venv/bin/activate
            echo "RAG development environment loaded"
            echo "Ollama available at: $(which ollama)"
          '';
        };
      }
    );
}
```

## Summary

- ✅ Always use `flake.nix` for development environments
- ✅ Include all system dependencies in the flake
- ✅ Test with `nix develop` before committing
- ✅ Document dependencies with comments
- ✅ Commit `flake.lock` to version control


# Publishing Guide

This guide covers publishing `rank-relax` to crates.io.

## Prerequisites

All publishing uses **OIDC (OpenID Connect) authentication** - no manual tokens needed!

1. **crates.io account**: https://crates.io
   - Configure OIDC trusted publisher in GitHub Actions (when ready)
   - Uses `rust-lang/crates-io-auth-action@v1` for authentication

## Publishing Workflow

### Automated (Recommended)

1. **Create a GitHub release**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   Then create a release on GitHub with the same tag.

2. **GitHub Actions will automatically**:
   - Validate version
   - Run tests
   - Publish to crates.io

### Manual Publishing

```bash
# For automated publishing, use GitHub Actions (recommended)
# For manual publishing, you'll need a crates.io API token:
cargo publish --token YOUR_CRATES_IO_TOKEN
```

## Version Management

- Update version in `Cargo.toml`
- Use semantic versioning:
  - `MAJOR.MINOR.PATCH`
  - Breaking changes: increment MAJOR
  - New features: increment MINOR
  - Bug fixes: increment PATCH

## Pre-Publish Checklist

- [ ] All tests pass: `cargo test`
- [ ] Clippy passes: `cargo clippy -- -D warnings`
- [ ] Formatting is correct: `cargo fmt --check`
- [ ] Documentation builds: `cargo doc --no-deps`
- [ ] Version updated in `Cargo.toml`
- [ ] CHANGELOG.md updated (if applicable)

## Status

🚧 **Early development** - Publishing workflow is configured but package may not be ready for publication yet.

## OIDC Configuration

- ⚠️ **crates.io**: Requires OIDC trusted publisher setup (see GitHub Actions workflow)

If OIDC isn't configured, workflows will fail at publish step. Check GitHub repository settings.


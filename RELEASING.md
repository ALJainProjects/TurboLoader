# Releasing TurboLoader

Releases are **fully automated from a git tag**. You never edit a version number by
hand — `setuptools_scm` derives the version from the tag and writes it everywhere
(`turboloader/_version.py`, `version()`/`features()` in the C++ module via
`-DTURBOLOADER_VERSION`, and the wheel/sdist metadata).

## Cut a release

```bash
git checkout main && git pull
git tag -a v2.26.0 -m "v2.26.0"     # the tag IS the version
git push origin v2.26.0
```

Pushing a `v*` tag triggers `.github/workflows/build-wheels.yml`, which:

1. Builds wheels for **Linux x86_64/aarch64** and **macOS arm64/x86_64**
   (CPython 3.10–3.13) with cibuildwheel.
2. Builds the source distribution.
3. Smoke-tests every wheel (import + a transform + version check).
4. **Publishes wheels + sdist to PyPI** (`publish_pypi` job).

## One-time PyPI setup (required before the first automated publish)

The publish job uses **PyPI Trusted Publishing (OIDC)** — no API token is stored in the
repo. Configure it once on PyPI:

1. Log in at <https://pypi.org> → the `turboloader` project → **Settings → Publishing**.
2. **Add a new pending/trusted publisher → GitHub Actions** with:
   - Owner: `ALJainProjects`
   - Repository: `TurboLoader`
   - Workflow filename: `build-wheels.yml`
   - Environment: `pypi`
3. Save. Future tag pushes publish automatically.

### Alternative: API token

If you prefer a token instead of Trusted Publishing:

1. Create a project-scoped token at PyPI → Account → API tokens.
2. Add it as a GitHub repo secret named `PYPI_API_TOKEN`.
3. In `build-wheels.yml`, set the publish step's
   `with: { password: ${{ secrets.PYPI_API_TOKEN }} }`.

## CI on every push / PR

`.github/workflows/test.yml` runs the Python test suite (3.10–3.12), the C++ GoogleTest
suite, and lint (`black --check`, `flake8`). All checkouts use `fetch-depth: 0` so
setuptools_scm can resolve the version.

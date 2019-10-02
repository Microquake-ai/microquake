# README #

Microquake is an open source package licensed under the [GNU general Public License, version 3 (GPLv3)](http://www.gnu.org/licenses/gpl-3.0.html). Microquake is an extension of Obspy for the processing of microseismic data

### Development

```
pip install poetry
poetry config http-basic.microquake {user} {password}
poetry install
```

Running tests

```
poetry run pytest
```

### How to release a new version

```
poetry version
git add pyproject.toml
gc -m "bump version"
git tag newversion
git push --tags
```

### Automatic tagging and releasing

By adding the following command to your git config you can bump and release a new version with one command

```
git config --global alias.bump "\!version=\$(poetry version | awk '{print \$NF}' ) && git add pyproject.toml && git commit -m \"Bumping version to \$version\" && git tag \$version && git push --tags"
```

After running the above command you may release a new version with:

```
git bump
```


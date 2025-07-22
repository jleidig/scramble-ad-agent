## Develop

```bash
pipenv install

pipenv run python -m src.main
```

## Authenticate w/ CapTech AI Lab

```bash
azd auth login --scope api://ailab/Model.Access
```

## Build w/ Docker Compose

```bash
docker compose up --build
```

## Run E2E Tests

```bash
pipenv run robot src/tests/e2e.robot
```

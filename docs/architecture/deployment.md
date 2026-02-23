# Deployment Diagram

## Overview

This document describes the deployment topology for MangoMAS Demo across three environments: local development, Docker container, and HuggingFace Spaces. It also covers the CI/CD pipeline that automates linting, testing, building, and deployment through GitHub Actions.

## Deployment Environments

```mermaid
graph TB
    subgraph LocalDev["Local Development Environment"]
        direction TB
        DevMachine["<b>Developer Machine</b><br/><i>macOS / Linux / Windows</i>"]

        subgraph PythonLocal["Python 3.11+ Runtime"]
            AppLocal["<b>app.py</b><br/>Gradio entrypoint<br/>Port 7860"]
            PkgLocal["<b>mangomas_demo/</b><br/>Full package:<br/>features, models, cells,<br/>mcts, routing, agents, ui"]
            LogLocal["<b>Logging</b><br/>MANGOMAS_LOG_LEVEL=DEBUG<br/>MANGOMAS_LOG_FORMAT=text"]
        end

        subgraph DevDeps["Development Dependencies"]
            Ruff["ruff (linter)"]
            Mypy["mypy (type checker)"]
            Pytest["pytest + pytest-cov"]
            PreCommit["pre-commit hooks"]
        end

        DevMachine --> PythonLocal
        DevMachine --> DevDeps
    end

    subgraph DockerEnv["Docker Container"]
        direction TB
        DockerHost["<b>Docker Host</b><br/><i>Any Docker-capable machine</i>"]

        subgraph Container["python:3.11-slim Container"]
            Workdir["<b>WORKDIR /app</b>"]
            SysDeps["<b>System Deps</b><br/>gcc, g++"]
            Reqs["<b>requirements.txt</b><br/>gradio, plotly, numpy, torch"]
            AppDocker["<b>app.py</b><br/>CMD python app.py"]
            HealthCheck["<b>HEALTHCHECK</b><br/>interval=30s, timeout=10s<br/>retries=3, start-period=30s<br/>python -c import mangomas_demo"]
            Port["<b>EXPOSE 7860</b>"]
        end

        DockerHost --> Container
    end

    subgraph HFSpace["HuggingFace Spaces"]
        direction TB
        HFInfra["<b>HuggingFace Infrastructure</b><br/><i>Managed container runtime</i>"]

        subgraph SpaceRuntime["Mango-Metrics-NLM/MangoMAS Space"]
            HFApp["<b>Gradio App</b><br/>Served at public URL<br/>https://huggingface.co/spaces/<br/>Mango-Metrics-NLM/MangoMAS"]
            HFFiles["<b>Uploaded Files</b><br/>All source minus:<br/>.git, tests/, docs/,<br/>.github/, __pycache__/"]
        end

        HFInfra --> SpaceRuntime
    end

    User["<b>End User</b><br/><i>Browser</i>"]

    User -->|"http://localhost:7860"| AppLocal
    User -->|"http://host:7860"| Port
    User -->|"https://...huggingface.co/..."| HFApp

    style LocalDev fill:none,stroke:#2D6A4F,stroke-width:2px,stroke-dasharray:5
    style DockerEnv fill:none,stroke:#0077B6,stroke-width:2px,stroke-dasharray:5
    style HFSpace fill:none,stroke:#7B2CBF,stroke-width:2px,stroke-dasharray:5
    style User fill:#08427B,stroke:#052E56,color:#fff
    style PythonLocal fill:#D8F3DC,stroke:#2D6A4F
    style Container fill:#CAF0F8,stroke:#0077B6
    style SpaceRuntime fill:#E8DAEF,stroke:#7B2CBF
```

## CI/CD Pipeline

The project uses two GitHub Actions workflows: `ci.yml` for continuous integration and `cd.yml` for continuous deployment.

### CI Pipeline (ci.yml)

Triggered on every push and pull request to `main`. Three sequential jobs ensure code quality before any deployment.

```mermaid
graph LR
    subgraph Trigger["Trigger Events"]
        Push["push to main"]
        PR["pull_request to main"]
    end

    subgraph LintJob["Job: Lint & Type Check"]
        direction TB
        Checkout1["actions/checkout@v4"]
        SetupPy1["setup-python 3.11"]
        Install1["pip install -e .[dev]"]
        RuffLint["ruff check<br/>mangomas_demo/ tests/"]
        MypyCheck["mypy mangomas_demo/<br/>(strict mode)"]

        Checkout1 --> SetupPy1 --> Install1 --> RuffLint --> MypyCheck
    end

    subgraph TestJob["Job: Test (matrix: 3.11, 3.12)"]
        direction TB
        Checkout2["actions/checkout@v4"]
        SetupPy2["setup-python<br/>matrix.python-version"]
        Install2["pip install -e .[dev]"]
        RunTests["pytest tests/ -v<br/>--cov=mangomas_demo<br/>--cov-fail-under=80"]
        UploadCov["Upload coverage.xml<br/>(Python 3.11 only)"]

        Checkout2 --> SetupPy2 --> Install2 --> RunTests --> UploadCov
    end

    subgraph BuildJob["Job: Build Package"]
        direction TB
        Checkout3["actions/checkout@v4"]
        SetupPy3["setup-python 3.11"]
        BuildPkg["python -m build<br/>(sdist + wheel)"]
        VerifyInstall["pip install dist/*.whl<br/>python -c import mangomas_demo"]

        Checkout3 --> SetupPy3 --> BuildPkg --> VerifyInstall
    end

    Push --> LintJob
    PR --> LintJob
    LintJob -->|"needs: lint"| TestJob
    TestJob -->|"needs: test"| BuildJob

    style Trigger fill:#E9C46A,stroke:#C9A24A,color:#000
    style LintJob fill:#2A9D8F,stroke:#1E7A6E,color:#fff
    style TestJob fill:#264653,stroke:#1A3040,color:#fff
    style BuildJob fill:#E76F51,stroke:#C5533A,color:#fff
```

### CD Pipeline (cd.yml)

Triggered only on push to `main` (excluding docs and markdown changes). Deploys to HuggingFace Spaces.

```mermaid
graph LR
    subgraph CDTrigger["Trigger"]
        PushMain["push to main<br/>(paths-ignore: docs/**, *.md)"]
    end

    subgraph DeployJob["Job: Build & Deploy"]
        direction TB
        CDCheckout["actions/checkout@v4"]
        CDSetupPy["setup-python 3.11"]
        CDInstallHF["pip install huggingface_hub"]
        CDDeploy["HfApi.upload_folder()<br/>repo_id: Mango-Metrics-NLM/MangoMAS<br/>repo_type: space<br/>ignore: .git, tests, docs, .github"]
        CDSmoke["Smoke Test<br/>HTTP GET HuggingFace Space URL<br/>Assert status == 200"]

        CDCheckout --> CDSetupPy --> CDInstallHF --> CDDeploy --> CDSmoke
    end

    PushMain --> DeployJob

    subgraph HFTarget["HuggingFace Space"]
        LiveApp["Mango-Metrics-NLM/MangoMAS<br/>Public Gradio app"]
    end

    CDDeploy -->|"HF_TOKEN secret"| HFTarget

    style CDTrigger fill:#E9C46A,stroke:#C9A24A,color:#000
    style DeployJob fill:#7B2CBF,stroke:#5A189A,color:#fff
    style HFTarget fill:#E8DAEF,stroke:#7B2CBF
```

## Docker Build Stages

The `Dockerfile` uses `python:3.11-slim` as the base image with a single-stage build:

```mermaid
graph TB
    subgraph DockerBuild["Dockerfile Build Process"]
        direction TB
        Base["<b>FROM python:3.11-slim</b><br/>Minimal Python base image"]
        Workdir["<b>WORKDIR /app</b>"]
        SysPkgs["<b>RUN apt-get install</b><br/>gcc, g++<br/>(for compiled dependencies)"]
        CopyReqs["<b>COPY requirements.txt .</b>"]
        PipReqs["<b>RUN pip install -r requirements.txt</b><br/>gradio, plotly, numpy, torch"]
        CopyAll["<b>COPY . .</b><br/>Full application source"]
        PipEditable["<b>RUN pip install -e .</b><br/>Editable install via hatchling"]
        Expose["<b>EXPOSE 7860</b><br/>Gradio default port"]
        Health["<b>HEALTHCHECK</b><br/>Every 30s: python -c<br/>import mangomas_demo; print ok"]
        Cmd["<b>CMD python app.py</b><br/>Launch Gradio server<br/>on 0.0.0.0:7860"]

        Base --> Workdir --> SysPkgs --> CopyReqs --> PipReqs --> CopyAll --> PipEditable --> Expose --> Health --> Cmd
    end

    style Base fill:#0077B6,stroke:#005A8A,color:#fff
    style SysPkgs fill:#0096C7,stroke:#006F94,color:#fff
    style PipReqs fill:#48CAE4,stroke:#2BA3C0,color:#000
    style CopyAll fill:#90E0EF,stroke:#5BC0D0,color:#000
    style PipEditable fill:#48CAE4,stroke:#2BA3C0,color:#000
    style Cmd fill:#023E8A,stroke:#012A5E,color:#fff
    style Health fill:#0077B6,stroke:#005A8A,color:#fff
```

## Environment Variables

The following environment variables control runtime behavior across all deployment targets:

```mermaid
graph LR
    subgraph EnvVars["Environment Variables"]
        direction TB
        LogLevel["<b>MANGOMAS_LOG_LEVEL</b><br/>Default: INFO<br/>Options: DEBUG, INFO,<br/>WARNING, ERROR, CRITICAL"]
        LogFormat["<b>MANGOMAS_LOG_FORMAT</b><br/>Default: text<br/>Options: text, json"]
        HFToken["<b>HF_TOKEN</b><br/>GitHub Actions secret<br/>Used by CD pipeline for<br/>HuggingFace deployment"]
    end

    subgraph Targets["Deployment Targets"]
        Local["Local Dev"]
        Docker["Docker Container"]
        HFSpace["HuggingFace Space"]
        GHACI["GitHub Actions CI"]
    end

    LogLevel --> Local
    LogLevel --> Docker
    LogLevel --> HFSpace
    LogFormat --> Local
    LogFormat --> Docker
    LogFormat --> HFSpace
    HFToken --> GHACI

    style EnvVars fill:none,stroke:#D4A373,stroke-width:2px,stroke-dasharray:5
    style LogLevel fill:#E9C46A,stroke:#C9A24A,color:#000
    style LogFormat fill:#E9C46A,stroke:#C9A24A,color:#000
    style HFToken fill:#E76F51,stroke:#C5533A,color:#fff
```

## Network Topology

```mermaid
graph TB
    Browser["<b>Browser</b><br/>User's device"]

    subgraph Internet["Public Internet"]
        HFURL["https://huggingface.co/spaces/<br/>Mango-Metrics-NLM/MangoMAS"]
        GitHubURL["https://github.com/<br/>Mango-Metrics-NLM/MangoMas-Demo"]
    end

    subgraph HFInfra["HuggingFace Infrastructure"]
        HFProxy["HF Reverse Proxy"]
        HFContainer["Gradio Container<br/>Port 7860"]
    end

    subgraph GHAInfra["GitHub Actions Runners"]
        CIRunner["ubuntu-latest<br/>CI Jobs"]
        CDRunner["ubuntu-latest<br/>CD Job"]
    end

    Browser -->|"HTTPS"| HFURL
    HFURL --> HFProxy
    HFProxy -->|"HTTP :7860"| HFContainer

    GitHubURL -->|"Webhook"| CIRunner
    CDRunner -->|"HfApi (HTTPS)"| HFInfra

    style Browser fill:#08427B,stroke:#052E56,color:#fff
    style HFContainer fill:#7B2CBF,stroke:#5A189A,color:#fff
    style CIRunner fill:#2D6A4F,stroke:#1B4332,color:#fff
    style CDRunner fill:#2D6A4F,stroke:#1B4332,color:#fff
```

<#
.SYNOPSIS
    Windows PowerShell equivalent of the project Makefile.
    Usage: .\run.ps1 <command>

.EXAMPLE
    .\run.ps1 help
    .\run.ps1 install
    .\run.ps1 train
    .\run.ps1 test
    .\run.ps1 run-dashboard
    .\run.ps1 run-api
    .\run.ps1 run-frontend
    .\run.ps1 run-evaluate
    .\run.ps1 lint
    .\run.ps1 clean
    .\run.ps1 docker-build
    .\run.ps1 docker-up
    .\run.ps1 docker-down
    .\run.ps1 docker-logs
#>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# ── colour helpers ────────────────────────────────────────────────────────────
function Write-Info  { param($m) Write-Host "  $m" -ForegroundColor Cyan    }
function Write-Ok    { param($m) Write-Host "  ✓ $m" -ForegroundColor Green  }
function Write-Err   { param($m) Write-Host "  ✗ $m" -ForegroundColor Red    }
function Write-Step  { param($m) Write-Host "`n▶ $m" -ForegroundColor Yellow }

# ── activate venv if present ──────────────────────────────────────────────────
function Activate-Venv {
    $venvScript = ".\venv\Scripts\Activate.ps1"
    if (Test-Path $venvScript) {
        Write-Info "Activating virtual environment..."
        & $venvScript
    }
}

switch ($Command) {

    "help" {
        Write-Host ""
        Write-Host "  Undersea Cable Fault Detection — Windows Command Runner" -ForegroundColor Cyan
        Write-Host "  ─────────────────────────────────────────────────────────" -ForegroundColor DarkGray
        Write-Host ""
        $rows = @(
            @{ Cmd="install";       Desc="Install Python dependencies from requirements.txt" },
            @{ Cmd="train";         Desc="Train the Conv-Transformer autoencoder (python model.py)" },
            @{ Cmd="test";          Desc="Run pytest test suite" },
            @{ Cmd="run-dashboard"; Desc="Launch the Streamlit dashboard on :8501" },
            @{ Cmd="run-api";       Desc="Launch the FastAPI server on :8000" },
            @{ Cmd="run-frontend";  Desc="Launch the React/Vite dev server on :5173" },
            @{ Cmd="run-evaluate";  Desc="Run model evaluation and generate plots" },
            @{ Cmd="lint";          Desc="Syntax-check all Python source files" },
            @{ Cmd="clean";         Desc="Remove __pycache__, .pytest_cache, evaluation_plots" },
            @{ Cmd="docker-build";  Desc="Build Docker images" },
            @{ Cmd="docker-up";     Desc="Start all services with docker-compose" },
            @{ Cmd="docker-down";   Desc="Stop all docker-compose services" },
            @{ Cmd="docker-logs";   Desc="Tail docker-compose logs" }
        )
        foreach ($r in $rows) {
            Write-Host ("  {0,-18}" -f $r.Cmd) -NoNewline -ForegroundColor White
            Write-Host $r.Desc -ForegroundColor DarkGray
        }
        Write-Host ""
        Write-Host "  Usage: .\run.ps1 <command>" -ForegroundColor DarkGray
        Write-Host ""
    }

    "install" {
        Write-Step "Installing Python dependencies..."
        Activate-Venv
        pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) { Write-Ok "Dependencies installed." }
        else                      { Write-Err "pip install failed." }
    }

    "train" {
        Write-Step "Training Conv-Transformer Autoencoder..."
        Activate-Venv
        python model.py
        if ($LASTEXITCODE -eq 0) { Write-Ok "Training complete. Model saved to saved_model/." }
        else                      { Write-Err "Training failed." }
    }

    "test" {
        Write-Step "Running pytest..."
        Activate-Venv
        python -m pytest tests/ -v
        if ($LASTEXITCODE -eq 0) { Write-Ok "All tests passed." }
        else                      { Write-Err "Some tests failed." }
    }

    "run-dashboard" {
        Write-Step "Starting Streamlit dashboard on http://localhost:8501 ..."
        Activate-Venv
        python -m streamlit run dashboard.py
    }

    "run-api" {
        Write-Step "Starting FastAPI server on http://localhost:8000 ..."
        Activate-Venv
        uvicorn api:app --reload --port 8000
    }

    "run-frontend" {
        Write-Step "Starting React/Vite frontend on http://localhost:5173 ..."
        if (-not (Test-Path "frontend\package.json")) {
            Write-Err "frontend\package.json not found. Run from project root."
            exit 1
        }
        Set-Location frontend
        npm run dev
        Set-Location ..
    }

    "run-evaluate" {
        Write-Step "Running evaluation..."
        Activate-Venv
        python evaluate.py
        if ($LASTEXITCODE -eq 0) { Write-Ok "Evaluation complete. Plots saved to evaluation_plots/." }
        else                      { Write-Err "Evaluation failed." }
    }

    "lint" {
        Write-Step "Checking Python syntax..."
        Activate-Venv
        $files = @("model.py", "simulator.py", "api.py", "dashboard.py", "utils.py", "config.py")
        $allOk = $true
        foreach ($f in $files) {
            python -c "import py_compile; py_compile.compile('$f', doraise=True)" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Ok "$f" }
            else                      { Write-Err "$f  ← syntax error!"; $allOk = $false }
        }
        if ($allOk) { Write-Host "`n  All files OK." -ForegroundColor Green }
        else         { Write-Host "`n  Errors found above." -ForegroundColor Red }
    }

    "clean" {
        Write-Step "Cleaning generated files..."
        $targets = @(
            "__pycache__", ".pytest_cache", ".coverage",
            "htmlcov", "evaluation_plots", "model_registry"
        )
        foreach ($t in $targets) {
            if (Test-Path $t) {
                Remove-Item -Recurse -Force $t
                Write-Info "Removed $t"
            }
        }
        # Remove nested __pycache__ dirs
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Ok "Clean complete."
    }

    "docker-build" {
        Write-Step "Building Docker images..."
        docker-compose build
    }

    "docker-up" {
        Write-Step "Starting all services with docker-compose..."
        docker-compose up -d
        Write-Ok "Services started. Use '.\run.ps1 docker-logs' to tail logs."
    }

    "docker-down" {
        Write-Step "Stopping docker-compose services..."
        docker-compose down
        Write-Ok "Services stopped."
    }

    "docker-logs" {
        Write-Step "Tailing docker-compose logs (Ctrl+C to stop)..."
        docker-compose logs -f
    }

    default {
        Write-Err "Unknown command: '$Command'"
        Write-Host "  Run '.\run.ps1 help' to see available commands." -ForegroundColor DarkGray
        exit 1
    }
}

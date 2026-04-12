<#
.SYNOPSIS
    Windows PowerShell project runner.
    Usage: .\run.ps1 <command>
#>

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# -- utility functions (ASCII only for max compatibility) --
function Write-Info  { param($m) Write-Host "  [INFO] $m" -ForegroundColor Cyan    }
function Write-Ok    { param($m) Write-Host "  [OK]   $m" -ForegroundColor Green   }
function Write-Err   { param($m) Write-Host "  [ERR]  $m" -ForegroundColor Red     }
function Write-Step  { param($m) Write-Host "`n[STEP] $m" -ForegroundColor Yellow }

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
        Write-Host "  Undersea Cable Monitor - Command Runner" -ForegroundColor Cyan
        Write-Host "  ---------------------------------------" -ForegroundColor DarkGray
        Write-Host ""
        $rows = @(
            @{ Cmd="install";       Desc="Install Python dependencies" },
            @{ Cmd="train";         Desc="Train the Conv-Transformer model" },
            @{ Cmd="test";          Desc="Run pytest suite" },
            @{ Cmd="run-api";       Desc="Launch FastAPI backend (:8000)" },
            @{ Cmd="run-frontend";  Desc="Launch React frontend (:5173+)" },
            @{ Cmd="clean";         Desc="Clean temporary files" }
        )
        foreach ($r in $rows) {
            $c = $r.Cmd
            Write-Host ("  {0,-18}" -f $c) -NoNewline -ForegroundColor White
            Write-Host $r.Desc -ForegroundColor DarkGray
        }
        Write-Host ""
        Write-Host "  Usage: .\run.ps1 `<command`>" -ForegroundColor DarkGray
        Write-Host ""
    }

    "install" {
        Write-Step "Installing dependencies..."
        Activate-Venv
        pip install -r requirements.txt
    }

    "train" {
        Write-Step "Training model..."
        Activate-Venv
        python model.py
    }

    "test" {
        Write-Step "Running tests..."
        Activate-Venv
        python -m pytest test_core.py -v
    }

    "run-api" {
        Write-Step "Starting FastAPI..."
        Activate-Venv
        # Check for port conflict and kill if necessary
        $proc = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
        if ($proc) { 
            Write-Info "Cleaning up port 8000..."
            Stop-Process -Id $proc.OwningProcess -Force 
        }
        python api.py
    }

    "run-frontend" {
        Write-Step "Starting Frontend..."
        if (-not (Test-Path "frontend")) { Write-Err "Frontend dir missing"; exit 1 }
        Set-Location frontend
        npm run dev
    }

    "clean" {
        Write-Step "Cleaning..."
        Remove-Item -Recurse -Force __pycache__, .pytest_cache, generated_reports -ErrorAction SilentlyContinue
        Write-Ok "Clean done."
    }

    default {
        Write-Err "Unknown command: $Command. Use 'help'."
        exit 1
    }
}

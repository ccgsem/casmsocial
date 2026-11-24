# CASMSOCIAL: Native Windows Installation and Verification

For Windows x64 with 64-bit CPython 3.12. Use PowerShell unless a step specifies CMD.

> **Not the locked install.** The README's `uv sync --locked` installs the Repast4Py and DuckDB versions in `uv.lock`, which have no native Windows support. This guide instead creates its own environments and installs with `uv pip install`, using Repast4Py 1.3.0 (a prebuilt no-MPI wheel or a source build against Microsoft MPI). Dependency versions can therefore differ from `uv.lock`; the README's version notes do not apply here.

Complete Sections 1 and 2, then choose [single-rank installation](#3-path-a-single-rank-installation) or [distributed MPI installation](#4-path-b-distributed-mpi-installation).

## 1. Shared prerequisites

**Shell: PowerShell. Location: any directory.** Finish software installations and reopen PowerShell as needed before recording repository paths in Section 2.

### 1.1 Windows Package Manager (`winget`): optional installer convenience

**Check:**

```powershell
winget --version
```

**Expected:** a version beginning with `v`. If it prints a version, continue to Python.

**If it is not recognized:** install or update Microsoft's **App Installer** through the Microsoft Store or an approved software catalog, then reopen PowerShell and repeat the check. Microsoft distributes WinGet with App Installer. [R1]

**Decision:** WinGet is not a CASMSOCIAL runtime dependency. If your organization does not provide it, use the official/approved installer alternatives below instead. Do not stop the whole installation solely because WinGet is unavailable.

### 1.2 Python 3.12, x64

**Check:**

```powershell
py -3.12 --version
py -3.12 -c "import struct, platform; print(struct.calcsize('P') * 8); print(platform.machine())"
```

**Expected:**

```text
Python 3.12.x
64
AMD64
```

`x` represents a patch version, not text to type. Use an approved maintained Python 3.12 build; there is no requirement to install the old patch version used in the original test transcript. This guide's wheel/build workflow targets x64, not native ARM64 or 32-bit Python.

**Install only if Python 3.12 x64 is missing:**

```powershell
winget install --id Python.Python.3.12 -e --architecture x64
```

The WinGet package identifier and x64 architecture selector are documented in the package manifest and WinGet command reference. [R12] [R13]

Without WinGet, use an approved Python 3.12 x64 installation from Python.org. For the traditional installer, select **Windows installer (64-bit)**, not the embeddable package, and include the Python launcher. The Python 3.12.10 release page provides that installer; later 3.12 security updates may have different distribution options. Follow your organization's update policy. [R2]

**Verify / continue:** reopen PowerShell and repeat both checks. Continue only when the selected interpreter reports Python 3.12 and 64 bits. If `py` is not recognized but Python is installed, repair/enable the Python launcher through the approved installer rather than repeatedly installing Python. A 32-bit result requires an x64 interpreter before proceeding.

### 1.3 Git

**Check:**

```powershell
git --version
```

**Expected:** `git version ...`. The exact version does not need to match a sample transcript.

**Install only if missing:**

```powershell
winget install --id Git.Git -e --source winget
```

Without WinGet, use the official **Git for Windows/x64 Setup** installer and enable Git for use from the command line. [R3]

**Verify / continue:** reopen PowerShell and repeat `git --version`. Skip installation when that check already succeeds.

### 1.4 uv

**Check:**

```powershell
uv --version
```

**Expected:** `uv ...` followed by a version.

**Install only if missing:**

```powershell
winget install --id astral-sh.uv -e
```

An alternative, only where running the official installer is permitted, is:

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Use one installation method, not both. Astral also provides standalone release binaries for organizations that require a different approved deployment method. [R5]

**Verify / continue:** reopen PowerShell and repeat `uv --version`. If installation is blocked, use the approved software channel; do not change execution policy or disable TLS verification. Certificate troubleshooting is optional and appears in Section 7.1.

---

## 2. Clone CASMSOCIAL and establish repository paths

### 2.1 Clone the repository

**Shell: PowerShell.** Navigate to an existing parent directory of your choice. Replace the example path below before running it.

```powershell
cd "C:\path\to\your\chosen\parent"
git clone https://github.com/ccgsem/casmsocial.git
cd casmsocial
```

**Expected:** clone progress. The clone is on `main`, which contains the Windows compatibility changes (`casmsocial/mpi_compat.py`). The prompt's directory should now end in `casmsocial`.

**Decision:** if cloning fails, stop here. For an authorized fork, use its clone URL and a branch containing the same changes.

If you already have a checkout, navigate to it instead of cloning over it, and update it with `git switch main` and `git pull --ff-only`. Preserve local changes first. A separate checkout is preferable for a clean-install test; deleting a working repository is not required.

### 2.2 Record paths in the current PowerShell session

**Required now and whenever you resume in a new PowerShell window.** First `cd` into the actual CASMSOCIAL repository, not its parent directory.

Run this block as a whole:

```powershell
& {
    if (-not (Test-Path .\pyproject.toml -PathType Leaf) -or
        -not (Test-Path .\casmsocial\mpi_compat.py -PathType Leaf)) {
        throw "Stop: enter the CASMSOCIAL checkout containing mpi_compat.py first."
    }
    $env:CASMSOCIAL_REPO = (Get-Location).Path
    $env:CASM_SINGLE_PYTHON = Join-Path $env:CASMSOCIAL_REPO ".venv-repast-test\Scripts\python.exe"
    $env:CASM_MPI_PYTHON = Join-Path $env:CASMSOCIAL_REPO ".venv-mpi-test\Scripts\python.exe"
    Write-Host "CASMSOCIAL repository: $env:CASMSOCIAL_REPO"
}
```

**Expected:** one line identifying your chosen CASMSOCIAL checkout. The two Python executables do not need to exist yet.

**Decision:** if the block stops, correct the directory or update the checkout to a `main` that contains `mpi_compat.py`. Do not create environments in an unrelated directory. Later environment-creation commands use these absolute paths specifically to prevent that mistake.

**Next:** follow **Section 3 for Path A** or **Section 4 for Path B**. There is no need to complete both.

---

## 3. Path A: Single-rank installation

**Shell: PowerShell. Location: CASMSOCIAL repository.** This path uses the prebuilt no-MPI Repast4Py wheel. It does not enable distributed execution.

```powershell
cd "$env:CASMSOCIAL_REPO"
```

### 3.1 Create or verify the single-rank environment

**Check:**

```powershell
Test-Path "$env:CASM_SINGLE_PYTHON"
```

**If `False`, create it:**

```powershell
uv venv (Join-Path $env:CASMSOCIAL_REPO ".venv-repast-test") --python 3.12
```

**Expected on creation:** `Creating virtual environment at: ...\.venv-repast-test`. The printed activation suggestion can be ignored; this guide invokes the environment's interpreter directly.

**If `True`:** skip creation. Do not recreate an existing environment merely because the guide contains a creation command.

**Verify in either case:**

```powershell
& "$env:CASM_SINGLE_PYTHON" --version
& "$env:CASM_SINGLE_PYTHON" -c "import sys, struct; print(sys.executable); print(struct.calcsize('P') * 8)"
```

**Continue when:** Python is 3.12, the executable is inside the chosen checkout's `.venv-repast-test`, and bitness is `64`. Stop and correct a wrong-version or wrong-location environment before installing packages.

### 3.2 Install or verify the single-rank Repast4Py wheel

**Check:**

```powershell
uv pip show --python "$env:CASM_SINGLE_PYTHON" repast4py
```

**Decision:** if Repast4Py is absent, use the installation below. If present, run the backend verification below before deciding to skip installation: version `1.3.0` alone does not identify which MPI build is installed.

**Install only if missing:** open the official [Repast4Py Windows no-MPI installation page][R4] and download its **Python 3.12** wheel:

```text
repast4py-1.3.0-cp312-cp312-win_amd64.whl
```

Enter the actual path to that downloaded file; no particular Downloads folder is required.

```powershell
$wheel = Read-Host "Enter the full path to the downloaded wheel, without surrounding quotes"
Test-Path -LiteralPath $wheel
```

**Expected:** `True`. If `False`, correct the path before running the installation.

```powershell
$env:R4PY_NO_MPI = "1"
uv pip install --python "$env:CASM_SINGLE_PYTHON" "$wheel"
```

**Verify in either case:**

```powershell
& "$env:CASM_SINGLE_PYTHON" -c "from importlib.metadata import version; from repast4py import MPI, core, space, _mpi_config; print('Repast4Py:', version('repast4py')); print('Single-rank build:', _mpi_config.SINGLE_RANK); print('World size:', MPI.COMM_WORLD.Get_size()); assert version('repast4py') == '1.3.0'; assert _mpi_config.SINGLE_RANK; assert MPI.COMM_WORLD.Get_size() == 1"
```

**Expected:**

```text
Repast4Py: 1.3.0
Single-rank build: True
World size: 1
```

A warning that a single-rank/mock MPI substitute is being used is expected for this path. If the marker is `False`, stop: this is a real-MPI build, not the intended Path A environment. Do not mix both builds in one environment.

### 3.3 Install CASMSOCIAL and its declared dependencies

**Location: CASMSOCIAL repository, not the Repast4Py source directory.**

```powershell
cd "$env:CASMSOCIAL_REPO"
uv pip show --python "$env:CASM_SINGLE_PYTHON" casmsocial
```

**Expected if installed:** package details, including an editable project location corresponding to this checkout. If it is absent or points to another checkout, use the installation below. Otherwise skip to Section 3.5 and still perform the remaining verification.

**Before installing, preview the change:**

```powershell
uv pip install --dry-run --python "$env:CASM_SINGLE_PYTHON" -e .
```

**Expected on a fresh setup:** `Would install ...` and a CASMSOCIAL entry. **This does not install anything.** Check that it does not propose replacing or rebuilding the working Repast4Py package. If it does, stop and resolve the dependency/build selection before proceeding.

**Install after a satisfactory dry run:**

```powershell
uv pip install --python "$env:CASM_SINGLE_PYTHON" -e .
uv pip show --python "$env:CASM_SINGLE_PYTHON" casmsocial
```

**Expected:** a completed installation and CASMSOCIAL package details for this checkout. The reference run used CASMSOCIAL `2.11.0`; another revision requires rerunning validation rather than assuming the same results.

### 3.4 Verify logging and environment-variable dependencies

CASMSOCIAL declares **`loguru`** (logging) and **`python-dotenv`** (loads optional `.env` configuration) as project dependencies. They should normally be installed by the editable CASMSOCIAL installation above, but verify them explicitly. **Run these commands in PowerShell from the CASMSOCIAL repository.**

**Check:**

```powershell
uv pip show --python "$env:CASM_SINGLE_PYTHON" loguru python-dotenv
```

**Expected if installed:** two package entries, one with `Name: loguru` and one with `Name: python-dotenv`, each showing a version and installation location. Exact versions may differ. A warning such as `Package(s) not found for: python-dotenv` means that package is missing.

**Decision:** if **both** are listed, **skip the installation command** and continue to the import verification. If **either** is missing, install the dependencies into this same environment:

```powershell
uv pip install --python "$env:CASM_SINGLE_PYTHON" loguru python-dotenv
```

**Verify (required whether or not installation was skipped):**

```powershell
& "$env:CASM_SINGLE_PYTHON" -c "from loguru import logger; from dotenv import load_dotenv; print('Logging and dotenv dependencies working')"
```

**Expected:** `Logging and dotenv dependencies working` and no traceback. If `ModuleNotFoundError` occurs, **stop** and confirm that the install and verification commands target the same virtual environment. Do not proceed to tests until the imports work.

**`.env` file clarification:** `python-dotenv` is a Python **package**; `.env` is an optional **configuration file**. The basic local MVP does **not** require creating a `.env` file or providing remote-provider API credentials. Other scenarios may document additional environment variables. Never commit a `.env` containing secrets.

### 3.5 Check DuckDB, PyArrow, and pytest

**Check:**

```powershell
uv pip show --python "$env:CASM_SINGLE_PYTHON" duckdb pyarrow pytest
```

**Decision:** the reference workflow requires **DuckDB 1.5.5**. Run the following installation if DuckDB is absent, its version differs from `1.5.5`, or PyArrow/pytest is missing. If all three are present and DuckDB matches, skip the installation but verify below.

```powershell
uv pip install --python "$env:CASM_SINGLE_PYTHON" "duckdb==1.5.5" pyarrow pytest
```

A downgrade from a different DuckDB version is intentional for this reference workflow. This is not a claim that every newer release fails; it keeps the known-good version used in the validated run.

**Verify the version and the exact Arrow operation used by the model:**

```powershell
& "$env:CASM_SINGLE_PYTHON" -c "import duckdb, pyarrow, pytest; print('DuckDB:', duckdb.__version__); c=duckdb.connect(':memory:'); print(c.execute('SELECT 1 AS x').arrow().read_all().to_pylist()); c.close(); assert duckdb.__version__ == '1.5.5'"
```

**Expected:**

```text
DuckDB: 1.5.5
[{'x': 1}]
```

A missing PyArrow import or missing `.arrow()` method is a failure at this step, not a reason to continue to simulation tests.

### 3.6 Verify the installed application

```powershell
uv pip check --python "$env:CASM_SINGLE_PYTHON"
& "$env:CASM_SINGLE_PYTHON" -m casmsocial --help
```

**Expected:** a nonempty installed environment with no dependency conflicts, followed by CLI help containing `parameters_file` and optional `parameters` arguments. The single-rank warning remains expected.

**Decision:** help output proves that the entry point loads; it is not a completed simulation. Continue to the regression tests only after these checks pass.

### 3.7 Run the selected single-rank regression suite

```powershell
& "$env:CASM_SINGLE_PYTHON" -m pytest `
    tests/test_mpi_compat.py `
    tests/test_mvp_smoke.py `
    tests/test_model.py `
    tests/test_person.py `
    tests/test_event_place_resolution.py `
    -q --tb=short
```

**Reference result:**

```text
142 passed, 1 warning in ...
```

The expected warning identifies Repast4Py's single-rank substitute. Timing varies. The count applies to the tested repository revision, not every future version.

**Continue/finish:** if this selected suite passes, Path A is complete for the tested scope. No MPI SDK/compiler installation or custom Windows launcher is needed. This is not a claim that every test in `tests/` has passed. Go to Section 8 for scope and recordkeeping; only start Path B if you also want distributed MPI.

---

## 4. Path B: Distributed MPI installation

**Start in PowerShell.** Complete Sections 1-2 first. Keep the single-rank environment untouched.

This path uses Microsoft MPI v10.1.3, `mpi4py`, the MPI SDK, and a source-built Repast4Py v1.3.0. The demonstrated distributed workload runs two ranks on **one computer**; this guide does not configure a multi-computer cluster.

### 4.1 Microsoft MPI runtime

**Check in PowerShell, from any directory:**

```powershell
Get-Command mpiexec -ErrorAction SilentlyContinue | Select-Object Source
Test-Path "$env:WINDIR\System32\msmpi.dll"
```

**Expected:** a launcher path identifying Microsoft MPI, usually ending in `Microsoft MPI\Bin\mpiexec.exe`, and `True` for the DLL check.

| Result | Action |
| --- | --- |
| Microsoft MPI launcher found and DLL present | Skip runtime installation; still run the `mpi4py` and two-rank checks in Section 4.4. |
| No launcher result, but DLL is present | Not a complete verification. Reopen PowerShell and check the launcher location below. A leftover DLL alone is insufficient. |
| Launcher or DLL is missing | Install/repair the runtime using the approved installer. |
| Launcher belongs to another MPI distribution | Stop and select the intended MPI runtime. Do not combine an arbitrary launcher with the MS-MPI build steps. |

For a launcher that is installed but not on the current PATH, check:

```powershell
Test-Path "$env:ProgramFiles\Microsoft MPI\Bin\mpiexec.exe"
```

If this is `True`, reopen PowerShell first. If still necessary, add that **existing** directory to the current session only, then repeat `Get-Command`:

```powershell
$env:PATH = "$env:ProgramFiles\Microsoft MPI\Bin;$env:PATH"
Get-Command mpiexec | Select-Object Source
```

**Install only if needed:** open the official [MS-MPI v10.1.3 download page][R6], click **Download**, select **`msmpisetup.exe`**, and complete the runtime installation.

> **Installer distinction:** Microsoft's download page lists `msmpisetup.exe` for the runtime and `msmpisdk.msi` for the SDK. They are separate components. Do not assume that downloading the SDK MSI installs the complete runtime. [R6]

**Verify / continue:** reopen PowerShell and repeat the two initial checks. A DLL in `System32`, rather than in `Microsoft MPI\Bin`, is not an error. Do not copy or delete DLLs in system directories as a troubleshooting shortcut. Repeat Section 2.2 if you opened a new shell.

### 4.2 Microsoft MPI SDK

The runtime executes MPI programs; the SDK supplies headers and libraries for the Repast4Py build.

**Check:**

```powershell
Test-Path "${env:ProgramFiles(x86)}\Microsoft SDKs\MPI\Include\mpi.h"
Test-Path "${env:ProgramFiles(x86)}\Microsoft SDKs\MPI\Lib\x64\msmpi.lib"
```

**Expected:**

```text
True
True
```

**If both are `True`:** skip the SDK installer. **If either is `False`:** install/repair the SDK, unless your approved installation uses a different location that you can verify.

**Install only if needed:** on the same [MS-MPI v10.1.3 download page][R6], select **`msmpisdk.msi`** and complete installation. Reopen PowerShell, restore the repository variables in Section 2.2, and repeat the checks.

**Set and verify the SDK directories in this session:**

```powershell
$env:MSMPI_INC = "${env:ProgramFiles(x86)}\Microsoft SDKs\MPI\Include\"
$env:MSMPI_LIB64 = "${env:ProgramFiles(x86)}\Microsoft SDKs\MPI\Lib\x64\"
Test-Path (Join-Path $env:MSMPI_INC "mpi.h")
Test-Path (Join-Path $env:MSMPI_LIB64 "msmpi.lib")
```

For an approved nondefault SDK installation, set these variables to its actual include and x64 library directories instead. Continue only when **both checks return `True`**. Do not use the 32-bit library for this Python x64 build.

### 4.3 Visual Studio C++ tools and compiler

**Check:**

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
Test-Path $vswhere
if (Test-Path $vswhere) {
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
```

**Pass condition:** the first check is `True` **and** the query prints a Visual Studio installation directory. The path may identify Community, Professional, Enterprise, or Build Tools; do not copy someone else's versioned path. Microsoft's `vswhere` query locates an installation with the requested C++ tools. [R7]

**Install/modify only if needed:**

- If Visual Studio is already installed but the query prints no directory, open **Visual Studio Installer**, choose **Modify**, and add **Desktop development with C++**.
- If Visual Studio is absent, install **Build Tools for Visual Studio** from the official downloads page, then select that workload. A full Visual Studio IDE is not required.

Include the MSVC x64/x86 build tools and the applicable Windows SDK. Keep the recommended C++ workload components unless your organization supplies an approved alternative. [R8]

**Verify after installation or when already present:**

```powershell
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Test-Path "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vsPath\VC\Auxiliary\Build\vcvars64.bat`" >nul && cl"
```

**Expected:** `True`, followed by a Microsoft C/C++ compiler banner for `x64` and a `usage: cl ...` message. That usage message is normal because no source file was supplied. Exact compiler and Visual Studio version numbers vary.

**Decision:** if the batch file is missing or the compiler is not found, stop and repair/select the C++ workload. Having `vswhere.exe` alone is not sufficient. An ordinary PowerShell `Get-Command cl.exe` can return nothing even when the compiler is installed; its developer environment must be initialized. No MinGW GCC or `mpicxx.exe` is required for the MSVC/MS-MPI route used here.

**Important:** the `cmd /c` verification exits afterward. It does not initialize the compiler in the parent PowerShell session. Section 4.7 opens the developer shell that stays active for the build.

### 4.4 Create the MPI environment and verify mpi4py

**Shell: PowerShell. Location: CASMSOCIAL repository.** If a new terminal was opened, restore Section 2.2 before proceeding.

```powershell
cd "$env:CASMSOCIAL_REPO"
Test-Path "$env:CASM_MPI_PYTHON"
```

**If `False`, create the environment:**

```powershell
uv venv (Join-Path $env:CASMSOCIAL_REPO ".venv-mpi-test") --python 3.12
```

**If `True`, skip creation. Verify in either case:**

```powershell
& "$env:CASM_MPI_PYTHON" --version
& "$env:CASM_MPI_PYTHON" -c "import sys, struct; print(sys.executable); print(struct.calcsize('P') * 8)"
```

**Expected:** Python 3.12, the executable under this checkout's `.venv-mpi-test`, and `64`. Stop if any differs. Do not use `.venv-repast-test` for this path.

**Check mpi4py:**

```powershell
uv pip show --python "$env:CASM_MPI_PYTHON" mpi4py
```

**Install only if missing:**

```powershell
uv pip install --python "$env:CASM_MPI_PYTHON" mpi4py
```

**Verify the actual runtime, even if already installed:**

```powershell
& "$env:CASM_MPI_PYTHON" -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

**Expected for the reference MS-MPI release:**

```text
Microsoft MPI 10.1.12498.52
```

The download's release name, `10.1.3`, and the reported library version, `10.1.12498.52`, are different forms of the version identifier. A different approved runtime version needs its own validation; `cannot load MPI library` is a failure, not an install success.

**Now verify two processes:**

```powershell
mpiexec -n 2 "$env:CASM_MPI_PYTHON" -c "from mpi4py import MPI; c=MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size(), flush=True)"
```

**Expected, in either order:**

```text
0 2
1 2
```

The first number is the rank ID; the second is the total rank count. Two lines, one for each rank, both reporting `2`, are required. If this passes, continue. If it fails or hangs, resolve the runtime/launcher problem before attempting a Repast4Py build. Do not substitute `mpirun --oversubscribe`; that is not the launcher command for this MS-MPI workflow.

### 4.5 Decide whether a Repast4Py build is needed

**Check:**

```powershell
uv pip show --python "$env:CASM_MPI_PYTHON" repast4py
```

**If missing:** continue to Sections 4.6-4.7 to clone and build it.

**If present:** run this backend check:

```powershell
& "$env:CASM_MPI_PYTHON" -c "from importlib.metadata import version; from mpi4py import MPI as native; from repast4py import MPI, core, space, _mpi_config; print('Repast4Py:', version('repast4py')); print('Single-rank build:', _mpi_config.SINGLE_RANK); print(MPI.Get_library_version()); assert version('repast4py') == '1.3.0'; assert not _mpi_config.SINGLE_RANK; assert MPI is native"
```

**Expected:** Repast4Py `1.3.0`, `Single-rank build: False`, and Microsoft MPI library information with no mock-MPI warning. If this passes, **skip the source/build steps and go to Section 4.8**.

If a single-rank warning appears, the import fails, or the marker is `True`, do not proceed as though distributed Repast4Py is installed. Use the separate intended MPI environment and build the correct variant. Installing Microsoft MPI does not convert an existing no-MPI wheel into a distributed build.

### 4.6 Clone Repast4Py and install build prerequisites

**Shell: PowerShell.** Choose an existing parent directory for a separate Repast4Py checkout. It need not be the same parent as CASMSOCIAL.

```powershell
cd "C:\path\to\your\chosen\source-parent"
Test-Path .\repast4py\setup.py
```

**If the checkout does not exist, clone it:**

```powershell
git clone https://github.com/Repast/repast4py.git
```

If a `repast4py` folder already exists but is not a valid source checkout, stop and inspect it or choose another parent directory; do not clone over or delete unknown work.

**For a new or existing checkout:**

```powershell
cd repast4py
git status --short
```

**Expected for a fresh checkout:** no output. If there are local changes, preserve them before switching releases.

```powershell
git switch --detach v1.3.0
git describe --tags --exact-match
$env:REPAST4PY_REPO = (Get-Location).Path
```

**Expected:** a detached-HEAD message, followed by `v1.3.0`. Detached HEAD is intentional when building a release tag, not a Git error. If the tag cannot be selected, stop; do not silently substitute the development branch.

**Check build prerequisites in the MPI environment:**

```powershell
uv pip show --python "$env:CASM_MPI_PYTHON" setuptools wheel numpy Cython
```

**Install if any are missing:**

```powershell
uv pip install --python "$env:CASM_MPI_PYTHON" setuptools wheel numpy Cython
```

**Verify in either case:**

```powershell
& "$env:CASM_MPI_PYTHON" -c "import setuptools, wheel, numpy, Cython, mpi4py; print('Build dependencies available')"
```

**Expected:** `Build dependencies available`, with no traceback. Extra dependency packages in the installer output are normal. Stop if an import fails. The upcoming `--no-build-isolation` build requires these packages in this environment before it begins. [R9]

### 4.7 Build MPI-enabled Repast4Py in the developer shell

#### Prepare in PowerShell

**Location: Repast4Py checkout.** Run this only after the SDK, compiler, and build-dependency checks pass.

```powershell
Remove-Item Env:R4PY_NO_MPI -ErrorAction SilentlyContinue
Test-Path (Join-Path $env:MSMPI_INC "mpi.h")
Test-Path (Join-Path $env:MSMPI_LIB64 "msmpi.lib")
Test-Path "$env:CASM_MPI_PYTHON"
Test-Path (Join-Path $env:REPAST4PY_REPO "setup.py")
```

**Expected:** four `True` results. If variables are empty or any result is `False`, restore the relevant paths from Sections 2.2, 4.2, and 4.6. Do not start the build yet.

Locate the compiler again so this step does not depend on a forgotten `$vsPath` variable:

```powershell
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
cmd /k "`"$vsPath\VC\Auxiliary\Build\vcvars64.bat`""
```

**Expected:** a developer environment banner and `Environment initialized for: 'x64'`. The prompt changes from PowerShell to CMD, often in the **same terminal window**. The shell inherits the repository, interpreter, and SDK environment variables set above.

#### Run these commands in CMD, not PowerShell

**Check inherited paths first:**

```cmd
echo %REPAST4PY_REPO%
echo %CASM_MPI_PYTHON%
echo %MSMPI_INC%
echo %MSMPI_LIB64%
```

**Expected:** four real filesystem paths, not literal `%VARIABLE_NAME%` strings. If a variable is unexpanded, type `exit`, restore it in PowerShell, and reopen the developer shell. Do not replace it with another user's path.

**Build:**

```cmd
cd /d "%REPAST4PY_REPO%"
set "R4PY_NO_MPI="
uv pip install --python "%CASM_MPI_PYTHON%" --no-build-isolation .
echo %ERRORLEVEL%
```

**Expected:** successful build/installation output for local-source `repast4py==1.3.0`, followed by exit code `0`. Download counts, package counts, and timings vary. A suitable cached build may produce different progress wording; the backend verification below remains mandatory.

**If the build fails:** stop and keep the complete error. Do not edit build files based only on a generic installer hint. In particular, `No module named 'setuptools'` here means the no-isolation build prerequisites were not installed into the selected environment; return to Section 4.6.

**Verify the installed package and compiled extension imports (CMD):**

```cmd
uv pip show --python "%CASM_MPI_PYTHON%" repast4py
"%CASM_MPI_PYTHON%" -c "from importlib.metadata import version; from mpi4py import MPI as native; from repast4py import MPI, core, space, _mpi_config; print('Repast4Py:', version('repast4py')); print('Single-rank build:', _mpi_config.SINGLE_RANK); print(MPI.Get_library_version()); assert version('repast4py') == '1.3.0'; assert not _mpi_config.SINGLE_RANK; assert MPI is native"
```

**Pass condition:** version `1.3.0`, package location inside `.venv-mpi-test`, native extensions import, marker `False`, and Microsoft MPI is reported. The package's dependency list should include `mpi4py`.

**Return to PowerShell now:**

```cmd
exit
```

**Expected:** the original `PS ...>` prompt returns. The C++ developer-shell variables do not need to remain active to run the installed wheel. If this closed a separately opened CMD window instead, resume PowerShell and restore Section 2.2 before continuing.

### 4.8 Install CASMSOCIAL into the MPI environment

**Shell: PowerShell. Location: CASMSOCIAL repository.**

```powershell
cd "$env:CASMSOCIAL_REPO"
uv pip show --python "$env:CASM_MPI_PYTHON" casmsocial
```

**If already installed from this editable checkout:** skip installation and continue to Section 4.10. **If missing or installed from another checkout:** preview the intended editable installation:

```powershell
uv pip install --dry-run --python "$env:CASM_MPI_PYTHON" -e .
```

**Expected:** `Would install ...` on a fresh setup. This is only a preview. Confirm the plan does **not replace or rebuild Repast4Py**. If it does, stop so that the source-built MPI variant is not accidentally replaced.

**If the preview is satisfactory, install:**

```powershell
uv pip install --python "$env:CASM_MPI_PYTHON" -e .
uv pip show --python "$env:CASM_MPI_PYTHON" casmsocial
```

**Expected:** CASMSOCIAL package details for this checkout after a successful installation. The reference version is `2.11.0`. Package counts and other dependency versions are not pass/fail criteria.

### 4.9 Verify logging and environment-variable dependencies

CASMSOCIAL declares **`loguru`** (logging) and **`python-dotenv`** (loads optional `.env` configuration) as project dependencies. They should normally be installed by the editable CASMSOCIAL installation above, but verify them explicitly. **Run these commands in PowerShell from the CASMSOCIAL repository.**

**Check:**

```powershell
uv pip show --python "$env:CASM_MPI_PYTHON" loguru python-dotenv
```

**Expected if installed:** two package entries, one with `Name: loguru` and one with `Name: python-dotenv`, each showing a version and installation location. Exact versions may differ. A warning such as `Package(s) not found for: python-dotenv` means that package is missing.

**Decision:** if **both** are listed, **skip the installation command** and continue to the import verification. If **either** is missing, install the dependencies into this same environment:

```powershell
uv pip install --python "$env:CASM_MPI_PYTHON" loguru python-dotenv
```

**Verify (required whether or not installation was skipped):**

```powershell
& "$env:CASM_MPI_PYTHON" -c "from loguru import logger; from dotenv import load_dotenv; print('Logging and dotenv dependencies working')"
```

**Expected:** `Logging and dotenv dependencies working` and no traceback. If `ModuleNotFoundError` occurs, **stop** and confirm that the install and verification commands target the same virtual environment. Do not proceed to tests until the imports work.

**`.env` file clarification:** `python-dotenv` is a Python **package**; `.env` is an optional **configuration file**. The basic local MVP does **not** require creating a `.env` file or providing remote-provider API credentials. Other scenarios may document additional environment variables. Never commit a `.env` containing secrets.

### 4.10 Check and verify DuckDB, PyArrow, and pytest

**Check:**

```powershell
uv pip show --python "$env:CASM_MPI_PYTHON" duckdb pyarrow pytest
```

**Decision:** run the installation below if **any** listed package is missing **or DuckDB is not exactly `1.5.5`**. For example, a result that shows DuckDB `1.5.6`, PyArrow installed, and pytest missing still requires this step. If all three are present and DuckDB matches, skip installation, not verification.

```powershell
uv pip install --python "$env:CASM_MPI_PYTHON" "duckdb==1.5.5" pyarrow pytest
```

**Expected when replacing DuckDB:** an uninstall line for the previous version and an install line for `duckdb==1.5.5`. That is intentional.

**Verify:**

```powershell
uv pip check --python "$env:CASM_MPI_PYTHON"
& "$env:CASM_MPI_PYTHON" -c "from casmsocial.mpi_compat import MPI; from casmsocial import casmpop; import duckdb, pyarrow, pytest; print(MPI.Get_library_version()); print('DuckDB:', duckdb.__version__); c=duckdb.connect(':memory:'); print(c.execute('SELECT 1 AS x').arrow().read_all().to_pylist()); c.close(); assert duckdb.__version__ == '1.5.5'"
& "$env:CASM_MPI_PYTHON" -m casmsocial --help
```

**Expected:** no dependency conflicts; Microsoft MPI information; `DuckDB: 1.5.5`; `[{'x': 1}]`; and CASMSOCIAL CLI help. Model-registration INFO messages are normal. A mock/single-rank MPI warning is **not** expected for this path.

If PyArrow is missing after an otherwise supposedly complete CASMSOCIAL installation, check the editable installation and target interpreter. Do not rely on a successful top-level `import casmsocial` alone to establish that every dependency is installed.

### 4.11 Verify Repast4Py and CASMSOCIAL on two ranks

**Shell: PowerShell. Location: CASMSOCIAL repository.**

```powershell
mpiexec -n 2 "$env:CASM_MPI_PYTHON" -c "from mpi4py import MPI as native; from repast4py import MPI as repast_mpi, core, space; from casmsocial.mpi_compat import MPI; from casmsocial import casmpop; c=MPI.COMM_WORLD; assert MPI is native and repast_mpi is native; assert c.Get_size() == 2; total=c.allreduce(c.Get_rank()+1, op=MPI.SUM); assert total == 3; print('CASMSOCIAL rank', c.Get_rank(), 'of', c.Get_size(), 'sum', total, flush=True)"
```

**Expected, in either order, in addition to possible INFO messages:**

```text
CASMSOCIAL rank 0 of 2 sum 3
CASMSOCIAL rank 1 of 2 sum 3
```

**Continue when:** both rank lines appear and no traceback occurs. This verifies imports and a collective, not a complete simulation. Continue to Section 5.

---

## 5. Run the distributed integration tests

**Path B only. Shell: PowerShell. Location: CASMSOCIAL repository.**

These test files launch `mpiexec` internally. **Run pytest once normally; do not put `mpiexec -n 2` before this command.**

```powershell
cd "$env:CASMSOCIAL_REPO"
& "$env:CASM_MPI_PYTHON" -m pytest `
    tests/test_person_migration_mpi.py `
    tests/test_runner_mpi_failures.py `
    tests/test_runner_mpi_cancellation.py `
    tests/test_grpc_runner_mpi.py `
    -v --tb=short -rs
```

**Reference result:**

```text
collected 27 items
...
SKIPPED [4] ... fixture cleanup requires POSIX process groups
23 passed, 4 skipped in ...
```

| Outcome | Decision |
| --- | --- |
| 23 passed, 4 skipped for the stated POSIX cleanup reason | Continue to the two-rank MVP. |
| Tests skipped because MPI Python/runtime or launcher is unavailable | Not a successful MPI validation. Return to Sections 4.1 and 4.4. |
| Tests skipped because nested MPI subprocess launching is disallowed | Rerun pytest normally, without an outer `mpiexec`. |
| Any `FAILED`, collection `ERROR`, or timeout | Stop and examine the first failure before running the MVP. |
| A different count on a newer code revision | Review which tests changed. Do not silently reinterpret missing tests as passing. |

The four gRPC runner skips are test-harness restrictions on Windows, not evidence that those features passed. The 23 passing cases cover six migration/synchronization checks, eleven failure-handling checks, and six cancellation checks. They are not the whole repository test suite.

---

## 6. Run and validate the complete two-rank MVP

**Path B only. Shell: PowerShell, not CMD. Location: CASMSOCIAL repository.**

```powershell
cd "$env:CASMSOCIAL_REPO"
```

If this is a new shell, first repeat Section 2.2 and confirm Section 4.4's MPI runtime check still passes. Use the repository's data generator and normal `python -m casmsocial` entry point; no custom launcher package is needed.

### 6.1 Verify the Windows DuckLake path correction before generating data

**Check:**

```powershell
Select-String -Path .\casmsocial\ducklake_utils.py -Pattern '^\s*data_url\s*='
```

**Required line for the tested workflow:**

```python
data_url = (ducklake_path / "storage").as_uri()
```

**If already present:** skip any code edit and continue.

**If the old manual concatenation is present:**

```python
data_url = "".join(["file://", str(ducklake_path / "storage")])
```

Stop before generating data: the checkout is older than the Windows changes. Update it (Section 2.1) and repeat the check.

This correction was exercised in the successful Windows MVP run. It uses Python's path-to-URI conversion instead of concatenating a URI prefix with a Windows path. [R10] Broader path coverage and Linux regression testing remain separate validation work.

### 6.2 Set local data paths and check the dataset

Run these assignments every new simulation session; they normally print no output:

```powershell
$env:CASMSOCIAL_DATA_PATH = Join-Path $env:CASMSOCIAL_REPO "examples\mvp"
$env:CASMSOCIAL_DUCKLAKE_PATH = Join-Path $env:CASMSOCIAL_DATA_PATH "mvp_windows_test.ducklake"
Test-Path (Join-Path $env:CASMSOCIAL_DUCKLAKE_PATH "metadata.sqlite")
```

**If `False`, generate the test lake:**

```powershell
& "$env:CASM_MPI_PYTHON" scripts\create_mvp_ducklake.py --ducklake-path "$env:CASMSOCIAL_DUCKLAKE_PATH"
```

**Expected:** `Created MVP DuckLake at ...\mvp_windows_test.ducklake`, with no traceback.

**If `True`:** skip regeneration for now, but still validate the tables below. A metadata file can exist after a partially failed generation and is not proof of complete data.

**Verify in either case:** copy the entire block, including the first and last lines.

```powershell
@'
import os
from pathlib import Path
from casmsocial.ducklake_utils import get_ducklake_connection

lake = Path(os.environ["CASMSOCIAL_DUCKLAKE_PATH"])
conn = get_ducklake_connection(lake)
try:
    expected_counts = {
        "casmsocial_mvp.places": 3,
        "casmsocial_mvp.persons": 2,
        "casmsocial_mvp.hh": 2,
        "casmsocial_mvp.activities": 6,
        "partitions.mvp_two_rank_place_partitions": 3,
    }
    for table, expected in expected_counts.items():
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} rows")
        assert count == expected, (table, count, expected)
    rows = conn.execute(
        "SELECT place_id, rank FROM partitions.mvp_two_rank_place_partitions "
        "ORDER BY place_id"
    ).fetchall()
    print(rows)
    assert rows == [(100, 0), (200, 1), (300, 0)], rows
    print("MVP dataset verification passed")
finally:
    conn.close()
'@ | & "$env:CASM_MPI_PYTHON" -
```

**Expected:**

```text
casmsocial_mvp.places: 3 rows
casmsocial_mvp.persons: 2 rows
casmsocial_mvp.hh: 2 rows
casmsocial_mvp.activities: 6 rows
partitions.mvp_two_rank_place_partitions: 3 rows
[(100, 0), (200, 1), (300, 0)]
MVP dataset verification passed
```

The tuples mean Home A/place 100 belongs to rank 0, Home B/place 200 to rank 1, and workplace/place 300 to rank 0.

**Decision:** continue only after the queries and assertions succeed. A count of zero visible `.parquet` files under this tiny input lake is not by itself a failure: the earlier test lake was queryable in that situation. Use the table checks, not a file-count assumption, as the input-data verification.

If an existing lake fails these checks, preserve it and investigate; do not automatically delete it or overwrite shared data. For a fresh attempt, choose another unused lake directory, update `CASMSOCIAL_DUCKLAKE_PATH`, regenerate there, and rerun this verification. Do not regenerate a lake while simulation processes are using it.

### 6.3 Create or verify the two-rank YAML configuration

Use a separate configuration file rather than a JSON command-line override. This avoids the quoting failure observed when JSON passed through PowerShell and `mpiexec`.

**Check:**

```powershell
Test-Path .\config\mvp_windows_2rank.yaml
```

**If `True`:** skip creation and run verification below. **If `False`:** create it from the repository's `mvp.yaml` with the following guarded replacements. This leaves the original configuration unchanged and does not append duplicate YAML keys.

Run this block as a whole:

```powershell
& {
    $destination = Join-Path $env:CASMSOCIAL_REPO "config\mvp_windows_2rank.yaml"
    if (Test-Path -LiteralPath $destination) {
        throw "Configuration already exists. Verify it instead of overwriting it."
    }
    $sourcePath = Join-Path $env:CASMSOCIAL_REPO "config\mvp.yaml"
    $source = [IO.File]::ReadAllText($sourcePath)
    $changes = [ordered]@{
        "partition.table: ''" = "partition.table: 'partitions.mvp_two_rank_place_partitions'"
        "partition.require_full_coverage: false" = "partition.require_full_coverage: true"
        "observers.agent_log_file: 'mvp_agent_log.parquet'" = "observers.agent_log_file: 'mvp_2rank_agent_log.parquet'"
        "observers.behavior_log_file: 'mvp_behavior_log.parquet'" = "observers.behavior_log_file: 'mvp_2rank_behavior_log.parquet'"
    }
    foreach ($old in $changes.Keys) {
        if ([regex]::Matches($source, [regex]::Escape($old)).Count -ne 1) {
            throw "Expected exactly one configuration entry: $old. No file written."
        }
        $source = $source.Replace($old, $changes[$old])
    }
    $source = $source.Replace(
        "# Single-rank MVP runs do not require a partition table.",
        "# Use the generated two-rank place partition table."
    )
    $source = $source.Replace(
        "# Run with one MPI rank, for example:",
        "# Run with two MPI ranks on native Windows, for example:"
    )
    $source = $source.Replace(
        "# uv run mpirun -n 1 python -m casmsocial config/mvp.yaml",
        "# mpiexec -n 2 .venv-mpi-test\Scripts\python.exe -m casmsocial config/mvp_windows_2rank.yaml"
    )
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($destination, $source, $utf8)
    Write-Host "Created config/mvp_windows_2rank.yaml"
}
```

**Expected:** `Created config/mvp_windows_2rank.yaml`. If a replacement is not found, stop and inspect the current base configuration rather than running a file with partial overrides.

**Verify a new or existing file:**

```powershell
@'
from pathlib import Path
import yaml

params = yaml.safe_load(Path("config/mvp_windows_2rank.yaml").read_text(encoding="utf-8-sig"))
expected = {
    "partition.table": "partitions.mvp_two_rank_place_partitions",
    "partition.require_full_coverage": True,
    "observers.output_dir": "data/output",
    "observers.agent_log_file": "mvp_2rank_agent_log.parquet",
    "observers.behavior_log_file": "mvp_2rank_behavior_log.parquet",
    "duration.hours": 24,
    "time.step.minutes": 60,
}
for key, value in expected.items():
    actual = params.get(key)
    print(f"{key}: {actual}")
    assert actual == value, (key, actual, value)
print("Two-rank configuration verification passed")
'@ | & "$env:CASM_MPI_PYTHON" -
```

**Expected:** the listed values, including `partition.require_full_coverage: True`, followed by `Two-rank configuration verification passed`. Continue only after this succeeds. A `True` file-existence result alone is not sufficient.

### 6.4 Keep previous outputs separate from the new validation run

**Check:**

```powershell
Test-Path .\data\output\mvp_2rank_agent_log.parquet
Test-Path .\data\output\mvp_2rank_behavior_log.parquet
```

**If both are `False`:** skip archiving and launch the simulation.

**If either is `True`:** preserve previous results before repeating the reference validation. The following optional block moves only these named output datasets aside; it does not delete them or the rest of `data/output`.

```powershell
& {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $outputs = @(
        "data\output\mvp_2rank_agent_log.parquet",
        "data\output\mvp_2rank_behavior_log.parquet"
    )
    foreach ($relative in $outputs) {
        $path = Join-Path $env:CASMSOCIAL_REPO $relative
        if (Test-Path -LiteralPath $path) {
            Move-Item -LiteralPath $path -Destination "$path.previous-$stamp" -ErrorAction Stop
            Write-Host "Archived $relative"
        }
    }
}
```

**Verify:** repeat both file checks. Both should now be `False`. If archiving fails, close the process using the outputs or select separate output names and corresponding validator arguments; do not delete unrelated results.

### 6.5 Run the complete two-rank simulation

**Shell: PowerShell. Location: CASMSOCIAL repository.** Confirm the data-path assignments from Section 6.2 are still set in this shell.

```powershell
mpiexec -n 2 "$env:CASM_MPI_PYTHON" -m casmsocial config/mvp_windows_2rank.yaml
$runExitCode = $LASTEXITCODE
Write-Host "MPI exit code: $runExitCode"
```

**Expected:** model startup, simulation steps, a final `Simulation took ... seconds.` message, return to the prompt, and `MPI exit code: 0`. The elapsed time is not a pass/fail criterion.

The following messages are normal for the reference MVP:

- Warnings that optional parameters are absent and defaults are being used.
- A message that the social-network table is not specified: this scenario intentionally disables that input.
- Most INFO messages showing rank 0 because `logging.rank0_only` is enabled. That alone does not mean rank 1 failed; output validation checks rank coverage.

**Decision:** if there is a traceback, the job hangs, or the exit code is not `0`, stop and diagnose the run before validation. Do not accept old outputs as evidence for a failed new run. A mock/single-rank backend warning is not expected here.

### 6.6 Validate output content, not just file existence

Run only after the simulation completes successfully:

```powershell
& "$env:CASM_MPI_PYTHON" scripts\validate_mvp_output.py `
    --agent-log data/output/mvp_2rank_agent_log.parquet `
    --behavior-log data/output/mvp_2rank_behavior_log.parquet `
    --expected-ranks 2
$validationExitCode = $LASTEXITCODE
Write-Host "Validation exit code: $validationExitCode"
```

**Expected for the unchanged reference scenario:**

```text
MVP agent log valid: data\output\mvp_2rank_agent_log.parquet (rows=48, runs=1, agents=2, ticks=24, ranks=2)
MVP behavior log valid: data\output\mvp_2rank_behavior_log.parquet (rows=48, runs=1, agents=2, ticks=24, ranks=2)
Validation exit code: 0
```

**Completion gate:** both datasets must be reported valid, both must include `ranks=2`, and validation must exit successfully. The row count is **48 per dataset across both ranks**, not 48 per rank. The `.parquet` output paths can be dataset directories containing partitioned files; they need not each be a single file.

If validation reports missing outputs, fewer ranks, duplicate rows, or another error, the end-to-end check has not passed. Resolve that problem rather than treating successful installation or runtime startup as equivalent validation.

---

## 7. Troubleshooting and resuming a session

### 7.1 Optional: certificate trust errors

Use this section only for errors such as `UnknownIssuer` or a warning that `SSL_CERT_FILE` names an invalid path. Do not change a working certificate configuration unnecessarily.

**Check in PowerShell:**

```powershell
Get-ChildItem Env:SSL_CERT_FILE,Env:SSL_CERT_DIR,Env:UV_SYSTEM_CERTS -ErrorAction SilentlyContinue
uv --version
```

If `SSL_CERT_FILE` is an obsolete/placeholder override rather than a required approved certificate bundle, remove it for this session and use Windows trust:

```powershell
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
$env:UV_SYSTEM_CERTS = "true"
```

**Expected:** normally no output. Retry the exact installation that failed. Success means the certificate error is gone and the installation itself completes, not merely that the assignments ran.

Astral documents system certificate trust and custom certificate overrides. An invalid override can prevent any default certificates from being trusted. [R11] If your organization requires a PEM bundle or an internal package index, use its approved settings instead. If the error persists, contact the administrator; do not use insecure-host flags or disable certificate checks. An obsolete uv version may need an approved update before supporting the documented setting.

These changes are session-local. If a Developer Command Prompt is already open, it will not receive later parent-shell changes; exit it and reopen it from the corrected PowerShell session before retrying the build.

### 7.2 Common results and the next action

| Result | Interpretation | Next action |
| --- | --- | --- |
| Environment created in an unintended directory | A relative environment path was used from the wrong location. | Return to the intended checkout and restore Section 2.2. Use the absolute environment paths in this guide. Do not automatically delete an unfamiliar directory. |
| `$env:CASMSOCIAL_REPO` is empty, or CMD prints `%CASM_MPI_PYTHON%` literally | Session variables were not set/inherited. | In PowerShell, restore Section 2.2. For building, also restore the source/SDK paths and reopen CMD from that session. |
| `No module named pip` | The uv-created environment does not contain pip. | Use the documented `uv pip` commands; installing pip is unnecessary. |
| Missing `setuptools`, `wheel`, or `numpy` during the no-isolation build | Build dependencies are not available in the selected environment. | Complete Section 4.6 and rerun the build from the x64 developer shell. Do not assume the source forgot to declare them. |
| `cl` is not recognized in ordinary PowerShell | Developer compiler variables may not be initialized. | Use the `vswhere`/`vcvars64.bat` checks, then the CMD build shell in Section 4.7. |
| `cannot load MPI library` | mpi4py cannot load an appropriate native runtime. | Verify Section 4.1 and Python x64. Runtime and SDK are separate installations. |
| `mpirun` is not recognized | The Linux/Open MPI launcher command was used. | Use the documented Microsoft `mpiexec -n 2 ...` command. |
| Mock MPI warning in Path A | Expected single-rank backend. | Continue if that step's tests pass. |
| Mock MPI warning in Path B | Wrong interpreter or Repast4Py build. | Stop. Verify `.venv-mpi-test` and its build marker; clear `R4PY_NO_MPI` before a native-MPI rebuild. |
| JSON decoding error at startup | Command-line JSON quoting did not survive the shell/launcher. | Use the separate verified YAML file in Section 6.3. |
| DuckLake directory error contains `file://C:\...` | Old Windows URI construction is still in use. | Verify Section 6.1. Preserve failed datasets until you choose a safe fresh location. |
| Metadata file exists but table queries fail | Dataset may be incomplete or incompatible. | Do not skip data verification. Investigate or create a separate fresh test lake. |
| Output validation says a path does not exist | The simulation may not have completed, or output paths differ. | Check the run exit code, configuration, and repository working directory first. |
| A directory cannot be removed during manual cleanup | A process may still be using it, including a developer shell. | Close only the relevant sessions after preserving work. Do not terminate unrelated processes or delete system DLLs. |

### 7.3 Package updates and clean-install tests

Keep the two environments separate. Do not run an unreviewed `uv sync`, `uv run`, upgrade, or generic Repast4Py reinstall against the validated environment: a different environment or build can be selected. This guide uses explicit interpreters and package checks instead.

A new virtual environment does not necessarily imply a new compiler invocation; installers can reuse cached artifacts. For a test specifically intended to verify recompilation, use a fresh Repast4Py source checkout with no old build output, initialize the x64 developer shell as in Section 4.7, and use this optional CMD command:

```cmd
uv pip install --python "%CASM_MPI_PYTHON%" --no-build-isolation --no-cache --reinstall-package repast4py .
```

This requests a fresh local-source reinstall even if Repast4Py is already installed. Recheck the backend and rerun the relevant tests afterward. It is a stricter rebuild check, not a required ordinary-installation step. [R9]

Do not remove shared system software merely to test the **already installed** branch of a check. A clean-machine installation, a clean virtual environment, and a clean source rebuild are different tests; record which one was performed.

---

## 8. Completion criteria and validation scope

| Path/check | Reference result | What it establishes |
| --- | --- | --- |
| Single-rank selected suite | 142 passed, expected mock-MPI warning | The selected five test files work with the no-MPI Repast4Py build. |
| mpi4py launcher check | Rank 0 and rank 1 both report size 2 | Two MPI Python processes start on the local computer. |
| Distributed targeted tests | 23 passed, 4 POSIX-harness skips | The tested migration, failure-handling, and cancellation cases pass. |
| Complete two-rank MVP | Simulation completes; both logs validate with 48 rows, 2 agents, 24 ticks, and 2 ranks | The tested complete scenario runs with real MPI and produces the expected outputs. |

These are **reference outcomes**, not a promise that all future revisions produce identical counts. The latest supplied reinstall transcript confirmed the distributed tests and the complete two-rank MVP again. The single-rank count comes from the earlier separate verification; the counts are not one combined full-suite run.

The four skipped gRPC tests remain unverified on Windows. Larger populations, more ranks, multi-computer deployment, all other repository tests, Windows CI, and cross-platform regression coverage require separate work. Do not label the whole framework universally validated based on this MVP.

### Record a successful installation

From the CASMSOCIAL checkout in PowerShell:

```powershell
git rev-parse HEAD
uv --version
```

For Path A:

```powershell
& "$env:CASM_SINGLE_PYTHON" --version
uv pip freeze --python "$env:CASM_SINGLE_PYTHON"
```

For Path B:

```powershell
& "$env:CASM_MPI_PYTHON" --version
& "$env:CASM_MPI_PYTHON" -c "from mpi4py import MPI; print(MPI.Get_library_version())"
uv pip freeze --python "$env:CASM_MPI_PYTHON"
```

Also record the Repast4Py source commit when built locally. Retain the no-MPI wheel's source and checksum when using Path A. A package-version list alone does not capture the difference between the two Repast4Py builds. Review exported records for personal paths or credentials before publishing them.

## 9. References and revision notes

The successful commands, expected model outputs, and test counts in this guide are grounded in the installation and simulation transcripts. Absolute path variables, stop/continue instructions, guard checks, and output archiving were added to make the workflow less dependent on the operator's current directory. These revised instruction blocks should still undergo a fresh user validation; no additional Windows execution is implied by the editorial rewrite.

Installer naming is clarified against Microsoft's download page: the runtime is `msmpisetup.exe`, while the SDK is `msmpisdk.msi`. The source walkthrough's successful runtime checks do not identify which installer originally supplied every component. Other current installer details and command behavior are supported by the official references below rather than inferred from the test transcript.

[R1]: https://learn.microsoft.com/en-us/windows/package-manager/winget/ "Microsoft: WinGet and App Installer"
[R2]: https://www.python.org/downloads/release/python-31210/ "Python 3.12.10 release and Windows installer files"
[R3]: https://git-scm.com/install/windows "Git: Windows installation"
[R4]: https://repast.github.io/repast4py.site/windows_install.html "Repast4Py: Windows no-MPI wheel installation"
[R5]: https://docs.astral.sh/uv/getting-started/installation/ "Astral: Installing uv"
[R6]: https://www.microsoft.com/en-us/download/details.aspx?id=105289 "Microsoft MPI v10.1.3 runtime and SDK downloads"
[R7]: https://github.com/microsoft/vswhere/wiki/Find-VC "Microsoft: Locate Visual C++ tools with vswhere"
[R8]: https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line "Microsoft: C++ Build Tools and developer command environments"
[R9]: https://docs.astral.sh/uv/reference/cli/ "Astral: uv CLI, build isolation, and cache controls"
[R10]: https://docs.python.org/3.12/library/pathlib.html "Python 3.12: pathlib and file URI conversion"
[R11]: https://docs.astral.sh/uv/concepts/authentication/certificates/ "Astral: TLS certificates and system trust"

[R12]: https://raw.githubusercontent.com/microsoft/winget-pkgs/master/manifests/p/Python/Python/3/12/3.12.10/Python.Python.3.12.yaml "Microsoft WinGet Python 3.12 package manifest"
[R13]: https://learn.microsoft.com/en-us/windows/package-manager/winget/install "Microsoft: WinGet install options"

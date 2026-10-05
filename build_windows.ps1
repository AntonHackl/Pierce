param(
    [ValidateSet('preprocess', 'query')]
    [string[]]$Only = @('preprocess', 'query'),
    [string]$Environment = 'pierce_windows_build',
    [string]$OptixInstallDir = $env:OptiX_INSTALL_DIR,
    [string]$CondaExe,
    [int]$Jobs = 2,
    [string]$CudaArchitectures = '75',
    [switch]$Fresh
)

$ErrorActionPreference = 'Stop'
if ($Jobs -lt 1) { throw 'Jobs must be positive.' }
if (-not $CondaExe) {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) { $CondaExe = $command.Source }
    else { $CondaExe = Join-Path $env:USERPROFILE 'miniconda3/Scripts/conda.exe' }
}
if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw 'Conda executable not found. Supply -CondaExe with its full path.'
}
$activation = & $CondaExe shell.powershell activate $Environment
if ($LASTEXITCODE -ne 0) { throw "Could not activate Conda environment '$Environment'." }
Invoke-Expression ($activation -join [Environment]::NewLine)

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio 2022 with Desktop development with C++.' }
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Visual Studio C++ tools were not found.' }
Import-Module (Join-Path $vsPath 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null

if (-not $OptixInstallDir) {
    $sdk = Get-ChildItem -LiteralPath (Join-Path $env:ProgramData 'NVIDIA Corporation') -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'OptiX SDK *' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'include/optix.h')) } |
        Sort-Object { [version]($_.Name -replace '^OptiX SDK ', '') } -Descending | Select-Object -First 1
    if ($sdk) { $OptixInstallDir = $sdk.FullName }
}
if ($Only -contains 'query' -and (-not $OptixInstallDir -or
    -not (Test-Path -LiteralPath (Join-Path $OptixInstallDir 'include/optix.h')))) {
    throw 'Supply -OptixInstallDir with the OptiX SDK directory (7.7 or newer; 9.0 tested).'
}

$prefix = $env:CONDA_PREFIX
$cmake = Join-Path $prefix 'Library/bin/cmake.exe'
$nvcc = Join-Path $prefix 'Library/bin/nvcc.exe'
if (-not (Test-Path -LiteralPath $cmake) -or -not (Test-Path -LiteralPath $nvcc)) {
    throw 'The environment must contain CMake and the CUDA toolkit.'
}
foreach ($component in $Only) {
    $source = Join-Path $PSScriptRoot "pierce/$component"
    $build = Join-Path $source 'build'
    $arguments = @('-S', $source, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_PREFIX_PATH=$prefix/Library;$prefix", "-DCUDAToolkit_ROOT=$prefix/Library")
    if ($Fresh) { $arguments += '--fresh' }
    if ($component -eq 'query') {
        $arguments += @("-DCMAKE_CUDA_COMPILER=$nvcc", "-DOptiX_INSTALL_DIR=$OptixInstallDir",
                        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures")
    }
    Write-Host "Building Pierce $component using $Environment"
    & $cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed for $component." }
    & $cmake --build $build --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "Build failed for $component." }
}
Write-Host 'Pierce Windows build completed.'

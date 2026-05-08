param(
    [string]$ColmapExe = "",
    [string]$DatasetPath = "data",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ColmapExe)) {
    $downloadDir = "F:\" + [char]0x8FC5 + [char]0x96F7 + [char]0x4E0B + [char]0x8F7D
    $localColmap = Join-Path $downloadDir "colmap-x64-windows-cuda\bin\colmap.exe"
    $pathColmap = Get-Command colmap.exe -ErrorAction SilentlyContinue
    if (Test-Path -Path $localColmap) {
        $ColmapExe = $localColmap
    } elseif ($pathColmap) {
        $ColmapExe = $pathColmap.Source
    } else {
        $ColmapExe = "colmap.exe"
    }
}

if (-not (Test-Path -Path $ColmapExe)) {
    throw "COLMAP executable not found: $ColmapExe"
}

$ImagePath = Join-Path $DatasetPath "images"
$ColmapPath = Join-Path $DatasetPath "colmap"
$SparsePath = Join-Path $ColmapPath "sparse"
$DensePath = Join-Path $ColmapPath "dense"
$DatabasePath = Join-Path $ColmapPath "database.db"
$SparseModelPath = Join-Path $SparsePath "0"
$DenseCloudPath = Join-Path $DensePath "fused.ply"

if ((Test-Path -Path $DenseCloudPath) -and -not $Force) {
    Write-Host "Existing COLMAP result found: $DenseCloudPath"
    Write-Host "Use -Force to delete data/colmap and rebuild from scratch."
    if (Test-Path -Path $SparseModelPath) {
        Write-Host "=== Sparse Model Analysis ==="
        & $ColmapExe model_analyzer --path $SparseModelPath
    }
    exit 0
}

if ($Force -and (Test-Path -Path $ColmapPath)) {
    Remove-Item -Path $ColmapPath -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $SparsePath | Out-Null
New-Item -ItemType Directory -Force -Path $DensePath | Out-Null

Write-Host "=== Step 1: Feature Extraction ==="
& $ColmapExe feature_extractor `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1

Write-Host "=== Step 2: Exhaustive Matching ==="
& $ColmapExe exhaustive_matcher `
    --database_path $DatabasePath

Write-Host "=== Step 3: Sparse Reconstruction ==="
& $ColmapExe mapper `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --output_path $SparsePath

Write-Host "=== Step 4: Image Undistortion ==="
& $ColmapExe image_undistorter `
    --image_path $ImagePath `
    --input_path $SparseModelPath `
    --output_path $DensePath

Write-Host "=== Step 5: Patch Match Stereo ==="
& $ColmapExe patch_match_stereo `
    --workspace_path $DensePath

Write-Host "=== Step 6: Stereo Fusion ==="
& $ColmapExe stereo_fusion `
    --workspace_path $DensePath `
    --output_path $DenseCloudPath

Write-Host "=== Step 7: Sparse Model Analysis ==="
& $ColmapExe model_analyzer `
    --path $SparseModelPath

Write-Host "=== Done ==="
Write-Host "Sparse model: $SparseModelPath"
Write-Host "Dense cloud:  $DenseCloudPath"

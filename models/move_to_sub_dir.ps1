$sourcePath = Split-Path $MyInvocation.MyCommand.Definition

Get-ChildItem -Path $sourcePath -Filter "" -File | ForEach-Object {
    # Get the file's base name (name without extension)
    $folderName = $_.BaseName
    # Construct the full path for the new folder
    $newFolderPath = Join-Path -Path $sourcePath -ChildPath $folderName
    # Check if the folder already exists before creating it
    if (-not (Test-Path -Path $newFolderPath)) {
        # Create the new directory
        New-Item -Path $newFolderPath -ItemType Directory
        Write-Host "Created directory: $newFolderPath"
    }
    # Move the file into the new directory
    Move-Item -Path $_.FullName -Destination $newFolderPath
    Write-Host "Moved file: $($_.Name) to $newFolderPath"
}
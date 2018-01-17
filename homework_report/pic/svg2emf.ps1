$env:path+= ";A:\inkscape"

Get-ChildItem -Path . -Recurse -ErrorAction SilentlyContinue -Filter *.svg | Where-Object { $_.Extension -eq '.svg' }|ForEach-Object {
        $name=$_.BaseName;
        write-host $name;
        inkscape.exe -z -f $_.Name --export-pdf="$name.pdf";
}

pause
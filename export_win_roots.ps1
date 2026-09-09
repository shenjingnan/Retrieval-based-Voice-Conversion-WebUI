# Export Windows CA certificates (roots + intermediates) to a PEM bundle
# so Python (requests/certifi) can verify TLS chains rooted in a CA that only
# exists in the Windows trust store (e.g. corporate SSL-inspection proxies).
$out = @()
$count = 0
foreach ($storeName in 'Root', 'CA') {
    foreach ($location in 'LocalMachine', 'CurrentUser') {
        $store = New-Object System.Security.Cryptography.X509Certificates.X509Store($storeName, $location)
        try {
            $store.Open('ReadOnly')
            foreach ($c in $store.Certificates) {
                if ($null -eq $c) { continue }
                $b64 = [Convert]::ToBase64String($c.RawData, 'InsertLineBreaks')
                $out += "-----BEGIN CERTIFICATE-----`n$b64`n-----END CERTIFICATE-----`n"
                $count++
            }
        } catch {
            Write-Host "skip $location\$storeName : $($_.Exception.Message)"
        } finally {
            $store.Close()
        }
    }
}
Set-Content -Path "$PSScriptRoot\.win_roots.pem" -Value $out -Encoding ASCII
Write-Host "Exported $count certificates"

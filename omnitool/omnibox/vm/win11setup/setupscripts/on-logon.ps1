$scriptFolder = "Z:\omnitool\omnibox\vm\win11setup\setupscripts"
$pythonScriptFile = "$scriptFolder\server\main.py"
$pythonServerPort = 5000

# Start the flask computer use server
Write-Host "Running the server on port $pythonServerPort"
python $pythonScriptFile --port $pythonServerPort

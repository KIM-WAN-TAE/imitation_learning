function addLog(msg, isError = false) {
    const consoleDiv = document.getElementById('console');
    const entry = document.createElement('div');
    const time = new Date().toLocaleTimeString();
    entry.className = isError ? 'log-entry log-error' : 'log-entry';
    entry.innerText = `[${time}] ${msg}`;
    consoleDiv.prepend(entry);
}

function sendCommand(cmd) {
    addLog(`CMD: ${cmd.toUpperCase()} INITIATED`);
    
    fetch(`/send?command=${cmd}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                addLog(`SUCCESS: ${data.message}`);
                document.getElementById('state-val').innerText = cmd.toUpperCase();
                document.getElementById('state-val').style.color = '#10b981';
            } else {
                addLog(`ERROR: ${data.message}`, true);
                document.getElementById('state-val').innerText = 'FAULT';
                document.getElementById('state-val').style.color = '#ef4444';
            }
        })
        .catch(err => {
            addLog(`CRITICAL: Connection failed`, true);
            document.getElementById('conn-val').innerText = 'OFFLINE';
            document.getElementById('conn-val').style.color = '#ef4444';
        });
}

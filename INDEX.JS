const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');

const app = express();
const PORT = 3000;

app.use(express.static('public'));
app.use(bodyParser.json());

const server = app.listen(PORT, () => {
  console.log(`Servidor escuchando en http://localhost:${PORT}`);
});

app.post('/python', (req, res) => {
  // Enviar el JSON al script de Python
  const pythonProcess = spawn('python', ['python.py']);

  // Convertir el JSON a una cadena y enviarlo al proceso de Python
  const jsonData = JSON.stringify(req.body);
  pythonProcess.stdin.write(jsonData);
  pythonProcess.stdin.end();

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Child process exited with code ${code}`);
    // En este punto, la imagen ya debería haber sido generada y guardada
    res.sendFile(path.join(__dirname, 'graficos', 'clusters_clientes.png'));
    
    // Cerrar el servidor después de completar el proceso de Python
    server.close();
  });
});
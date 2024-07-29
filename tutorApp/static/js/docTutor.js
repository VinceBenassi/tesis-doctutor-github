// Traductor
let audioContext = null;
let audioStream = null;
let audioData = [];
let isRecording = false;

function startRecording() {
    if (isRecording) return;

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            audioStream = stream;
            
            // Obtener la frecuencia de muestreo del stream de audio
            const trackSettings = audioStream.getAudioTracks()[0].getSettings();
            const streamSampleRate = trackSettings.sampleRate;

            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: streamSampleRate
            });

            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(2048, 1, 1);

            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                audioData.push(new Float32Array(channelData));
            };

            document.getElementById('status').textContent = 'Grabando...';
            document.getElementById('status').style.display = 'block';
            document.getElementById('startButton').textContent = 'Detener Grabación';
            isRecording = true;
        })
        .catch(err => {
            console.error("Error en getUserMedia:", err);
            alert('Error accediendo al micrófono: ' + err.message);
        });
}

function stopRecording() {
    if (!isRecording) return;

    isRecording = false;
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    if (audioContext) {
        audioContext.close();
    }

    const audioBlob = exportWAV(audioData, audioContext.sampleRate);
    audioData = [];

    document.getElementById('status').textContent = 'Procesando...';
    sendAudioToServer(audioBlob);

    document.getElementById('startButton').textContent = 'Iniciar Grabación';
}

function exportWAV(audioData, sampleRate) {
    const bufferLength = audioData.reduce((acc, curr) => acc + curr.length, 0);
    const buffer = new Float32Array(bufferLength);
    let offset = 0;
    for (let i = 0; i < audioData.length; i++) {
        buffer.set(audioData[i], offset);
        offset += audioData[i].length;
    }
    const wav = encodeWAV(buffer, sampleRate);
    return new Blob([wav], { type: 'audio/wav' });
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);

    return view;
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function sendAudioToServer(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.wav');

    fetch('/traductor/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        document.getElementById('idioma').textContent = data.idioma_detectado || 'No se pudo detectar el idioma';
        document.getElementById('textoOriginal').textContent = data.mensaje || 'No se detectó texto';
        document.getElementById('textoTraducido').textContent = data.texto || 'No se pudo traducir';
    })
    .catch(error => {
        console.error('Error en sendAudioToServer:', error);
        document.getElementById('status').textContent = 'Error al procesar el audio: ' + error.message;
    })
    .finally(() => {
        document.getElementById('status').style.display = 'none';
    });
}

document.getElementById('startButton').addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
// Fin Traductor



// Chatbot
var chatHistory = document.getElementById('chat-history');
chatHistory.scrollTop = chatHistory.scrollHeight;
// Fin chatbot



// Documentación
$('#tabla_pdf').DataTable({
    info: false,
    infoFiltered: false,
    countFiltered: false,
    autoWidth: false,
    ordering: false,
    "scrollX": true,
    stateSave: true,
    language: {
        processing: "Procesando...",
        lengthMenu: "Mostrar _MENU_ registros",
        zeroRecords: "No se encontraron resultados",
        emptyTable: "Ningún dato disponible en esta tabla",
        infoFiltered: "(filtrado de un total de _MAX_ registros)",
        search: "Buscar:",
        infoThousands: ",",
        loadingRecords: "Cargando...",
        paginate: {
            first: "Primero",
            last: "Último",
            next: "&raquo;",
            previous: "&laquo;"
        },
        info: "Mostrando página _PAGE_ de _PAGES_"
    }
});
// Fin Documentación



// Evaluaciones

// Fin evaluaciones




// Administrador

// Fin administrador
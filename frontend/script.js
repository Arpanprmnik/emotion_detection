// Get elements
const uploadButton = document.getElementById("uploadButton");
const audioFileInput = document.getElementById("audioFile");
const fileError = document.getElementById("fileError");
const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const audioPlayback = document.getElementById("audioPlayback");

// File upload handling
uploadButton.addEventListener("click", () => {
    audioFileInput.click(); // Trigger file input click
});

// When a file is selected, upload it
audioFileInput.addEventListener("change", () => {
    const file = audioFileInput.files[0];
    if (!file || file.type !== "audio/wav") {
        fileError.textContent = "Please select a valid .wav audio file.";
        return;
    }
    uploadAudio(file);
});

async function uploadAudio(file) {
    const formData = new FormData();
    formData.append("audio", file);

    try {
        const response = await fetch("http://localhost:5000/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        console.log("Upload response:", data);
        alert("Audio uploaded successfully!");
    } catch (error) {
        console.error("Error uploading audio:", error);
        alert("Error uploading audio.");
    }
}

// Recording handling
let mediaRecorder;
let audioChunks = [];

recordButton.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        recordButton.disabled = true;
        stopButton.disabled = false;

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            const audioURL = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioURL;
            audioPlayback.style.display = "block";
            uploadAudio(audioBlob);
            audioChunks = [];
        };

        mediaRecorder.start();
    } catch (error) {
        fileError.textContent = "Microphone access denied or not supported.";
    }
});

// Stop recording
stopButton.addEventListener("click", () => {
    mediaRecorder.stop();
    recordButton.disabled = false;
    stopButton.disabled = true;
});

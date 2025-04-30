const express = require("express");
const multer = require("multer");
const mongoose = require("mongoose");
const path = require("path");
const fetch = require("node-fetch"); // Ensure node-fetch is installed: npm install node-fetch@2
const { ObjectId } = require("mongodb");

const router = express.Router();

// MongoDB connection URL
const mongoURI = "mongodb://127.0.0.1:27017/audioDB";

// Create a connection instance for MongoDB
const conn = mongoose.createConnection(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

let gfs;

// Ensure MongoDB connection is open before using GridFS
conn.once("open", () => {
  gfs = new mongoose.mongo.GridFSBucket(conn.db, {
    bucketName: "audios",
  });
  console.log("GridFS initialized");
});

// Multer memory storage (holds file in memory before uploading to GridFS)
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Upload endpoint for audio files
router.post("/upload", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file uploaded" });
  }

  try {
    const filename = `audio-${Date.now()}${path.extname(req.file.originalname)}`;
    const uploadStream = gfs.openUploadStream(filename, {
      contentType: req.file.mimetype,
    });

    // Attach error handler first
    uploadStream.on("error", (err) => {
      console.error("Upload error:", err);
      return res.status(500).json({ error: "File upload failed" });
    });

    // Once upload is finished, call the Flask server
    uploadStream.on("finish", async () => {
      console.log(`✅ Uploaded to GridFS: ${filename}`);
      console.log("🆔 GridFS File ID:", uploadStream.id);

      try {
        const response = await fetch("http://localhost:5001/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fileId: uploadStream.id.toString() }),
        });

        const result = await response.json();
        console.log("🎯 Prediction Result:", result);

        if (result.error) {
          return res.status(500).json({ error: result.error });
        }

        res.status(200).json({
          message: "Audio uploaded and prediction complete",
          file: {
            id: uploadStream.id,
            filename: filename,
          },
          emotion: result.emotion,
        });
      } catch (err) {
        console.error("❌ Prediction error:", err);
        res.status(500).json({ error: "Prediction failed" });
      }
    });

    // Now start the upload
    uploadStream.end(req.file.buffer);

  } catch (error) {
    console.error("Unexpected upload error:", error);
    res.status(500).json({ error: "Error uploading audio file" });
  }
});


// Get all audio files
router.get("/files", async (req, res) => {
  try {
    const files = await gfs.find().toArray();  // This is correct for the 'audios' bucket
    if (!files || files.length === 0) {
      return res.status(404).json({ error: "No files found" });
    }
    res.json(files);
  } catch (error) {
    console.error("Error fetching files:", error);
    res.status(500).json({ error: "Error retrieving files" });
  }
});


// Download audio file by filename
router.get("/files/:filename", async (req, res) => {
  try {
    const file = await gfs.find({ filename: req.params.filename }).toArray();
    if (!file || file.length === 0) {
      return res.status(404).json({ error: "File not found" });
    }

    const downloadStream = gfs.openDownloadStreamByName(req.params.filename);
    downloadStream.pipe(res);

    downloadStream.on("error", (err) => {
      console.error("Download error:", err);
      res.status(500).json({ error: "Error downloading file" });
    });
  } catch (error) {
    console.error("Error retrieving file:", error);
    res.status(500).json({ error: "Error retrieving file" });
  }
});

module.exports = router;

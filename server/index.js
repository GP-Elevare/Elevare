const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const cors = require("cors");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 5000;

// Create folders if they don't exist
const dirs = ["uploads", "outputs"];
dirs.forEach((dir) => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir);
});

app.use(cors());
app.use("/outputs", express.static(path.join(__dirname, "outputs")));
app.use(express.json());

// Set up storage
const upload = multer({ dest: "uploads/" });

// --- VIDEO ROUTE ---
app.post("/process-video", upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).send("No video file.");
  const inputPath = req.file.path;
  const fps = req.body.fps || 30;
  const outputName = `processed-${Date.now()}.mp4`;
  const outputPath = path.join(__dirname, "outputs", outputName);

  ffmpeg(inputPath)
    .videoCodec("h264_nvenc") // Using your RTX 3060
    .outputOptions([`-filter:v fps=fps=${fps}`])
    .on("end", () => {
      fs.unlinkSync(inputPath);
      res.json({ downloadUrl: `/outputs/${outputName}` });
    })
    .on("error", (err) => {
      console.error(err);
      res.status(500).send("Processing failed");
    })
    .save(outputPath);
});

// --- POWERPOINT ROUTE (Simple Upload) ---
app.post("/upload-ppt", upload.single("powerpoint"), (req, res) => {
  if (!req.file) return res.status(400).send("No PPTX file.");

  console.log(`Received PowerPoint: ${req.file.originalname}`);

  // For now, we just acknowledge the upload
  res.json({
    message: "PowerPoint uploaded successfully!",
    fileName: req.file.originalname,
  });
});

app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`),
);

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
    .videoCodec("libx264")
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

const { spawn } = require("child_process");

// --- AI VIDEO ROUTE WITH FPS (FRAMES PER WINDOW) SUPPORT ---
app.post("/process-video-ai", upload.single("video"), async (req, res) => {
  if (!req.file) return res.status(400).send("No video file.");
  
  const inputPath = req.file.path;
  const fps = req.body.fps || 5;  // fps = frames per window (default 5)
  const timestamp = Date.now();
  const outputJson = path.join(__dirname, "outputs", `emotion-analysis-${timestamp}.json`);

  // Pass fps (frames per window) as third argument to Python script
  const pyProcess = spawn("python", ["-u", "ai_pipeline.py", inputPath, outputJson, fps.toString()]);

  pyProcess.stdout.on("data", (data) => {
    console.log(`Python stdout: ${data.toString().trim()}`);
  });

  pyProcess.stderr.on("data", (data) => {
    console.error(`Python stderr: ${data.toString().trim()}`);
  });

  pyProcess.on("close", (code) => {
    fs.unlinkSync(inputPath); // remove uploaded video
    if (code === 0) {
      res.json({ downloadUrl: `/outputs/${path.basename(outputJson)}` });
    } else {
      res.status(500).send("Processing failed");
    }
  });
});


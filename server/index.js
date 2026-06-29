const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

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

// --- POWERPOINT ROUTE (Upload + Call Python QG Pipeline) ---
app.post("/upload-ppt", upload.single("powerpoint"), (req, res) => {
  if (!req.file) return res.status(400).send("No PPTX file.");

  console.log(`Received PowerPoint: ${req.file.originalname}`);

  // Full file path to uploaded PPTX
  const pptxPath = path.resolve(req.file.path);

  // Call python script
  const pythonProcess = spawn("python", ["qg_pipeline.py", pptxPath]);

  console.log("Python process started, PID:", pythonProcess.pid);

  pythonProcess.on("error", (err) => {
    console.error("Failed to start Python process:", err);
  });

  let outputData = "";
  let errorData = "";

  pythonProcess.stdout.on("data", (data) => {
    console.log("Python stdout:", data.toString());
    outputData += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    console.log("Python stderr:", data.toString()); // log always, not just on error
    errorData += data.toString();
  });

  pythonProcess.on("close", (code) => {
    if (code !== 0) {
      console.error("Python error:", errorData);
      return res.status(500).json({
        message: "Python script failed",
        error: errorData,
      });
    }

    console.log("Python script executed successfully!");
    console.log("Output:", outputData);

    // If python returns JSON, parse it
    try {
      const parsedOutput = JSON.parse(outputData);
      console.log("JSON parsed successfully!");

      return res.json({
        message: "PowerPoint processed successfully!",
        fileName: req.file.originalname,
        result: parsedOutput,
      });
    } catch (err) {
      console.log("Python output is not JSON, returning raw text.");

      return res.json({
        message: "PowerPoint processed successfully!",
        fileName: req.file.originalname,
        result: outputData,
      });
    }
  });
});

// --- AI VIDEO ROUTE WITH FPS (FRAMES PER WINDOW) SUPPORT ---
app.post("/process-video-ai", upload.single("video"), async (req, res) => {
  if (!req.file) return res.status(400).send("No video file.");

  const inputPath = req.file.path;
  const fps = req.body.fps || 5; // fps = frames per window (default 5)
  const intervalSec = req.body.intervalSec || 1; // feedback metrics interval in seconds (default 1)
  const timestamp = Date.now();
  const outputJson = path.join(
    __dirname,
    "outputs",
    `emotion-analysis-${timestamp}.json`,
  );

  // Pass fps (frames per window) as third argument to Python script
  const pyProcess = spawn("python", [
    "-u",
    "ai_pipeline.py",
    inputPath,
    outputJson,
    fps.toString(),
    intervalSec.toString(),
  ]);

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

// --- SIMPLE JSON RETRIEVAL ROUTES ---
app.get("/feedback", (req, res) => {
  const filePath = path.join(__dirname, "final_feedback.json");
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).json({ error: "feedback file not found" });
  }
});

app.get("/questions", (req, res) => {
  // Update this to point to the Bloom JSON file instead of the old one
  const filePath = path.join(__dirname, "QA_pairs_bloom.json");
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).json({ error: "questions file not found" });
  }
});

app.get("/qa-results", (req, res) => {
  const filePath = path.join(__dirname, "qa_results.json");
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).json({ error: "QA results file not found" });
  }
});

// --- SAVE ANSWERS & GRADE ROUTE ---
app.post("/save-answers", (req, res) => {
  const { updatedData } = req.body;

  if (!updatedData) {
    return res.status(400).json({ error: "No updated data provided" });
  }

  // 1. Define our file paths
  const inputFilePath = path.join(__dirname, "QA_pairs_bloom.json");
  const outputResultsPath = path.join(__dirname, "qa_results.json");
  const pythonScriptPath = path.join(__dirname, "QA_pipeline.py");

  // 2. Overwrite the Bloom file with the new student answers
  fs.writeFile(inputFilePath, JSON.stringify(updatedData, null, 2), (err) => {
    if (err) {
      console.error("Failed to save answers:", err);
      return res.status(500).json({ error: "Failed to save answers" });
    }

    console.log(
      "Answers saved to QA_pairs_bloom.json. Starting grading pipeline...",
    );

    // 3. Start the Python QA pipeline using the active environment's 'python'
    const qaProcess = spawn("python", [
      pythonScriptPath,
      inputFilePath,
      outputResultsPath,
    ]);

    // Optional: Catch spawn errors so the server doesn't crash completely
    qaProcess.on("error", (err) => {
      console.error("Failed to start the Python process:", err.message);
      return res
        .status(500)
        .json({ error: "Failed to start grading pipeline." });
    });

    let pythonError = "";

    qaProcess.stdout.on("data", (data) => {
      console.log(`QA Pipeline: ${data.toString().trim()}`);
    });

    qaProcess.stderr.on("data", (data) => {
      console.error(`QA Pipeline Error: ${data.toString().trim()}`);
      pythonError += data.toString();
    });

    // 4. Wait for Python to finish grading
    qaProcess.on("close", (code) => {
      if (code !== 0) {
        console.error("Grading failed:", pythonError);
        return res.status(500).json({
          error: "Answers were saved, but the grading pipeline failed.",
          details: pythonError,
        });
      }

      console.log("Grading complete! Results saved to qa_results.json");

      // 5. Finally, tell React that everything worked!
      res.json({ message: "Answers saved and graded successfully!" });
    });
  });
});

// Start the server (MUST be at the bottom)
app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`),
);

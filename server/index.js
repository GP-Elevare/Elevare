const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const db = require("./db"); // The pool we created
const protect = require("./auth"); // The middleware
require('./init_db'); // This will run the table/admin check on startup
require('dotenv').config();

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

//Database

app.post('/auth/register', async (req, res) => {
  const { first_name, last_name, email, password } = req.body;
  try {
    const hashed = await bcrypt.hash(password, 10);
    await db.query(
      'INSERT INTO users (first_name, last_name, email, password_hash) VALUES ($1, $2, $3, $4)',
                   [first_name, last_name, email, hashed]
    );
    res.status(201).json({ message: 'User registered' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/auth/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const result = await db.query('SELECT * FROM users WHERE email = $1', [email]);
    if (result.rows.length === 0) return res.status(400).json({ error: 'Invalid credentials' });

    const user = result.rows[0];
    const valid = await bcrypt.compare(password, user.password_hash);
    if (!valid) return res.status(400).json({ error: 'Invalid credentials' });

    const token = jwt.sign({ id: user.id, email: user.email }, process.env.JWT_SECRET, { expiresIn: '1d' });
    res.json({ token, user: { id: user.id, first_name: user.first_name } });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

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
app.post("/process-video-ai", protect, upload.single("video"), (req, res) => {
  if (!req.file) return res.status(400).send("No video file.");

  const inputPath = req.file.path;
  const fps = req.body.fps || 5;
  const intervalSec = req.body.intervalSec || 1;
  const timestamp = Date.now();
  const outputJsonPath = path.join(__dirname, "outputs", `emotion-analysis-${timestamp}.json`);
  const userId = req.user.id;

  const pyProcess = spawn("python", ["-u", "ai_pipeline.py", inputPath, outputJsonPath, fps.toString(), intervalSec.toString()]);

  pyProcess.on("close", async (code) => {
    // Delete the heavy video file immediately
    fs.unlinkSync(inputPath);

    if (code === 0) {
      try {
        const analysisData = fs.readFileSync(outputJsonPath, 'utf8');

        await db.query(
          'INSERT INTO reports (user_id, report_type, report_data) VALUES ($1, $2, $3)',
                       [userId, 'AI_Video_Analysis', analysisData]
        );

        res.json({ message: "Video analyzed and saved to your account." });
      } catch (err) {
        res.status(500).json({ error: "Failed to save analysis to DB" });
      }

      // Delete the generated JSON file from the outputs folder
      if (fs.existsSync(outputJsonPath)) fs.unlinkSync(outputJsonPath);

    } else {
      res.status(500).send("Processing failed");
    }
  });
});

app.get('/api/my-reports', protect, async (req, res) => {
  try {
    const result = await db.query(
      'SELECT id, report_type, report_data, created_at FROM reports WHERE user_id = $1 ORDER BY created_at DESC',
      [req.user.id]
    );
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch reports" });
  }
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
app.post("/save-answers", protect, (req, res) => {
  const { updatedData } = req.body;
  const userId = req.user.id; // From the auth middleware
  const timestamp = Date.now();

  // Create temporary files specific to this request
  const inputFilePath = path.join(__dirname, `temp_qa_input_${timestamp}.json`);
  const outputResultsPath = path.join(__dirname, `temp_qa_output_${timestamp}.json`);
  const pythonScriptPath = path.join(__dirname, "QA_pipeline.py");

  // Write the user's specific answers to a temp file
  fs.writeFile(inputFilePath, JSON.stringify(updatedData, null, 2), (err) => {
    if (err) return res.status(500).json({ error: "Failed to write temp file" });

    const qaProcess = spawn("python", [pythonScriptPath, inputFilePath, outputResultsPath]);

    qaProcess.on("close", async (code) => {
      if (code === 0) {
        try {
          // Read the output from Python
          const resultData = fs.readFileSync(outputResultsPath, 'utf8');

          // Insert straight into PostgreSQL as JSONB
          await db.query(
            'INSERT INTO reports (user_id, report_type, report_data) VALUES ($1, $2, $3)',
                         [userId, 'QA_Evaluation', resultData] // resultData is auto-parsed to JSONB by pg
          );

          res.json({ message: "Graded and saved to database successfully!" });
        } catch (dbErr) {
          res.status(500).json({ error: "Failed to save to database" });
        }
      } else {
        res.status(500).json({ error: "Python pipeline failed" });
      }

      // CLEANUP: Delete the temp files so they don't clutter your server
      if (fs.existsSync(inputFilePath)) fs.unlinkSync(inputFilePath);
      if (fs.existsSync(outputResultsPath)) fs.unlinkSync(outputResultsPath);
    });
  });
});
// Start the server (MUST be at the bottom)
app.listen(PORT, () =>
  console.log(`Server running on http://localhost:${PORT}`),
);

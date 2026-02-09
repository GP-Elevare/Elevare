const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const cors = require("cors");
const path = require("path");
const fs = require("fs");

const app = express();
app.use(cors());
app.use("/outputs", express.static("outputs"));

// Setup storage for uploads
const upload = multer({ dest: "uploads/" });

app.post("/process-video", upload.single("video"), (req, res) => {
  const inputPath = req.file.path;
  const fps = req.body.fps || 30;
  const outputName = `processed-${Date.now()}.mp4`;
  const outputPath = path.join(__dirname, "outputs", outputName);

  ffmpeg(inputPath)
    .outputOptions([`-filter:v fps=fps=${fps}`])
    .on("end", () => {
      // Clean up the original upload
      fs.unlinkSync(inputPath);
      res.json({ downloadUrl: `/outputs/${outputName}` });
    })
    .on("error", (err) => {
      console.error(err);
      res.status(500).send("Error processing video");
    })
    .save(outputPath);
});

app.listen(5000, () => console.log("Server running on port 5000"));

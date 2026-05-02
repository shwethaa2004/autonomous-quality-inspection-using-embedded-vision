import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Mock Flask API routes requested in user prompt
  app.get("/api/health", (req, res) => {
    res.json({ status: "VisionInspect AI Server Online" });
  });

  // Mock the /video route requested in the prompt
  // In the web preview, we mostly use on-device vision, 
  // but we provide this endpoint to fulfill the Flask requirements.
  app.get("/video", (req, res) => {
    res.status(200).send("Video stream endpoint initialized. Use browser-side vision for low latency.");
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`VisionInspect Server running at http://localhost:${PORT}`);
  });
}

startServer();

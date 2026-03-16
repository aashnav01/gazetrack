/**
 * GazeTrack - MongoDB Backend
 * ────────────────────────────
 * Receives session data (CSV + metadata) from the deployed website
 * and saves it to MongoDB Atlas.
 *
 * SETUP:
 *   1. npm install
 *   2. Create a .env file:
 *        MONGODB_URI=mongodb+srv://youruser:yourpass@yourcluster.mongodb.net/gazetrack
 *        PORT=3001
 *   3. node server.js
 *
 * DEPLOY:
 *   - Railway / Render / Fly.io — push this folder, set env vars in dashboard
 *   - Then update MONGO_API_URL in app.js to your deployed URL
 */

require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const { MongoClient } = require('mongodb');

const app  = express();
const PORT = process.env.PORT || 3001;
const MONGODB_URI = process.env.MONGODB_URI;

if (!MONGODB_URI) {
  console.error('ERROR: MONGODB_URI not set in .env file');
  process.exit(1);
}

// ── Middleware ──────────────────────────────────────────────
app.use(cors()); // Allow all origins (your deployed site + local dev)
app.use(express.json({ limit: '10mb' })); // Sessions can be ~1-2 MB

// ── MongoDB connection ──────────────────────────────────────
let db;
async function connectDB() {
  const client = new MongoClient(MONGODB_URI);
  await client.connect();
  db = client.db('gazetrack');
  console.log('Connected to MongoDB');
}

// ── Routes ──────────────────────────────────────────────────

// Health check
app.get('/', (req, res) => {
  res.json({ status: 'ok', service: 'GazeTrack API' });
});

// Save a session
app.post('/api/sessions', async (req, res) => {
  try {
    const {
      filename, pid, age, group,
      clinician, location, notes,
      timestamp, csv
    } = req.body;

    if (!pid || !csv) {
      return res.status(400).json({ error: 'pid and csv are required' });
    }

    const doc = {
      filename:   filename  || `gaze_${pid}_${Date.now()}.csv`,
      pid,
      age:        age || null,
      group:      group || null,
      clinician:  clinician || null,
      location:   location || null,
      notes:      notes || null,
      timestamp:  timestamp ? new Date(timestamp) : new Date(),
      csv,                        // full CSV text
      createdAt:  new Date()
    };

    const result = await db.collection('sessions').insertOne(doc);
    console.log(`Session saved: ${pid} | ${group} | ${filename} | _id: ${result.insertedId}`);

    res.json({ success: true, id: result.insertedId });
  } catch (err) {
    console.error('Save error:', err);
    res.status(500).json({ error: err.message });
  }
});

// List all sessions (for your own dashboard/review)
app.get('/api/sessions', async (req, res) => {
  try {
    const sessions = await db.collection('sessions')
      .find({}, { projection: { csv: 0 } }) // exclude CSV from listing
      .sort({ createdAt: -1 })
      .limit(100)
      .toArray();
    res.json(sessions);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Download a specific session's CSV
app.get('/api/sessions/:id/csv', async (req, res) => {
  try {
    const { ObjectId } = require('mongodb');
    const session = await db.collection('sessions').findOne(
      { _id: new ObjectId(req.params.id) }
    );
    if (!session) return res.status(404).json({ error: 'Not found' });
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="${session.filename}"`);
    res.send(session.csv);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ── Start ───────────────────────────────────────────────────
connectDB().then(() => {
  app.listen(PORT, () => {
    console.log(`GazeTrack API running on port ${PORT}`);
    console.log(`POST sessions to: http://localhost:${PORT}/api/sessions`);
  });
}).catch(err => {
  console.error('MongoDB connection failed:', err);
  process.exit(1);
});

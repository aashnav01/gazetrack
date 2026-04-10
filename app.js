console.log('%c GazeTrack v22 (Star Keeper — TabPFN Compatible)','background:#00e5b0;color:#000;font-weight:bold;font-size:14px');
import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs';

// ─── DEVICE ────────────────────────────────────────────────────────────────────
const isMobile = /Android|iPad|iPhone|iPod|Mobile/i.test(navigator.userAgent)
  || (navigator.maxTouchPoints > 1 && window.innerWidth < 1200);
const MP_DELEGATE = isMobile ? 'CPU' : 'GPU';
const IS_TABLET = /iPad|Android(?!.*Mobile)/i.test(navigator.userAgent)
  || (window.innerWidth >= 768 && window.innerWidth <= 1280);

// ─── CONSTANTS ─────────────────────────────────────────────────────────────────
const CALIB_SAMPLE_MS          = 600;
const CALIB_TOTAL_PTS          = 5;
const CALIB_GAP_MS             = 500;
const MIN_SAMPLES              = 25;
const RIDGE_ALPHA              = 0.01;
const CALIB_IRIS_BRIDGE_MS     = 150;
const CALIB_DWELL_REQUIRED_MS  = 2500;
const CALIB_FORCE_SKIP_MS      = 8000;
const LEFT_IRIS  = [468,469,470,471];
const RIGHT_IRIS = [473,474,475,476];
const L_CORNERS  = [33,133];
const R_CORNERS  = [362,263];
const VAL_DWELL_MS    = 3000;
const VAL_GAP_MS      = 600;
const VAL_STAR_RADIUS = 48;
const VAL_SAMPLE_START= 0.60;
const VAL_INTRO_MS    = 2000;
const GOOD_STREAK_NEEDED = 8;
const BAD_STREAK_HIDE    = 12;
const MONGO_API_URL = 'https://gazetrack-api.onrender.com/api/sessions';

// ─── MONOTONIC TIMESTAMP ───────────────────────────────────────────────────────
let _mpLastTs = -1;
function mpNow() {
  const t = performance.now();
  _mpLastTs = t > _mpLastTs ? t : _mpLastTs + 0.001;
  return _mpLastTs;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ─── I-DT GAZE EVENT CLASSIFIER ───────────────────────────────────────────────
// Converts raw gaze XY stream → "Fixation" | "Saccade" | "Blink"
// Compatible with extract_features_tabpfn.py Category Right column.
//
// Algorithm: I-DT (Identification by Dispersion Threshold) — Salvucci & Goldberg 2000
//   Window of recent samples → if spatial dispersion ≤ threshold → Fixation
//   Otherwise: check inter-sample velocity → if high → Saccade, else transitional Fixation
//
// Tuning:
//   IDT_DISPERSION_PX  — increase if too many saccades on a 4K screen
//   IDT_MIN_WINDOW_MS  — minimum time window for fixation decision (≥ 2 frames at 28 Hz)
//   IDT_SACCADE_VEL    — px/ms threshold; 0.3 is conservative for webcam noise
// ═══════════════════════════════════════════════════════════════════════════════
const IDT_WINDOW_MS       = 100;   // rolling window for dispersion calc
const IDT_DISPERSION_PX   = 60;    // max spatial dispersion for fixation (pixels)
const IDT_SACCADE_VEL     = 0.25;  // px/ms inter-sample velocity → saccade
const IDT_MIN_SAMPLES     = 2;     // need at least this many samples in window

let _idtBuffer = [];          // [{x, y, t}]  — rolling gaze window
let _fixationIndex = 0;       // increments each time a new fixation starts
let _lastCategory  = null;    // previous frame's category

function idtClassify(gazeX, gazeY, timestamp, isBlink) {
  // ── Blink takes priority ──────────────────────────────────────────────────
  if (isBlink) {
    _idtBuffer = [];           // clear window on blink; velocity is garbage after
    _lastCategory = 'Blink';
    return { category: 'Blink', fixationIndex: _fixationIndex };
  }

  // ── Push into rolling window ──────────────────────────────────────────────
  _idtBuffer.push({ x: gazeX, y: gazeY, t: timestamp });
  _idtBuffer = _idtBuffer.filter(s => timestamp - s.t <= IDT_WINDOW_MS);

  if (_idtBuffer.length < IDT_MIN_SAMPLES) {
    // Not enough data yet — treat as fixation continuation
    return { category: _lastCategory ?? 'Fixation', fixationIndex: _fixationIndex };
  }

  // ── Dispersion: (maxX-minX) + (maxY-minY) — Salvucci & Goldberg formula ──
  const xs = _idtBuffer.map(s => s.x);
  const ys = _idtBuffer.map(s => s.y);
  const dispersion = (Math.max(...xs) - Math.min(...xs))
                   + (Math.max(...ys) - Math.min(...ys));

  let category;
  if (dispersion <= IDT_DISPERSION_PX) {
    category = 'Fixation';
  } else {
    // High dispersion — check inter-sample velocity for Saccade vs noisy fixation
    const last = _idtBuffer[_idtBuffer.length - 1];
    const prev = _idtBuffer[_idtBuffer.length - 2];
    const dt   = Math.max(last.t - prev.t, 1);
    const vel  = Math.hypot(last.x - prev.x, last.y - prev.y) / dt;
    category   = vel >= IDT_SACCADE_VEL ? 'Saccade' : 'Fixation';
  }

  // ── Track fixation index (increment on Fixation start after non-Fixation) ─
  if (category === 'Fixation' && _lastCategory !== 'Fixation') {
    _fixationIndex++;
  }
  _lastCategory = category;
  return { category, fixationIndex: _fixationIndex };
}

// ─── IRIS PIXEL RADIUS → PUPIL PROXY ─────────────────────────────────────────
// MediaPipe gives iris landmarks in normalised coords.
// We compute pixel radius from the 4 iris points and use it as a
// relative pupil size proxy (z-scored per participant in Python anyway).
// Column written as "Pupil Diameter Right [mm]" to match pipeline header
// even though the unit is pixels — Python z-scores it, so absolute scale
// doesn't matter, only within-session relative change does.
function irisRadiusPx(lmList, irisIndices, canvasW, canvasH) {
  try {
    const pts = irisIndices.map(i => ({
      x: lmList[i].x * canvasW,
      y: lmList[i].y * canvasH
    }));
    // Diameter = distance between opposite iris edge points (idx 0↔2, 1↔3)
    const d1 = Math.hypot(pts[0].x - pts[2].x, pts[0].y - pts[2].y);
    const d2 = Math.hypot(pts[1].x - pts[3].x, pts[1].y - pts[3].y);
    return ((d1 + d2) / 2).toFixed(3);
  } catch(e) { return ''; }
}

// ─── FACE CONFIDENCE → TRACKING RATIO ─────────────────────────────────────────
// MediaPipe doesn't expose a single confidence float the same way SMI does.
// We derive a proxy: fraction of iris landmarks that are valid (non-zero).
// Output: 0–100, matching "Tracking Ratio [%]" column.
function trackingRatio(lmList, canvasW, canvasH) {
  if (!lmList || lmList.length === 0) return 0;
  const irisAll = [...LEFT_IRIS, ...RIGHT_IRIS];
  let valid = 0;
  for (const i of irisAll) {
    const lm = lmList[i];
    if (lm && lm.x > 0 && lm.y > 0) valid++;
  }
  return ((valid / irisAll.length) * 100).toFixed(1);
}

// ─── CSV ROW BUILDER ───────────────────────────────────────────────────────────
// Builds one row per frame in the exact column format expected by
// extract_features_tabpfn.py. Column names must match exactly.
const CSV_COLUMNS = [
  'RecordingTime [ms]',
  'Time of Day [h:m:s:ms]',
  'Trial',
  'Stimulus',
  'Participant',
  'Category Group',
  'Category Right',
  'Category Left',
  'Index Right',
  'Index Left',
  'Point of Regard Right X [px]',
  'Point of Regard Right Y [px]',
  'Point of Regard Left X [px]',
  'Point of Regard Left Y [px]',
  'Pupil Diameter Right [mm]',
  'Pupil Diameter Left [mm]',
  'Tracking Ratio [%]',
];

function buildCsvRow(opts) {
  // opts: { ts, gazeX, gazeY, category, fixIdx, pupilR, pupilL, trackRatio,
  //         trial, stimulus, participant, isFacePresent }
  const now = new Date();
  const tod = `${now.getHours()}:${String(now.getMinutes()).padStart(2,'0')}:`
            + `${String(now.getSeconds()).padStart(2,'0')}:`
            + `${String(now.getMilliseconds()).padStart(3,'0')}`;

  const catGroup = opts.isFacePresent ? 'Eye' : 'Information';
  // Webcam is monocular — mirror right gaze to left (same as ETSDS binocular)
  const gx = (opts.gazeX != null && opts.isFacePresent) ? opts.gazeX.toFixed(2) : '';
  const gy = (opts.gazeY != null && opts.isFacePresent) ? opts.gazeY.toFixed(2) : '';

  return [
    opts.ts.toFixed(3),
    tod,
    opts.trial,
    opts.stimulus,
    opts.participant,
    catGroup,
    opts.isFacePresent ? opts.category : '-',     // '-' matches original format
    opts.isFacePresent ? opts.category : '-',     // left mirrors right
    opts.isFacePresent ? opts.fixIdx   : '-',
    opts.isFacePresent ? opts.fixIdx   : '-',
    gx, gy,   // right
    gx, gy,   // left (mirrored)
    opts.isFacePresent ? opts.pupilR : '',
    opts.isFacePresent ? opts.pupilL : '',
    opts.trackRatio,
  ];
}

function csvEscape(v) {
  const s = String(v ?? '');
  return s.includes(',') || s.includes('"') || s.includes('\n')
    ? `"${s.replace(/"/g, '""')}"`
    : s;
}

function rowsToCsv(rows) {
  return [CSV_COLUMNS, ...rows].map(r => r.map(csvEscape).join(',')).join('\r\n');
}

// ─── STATE ─────────────────────────────────────────────────────────────────────
let phase          = 'intake';
let faceLandmarker = null;
let camStream      = null;
let sessionStream  = null;
let gazeModel      = null;
let calibSamples   = [];
let affineBias     = {dx:0, dy:0, sx:1, sy:1};
// Calibration
let calibPoints     = [];
let calibIdx        = 0;
let calibRaf        = null;
let calibState      = 'idle';
let calibFailCount  = 0;
let calibSkipActive = false;
let _calibCurrentGaze   = null;
let _calibLastGaze      = null;
let _calibLastGazeTs    = 0;
let _calibLastFeat      = null;
let _calibHoldStart     = null;
let _calibHoldLostAt    = null;
let _calibSampling      = false;
let _calibSamplingStart = 0;
let _calibPointSamples  = [];
let _calibSparkled      = false;
let _calibSkipTimer     = null;
let _calibLoopT         = 0;
let _calibDwellAccum    = 0;
let _calibLastFrameTs   = 0;
let _calibProcTs        = -1;
let _calibTargetY       = -1;
// Creature calibration
let creatureEls     = [];
let doneCalibPoints = new Set();
let calibParticles  = [];
let calibFloaties   = [];
// Validation
let valPoints=[],valIdx=0,valSamples=[],valRaf=null,valStart=0;
const VAL_PARTICLES = [];
let _valLastDetectTs = -1;
// Recording — recordedRows now stores raw arrays (not objects) for CSV
let recordedRows=[], recordedFrames=[];   // recordedRows = CSV rows; recordedFrames kept for back-compat
let totalF=0,trackedF=0;
let sessionStart=0,timerInt=null;
let csvData=null;
let META={pid:'',age:'',group:'',clinician:'',location:'',notes:'',stimulus:''};
// Playlist / multi-trial
let playlist         = [];
let playlistTrialIdx = 0;
let trialStartMs     = 0;
let photoDurSec      = 5;
let _photoTimer      = null;
let _dragIdx         = null;
// Saccade
let prevGaze=null,prevGazeTime=null;
// Face
let calibFacePresent=false;
// Adaptive EAR threshold
let _earCalibSamples=[];
let _earThreshold=0.22;
// Pupil hold
let _lastKnownPupilDiamR=3.5;
let _lastKnownPupilDiamL=3.5;
// Position banner
let _goodFrameStreak=0,_badFrameStreak=0,_allclearShowing=false,_allclearHideTimer=null;
// Preview
let previewFl=null,previewRaf=null,lastPreviewTs=-1,_prevLastRun=0;
let _lastVideoTime=-1;
let procRaf=null;
// Stimulus throttle
let _stimLastTs = -1;
let _cachedVideoW = 640;
let _cachedVideoH = 480;
// Pre-flight
const pfState={cam:'scanning',face:'scanning',light:'scanning',browser:'scanning'};
let pfRaf=null,_pfSamples=[],_pfThrottle=0,_pfCanvas=null,_pfCtx=null;
// Star background
let calibStarCvs=null,calibStarCtx=null,calibStars=[];
let calibFxCvs=null,calibFxCtx=null;
let calibStoryBanner=null,calibHud=null;

// ─── DOM REFS ──────────────────────────────────────────────────────────────────
const screens = {
  intake:   document.getElementById('s-intake'),
  loading:  document.getElementById('s-loading'),
  calib:    document.getElementById('s-calib'),
  stimulus: document.getElementById('s-stimulus'),
  done:     document.getElementById('s-done'),
};
const camPreview  = document.getElementById('cam-preview');
const camCanvas   = document.getElementById('cam-canvas');
const camCtx      = camCanvas?.getContext('2d');
const calibCanvas = document.getElementById('calib-canvas');
const calibCtx    = calibCanvas?.getContext('2d');
const gazeCanvas  = document.getElementById('gaze-canvas');
const gazeCtx     = gazeCanvas?.getContext('2d');
const webcam      = document.getElementById('webcam');
const stimVideo   = document.getElementById('stim-video');
const stimImage   = document.getElementById('stim-image');
const allclearBanner = document.getElementById('position-allclear');

function showScreen(n) {
  Object.values(screens).forEach(s => s?.classList.remove('active'));
  screens[n]?.classList.add('active');
}
if (IS_TABLET) {
  [calibCanvas, gazeCanvas, camCanvas].forEach(c => { if (c) c.style.touchAction = 'none'; });
  document.querySelectorAll('button,.clickable').forEach(el => { if (el) el.style.minHeight = '48px'; });
}

// ─── INJECT CALIBRATION CSS ────────────────────────────────────────────────────
(function injectCalibCSS(){
  if (document.getElementById('gazetrack-calib-css')) return;
  const style = document.createElement('style');
  style.id = 'gazetrack-calib-css';
  style.innerHTML = `
    @keyframes fall {
      0%   { transform:translateY(0) rotate(0deg) scale(1); opacity:1; }
      80%  { opacity:0.8; }
      100% { transform:translateY(110vh) rotate(520deg) scale(0.4); opacity:0; }
    }
    @keyframes fall-gentle {
      0%   { transform:translateY(0) rotate(0deg); opacity:0.9; }
      100% { transform:translateY(80vh) rotate(180deg); opacity:0; }
    }
    @keyframes break-pulse {
      0%,100% { box-shadow:0 0 0 0 rgba(255,200,0,0.4); }
      50%     { box-shadow:0 0 0 18px rgba(255,200,0,0); }
    }
    @keyframes creature-enter {
      0%   { transform:translate(-50%,-50%) scale(0) rotate(-20deg); opacity:0; }
      70%  { transform:translate(-50%,-50%) scale(1.15) rotate(3deg); opacity:1; }
      100% { transform:translate(-50%,-50%) scale(1) rotate(0deg); opacity:1; }
    }
    @keyframes creature-done {
      0%   { transform:translate(-50%,-50%) scale(1); }
      50%  { transform:translate(-50%,-50%) scale(1.3) rotate(10deg); }
      100% { transform:translate(-50%,-50%) scale(1.1); }
    }
    .calib-creature-wrap {
      position:absolute; transform:translate(-50%,-50%);
      pointer-events:none; z-index:30; opacity:0; transition:opacity 0.3s;
    }
    .calib-creature-wrap.visible {
      animation:creature-enter 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards; opacity:1;
    }
    .calib-creature-wrap.done { animation:creature-done 0.4s ease forwards; }
    .calib-story-banner {
      position:absolute; top:18px; left:50%; transform:translateX(-50%);
      background:rgba(10,5,40,0.88); border:1.5px solid rgba(160,130,255,0.45);
      border-radius:20px; padding:10px 28px; z-index:60; color:#ddd5ff;
      font-size:15px; text-align:center; max-width:520px; min-width:280px;
      transition:opacity 0.4s; pointer-events:none;
      font-family:'Comic Sans MS','Chalkboard SE',cursive;
    }
    .calib-hud {
      position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
      display:flex; gap:10px; z-index:60; pointer-events:none;
    }
    .calib-hud-paw { font-size:20px; transition:transform 0.3s, filter 0.3s; }
    .calib-hud-paw.done { transform:scale(1.4); filter:drop-shadow(0 0 6px #ffd700); }
    .calib-bg {
      position:absolute; inset:0;
      background:radial-gradient(ellipse at 50% 40%, #0d0d2b 0%, #060614 100%); z-index:0;
    }
    #calib-fx-canvas   { position:absolute; inset:0; pointer-events:none; z-index:40; }
    #calib-star-canvas { position:absolute; inset:0; pointer-events:none; z-index:1;  }
    .playlist-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
    .playlist-label  { font-size:12px; font-weight:700; color:rgba(255,255,255,0.5); }
    .playlist-hint   { font-size:11px; color:rgba(255,255,255,0.3); }
    #playlist-list   { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:4px; }
    .playlist-item   {
      display:flex; align-items:center; gap:8px; padding:7px 10px;
      background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
      border-radius:8px; cursor:grab; user-select:none;
      font-size:12.5px; transition:background 0.15s;
    }
    .playlist-item:hover { background:rgba(0,229,176,0.06); }
    .pl-drag   { color:rgba(255,255,255,0.25); font-size:16px; cursor:grab; }
    .pl-icon   { font-size:15px; }
    .pl-name   { flex:1; color:#fff; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .pl-type   { font-size:10px; color:rgba(255,255,255,0.35); font-family:monospace; }
    .pl-remove { background:none; border:none; color:rgba(255,92,58,0.7); font-size:16px; cursor:pointer; padding:0 4px; line-height:1; }
    .pl-remove:hover { color:#ff5c3a; }
    .playlist-photo-dur { display:flex; align-items:center; gap:8px; margin-top:8px; font-size:12px; color:rgba(255,255,255,0.5); }
    .playlist-photo-dur input { width:52px; padding:4px 8px; border-radius:6px; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.15); color:#fff; font-size:12px; text-align:center; }
  `;
  document.head.appendChild(style);
})();

// ─── AUDIO ─────────────────────────────────────────────────────────────────────
let _audioCtx = null;
function getAudioCtx() {
  if (!_audioCtx) { try { _audioCtx = new AudioContext(); } catch(e) {} }
  return _audioCtx;
}
function playTone(freq, vol, duration, type='sine', delay=0) {
  const a = getAudioCtx(); if (!a) return;
  setTimeout(() => {
    try {
      const o = a.createOscillator(), g = a.createGain();
      o.connect(g); g.connect(a.destination);
      o.type = type; o.frequency.value = freq;
      g.gain.setValueAtTime(0, a.currentTime);
      g.gain.linearRampToValueAtTime(vol, a.currentTime + 0.025);
      g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + duration);
      o.start(); o.stop(a.currentTime + duration);
    } catch(e) {}
  }, delay);
}
function playAnimalJingle(idx) {
  const scales = [[523,659,784,1047],[587,698,880,1175],[659,784,988,1319],[698,880,1047,1397],[784,988,1175,1568]];
  scales[idx % scales.length].forEach((f,i) => playTone(f, 0.08, 0.2, 'sine', i*60));
}
function playSuccessChime() {
  [523,659,784,880,1047,1319].forEach((f,i) => playTone(f, 0.10, 0.28, 'sine', i*65));
  setTimeout(() => playTone(1568, 0.12, 0.6, 'sine'), 420);
}
function playHappyJingle(idx) {
  const jingles = [[784,880,1047,1175],[659,784,880,1047],[523,659,784,880],[698,784,880,1047],[523,659,784,1047]];
  jingles[idx % jingles.length].forEach((f,i) => playTone(f, 0.09, 0.25, 'sine', i*55));
}
function playChime(freq, vol, duration) {
  const a = getAudioCtx(); if (!a) return;
  try {
    const o = a.createOscillator(), g = a.createGain();
    o.connect(g); g.connect(a.destination);
    o.type = 'sine';
    o.frequency.setValueAtTime(freq * 0.8, a.currentTime);
    o.frequency.linearRampToValueAtTime(freq, a.currentTime + 0.1);
    g.gain.setValueAtTime(0, a.currentTime);
    g.gain.linearRampToValueAtTime(vol, a.currentTime + 0.05);
    g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + duration);
    o.start(); o.stop(a.currentTime + duration);
  } catch(e) {}
}

// ─── CONFETTI ───────────────────────────────────────────────────────────────────
function startConfettiLight() {
  const colors = ['#f4a261','#e9c46a','#90e0ef','#ffb3c1','#c77dff','#ff6b6b','#4cc9f0'];
  for (let i = 0; i < 8; i++) {
    setTimeout(() => {
      const conf = document.createElement('div');
      conf.style.cssText = `position:fixed;top:-10px;left:${Math.random()*100}vw;
        width:5px;height:5px;background:${colors[Math.floor(Math.random()*colors.length)]};
        border-radius:50%;z-index:9999;pointer-events:none;
        animation:fall-gentle ${2.5+Math.random()*1.5}s linear forwards;`;
      document.body.appendChild(conf);
      setTimeout(() => conf.remove(), 4100);
    }, i * 40);
  }
}
function startConfettiBig() {
  const colors = ['#f4a261','#e9c46a','#90e0ef','#ffb3c1','#c77dff','#ff6b6b','#4cc9f0','#ffd700','#00e5b0','#ff9f43'];
  for (let i = 0; i < 25; i++) {
    setTimeout(() => {
      const conf = document.createElement('div');
      const size = 5 + Math.random() * 8;
      conf.style.cssText = `position:fixed;top:-12px;left:${Math.random()*100}vw;
        width:${size}px;height:${size}px;
        background:${colors[Math.floor(Math.random()*colors.length)]};
        border-radius:${Math.random()>0.5?'50%':'3px'};
        z-index:9999;pointer-events:none;
        animation:fall ${3.5+Math.random()*2.5}s linear forwards;`;
      document.body.appendChild(conf);
      setTimeout(() => conf.remove(), 6200);
    }, i * 18);
  }
  playSuccessChime();
}

// ─── POSITION ALL-CLEAR ────────────────────────────────────────────────────────
function updateAllClear(bright) {
  const allOk = pfState.face === 'pass' && pfState.light === 'pass';
  if (allOk) {
    _badFrameStreak = 0;
    _goodFrameStreak = Math.min(_goodFrameStreak + 1, GOOD_STREAK_NEEDED + 1);
    if (_goodFrameStreak >= GOOD_STREAK_NEEDED && !_allclearShowing) showAllClear(bright);
  } else {
    _goodFrameStreak = 0;
    _badFrameStreak = Math.min(_badFrameStreak + 1, BAD_STREAK_HIDE + 1);
    if (_badFrameStreak >= BAD_STREAK_HIDE && _allclearShowing) hideAllClear();
  }
}
function showAllClear(bright) {
  _allclearShowing = true;
  clearTimeout(_allclearHideTimer);
  const tagsEl = document.getElementById('allclear-tags');
  if (tagsEl) tagsEl.innerHTML = ['✓ Face visible · Good distance','✓ Lighting OK']
    .map(t=>`<span class="allclear-tag">${t}</span>`).join('');
  const det = document.getElementById('allclear-detail');
  if (det) det.textContent = `Brightness ${Math.round(bright)}/255`;
  allclearBanner?.classList.add('show');
  playChime(660, 0.08, 0.4);
}
function hideAllClear() {
  _allclearShowing = false;
  allclearBanner?.classList.remove('show');
}

// ─── PRE-FLIGHT ────────────────────────────────────────────────────────────────
function pfSet(id, state, msg) {
  pfState[id] = state;
  const card = document.getElementById('pfc-' + id);
  const stat = document.getElementById('pfc-' + id + '-s');
  if (card) card.className = 'pf-check ' + state;
  if (stat) stat.innerHTML = msg;
  pfUpdateScore();
}
function pfUpdateScore() {
  const vals   = Object.values(pfState);
  const done   = vals.filter(v => v !== 'scanning').length;
  const passes = vals.filter(v => v === 'pass').length;
  const warns  = vals.filter(v => v === 'warn').length;
  const total  = vals.length;
  const score  = Math.round(((passes + warns * 0.6) / total) * 100);
  const fill   = document.getElementById('pf-score-fill');
  const pct    = document.getElementById('pf-score-pct');
  if (fill) { fill.style.width = score+'%'; fill.style.background = score>=75?'var(--accent)':score>=50?'var(--gold)':'var(--warn)'; }
  if (pct)  pct.textContent = done===total ? score+'%' : '...';
  const tips = [];
  if (pfState.light==='fail')   tips.push('<strong>💡 Too dark:</strong> Add a front-facing lamp.');
  if (pfState.light==='warn')   tips.push('<strong>💡 Lighting:</strong> Brighter room helps iris detection.');
  if (pfState.face==='fail')    tips.push('<strong>👤 No face:</strong> Make sure child is in frame, camera at eye level.');
  if (pfState.browser==='warn') tips.push('<strong>🌐 Browser:</strong> Use Chrome for best webcam performance.');
  if (window.innerHeight < 700) {
    tips.push(`<strong>📐 Small window (${window.innerHeight}px):</strong> Press <kbd style="background:#333;padding:1px 5px;border-radius:4px;font-size:11px;">F11</kbd> for fullscreen.`);
  }
  const adv = document.getElementById('pf-advice');
  if (adv) { adv.innerHTML = tips.join('<br>'); adv.className = 'pf-advice' + (tips.length ? ' show' : ''); }
  const btn = document.getElementById('start-btn');
  if (btn && !btn.disabled) {
    const critFails = ['cam','face'].filter(k => pfState[k]==='fail').length;
    if      (critFails>0)  { btn.textContent='⚠ Proceed Anyway'; btn.style.background='linear-gradient(135deg,#ff9f43,#e17f20)'; }
    else if (done<total)   { btn.textContent='Begin Session →'; btn.style.background=''; }
    else if (score>=75)    { btn.textContent='✅ All Clear - Begin Session'; btn.style.background=''; }
    else                   { btn.textContent='⚠ Proceed with Warnings'; btn.style.background='linear-gradient(135deg,#ca8a04,#a16207)'; }
  }
}
function pfAnalyseFrame() {
  if (phase !== 'intake') return;
  const now = performance.now();
  if (now - _pfThrottle < 200) { pfRaf = requestAnimationFrame(pfAnalyseFrame); return; }
  _pfThrottle = now;
  if (!camPreview || camPreview.readyState < 2) { pfRaf = requestAnimationFrame(pfAnalyseFrame); return; }
  if (!_pfCanvas) {
    _pfCanvas = document.createElement('canvas'); _pfCanvas.width=80; _pfCanvas.height=60;
    _pfCtx = _pfCanvas.getContext('2d', {willReadFrequently:true});
  }
  try {
    _pfCtx.drawImage(camPreview, 0, 0, 80, 60);
    const d = _pfCtx.getImageData(0, 0, 80, 60).data;
    let sumR=0,sumG=0,sumB=0,n=0;
    for (let i=0;i<d.length;i+=4){sumR+=d[i];sumG+=d[i+1];sumB+=d[i+2];n++;}
    const brightness = (sumR+sumG+sumB)/(n*3);
    _pfSamples.push(brightness);
    if (_pfSamples.length>3) _pfSamples.shift();
    const avgBright = _pfSamples.reduce((a,b)=>a+b,0)/_pfSamples.length;
    if      (avgBright>=60&&avgBright<=220) pfSet('light','pass',`✓ Good (${Math.round(avgBright)}/255)`);
    else if (avgBright<40)                  pfSet('light','fail',`✗ Too dark (${Math.round(avgBright)}) - add light`);
    else if (avgBright<60)                  pfSet('light','warn',`⚠ Dim (${Math.round(avgBright)}) - improve lighting`);
    else                                    pfSet('light','warn',`⚠ Bright (${Math.round(avgBright)}) - reduce backlight`);
    updateAllClear(avgBright);
  } catch(e) {}
  pfRaf = requestAnimationFrame(pfAnalyseFrame);
}
function pfCheckBrowser() {
  const ua = navigator.userAgent;
  const isChrome  = /Chrome/.test(ua) && !/Edg/.test(ua) && !/OPR/.test(ua);
  const isEdge    = /Edg/.test(ua);
  const isFirefox = /Firefox/.test(ua);
  if      (isChrome)  pfSet('browser','pass','✓ Chrome - optimal');
  else if (isEdge)    pfSet('browser','pass','✓ Edge - good');
  else if (isFirefox) pfSet('browser','warn','⚠ Firefox - use Chrome for best results');
  else                pfSet('browser','warn','⚠ Use Chrome for best results');
}

// ─── CAMERA INIT ───────────────────────────────────────────────────────────────
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:640,max:1280},height:{ideal:480,max:720},facingMode:'user',frameRate:{ideal:60,min:30}},
      audio:false
    });
    camStream = stream;
    if (camPreview) { camPreview.srcObject = stream; await camPreview.play(); }
    if (webcam)     { webcam.srcObject = stream; await webcam.play(); }
    pfSet('cam','pass','✓ Camera ready');
    pfCheckBrowser();
    pfRaf = requestAnimationFrame(pfAnalyseFrame);
    return stream;
  } catch(e) {
    pfSet('cam','fail',`✗ Camera error: ${e.message}`);
    throw e;
  }
}

// ─── MEDIAPIPE INIT ────────────────────────────────────────────────────────────
async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: MP_DELEGATE,
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  });
}

// ─── EAR (EYE ASPECT RATIO) ────────────────────────────────────────────────────
function getEAR(lm, topI, botI, leftI, rightI, W, H) {
  const p = i => ({ x: lm[i].x * W, y: lm[i].y * H });
  const vert  = Math.hypot(p(topI).x-p(botI).x, p(topI).y-p(botI).y);
  const horiz = Math.hypot(p(leftI).x-p(rightI).x, p(leftI).y-p(rightI).y);
  return horiz > 0 ? vert / horiz : 1;
}
function detectBlink(lm, W, H) {
  try {
    const earL = getEAR(lm,386,374,362,263,W,H);
    const earR = getEAR(lm,159,145,33,133,W,H);
    const ear  = (earL + earR) / 2;
    if (phase === 'calib') {
      _earCalibSamples.push(ear);
      if (_earCalibSamples.length > 60) {
        _earThreshold = Math.min(..._earCalibSamples) * 1.4 + 0.04;
        _earCalibSamples = _earCalibSamples.slice(-60);
      }
    }
    return ear < _earThreshold;
  } catch(e) { return false; }
}

// ─── GAZE FEATURE EXTRACTION ───────────────────────────────────────────────────
function extractFeatures(lm, W, H) {
  try {
    const iris = (indices) => {
      const pts = indices.map(i => ({ x: lm[i].x * W, y: lm[i].y * H }));
      return { x: pts.reduce((s,p)=>s+p.x,0)/pts.length, y: pts.reduce((s,p)=>s+p.y,0)/pts.length };
    };
    const lc = (is) => is.map(i => ({ x: lm[i].x * W, y: lm[i].y * H }));
    const li = iris(LEFT_IRIS),  ri = iris(RIGHT_IRIS);
    const lcs = lc(L_CORNERS),   rcs = lc(R_CORNERS);
    const lw = Math.hypot(lcs[1].x-lcs[0].x, lcs[1].y-lcs[0].y);
    const rw = Math.hypot(rcs[1].x-rcs[0].x, rcs[1].y-rcs[0].y);
    if (lw < 5 || rw < 5) return null;
    const lnx = (li.x - lcs[0].x) / lw, lny = (li.y - lcs[0].y) / lw;
    const rnx = (ri.x - rcs[0].x) / rw, rny = (ri.y - rcs[0].y) / rw;
    return [lnx, lny, rnx, rny, lw / W, rw / W];
  } catch(e) { return null; }
}

// ─── RIDGE REGRESSION ─────────────────────────────────────────────────────────
function fitRidge(samples, alpha) {
  const n = samples.length, f = samples[0].feat.length;
  const X = samples.map(s => [...s.feat, 1]);
  const Xt = X[0].map((_,j) => X.map(r => r[j]));
  const XtX = Xt.map(row => Xt[0].map((_,k) => row.reduce((s,v,i) => s + v * Xt[k][i], 0)));
  for (let i = 0; i < f; i++) XtX[i][i] += alpha;
  const Xty_x = Xt.map(row => row.reduce((s,v,i) => s + v * samples[i].tx, 0));
  const Xty_y = Xt.map(row => row.reduce((s,v,i) => s + v * samples[i].ty, 0));
  const inv = invertMatrix(XtX);
  if (!inv) return null;
  const wx = inv.map(r => r.reduce((s,v,i) => s + v * Xty_x[i], 0));
  const wy = inv.map(r => r.reduce((s,v,i) => s + v * Xty_y[i], 0));
  return { wx, wy };
}
function invertMatrix(M) {
  const n = M.length;
  const A = M.map(r => [...r]);
  const I = M.map((_,i) => M[0].map((_,j) => i===j ? 1 : 0));
  for (let c = 0; c < n; c++) {
    let max = c;
    for (let r = c+1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[max][c])) max = r;
    [A[c], A[max]] = [A[max], A[c]]; [I[c], I[max]] = [I[max], I[c]];
    const d = A[c][c]; if (Math.abs(d) < 1e-12) return null;
    for (let j = 0; j < n; j++) { A[c][j] /= d; I[c][j] /= d; }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      for (let j = 0; j < n; j++) { A[r][j] -= f * A[c][j]; I[r][j] -= f * I[c][j]; }
    }
  }
  return I;
}
function predictGaze(feat) {
  if (!gazeModel) return null;
  const x = [...feat, 1];
  const gx = x.reduce((s,v,i) => s + v * gazeModel.wx[i], 0);
  const gy = x.reduce((s,v,i) => s + v * gazeModel.wy[i], 0);
  return {
    x: gx * affineBias.sx + affineBias.dx,
    y: gy * affineBias.sy + affineBias.dy
  };
}

// ─── CALIBRATION ──────────────────────────────────────────────────────────────
function buildCalibPoints() {
  const W = window.innerWidth, H = window.innerHeight;
  const mx = 80, my = 80;
  return [
    { x: W/2,    y: H/2    },
    { x: mx,     y: my     },
    { x: W-mx,   y: my     },
    { x: mx,     y: H-my   },
    { x: W-mx,   y: H-my   },
  ];
}

function spawnCalibCreature(idx, x, y) {
  const animals = ['🐱','🐶','🐸','🐰','🦊'];
  const wrap = document.createElement('div');
  wrap.className = 'calib-creature-wrap';
  wrap.style.left = x + 'px'; wrap.style.top = y + 'px';
  wrap.innerHTML = `<span style="font-size:52px;display:block;text-align:center">${animals[idx % animals.length]}</span>`;
  document.getElementById('s-calib')?.appendChild(wrap);
  requestAnimationFrame(() => wrap.classList.add('visible'));
  creatureEls[idx] = wrap;
  return wrap;
}

function sparkleCalibPoint(x, y) {
  if (!calibFxCtx) return;
  const colors = ['#ffd700','#00e5b0','#ff6b9d','#c77dff','#4cc9f0'];
  for (let i = 0; i < 18; i++) {
    const angle = (i / 18) * Math.PI * 2;
    const speed = 2 + Math.random() * 4;
    const life  = 40 + Math.floor(Math.random() * 30);
    calibParticles.push({ x, y, vx: Math.cos(angle)*speed, vy: Math.sin(angle)*speed,
      life, maxLife: life, color: colors[Math.floor(Math.random()*colors.length)], r: 3+Math.random()*3 });
  }
}

function startCalibration() {
  phase = 'calib';
  showScreen('calib');
  calibPoints = buildCalibPoints();
  calibIdx = 0;
  calibSamples = [];
  doneCalibPoints.clear();
  creatureEls = [];
  calibParticles = [];
  calibFloaties = [];
  _fixationIndex = 0;
  _lastCategory  = null;
  _idtBuffer     = [];

  // Star canvas
  const scr = document.getElementById('s-calib');
  if (scr) {
    calibStarCvs = document.getElementById('calib-star-canvas');
    if (!calibStarCvs) {
      calibStarCvs = document.createElement('canvas');
      calibStarCvs.id = 'calib-star-canvas';
      scr.appendChild(calibStarCvs);
    }
    calibStarCvs.width  = window.innerWidth;
    calibStarCvs.height = window.innerHeight;
    calibStarCtx = calibStarCvs.getContext('2d');
    calibStars = Array.from({length:80}, () => ({
      x: Math.random()*window.innerWidth, y: Math.random()*window.innerHeight,
      r: 0.5+Math.random()*1.5, a: Math.random(), da: 0.005+Math.random()*0.01
    }));
    // FX canvas
    calibFxCvs = document.getElementById('calib-fx-canvas');
    if (!calibFxCvs) {
      calibFxCvs = document.createElement('canvas');
      calibFxCvs.id = 'calib-fx-canvas';
      scr.appendChild(calibFxCvs);
    }
    calibFxCvs.width  = window.innerWidth;
    calibFxCvs.height = window.innerHeight;
    calibFxCtx = calibFxCvs.getContext('2d');
    // Story banner
    calibStoryBanner = document.getElementById('calib-story-banner');
    if (!calibStoryBanner) {
      calibStoryBanner = document.createElement('div');
      calibStoryBanner.className = 'calib-story-banner';
      calibStoryBanner.id = 'calib-story-banner';
      scr.appendChild(calibStoryBanner);
    }
    calibStoryBanner.textContent = '✨ Look at each friend to wake them up!';
    // HUD
    calibHud = document.getElementById('calib-hud');
    if (!calibHud) {
      calibHud = document.createElement('div');
      calibHud.className = 'calib-hud';
      calibHud.id = 'calib-hud';
      scr.appendChild(calibHud);
    }
    calibHud.innerHTML = calibPoints.map((_,i) =>
      `<span class="calib-hud-paw" id="hud-paw-${i}">🐾</span>`).join('');
  }
  showCalibPoint(0);
  calibRaf = requestAnimationFrame(calibLoop);
}

function showCalibPoint(idx) {
  calibState = 'waiting';
  _calibHoldStart = null;
  _calibDwellAccum = 0;
  _calibSampling = false;
  _calibSparkled = false;
  _calibPointSamples = [];
  clearTimeout(_calibSkipTimer);
  if (idx < calibPoints.length) {
    const pt = calibPoints[idx];
    spawnCalibCreature(idx, pt.x, pt.y);
    playAnimalJingle(idx);
    _calibSkipTimer = setTimeout(() => {
      if (calibIdx === idx && calibState !== 'done') forceSkipCalibPoint();
    }, CALIB_FORCE_SKIP_MS);
  }
}

function forceSkipCalibPoint() {
  calibFailCount++;
  advanceCalibPoint();
}

function advanceCalibPoint() {
  doneCalibPoints.add(calibIdx);
  const paw = document.getElementById(`hud-paw-${calibIdx}`);
  if (paw) paw.classList.add('done');
  const wrap = creatureEls[calibIdx];
  if (wrap) { wrap.classList.add('done'); setTimeout(() => wrap.remove(), 500); }
  calibIdx++;
  if (calibIdx >= calibPoints.length) {
    finishCalibration();
  } else {
    setTimeout(() => showCalibPoint(calibIdx), CALIB_GAP_MS);
  }
}

function finishCalibration() {
  calibState = 'done';
  cancelAnimationFrame(calibRaf);
  clearTimeout(_calibSkipTimer);
  if (calibSamples.length >= MIN_SAMPLES) {
    gazeModel = fitRidge(calibSamples, RIDGE_ALPHA);
    computeAffineBias();
    playSuccessChime();
    startConfettiBig();
    if (calibStoryBanner) calibStoryBanner.textContent = '🎉 All friends woken up! Amazing!';
    setTimeout(() => startValidation(), 1800);
  } else {
    if (calibStoryBanner) calibStoryBanner.textContent = '⚠ Not enough gaze data. Please try again.';
    setTimeout(() => startCalibration(), 2500);
  }
}

function computeAffineBias() {
  if (!gazeModel || calibSamples.length === 0) return;
  let sx=0,sy=0,n=0;
  for (const s of calibSamples) {
    const pred = predictGaze(s.feat);
    if (!pred) continue;
    sx += (s.tx - pred.x); sy += (s.ty - pred.y); n++;
  }
  affineBias = { dx: n>0?sx/n:0, dy: n>0?sy/n:0, sx:1, sy:1 };
}

function calibLoop(t) {
  calibRaf = requestAnimationFrame(calibLoop);
  const dt = t - _calibLoopT; _calibLoopT = t;

  // Draw starfield
  if (calibStarCtx) {
    calibStarCtx.clearRect(0,0,calibStarCvs.width,calibStarCvs.height);
    for (const s of calibStars) {
      s.a += s.da; if (s.a>1||s.a<0) s.da=-s.da;
      calibStarCtx.beginPath();
      calibStarCtx.arc(s.x,s.y,s.r,0,Math.PI*2);
      calibStarCtx.fillStyle=`rgba(255,255,255,${s.a.toFixed(2)})`;
      calibStarCtx.fill();
    }
  }

  // Draw particles
  if (calibFxCtx) {
    calibFxCtx.clearRect(0,0,calibFxCvs.width,calibFxCvs.height);
    calibParticles = calibParticles.filter(p => p.life > 0);
    for (const p of calibParticles) {
      p.x += p.vx; p.y += p.vy; p.vy += 0.1; p.life--;
      const a = p.life / p.maxLife;
      calibFxCtx.beginPath();
      calibFxCtx.arc(p.x,p.y,p.r*a,0,Math.PI*2);
      calibFxCtx.fillStyle = p.color + Math.round(a*255).toString(16).padStart(2,'0');
      calibFxCtx.fill();
    }
  }

  // Gaze-based dwell detection
  if (calibState !== 'waiting' && calibState !== 'dwelling') return;
  const feat = _calibLastFeat;
  if (!feat) return;
  const gaze = predictGaze(feat);
  if (!gaze) return;
  _calibCurrentGaze = gaze;

  const pt = calibPoints[calibIdx];
  if (!pt) return;
  const dist = Math.hypot(gaze.x - pt.x, gaze.y - pt.y);
  const threshold = 120;

  if (dist < threshold) {
    if (!_calibHoldStart) _calibHoldStart = t;
    _calibDwellAccum += dt;
    if (_calibDwellAccum >= CALIB_DWELL_REQUIRED_MS && !_calibSampling) {
      _calibSampling = true;
      _calibSamplingStart = t;
    }
    if (_calibSampling && (t - _calibSamplingStart) < CALIB_SAMPLE_MS) {
      if (_calibLastFeat) {
        _calibPointSamples.push({ feat: _calibLastFeat, tx: pt.x / window.innerWidth, ty: pt.y / window.innerHeight });
      }
    }
    if (_calibSampling && (t - _calibSamplingStart) >= CALIB_SAMPLE_MS) {
      if (!_calibSparkled) {
        _calibSparkled = true;
        for (const s of _calibPointSamples) calibSamples.push(s);
        sparkleCalibPoint(pt.x, pt.y);
        startConfettiLight();
        playHappyJingle(calibIdx);
        advanceCalibPoint();
      }
    }
  } else {
    _calibHoldStart = null;
    _calibDwellAccum = Math.max(0, _calibDwellAccum - dt * 1.5);
    if (_calibSampling) { _calibSampling = false; }
  }
}

// ─── VALIDATION ────────────────────────────────────────────────────────────────
function buildValPoints() {
  const W = window.innerWidth, H = window.innerHeight;
  return [
    { x: W*0.1, y: H*0.1 }, { x: W*0.9, y: H*0.1 },
    { x: W*0.5, y: H*0.5 },
    { x: W*0.1, y: H*0.9 }, { x: W*0.9, y: H*0.9 },
  ];
}

function startValidation() {
  phase = 'val';
  valPoints = buildValPoints();
  valIdx = 0; valSamples = [];
  showScreen('calib');
  if (calibStoryBanner) calibStoryBanner.textContent = '⭐ Now follow each star!';
  setTimeout(() => {
    valRaf = requestAnimationFrame(valLoop);
    valStart = performance.now();
  }, VAL_INTRO_MS);
}

function valLoop(t) {
  valRaf = requestAnimationFrame(valLoop);
  if (!calibCtx) return;
  const elapsed = t - valStart;
  const W = window.innerWidth, H = window.innerHeight;
  calibCtx.clearRect(0,0,W,H);
  if (valIdx >= valPoints.length) {
    finishValidation();
    return;
  }
  const pt = valPoints[valIdx];
  const phase_t = elapsed % (VAL_DWELL_MS + VAL_GAP_MS);

  // Draw star target
  calibCtx.save();
  calibCtx.translate(pt.x, pt.y);
  const scale = 0.8 + 0.2 * Math.sin(t * 0.005);
  calibCtx.scale(scale, scale);
  calibCtx.font = `${VAL_STAR_RADIUS}px serif`;
  calibCtx.textAlign = 'center'; calibCtx.textBaseline = 'middle';
  calibCtx.fillText('⭐', 0, 0);
  calibCtx.restore();

  // Sample gaze in back 40% of dwell window
  if (phase_t > VAL_DWELL_MS * VAL_SAMPLE_START && phase_t < VAL_DWELL_MS) {
    const gaze = _calibLastFeat ? predictGaze(_calibLastFeat) : null;
    if (gaze) valSamples.push({ target: pt, gaze });
  }
  if (phase_t > VAL_DWELL_MS + VAL_GAP_MS * 0.5) {
    valIdx++;
    valStart = t - (phase_t - VAL_DWELL_MS - VAL_GAP_MS * 0.5);
  }
}

function finishValidation() {
  cancelAnimationFrame(valRaf);
  if (calibCtx) calibCtx.clearRect(0,0,window.innerWidth,window.innerHeight);
  // Compute RMS error
  let rmse = 0;
  if (valSamples.length > 0) {
    rmse = Math.sqrt(valSamples.reduce((s,v) =>
      s + Math.pow(v.gaze.x - v.target.x, 2) + Math.pow(v.gaze.y - v.target.y, 2), 0
    ) / valSamples.length);
  }
  console.log(`Validation RMSE: ${rmse.toFixed(1)}px over ${valSamples.length} samples`);
  if (calibStoryBanner) calibStoryBanner.textContent = `✅ Ready! Accuracy ≈ ${Math.round(rmse)}px`;
  setTimeout(() => startStimulus(), 1500);
}

// ─── STIMULUS / RECORDING ─────────────────────────────────────────────────────
function startStimulus() {
  phase = 'stimulus';
  showScreen('stimulus');
  recordedRows   = [];
  recordedFrames = [];   // kept for any legacy references
  totalF = 0; trackedF = 0;
  sessionStart = performance.now();
  playlistTrialIdx = 0;
  _idtBuffer    = [];
  _fixationIndex = 0;
  _lastCategory  = null;
  loadPlaylistItem(0);
  timerInt = setInterval(updateTimer, 1000);
}

function loadPlaylistItem(idx) {
  if (idx >= playlist.length) { endSession(); return; }
  playlistTrialIdx = idx;
  trialStartMs = performance.now();
  const item = playlist[idx];
  if (item.type === 'video') {
    stimImage.style.display = 'none';
    stimVideo.style.display = '';
    stimVideo.src = item.objectURL;
    stimVideo.play();
    stimVideo.onended = () => {
      clearTimeout(_photoTimer);
      setTimeout(() => loadPlaylistItem(idx + 1), 500);
    };
  } else {
    stimVideo.style.display = 'none';
    stimImage.style.display = '';
    stimImage.src = item.objectURL;
    clearTimeout(_photoTimer);
    _photoTimer = setTimeout(() => loadPlaylistItem(idx + 1), photoDurSec * 1000);
  }
}

function updateTimer() {
  const elapsed = Math.floor((performance.now() - sessionStart) / 1000);
  const m = Math.floor(elapsed/60), s = elapsed%60;
  const el = document.getElementById('session-timer');
  if (el) el.textContent = `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

// ─── MAIN PROCESSING LOOP ──────────────────────────────────────────────────────
// This is called every frame via requestAnimationFrame during recording.
// It runs MediaPipe, extracts gaze, classifies the event with I-DT,
// and writes one CSV row per frame.
function processFrame(timestamp) {
  procRaf = requestAnimationFrame(processFrame);
  if (phase !== 'stimulus' && phase !== 'calib' && phase !== 'val') return;

  const src = webcam;
  if (!src || src.readyState < 2) return;
  if (!faceLandmarker) return;

  // Throttle to ~30fps to match training data
  if (timestamp - _stimLastTs < 30) return;

  // MediaPipe needs monotonically increasing timestamps
  const mpTs = mpNow();
  if (mpTs <= _mpLastTs) return;

  let result;
  try { result = faceLandmarker.detectForVideo(src, mpTs); }
  catch(e) { return; }

  const W = src.videoWidth  || 640;
  const H = src.videoHeight || 480;
  const lm = result?.faceLandmarks?.[0];
  const facePresent = !!(lm && lm.length > 0);

  totalF++;

  // ── Calibration phase: collect features only ──────────────────────────────
  if (phase === 'calib' || phase === 'val') {
    if (facePresent) {
      const feat = extractFeatures(lm, W, H);
      if (feat) _calibLastFeat = feat;
      calibFacePresent = facePresent;
      if (phase === 'intake') {
        pfSet('face', 'pass', '✓ Face detected');
      }
    } else {
      calibFacePresent = false;
    }
    _stimLastTs = timestamp;
    return;
  }

  // ── Stimulus / recording phase ────────────────────────────────────────────
  const ts  = mpTs - sessionStart;
  const trial    = playlistTrialIdx + 1;
  const stimulus = playlist[playlistTrialIdx]?.name ?? '';

  let gazeX = null, gazeY = null;
  let pupilR = '', pupilL = '';
  let isBlink = false;
  let tRatio  = 0;
  let category = 'Unclassified';
  let fixIdx   = _fixationIndex;

  if (facePresent) {
    trackedF++;
    isBlink  = detectBlink(lm, W, H);
    const feat = extractFeatures(lm, W, H);

    if (feat) {
      _calibLastFeat = feat;
      const gaze = predictGaze(feat);
      if (gaze && !isBlink) {
        gazeX = Math.max(0, Math.min(window.innerWidth,  gaze.x));
        gazeY = Math.max(0, Math.min(window.innerHeight, gaze.y));
      }
    }

    // Pupil proxy (pixel radius of iris)
    const pR = irisRadiusPx(lm, RIGHT_IRIS, W, H);
    const pL = irisRadiusPx(lm, LEFT_IRIS,  W, H);
    if (pR) { pupilR = pR; _lastKnownPupilDiamR = parseFloat(pR); }
    else      pupilR = _lastKnownPupilDiamR.toFixed(3);
    if (pL) { pupilL = pL; _lastKnownPupilDiamL = parseFloat(pL); }
    else      pupilL = _lastKnownPupilDiamL.toFixed(3);

    tRatio = trackingRatio(lm, W, H);

    // ── I-DT EVENT CLASSIFICATION ─────────────────────────────────────────
    const classified = idtClassify(
      gazeX ?? (window.innerWidth / 2),
      gazeY ?? (window.innerHeight / 2),
      ts,
      isBlink
    );
    category = classified.category;
    fixIdx   = classified.fixationIndex;

    // Draw gaze dot on overlay
    if (gazeCtx && gazeX != null) {
      gazeCtx.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
      gazeCtx.beginPath();
      gazeCtx.arc(gazeX, gazeY, 12, 0, Math.PI*2);
      gazeCtx.fillStyle = category === 'Saccade'
        ? 'rgba(255,120,60,0.55)'   // orange for saccade
        : category === 'Blink'
        ? 'rgba(100,100,255,0.3)'   // blue for blink
        : 'rgba(0,229,176,0.55)';   // teal for fixation
      gazeCtx.fill();
    }
  } else {
    // Face lost — clear IDT buffer (velocity after gap is garbage)
    _idtBuffer = [];
    _lastCategory = null;
    if (gazeCtx) gazeCtx.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
    tRatio = 0;
  }

  // ── Write CSV row ─────────────────────────────────────────────────────────
  const row = buildCsvRow({
    ts,
    gazeX,
    gazeY,
    category,
    fixIdx,
    pupilR,
    pupilL,
    trackRatio: tRatio,
    trial,
    stimulus,
    participant: META.pid,
    isFacePresent: facePresent,
  });
  recordedRows.push(row);
  // Keep recordedFrames in sync for any legacy code that reads it
  recordedFrames.push({
    ts, gazeX, gazeY, category, fixIdx, pupilR, pupilL,
    tRatio, facePresent, trial, stimulus
  });

  _stimLastTs = timestamp;
}

// ─── END SESSION ──────────────────────────────────────────────────────────────
function endSession() {
  phase = 'done';
  clearInterval(timerInt);
  cancelAnimationFrame(procRaf);
  clearTimeout(_photoTimer);

  // Build CSV string from recorded rows
  csvData = rowsToCsv(recordedRows);

  // Stats
  const durationSec = (performance.now() - sessionStart) / 1000;
  const trackPct    = totalF > 0 ? ((trackedF / totalF) * 100).toFixed(1) : '0';

  // Count event types from recorded frames for summary
  const cats = recordedFrames.reduce((acc, f) => {
    const c = f.category || 'Unclassified';
    acc[c] = (acc[c] || 0) + 1;
    return acc;
  }, {});
  const fixCount = cats['Fixation'] || 0;
  const sacCount = cats['Saccade']  || 0;
  const blkCount = cats['Blink']    || 0;

  showScreen('done');
  const sumEl = document.getElementById('session-summary');
  if (sumEl) {
    sumEl.innerHTML = `
      <p>Duration: <strong>${Math.round(durationSec)}s</strong></p>
      <p>Frames: <strong>${totalF}</strong> total · <strong>${trackedF}</strong> tracked (${trackPct}%)</p>
      <p>Events: <strong>${fixCount}</strong> fixation · <strong>${sacCount}</strong> saccade · <strong>${blkCount}</strong> blink frames</p>
      <p>Fixations detected: <strong>${_fixationIndex}</strong> unique</p>
      <p>Trials: <strong>${playlist.length}</strong></p>
      <p>CSV rows: <strong>${recordedRows.length}</strong></p>
    `;
  }
  playSuccessChime();
  startConfettiBig();
  autoUploadSession();
}

// ─── CSV DOWNLOAD ─────────────────────────────────────────────────────────────
function downloadCsv() {
  if (!csvData) return;
  const blob = new Blob([csvData], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  const pid  = META.pid || 'participant';
  a.href     = url;
  a.download = `gazetrack_${pid}_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── MONGO UPLOAD ─────────────────────────────────────────────────────────────
async function autoUploadSession() {
  try {
    const payload = {
      meta:    META,
      csvRows: recordedRows.length,
      summary: {
        totalFrames:   totalF,
        trackedFrames: trackedF,
        fixations:     _fixationIndex,
        durationMs:    performance.now() - sessionStart,
        playlist:      playlist.map(p => ({ name: p.name, type: p.type })),
      },
      csv: csvData,
    };
    const res = await fetch(MONGO_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      console.log('Session uploaded:', await res.json());
      const el = document.getElementById('upload-status');
      if (el) el.textContent = '✓ Saved to server';
    }
  } catch(e) {
    console.warn('Upload failed (offline?):', e.message);
  }
}

// ─── INTAKE / PLAYLIST UI ─────────────────────────────────────────────────────
function renderPlaylist() {
  const list = document.getElementById('playlist-list');
  if (!list) return;
  list.innerHTML = '';
  playlist.forEach((item, idx) => {
    const li = document.createElement('li');
    li.className = 'playlist-item';
    li.draggable = true;
    li.dataset.idx = idx;
    li.innerHTML = `
      <span class="pl-drag">⠿</span>
      <span class="pl-icon">${item.type === 'video' ? '🎬' : '🖼️'}</span>
      <span class="pl-name">${item.name}</span>
      <span class="pl-type">${item.type}</span>
      <button class="pl-remove" data-idx="${idx}">×</button>`;
    li.addEventListener('dragstart', e => { _dragIdx = idx; li.style.opacity='0.5'; });
    li.addEventListener('dragend',   e => { li.style.opacity=''; });
    li.addEventListener('dragover',  e => { e.preventDefault(); });
    li.addEventListener('drop',      e => {
      e.preventDefault();
      if (_dragIdx == null || _dragIdx === idx) return;
      const moved = playlist.splice(_dragIdx, 1)[0];
      playlist.splice(idx, 0, moved);
      _dragIdx = null;
      renderPlaylist();
    });
    list.appendChild(li);
  });
  list.querySelectorAll('.pl-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = parseInt(btn.dataset.idx);
      URL.revokeObjectURL(playlist[i].objectURL);
      playlist.splice(i, 1);
      renderPlaylist();
    });
  });
}

function handleFileAdd(files) {
  for (const file of files) {
    const type = file.type.startsWith('video') ? 'video' : 'image';
    playlist.push({ objectURL: URL.createObjectURL(file), name: file.name, type });
  }
  renderPlaylist();
}

// ─── INTAKE FORM ──────────────────────────────────────────────────────────────
function bindIntakeForm() {
  const fields = ['pid','age','group','clinician','location','notes'];
  fields.forEach(f => {
    const el = document.getElementById(`meta-${f}`);
    if (el) el.addEventListener('input', () => { META[f] = el.value.trim(); });
  });
  const fileInput = document.getElementById('stimulus-files');
  if (fileInput) {
    fileInput.addEventListener('change', e => handleFileAdd(e.target.files));
  }
  const dropZone = document.getElementById('drop-zone');
  if (dropZone) {
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
      handleFileAdd(e.dataTransfer.files);
    });
  }
  const photoDurEl = document.getElementById('photo-dur');
  if (photoDurEl) {
    photoDurEl.value = photoDurSec;
    photoDurEl.addEventListener('change', () => { photoDurSec = Math.max(1, parseInt(photoDurEl.value)||5); });
  }
  const startBtn = document.getElementById('start-btn');
  if (startBtn) {
    startBtn.addEventListener('click', async () => {
      if (!META.pid) { alert('Please enter a Participant ID.'); return; }
      startBtn.disabled = true;
      startBtn.textContent = 'Loading…';
      try {
        await initMediaPipe();
        procRaf = requestAnimationFrame(processFrame);
        startCalibration();
      } catch(e) {
        alert('Failed to start: ' + e.message);
        startBtn.disabled = false;
        startBtn.textContent = 'Begin Session →';
      }
    });
  }
  const dlBtn = document.getElementById('download-csv');
  if (dlBtn) dlBtn.addEventListener('click', downloadCsv);
  const reBtn = document.getElementById('new-session');
  if (reBtn) reBtn.addEventListener('click', () => location.reload());
}

// ─── BOOTSTRAP ────────────────────────────────────────────────────────────────
(async function init() {
  showScreen('intake');
  pfCheckBrowser();
  bindIntakeForm();
  try {
    await initCamera();
  } catch(e) {
    console.warn('Camera init failed:', e);
  }
})();

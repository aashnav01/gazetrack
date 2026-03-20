/**
 * GazeTrack v16 – Immersive Child Calibration Edition
 * =====================================================
 * All fixes from v15 retained, plus SMI-matching fixes:
 *   [FIX-V1] CRITICAL: validation used poly() (8 elems) but model trained with
 *            polyX/polyY (7 elems) → wx[7]=undefined → NaN gaze everywhere.
 *            Fixed: runStarDot now uses polyX()/polyY() matching trainModel().
 *   [FIX-V2] Blink rows: SMI outputs 0.0000 for PoR/PupilSize, not '-'
 *            Pupil Diameter preserved during blinks (last known value held)
 *   [FIX-V3] Pupil size px: SMI formula = diam_mm × 3.73 (derived from SMI
 *            RED reference data at normal viewing distance ~730mm)
 *   [FIX-V4] Depth scale 40→33: gives Eye Z ≈730mm matching SMI reference mean
 *   [FIX-V5] Eye position X left/right swap fixed (camera mirror convention:
 *            right eye → negative X, left eye → positive X)
 *   [FIX-V6] Separator row added at trial start (Category Group=Information)
 *   [FIX-V7] Removed 'Port Status' and "groupe d'enfants" cols (not in SMI format)
 *   [FIX-V8] Unclassified '-' category for borderline velocity frames (0.85-1.0)
 *   [FIX-V9] Pupil position scaled to SMI camera space (1280×1024)
 */

console.log('%c GazeTrack v17.1 (Star Keeper — production)','background:#00e5b0;color:#000;font-weight:bold;font-size:14px');

import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs';

// ─── DEVICE ────────────────────────────────────────────────────────────────
const isMobile = /Android|iPad|iPhone|iPod|Mobile/i.test(navigator.userAgent)
  || (navigator.maxTouchPoints > 1 && window.innerWidth < 1200);
const MP_DELEGATE = isMobile ? 'CPU' : 'GPU';
const IS_TABLET = /iPad|Android(?!.*Mobile)/i.test(navigator.userAgent)
  || (window.innerWidth >= 768 && window.innerWidth <= 1280);

// ─── CONSTANTS ──────────────────────────────────────────────────────────────
const CALIB_GAZE_HOLD        = 700;
const CALIB_SAMPLE_MS        = 600;
const CALIB_TOTAL_PTS        = 5;
const CALIB_GAP_MS           = 500;
const MIN_SAMPLES            = 25;
const RIDGE_ALPHA            = 0.01;

const CALIB_BLINK_FORGIVE_MS = 600;
const CALIB_IRIS_BRIDGE_MS   = 150;
const CALIB_CORNER_BONUS     = 2.2;

// Fatigue

const LEFT_IRIS  = [468,469,470,471];
const RIGHT_IRIS = [473,474,475,476];
const L_CORNERS  = [33,133];
const R_CORNERS  = [362,263];

// Validation
const VAL_DWELL_MS    = 3000;
const VAL_GAP_MS      = 600;
const VAL_STAR_RADIUS = 48;
const VAL_SAMPLE_START= 0.60;
const VAL_INTRO_MS    = 2000;

const GOOD_STREAK_NEEDED = 8;
const BAD_STREAK_HIDE    = 12;

// [FIX-V3] SMI pupil size scale factor derived from SMI RED reference data
// PupilSize_px = PupilDiam_mm × 3.73 at ~730mm viewing distance
const SMI_PUPIL_PX_PER_MM = 3.73;

// [FIX-V9] SMI RED camera resolution for pupil position scaling
const SMI_CAM_W = 1280;
const SMI_CAM_H = 1024;

// ─── MONOTONIC TIMESTAMP ─────────────────────────────────────────────────────
let _mpLastTs = -1;
function mpNow() {
  const t = performance.now();
  _mpLastTs = t > _mpLastTs ? t : _mpLastTs + 0.001;
  return _mpLastTs;
}

// MongoDB
const MONGO_API_URL = 'https://gazetrack-api.onrender.com/api/sessions';

// ─── STATE ──────────────────────────────────────────────────────────────────
let phase = 'intake';
let faceLandmarker = null;
let camStream = null;
let sessionStream = null;
let gazeModel = null;
let calibSamples = [];
let affineBias = {dx:0, dy:0, sx:1, sy:1};

// Calibration state
let calibPoints = [];
let calibIdx = 0;
let calibRaf = null;
let calibState = 'idle';
let calibFailCount = 0;
let calibSkipActive = false;

let _calibCurrentGaze = null;
let _calibLastGaze    = null;
let _calibLastGazeTs  = 0;
let _calibLastFeat    = null;
let _calibHoldStart   = null;
let _calibHoldLostAt  = null;
let _calibSampling    = false;
let _calibSamplingStart = 0;
let _calibPointSamples  = [];
let _calibSparkled    = false;
let _calibSkipTimer   = null;
let _calibLoopT       = 0;

// Creature calibration state
let creatureEls    = [];
let doneCalibPoints = new Set();
let calibParticles  = [];
let calibFloaties   = [];

// Fatigue

// Validation state
let valPoints=[],valIdx=0,valSamples=[],valRaf=null,valStart=0;
const VAL_PARTICLES = [];

// Recording
let recordedFrames=[],totalF=0,trackedF=0;
let sessionStart=0,timerInt=null;
let csvData=null;
let videoBlob=null;
let META={pid:'',age:'',group:'',clinician:'',location:'',notes:'',stimulus:''};

// [FIX-V2] Track last known pupil diameter to preserve during blinks
let _lastKnownPupilDiamR = 3.5;
let _lastKnownPupilDiamL = 3.5;

// Saccade
let prevGaze=null,prevGazeTime=null;

// Face presence
let calibFacePresent=false;
let _earCalibSamples=[];
let _earThreshold=0.20;

// ─── DIAGNOSTICS STATE ───────────────────────────────────────────────────────
const DIAG = {
  feat_ranges: {
    pitch_min:  Infinity, pitch_max: -Infinity,
    irisY_min:  Infinity, irisY_max: -Infinity,
    irisX_min:  Infinity, irisX_max: -Infinity,
    iod_min:    Infinity, iod_max:   -Infinity,
    ear_min:    Infinity, ear_max:   -Infinity,
  },
  pt_samples: [0,0,0,0,0],
  val_predictions: [],
  bias: null,
  ear_threshold_used: 0.20,
  ear_open_mean: 0,
  iod_at_calib_start: 0,
  iod_mm_estimate: 0,
  sample_intervals: [],
  force_skipped: [],
  calib_blink_count: 0,
  calib_frame_count: 0,
  model_wx: null,
  model_wy: null,
  gaze_y_values: [],
  gaze_x_values: [],
  version: 'v17+normY',
};

// Position banner
let _goodFrameStreak=0,_badFrameStreak=0,_allclearShowing=false,_allclearHideTimer=null;

// Preview detector
let previewFl=null,previewRaf=null,lastPreviewTs=-1,_prevLastRun=0;
let _lastVT=-1,_lastVideoTime=-1;
let procRaf=null;

// Pre-flight
const pfState={cam:'scanning',face:'scanning',light:'scanning',browser:'scanning'};
let pfRaf=null,_pfSamples=[],_pfThrottle=0,_pfCanvas=null,_pfCtx=null;

// Welfare

// ─── DOM REFS ────────────────────────────────────────────────────────────────
const screens = {
  intake:   document.getElementById('s-intake'),
  loading:  document.getElementById('s-loading'),
  calib:    document.getElementById('s-calib'),
  stimulus: document.getElementById('s-stimulus'),
  done:     document.getElementById('s-done'),
};
const camPreview      = document.getElementById('cam-preview');
const camCanvas       = document.getElementById('cam-canvas');
const camCtx          = camCanvas.getContext('2d');
const calibCanvas     = document.getElementById('calib-canvas');
const calibCtx        = calibCanvas.getContext('2d');
const gazeCanvas      = document.getElementById('gaze-canvas');
const gazeCtx         = gazeCanvas.getContext('2d');
const webcam          = document.getElementById('webcam');
const stimVideo       = document.getElementById('stim-video');
const allclearBanner  = document.getElementById('position-allclear');

function showScreen(n) {
  Object.values(screens).forEach(s => s && s.classList.remove('active'));
  const scr = screens[n];
  if (scr) scr.classList.add('active');
}

[calibCanvas, gazeCanvas, camCanvas].forEach(canvas => {
  if (canvas) canvas.style.touchAction = 'none';
});
document.querySelectorAll('button, .clickable').forEach(el => {
  if (IS_TABLET && el) el.style.minHeight = '48px';
});

// ─── INJECT CALIBRATION CSS ─────────────────────────────────────────────────
(function injectCalibCSS() {
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
      50%      { box-shadow:0 0 0 18px rgba(255,200,0,0); }
    }
    @keyframes creature-bounce {
      0%,100% { transform:translate(-50%,-50%) translateY(0); }
      50%      { transform:translate(-50%,-50%) translateY(-8px); }
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
    }    .calib-creature-wrap {
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
      border-radius:20px; padding:10px 28px; z-index:60;
      color:#ddd5ff; font-size:15px; text-align:center;
      max-width:520px; min-width:280px; transition:opacity 0.4s;
      pointer-events:none; font-family:'Comic Sans MS','Chalkboard SE',cursive;
    }
    .calib-hud {
      position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
      display:flex; gap:10px; z-index:60; pointer-events:none;
    }
    .calib-hud-paw { font-size:20px; transition:transform 0.3s filter 0.3s; }
    .calib-hud-paw.done { transform:scale(1.4); filter:drop-shadow(0 0 6px #ffd700); }
    .calib-bg {
      position:absolute; inset:0;
      background:radial-gradient(ellipse at 50% 40%, #0d0d2b 0%, #060614 100%); z-index:0;
    }
    #calib-fx-canvas { position:absolute; inset:0; pointer-events:none; z-index:40; }
    #calib-star-canvas { position:absolute; inset:0; pointer-events:none; z-index:1; }
  `;
  document.head.appendChild(style);
})();


// ─── LIVE DIAGNOSTIC PANEL ──────────────────────────────────────────────────



// ─── AUDIO ───────────────────────────────────────────────────────────────────
let _audioCtx = null;
function getAudioCtx() {
  if (!_audioCtx) { try { _audioCtx = new AudioContext(); } catch(e) {} }
  return _audioCtx;
}
function playTone(freq, vol, duration, type='sine', delay=0) {
  const a = getAudioCtx(); if (!a) return;
  setTimeout(() => {
    const o = a.createOscillator(), g = a.createGain();
    o.connect(g); g.connect(a.destination);
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(0, a.currentTime);
    g.gain.linearRampToValueAtTime(vol, a.currentTime + 0.025);
    g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + duration);
    o.start(); o.stop(a.currentTime + duration);
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
  const o = a.createOscillator(), g = a.createGain();
  o.connect(g); g.connect(a.destination);
  o.type = 'sine';
  o.frequency.setValueAtTime(freq * 0.8, a.currentTime);
  o.frequency.linearRampToValueAtTime(freq, a.currentTime + 0.1);
  g.gain.setValueAtTime(0, a.currentTime);
  g.gain.linearRampToValueAtTime(vol, a.currentTime + 0.05);
  g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + duration);
  o.start(); o.stop(a.currentTime + duration);
}

// ─── CONFETTI ────────────────────────────────────────────────────────────────
function startConfettiLight() {
  const colors = ['#f4a261','#e9c46a','#90e0ef','#ffb3c1','#c77dff','#ff6b6b','#4cc9f0'];
  for (let i = 0; i < 8; i++) {
    setTimeout(() => {
      const conf = document.createElement('div');
      conf.style.cssText = `position:fixed;top:-10px;left:${Math.random()*100}vw;width:5px;height:5px;background:${colors[Math.floor(Math.random()*colors.length)]};border-radius:50%;z-index:9999;pointer-events:none;animation:fall-gentle ${2.5+Math.random()*1.5}s linear forwards;`;
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
      conf.style.cssText = `position:fixed;top:-12px;left:${Math.random()*100}vw;width:${size}px;height:${size}px;background:${colors[Math.floor(Math.random()*colors.length)]};border-radius:${Math.random()>0.5?'50%':'3px'};z-index:9999;pointer-events:none;animation:fall ${3.5+Math.random()*2.5}s linear forwards;`;
      document.body.appendChild(conf);
      setTimeout(() => conf.remove(), 6200);
    }, i * 18);
  }
  playSuccessChime();
}

// ─── POSITION ALL-CLEAR ──────────────────────────────────────────────────────
function updateAllClear(bright) {
  const posOk = pfState.face === 'pass', lightOk = pfState.light === 'pass';
  const allOk = posOk && lightOk;
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
  if (tagsEl) tagsEl.innerHTML = ['✓ Face visible · Good distance','✓ Lighting OK'].map(t => `<span class="allclear-tag">${t}</span>`).join('');
  const detailEl = document.getElementById('allclear-detail');
  if (detailEl) detailEl.textContent = `Brightness ${Math.round(bright)}/255`;
  if (allclearBanner) allclearBanner.classList.add('show');
  playChime(660, 0.08, 0.4);
}
function hideAllClear() {
  _allclearShowing = false;
  if (allclearBanner) allclearBanner.classList.remove('show');
}

// ─── PRE-FLIGHT ──────────────────────────────────────────────────────────────
function pfSet(id, state, msg) {
  pfState[id] = state;
  const card = document.getElementById('pfc-' + id);
  const stat = document.getElementById('pfc-' + id + '-s');
  if (card && stat) { card.className = 'pf-check ' + state; stat.innerHTML = msg; }
  pfUpdateScore();
}
function pfUpdateScore() {
  const vals = Object.values(pfState);
  const done = vals.filter(v => v !== 'scanning').length;
  const passes = vals.filter(v => v === 'pass').length;
  const warns  = vals.filter(v => v === 'warn').length;
  const total  = vals.length;
  const score  = Math.round(((passes + warns * 0.6) / total) * 100);
  const fill = document.getElementById('pf-score-fill');
  const pct  = document.getElementById('pf-score-pct');
  if (fill) {
    fill.style.width = score + '%';
    fill.style.background = score >= 75 ? 'var(--accent)' : score >= 50 ? 'var(--gold)' : 'var(--warn)';
  }
  if (pct) pct.textContent = done === total ? score + '%' : '...';
  const tips = [];
  if (pfState.light === 'fail')   tips.push('<strong>💡 Too dark:</strong> Add a front-facing lamp.');
  if (pfState.light === 'warn')   tips.push('<strong>💡 Lighting:</strong> Brighter room helps iris detection.');
  if (pfState.face  === 'fail')   tips.push('<strong>👤 No face:</strong> Make sure child is in frame.');
  if (pfState.browser === 'warn') tips.push('<strong>🌐 Browser:</strong> Use Chrome for best webcam performance.');
  const adv = document.getElementById('pf-advice');
  if (adv) { adv.innerHTML = tips.join('<br>'); adv.className = 'pf-advice' + (tips.length ? ' show' : ''); }
  const btn = document.getElementById('start-btn');
  if (btn && !btn.disabled) {
    const critFails = ['cam','face'].filter(k => pfState[k] === 'fail').length;
    if (critFails > 0)     { btn.textContent = '⚠ Proceed Anyway'; btn.style.background = 'linear-gradient(135deg,#ff9f43,#e17f20)'; }
    else if (done < total) { btn.textContent = 'Begin Session →'; btn.style.background = ''; }
    else if (score >= 75)  { btn.textContent = '✅ All Clear - Begin Session'; btn.style.background = ''; }
    else                   { btn.textContent = '⚠ Proceed with Warnings'; btn.style.background = 'linear-gradient(135deg,#ca8a04,#a16207)'; }
  }
}
function pfAnalyseFrame() {
  if (phase !== 'intake') return;
  const now = performance.now();
  if (now - _pfThrottle < 200) { pfRaf = requestAnimationFrame(pfAnalyseFrame); return; }
  _pfThrottle = now;
  if (!camPreview || camPreview.readyState < 2) { pfRaf = requestAnimationFrame(pfAnalyseFrame); return; }
  if (!_pfCanvas) {
    _pfCanvas = document.createElement('canvas'); _pfCanvas.width = 80; _pfCanvas.height = 60;
    _pfCtx = _pfCanvas.getContext('2d', {willReadFrequently: true});
  }
  try {
    _pfCtx.drawImage(camPreview, 0, 0, 80, 60);
    const d = _pfCtx.getImageData(0, 0, 80, 60).data;
    let sumR=0,sumG=0,sumB=0,n=0;
    for (let i=0;i<d.length;i+=4){sumR+=d[i];sumG+=d[i+1];sumB+=d[i+2];n++;}
    const brightness = (sumR+sumG+sumB)/(n*3);
    _pfSamples.push(brightness);
    if (_pfSamples.length > 3) _pfSamples.shift();
    const avgBright = _pfSamples.reduce((a,b)=>a+b,0)/_pfSamples.length;
    if (avgBright >= 60 && avgBright <= 220)  pfSet('light','pass',`✓ Good (${Math.round(avgBright)}/255)`);
    else if (avgBright < 40)                  pfSet('light','fail',`✗ Too dark (${Math.round(avgBright)}) - add light`);
    else if (avgBright < 60)                  pfSet('light','warn',`⚠ Dim (${Math.round(avgBright)}) - improve lighting`);
    else                                      pfSet('light','warn',`⚠ Bright (${Math.round(avgBright)}) - reduce backlight`);
    updateAllClear(avgBright);
  } catch(e) {}
  pfRaf = requestAnimationFrame(pfAnalyseFrame);
}
function pfCheckBrowser() {
  const ua = navigator.userAgent;
  const isChrome  = /Chrome/.test(ua) && !/Edg/.test(ua) && !/OPR/.test(ua);
  const isEdge    = /Edg/.test(ua);
  const isFirefox = /Firefox/.test(ua);
  if (isChrome)       pfSet('browser','pass','✓ Chrome - optimal');
  else if (isEdge)    pfSet('browser','pass','✓ Edge - good');
  else if (isFirefox) pfSet('browser','warn','⚠ Firefox - use Chrome for best results');
  else                pfSet('browser','warn','⚠ Use Chrome for best results');
}

// ─── CAMERA INIT ─────────────────────────────────────────────────────────────
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:640,max:1280},height:{ideal:480,max:720},facingMode:'user',frameRate:{ideal:60,min:30}},audio:false
    });
    camStream = stream;
    camPreview.srcObject = stream; camPreview.play();
    const camDot = document.getElementById('cam-dot');
    if (camDot) camDot.classList.add('ok');
    const camStatus = document.getElementById('cam-status-txt');
    if (camStatus) camStatus.textContent = 'Camera active';
    const chkCam = document.getElementById('chk-cam');
    if (chkCam) { chkCam.classList.add('ok'); chkCam.textContent = '✓ Cam'; }
    const t = stream.getVideoTracks()[0].getSettings();
    pfSet('cam','pass',`✓ ${t.width||640}x${t.height||480}`);
    checkStartBtn();
    pfRaf = requestAnimationFrame(pfAnalyseFrame);
    pfCheckBrowser();
    loadPreviewDetector();
  } catch(e) {
    const camStatus = document.getElementById('cam-status-txt');
    if (camStatus) camStatus.textContent = '✗ Camera error - allow access';
    pfSet('cam','fail','✗ Camera denied or not found');
    pfSet('face','fail','✗ No camera');
    pfSet('light','fail','✗ No camera');
  }
}

// ─── PREVIEW MESH ────────────────────────────────────────────────────────────
async function loadPreviewDetector() {
  try {
    const resolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
    previewFl = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:MP_DELEGATE},
      runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
    });
    previewLoop();
  } catch(e) {}
}
function previewLoop() {
  if (phase !== 'intake') return;
  const now = performance.now();
  if (now - _prevLastRun < 125) { previewRaf = requestAnimationFrame(previewLoop); return; }
  _prevLastRun = now;
  if (camPreview.readyState >= 2 && previewFl) {
    const rect = camCanvas.getBoundingClientRect();
    const dW = Math.round(rect.width)||640, dH = Math.round(rect.height)||480;
    if (camCanvas.width !== dW || camCanvas.height !== dH) { camCanvas.width=dW; camCanvas.height=dH; }
    const ts = mpNow();
    lastPreviewTs = ts;
    try {
      const res = previewFl.detectForVideo(camPreview, ts);
      const hasFace = !!(res.faceLandmarks && res.faceLandmarks.length > 0);
      camCtx.clearRect(0,0,camCanvas.width,camCanvas.height);
      if (hasFace) {
        drawPreviewMesh(res.faceLandmarks[0]);
        const lm = res.faceLandmarks[0];
        const hasIris = !!(lm[468] && lm[473]);
        const chkFace = document.getElementById('chk-face');
        const chkIris = document.getElementById('chk-iris');
        if (chkFace) { chkFace.classList.add('ok'); chkFace.textContent = '✓ Face'; }
        if (chkIris) { chkIris.classList.toggle('ok',hasIris); chkIris.textContent = hasIris?'✓ Iris':'👁 Iris'; }
        if (hasIris) {
          const iodNorm = Math.hypot(lm[473].x-lm[468].x,lm[473].y-lm[468].y);
          const faceCX  = (lm[33].x+lm[263].x)/2;
          const offCentre = faceCX < 0.25 || faceCX > 0.75;
          const lEAR = Math.hypot(lm[159].x-lm[145].x,lm[159].y-lm[145].y);
          const rEAR = Math.hypot(lm[386].x-lm[374].x,lm[386].y-lm[374].y);
          const earPx = (lEAR+rEAR)/2;
          const qPct  = (earPx/(iodNorm+1e-6))>0.08?95:75;
          const qFill  = document.getElementById('q-fill');
          const qPctEl = document.getElementById('q-pct');
          if (qFill) qFill.style.width = qPct+'%';
          if (qPctEl) qPctEl.textContent = qPct+'%';
          if (iodNorm > 0.22)       pfSet('face','warn',`⚠ Too close - move back ~15 cm`);
          else if (iodNorm >= 0.13) pfSet('face','pass',offCentre?`✓ Good distance - Move to centre`:`✓ Face visible - Good distance (~50-70 cm)`);
          else if (iodNorm >= 0.07) pfSet('face','warn',`⚠ Too far - move ${offCentre?'closer & to centre':'~20 cm closer'}`);
          else                      pfSet('face','warn',`⚠ Very far or face at edge - move much closer`);
        } else {
          const qFill = document.getElementById('q-fill'); if(qFill) qFill.style.width='40%';
          const qPctEl = document.getElementById('q-pct'); if(qPctEl) qPctEl.textContent='40%';
          pfSet('face','warn','⚠ Face detected but iris not visible - look at camera');
        }
      } else {
        const chkFace = document.getElementById('chk-face');
        const chkIris = document.getElementById('chk-iris');
        if (chkFace) { chkFace.classList.remove('ok'); chkFace.textContent = '👤 Face'; }
        if (chkIris) { chkIris.classList.remove('ok'); chkIris.textContent = '👁 Iris'; }
        const qFill = document.getElementById('q-fill'); if(qFill) qFill.style.width='0%';
        const qPctEl = document.getElementById('q-pct'); if(qPctEl) qPctEl.textContent=' - ';
        pfSet('face','fail','✗ No face detected - check camera position');
      }
    } catch(e) {}
  }
  previewRaf = requestAnimationFrame(previewLoop);
}
function drawPreviewMesh(lm) {
  const W = camCanvas.width, H = camCanvas.height;
  const fx = x => (1-x)*W, fy = y => y*H;
  [[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33],
   [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]].forEach(pts => {
    camCtx.beginPath();
    pts.forEach((idx,i)=>{const p=lm[idx];i===0?camCtx.moveTo(fx(p.x),fy(p.y)):camCtx.lineTo(fx(p.x),fy(p.y));});
    camCtx.strokeStyle='#00e5b0';camCtx.lineWidth=0.4;camCtx.globalAlpha=0.2;camCtx.stroke();camCtx.globalAlpha=1;
  });
  [[468,469],[473,474]].forEach(([c,e])=>{
    if(!lm[c]||!lm[e])return;
    const cx=fx(lm[c].x),cy=fy(lm[c].y),ex2=fx(lm[e].x),ey2=fy(lm[e].y);
    const r=Math.hypot(ex2-cx,ey2-cy)+0.5;
    camCtx.beginPath();camCtx.arc(cx,cy,r,0,Math.PI*2);
    camCtx.strokeStyle='#00e5b0';camCtx.lineWidth=0.6;camCtx.globalAlpha=0.3;camCtx.stroke();camCtx.globalAlpha=1;
  });
}

// ─── FORM ────────────────────────────────────────────────────────────────────
function checkStartBtn() {
  const pidOk   = document.getElementById('f-pid').value.trim().length > 0;
  const groupOk = document.querySelector('input[name="group"]:checked') !== null;
  const btn     = document.getElementById('start-btn');
  if (btn) btn.disabled = !(pidOk && groupOk);
  if (pidOk && groupOk) pfUpdateScore();
}
document.getElementById('f-pid').addEventListener('input', checkStartBtn);
document.querySelectorAll('input[name="group"]').forEach(r => r.addEventListener('change', checkStartBtn));

document.getElementById('video-drop').addEventListener('click', () => document.getElementById('video-input').click());
document.getElementById('video-input').addEventListener('change', e => {
  const f = e.target.files[0]; if (!f) return;
  videoBlob = URL.createObjectURL(f);
  META.stimulus = f.name;
  const hint = document.getElementById('video-hint');
  if (hint) hint.style.display = 'none';
  const existing = document.getElementById('video-drop').querySelector('.chosen');
  if (existing) existing.remove();
  document.getElementById('video-drop').insertAdjacentHTML('beforeend',`<div class="chosen">✓ ${f.name}</div>`);
});

document.getElementById('start-btn').addEventListener('click', () => {
  META.pid       = document.getElementById('f-pid').value.trim();
  META.age       = document.getElementById('f-age').value;
  META.group     = document.querySelector('input[name="group"]:checked')?.value || '';
  META.clinician = document.getElementById('f-clinician').value.trim();
  META.location  = document.getElementById('f-location').value.trim();
  META.notes     = document.getElementById('f-notes').value.trim();
  cancelAnimationFrame(previewRaf);
  cancelAnimationFrame(pfRaf);
  phase = 'loading';
  showScreen('loading');
  beginSession();
});

initCamera();

// ─── SESSION START ────────────────────────────────────────────────────────────
async function beginSession() {
  try {
    if (previewFl) {
      faceLandmarker = previewFl;
      const msgEl = document.getElementById('load-msg');
      if (msgEl) msgEl.textContent = 'Model ready - starting camera...';
    } else {
      const msgEl = document.getElementById('load-msg');
      if (msgEl) msgEl.textContent = 'Loading eye tracking model...';
      const resolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
      faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:MP_DELEGATE},
        runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
      });
    }
    if (camStream) { sessionStream = camStream; camStream = null; }
    else {
      sessionStream = await navigator.mediaDevices.getUserMedia({
        video:{width:{ideal:640},height:{ideal:480},facingMode:'user',frameRate:{ideal:60,min:30}},audio:false
      });
    }
    webcam.srcObject = sessionStream;
    await new Promise(r => { webcam.onloadedmetadata = () => { webcam.play(); r(); }; });
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
    phase = 'calib-ready';
    showScreen('calib');
    procRaf = requestAnimationFrame(processingLoop);
  } catch(err) {
    console.error(err);
    const msgEl = document.getElementById('load-msg');
    if (msgEl) msgEl.textContent = '✗ ' + (err.message || 'Startup error');
  }
}
function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  [calibCanvas, gazeCanvas].forEach(canvas => {
    if (!canvas) return;
    canvas.width  = Math.round(window.innerWidth  * dpr);
    canvas.height = Math.round(window.innerHeight * dpr);
    canvas.style.width  = window.innerWidth  + 'px';
    canvas.style.height = window.innerHeight + 'px';
  });
  if (calibCtx) calibCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────
function extractFeatures(lm, mat) {
  const avg = ids => {
    const s = {x:0,y:0,z:0};
    ids.forEach(i=>{s.x+=lm[i].x;s.y+=lm[i].y;s.z+=(lm[i].z||0);});
    return {x:s.x/ids.length,y:s.y/ids.length,z:s.z/ids.length};
  };
  const li=avg(LEFT_IRIS),ri=avg(RIGHT_IRIS);
  const lIn=lm[L_CORNERS[0]],lOut=lm[L_CORNERS[1]];
  const rIn=lm[R_CORNERS[0]],rOut=lm[R_CORNERS[1]];
  const lW=Math.hypot(lOut.x-lIn.x,lOut.y-lIn.y)+1e-6;
  const rW=Math.hypot(rOut.x-rIn.x,rOut.y-rIn.y)+1e-6;
  const lCx=(lIn.x+lOut.x)/2,rCx=(rIn.x+rOut.x)/2;
  const liX=(li.x-lCx)/lW,riX=(ri.x-rCx)/rW;
  let pitchDeg=0;
  if (mat?.data) {
    const m=mat.data;
    pitchDeg=Math.asin(Math.max(-1,Math.min(1,-m[6])))*180/Math.PI/30;
  }
  const nose=lm[1],fore=lm[10],chin=lm[152];
  const pitchZ=((nose.z||0)-((fore.z||0)+(chin.z||0))/2)*10;
  const vertMain=(Math.abs(pitchDeg)>0.001)?pitchDeg:pitchZ;
  const faceCY=(fore.y+chin.y)/2,faceH=Math.abs(chin.y-fore.y)+1e-6;
  const irisY=((li.y+ri.y)/2-faceCY)/faceH;
  const lEAR=Math.hypot(lm[159].x-lm[145].x,lm[159].y-lm[145].y)/lW;
  const rEAR=Math.hypot(lm[386].x-lm[374].x,lm[386].y-lm[374].y)/rW;
  const ear=(lEAR+rEAR)/2;
  const iod=Math.hypot(ri.x-li.x,ri.y-li.y);
  const feat=[liX,riX,vertMain,fore.y,irisY,(li.y+ri.y)/2,(liX+riX)/2,ear,iod];

  feat._3d = null;
  if (mat?.data) {
    const m = mat.data;

    // [FIX-V4] Scale 40→33: gives Eye Z ≈730mm matching SMI RED reference mean (was 600mm)
    const SCALE = 33;
    const txMm =  m[12] / SCALE;
    const tyMm = -m[13] / SCALE;
    const tzMm =  Math.abs(m[14] / SCALE);

    const iodMm = Math.max(45, Math.min(75, iod * 950));

    const gvx =  m[8];
    const gvy = -m[9];
    const gvz = -m[10];
    const gmag = Math.sqrt(gvx*gvx+gvy*gvy+gvz*gvz)||1;

    const pupR = Math.max(0, Math.min(5.0, 2.8 + (ear - 0.30) * 6.5));
    const pupL = Math.max(0, Math.min(5.0, 2.7 + (ear - 0.30) * 6.5));

    // [FIX-V4] Updated fallback depth to 730mm
    const approxDist = Math.max(300, Math.min(900, tzMm || 730));
    const vergenceDisp = Math.round((iodMm * 10) / approxDist);

    feat._3d = {
      // [FIX-V5] Right eye → negative X (left side of mirrored camera image)
      //          Left eye  → positive X (right side of mirrored camera image)
      eyeRX: txMm - iodMm/2,  eyeRY: tyMm,  eyeRZ: tzMm || 730,
      eyeLX: txMm + iodMm/2,  eyeLY: tyMm,  eyeLZ: tzMm || 730,
      gazeVX: gvx/gmag,  gazeVY: gvy/gmag,  gazeVZ: gvz/gmag,
      pupilDiamR: pupR,
      pupilDiamL: pupL,
      vergenceDisp: Math.max(5, vergenceDisp),
    };
  }
  return feat;
}

// ─── RIDGE REGRESSION ────────────────────────────────────────────────────────
function polyX(f) {
  // f[0]=leftIrisOffset, f[1]=rightIrisOffset, f[6]=mean horizontal offset
  // f[0]-f[1] captures left/right asymmetry — key signal for horizontal gaze
  // Removed: f[6]^2 (symmetric, can't distinguish left/right) and f[8]=IOD (not gaze)
  return [1, f[0], f[1], f[6], f[0]*f[1], f[0]-f[1], f[0]+f[1]];
}
function polyY(f) {
  // [FIX-N2] irisY*2: amplifies iris vertical signal as fallback when pitch is flat.
  // Children often don't tilt head between top/bottom targets — iris moves instead.
  return [1, f[2]*3, f[3], f[4]*2, f[2]*f[3], (f[2]*3)*(f[2]*3), f[5]];
}
function ridgeFit(X,y,alpha=RIDGE_ALPHA) {
  const n=X[0].length;
  const XtX=Array.from({length:n},()=>new Array(n).fill(0));
  const Xty=new Array(n).fill(0);
  for(let r=0;r<X.length;r++){
    for(let i=0;i<n;i++){
      Xty[i]+=X[r][i]*y[r];
      for(let j=0;j<n;j++) XtX[i][j]+=X[r][i]*X[r][j];
    }
  }
  for(let i=0;i<n;i++) XtX[i][i]+=alpha;
  const aug=XtX.map((row,i)=>[...row,Xty[i]]);
  for(let c=0;c<n;c++){
    let p=c;
    for(let r=c+1;r<n;r++) if(Math.abs(aug[r][c])>Math.abs(aug[p][c])) p=r;
    [aug[c],aug[p]]=[aug[p],aug[c]];
    const pv=aug[c][c]; if(Math.abs(pv)<1e-12) continue;
    for(let j=c;j<=n;j++) aug[c][j]/=pv;
    for(let r=0;r<n;r++){
      if(r!==c){const f=aug[r][c];for(let j=c;j<=n;j++) aug[r][j]-=f*aug[c][j];}
    }
  }
  return aug.map(r=>r[n]);
}
function trainModel(samples) {
  if (samples.length < MIN_SAMPLES) return null;
  const W = window.innerWidth, H = window.innerHeight;
  const Xx = samples.map(s=>polyX(s.feat));
  const Xy = samples.map(s=>polyY(s.feat));
  // [FIX-N1] Normalise targets to [0,1] before ridge regression.
  // Ridge alpha=0.01 is calibrated for unit-scale outputs. With raw pixel
  // targets (0–1280px) the regularisation suppresses weights needed to fit
  // large values, causing the Y model to collapse toward a constant.
  // Normalising fixes this: weights now predict fractions, then we scale back.
  const wx = ridgeFit(Xx, samples.map(s=>s.sx/W));
  const wy = ridgeFit(Xy, samples.map(s=>s.sy/H));
  const pitchVals = samples.map(s=>s.feat[2]);
  const irisYVals = samples.map(s=>s.feat[4]);
  const irisXVals = samples.map(s=>s.feat[6]);
  const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  const std  = arr => { const m=mean(arr); return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length); };
  const pStd=std(pitchVals), iYStd=std(irisYVals), iXStd=std(irisXVals);
  const sxVals=samples.map(s=>s.sx), syVals=samples.map(s=>s.sy);
  if(pStd<0.01) console.warn(`[DIAG] ⚠️ pitch std very low — Y model poor. Child needs to move head toward creatures.`);
  if(iXStd<0.02) console.warn(`[DIAG] ⚠️ irisX std very low — X model poor.`);
  return { wx, wy };
}
function predictGaze(feat, model) {
  if (!model) return null;
  const W = window.innerWidth, H = window.innerHeight;
  // [FIX-N1] Model predicts normalised [0,1], scale back to pixels
  const gx = polyX(feat).reduce((s,v,i)=>s+v*model.wx[i],0) * W;
  const gy = polyY(feat).reduce((s,v,i)=>s+v*model.wy[i],0) * H;
  const cx = affineBias.sx*gx + affineBias.dx;
  const cy = affineBias.sy*gy + affineBias.dy;
  return {x:Math.max(0,Math.min(W,cx)), y:Math.max(0,Math.min(H,cy))};
}

// ─── GAZE ESTIMATION FROM IRIS (pre-calibration) ────────────────────────────
let _calibTargetY = -1;

function estimateGazeFromIris(feat) {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const rawX = (-feat[6] * 4870) + (W * 0.54);
  const rawY = _calibTargetY >= 0 ? _calibTargetY : H * 0.5;
  return { x: Math.max(0, Math.min(W, rawX)), y: rawY };
}

function computeAffineCorrection(pairs) {
  function linfit(ps,ts,scaleMin,scaleMax){
    const n=ps.length,mp=ps.reduce((a,b)=>a+b,0)/n,mt=ts.reduce((a,b)=>a+b,0)/n;
    let num=0,den=0;
    for(let i=0;i<n;i++){num+=(ps[i]-mp)*(ts[i]-mt);den+=(ps[i]-mp)**2;}
    const s=den>1e-6?num/den:1, sc=Math.max(scaleMin,Math.min(scaleMax,s));
    return {s:sc,d:mt-sc*mp};
  }
  const fx=linfit(pairs.map(p=>p.px),pairs.map(p=>p.tx), 0.5, 1.6);
  const fy=linfit(pairs.map(p=>p.py),pairs.map(p=>p.ty), 0.5, 2.0); // [FIX-N3] wider clamp for sessions with poor pitch
  return {sx:fx.s,dx:fx.d,sy:fy.s,dy:fy.d};
}

// ══════════════════════════════════════════════════════════════════════════════
//  IMMERSIVE CHILD CALIBRATION — STAR KEEPER
// ══════════════════════════════════════════════════════════════════════════════

const CREATURE_DEFS = [
  {name:'Starby',   color:'#ffd700',glow:'#ffe066',bodyFn:'star',  hunger:'star seeds'},
  {name:'Bubbles',  color:'#4fc3f7',glow:'#80deea',bodyFn:'blob',  hunger:'sparkle drops'},
  {name:'Fizzwick', color:'#69f0ae',glow:'#b9f6ca',bodyFn:'puff',  hunger:'moon cookies'},
  {name:'Roary',    color:'#ff7043',glow:'#ffab91',bodyFn:'round', hunger:'fire berries'},
  {name:'Shimmer',  color:'#ce93d8',glow:'#f3e5f5',bodyFn:'floof', hunger:'dream petals'},
];

function buildCalibPoints() {
  const W=window.innerWidth, H=window.innerHeight;
  const px = Math.max(120, W * 0.25);
  const pyTop = Math.max(80, H * 0.14);
  const pyBot = Math.max(80, H * 0.80);
  return [
    {x:W/2,    y:H/2,    isCorner:false},
    {x:px,     y:pyTop,  isCorner:true},
    {x:W-px,   y:pyTop,  isCorner:true},
    {x:W-px,   y:pyBot,  isCorner:true},
    {x:px,     y:pyBot,  isCorner:true},
  ];
}

function getCalibGazeRadius(pt) {
  if (!pt.isCorner) return Math.min(window.innerWidth, window.innerHeight) * 0.42;
  return 220;
}

const CALIB_DWELL_REQUIRED_MS = 2500;
const CALIB_FORCE_SKIP_MS     = 8000;

function ensureDebugDot()  { /* production */ }
function removeDebugDot()  {
  ['calib-debug-dot','calib-debug-hud','calib-debug-rings'].forEach(id => {
    const el = document.getElementById(id); if (el) el.remove();
  });
}
function updateDebugOverlay() { /* production */ }

let calibStarCvs=null, calibStarCtx=null, calibStars=[];
let calibFxCvs=null, calibFxCtx=null;
let calibStoryBanner=null, calibHud=null;

function initCalibScene() {
  document.querySelectorAll('.calib-creature-wrap').forEach(el=>el.remove());
  if (calibStarCvs) { calibStarCvs.remove(); calibStarCvs=null; }
  if (calibFxCvs)   { calibFxCvs.remove();   calibFxCvs=null;   }
  if (calibStoryBanner) { calibStoryBanner.remove(); calibStoryBanner=null; }
  if (calibHud) { calibHud.remove(); calibHud=null; }
  document.querySelectorAll('.calib-bg').forEach(el=>el.remove());

  const calibScreen = document.getElementById('s-calib');
  if (!calibScreen) return;

  const bg = document.createElement('div');
  bg.className = 'calib-bg';
  calibScreen.appendChild(bg);

  const W=window.innerWidth, H=window.innerHeight;
  calibStarCvs = document.createElement('canvas');
  calibStarCvs.id = 'calib-star-canvas';
  calibStarCvs.width=W; calibStarCvs.height=H;
  calibStarCvs.style.cssText=`position:absolute;inset:0;pointer-events:none;z-index:1;`;
  calibScreen.appendChild(calibStarCvs);
  calibStarCtx = calibStarCvs.getContext('2d');
  calibStars = Array.from({length:180},()=>({x:Math.random()*W,y:Math.random()*H,r:0.4+Math.random()*1.5,tw:Math.random()*Math.PI*2}));

  calibFxCvs = document.createElement('canvas');
  calibFxCvs.id = 'calib-fx-canvas';
  calibFxCvs.width=W; calibFxCvs.height=H;
  calibFxCvs.style.cssText=`position:absolute;inset:0;pointer-events:none;z-index:40;`;
  calibScreen.appendChild(calibFxCvs);
  calibFxCtx = calibFxCvs.getContext('2d');

  calibStoryBanner = document.createElement('div');
  calibStoryBanner.className = 'calib-story-banner';
  calibStoryBanner.textContent = '🌟 Help the magical creatures eat their star seeds!';
  calibScreen.appendChild(calibStoryBanner);

  calibHud = document.createElement('div');
  calibHud.className = 'calib-hud';
  calibScreen.appendChild(calibHud);
}

function setCalibBanner(txt) {
  if (!calibStoryBanner) return;
  calibStoryBanner.style.opacity='0';
  setTimeout(()=>{if(calibStoryBanner){calibStoryBanner.textContent=txt;calibStoryBanner.style.opacity='1';}},200);
}

function updateCalibHUD() {
  if (!calibHud) return;
  calibHud.innerHTML='';
  for(let i=0;i<CALIB_TOTAL_PTS;i++){
    const paw=document.createElement('span');
    paw.className='calib-hud-paw'+(doneCalibPoints.has(i)?' done':'');
    paw.textContent=doneCalibPoints.has(i)?'🐾':(i===calibIdx?'⭐':'○');
    calibHud.appendChild(paw);
  }
}

// ─── CREATURE DRAWING ────────────────────────────────────────────────────────
function drawCreatureOnCanvas(ctx, def, S, t, gazeNear, holdPct, isHappy, isDone) {
  const cx=S/2, cy=S/2;
  ctx.clearRect(0,0,S,S);
  ctx.save();
  ctx.translate(cx, cy);

  if (gazeNear && !isDone) {
    const g=ctx.createRadialGradient(0,0,20,0,0,60);
    g.addColorStop(0,def.glow+'bb'); g.addColorStop(1,def.glow+'00');
    ctx.beginPath(); ctx.arc(0,0,60,0,Math.PI*2);
    ctx.fillStyle=g; ctx.fill();
  }
  if (!gazeNear && !isDone) {
    const pulse=0.5+0.5*Math.abs(Math.sin(t*3.2));
    ctx.beginPath(); ctx.arc(0,0,48,0,Math.PI*2);
    ctx.strokeStyle=def.glow+'55'; ctx.lineWidth=2+pulse*4;
    ctx.setLineDash([7,5]); ctx.stroke(); ctx.setLineDash([]);
  }

  const r=isDone?34:30;
  const fn=def.bodyFn;
  ctx.beginPath();
  if(fn==='star'){
    for(let i=0;i<10;i++){
      const a=(i*Math.PI/5)-Math.PI/2,rad=i%2===0?r:r*0.52;
      i===0?ctx.moveTo(Math.cos(a)*rad,Math.sin(a)*rad):ctx.lineTo(Math.cos(a)*rad,Math.sin(a)*rad);
    } ctx.closePath();
  } else if(fn==='blob'){
    for(let a=0;a<=Math.PI*2+0.1;a+=0.05){
      const bump=r+Math.sin(a*4+t*2)*3;
      a===0?ctx.moveTo(Math.cos(a)*bump,Math.sin(a)*bump):ctx.lineTo(Math.cos(a)*bump,Math.sin(a)*bump);
    }
  } else if(fn==='puff'){
    ctx.arc(0,0,r,0,Math.PI*2);
  } else if(fn==='round'){
    for(let i=0;i<16;i++){
      const a=(i*Math.PI/8)-Math.PI/2,spk=r+(i%2===0?8:0)+Math.sin(t*4+i)*2;
      i===0?ctx.moveTo(Math.cos(a)*spk,Math.sin(a)*spk):ctx.lineTo(Math.cos(a)*spk,Math.sin(a)*spk);
    } ctx.closePath();
  } else {
    for(let i=0;i<5;i++){
      const a=(i*Math.PI*2/5)-Math.PI/2,spk=r+Math.sin(t*2+i)*3;
      i===0?ctx.moveTo(Math.cos(a)*spk,Math.sin(a)*spk):ctx.lineTo(Math.cos(a)*spk,Math.sin(a)*spk);
    } ctx.closePath();
  }
  ctx.fillStyle=isDone?'#ffd700':def.color;
  ctx.shadowColor=gazeNear?def.glow:'transparent'; ctx.shadowBlur=gazeNear?22:0;
  ctx.fill();
  ctx.strokeStyle='rgba(255,255,255,0.25)'; ctx.lineWidth=1.5; ctx.stroke();
  ctx.shadowBlur=0;

  const eyeY=-r*0.15,eyeX=r*0.28;
  [-eyeX,eyeX].forEach(ex=>{
    ctx.beginPath();ctx.arc(ex,eyeY,r*0.17,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
    ctx.beginPath();ctx.arc(ex+r*0.05,eyeY,r*0.10,0,Math.PI*2);ctx.fillStyle='#1a1030';ctx.fill();
    ctx.beginPath();ctx.arc(ex+r*0.09,eyeY-r*0.06,r*0.04,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
  });

  if(isHappy||isDone){
    ctx.beginPath();ctx.arc(0,r*0.25,r*0.26,0.1*Math.PI,0.9*Math.PI);
    ctx.strokeStyle='#2d1a40';ctx.lineWidth=2.5;ctx.stroke();
    [-0.38,0.38].forEach(ex=>{
      ctx.beginPath();ctx.arc(ex*r,r*0.28,r*0.13,0,Math.PI*2);
      ctx.fillStyle='rgba(255,100,120,0.45)';ctx.fill();
    });
  } else {
    ctx.beginPath();ctx.arc(0,r*0.38,r*0.14,Math.PI*1.1,Math.PI*1.9);
    ctx.strokeStyle='#2d1a40';ctx.lineWidth=2;ctx.stroke();
  }

  if(holdPct>0&&holdPct<1){
    const rr=r+15;
    ctx.beginPath();
    ctx.arc(0,0,rr,-Math.PI/2,-Math.PI/2+holdPct*Math.PI*2);
    ctx.strokeStyle='#ffd700';ctx.lineWidth=5;ctx.lineCap='round';ctx.stroke();ctx.lineCap='butt';
  }

  if(isDone){
    for(let i=0;i<5;i++){
      const a=(i*Math.PI*2/5)-Math.PI/2+t*0.5;
      const rx=Math.cos(a)*(r+18),ry=Math.sin(a)*(r+18);
      drawMiniStar(ctx,rx,ry,5,'#ffd700');
    }
  }

  ctx.restore();
  ctx.save();
  ctx.font='bold 11px "Comic Sans MS",cursive';
  ctx.fillStyle=isDone?'#ffd700':def.glow;
  ctx.textAlign='center';
  ctx.fillText(def.name,S/2,S-5);
  ctx.restore();
}

function drawMiniStar(ctx,x,y,r,color){
  ctx.save();ctx.fillStyle=color;
  ctx.beginPath();
  for(let i=0;i<10;i++){
    const a=(i*Math.PI/5)-Math.PI/2,rad=i%2===0?r:r*0.42;
    i===0?ctx.moveTo(x+Math.cos(a)*rad,y+Math.sin(a)*rad):ctx.lineTo(x+Math.cos(a)*rad,y+Math.sin(a)*rad);
  }
  ctx.closePath();ctx.fill();ctx.restore();
}

// ─── CALIBRATION FX ──────────────────────────────────────────────────────────
function addCalibBurst(x,y,color,n=30){
  if(!calibFxCtx) return;
  for(let i=0;i<n;i++){
    const a=Math.random()*Math.PI*2,s=3+Math.random()*8;
    calibParticles.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s-2,life:1,size:3+Math.random()*6,color,decay:0.02+Math.random()*0.01});
  }
}
function addCalibHearts(x,y,n=7){
  for(let i=0;i<n;i++){
    calibFloaties.push({x:x+(-40+Math.random()*80),y,vy:-1.2-Math.random(),vx:-0.4+Math.random()*0.8,life:1,size:10+Math.random()*8,decay:0.013,type:'heart'});
  }
}
function addCalibStars(x,y,n=10){
  for(let i=0;i<n;i++){
    calibFloaties.push({x:x+(-60+Math.random()*120),y:y+(-30+Math.random()*30),vy:-1.8-Math.random()*2,vx:-0.8+Math.random()*1.6,life:1,size:8+Math.random()*12,decay:0.016,type:'star'});
  }
}
function drawHeart(ctx,x,y,size,alpha){
  ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle='#ff69b4';
  ctx.beginPath();
  ctx.moveTo(x,y+size*0.3);
  ctx.bezierCurveTo(x,y,x-size,y,x-size,y+size*0.35);
  ctx.bezierCurveTo(x-size,y+size*0.7,x,y+size,x,y+size*1.1);
  ctx.bezierCurveTo(x,y+size,x+size,y+size*0.7,x+size,y+size*0.35);
  ctx.bezierCurveTo(x+size,y,x,y,x,y+size*0.3);
  ctx.fill();ctx.restore();
}
function updateCalibFX(){
  if(!calibFxCtx) return;
  const W=calibFxCvs.width,H=calibFxCvs.height;
  calibFxCtx.clearRect(0,0,W,H);
  for(let i=calibParticles.length-1;i>=0;i--){
    const p=calibParticles[i];
    p.x+=p.vx;p.y+=p.vy;p.vy+=0.14;p.life-=p.decay;
    if(p.life<=0){calibParticles.splice(i,1);continue;}
    calibFxCtx.save();calibFxCtx.globalAlpha=p.life;calibFxCtx.fillStyle=p.color;
    calibFxCtx.beginPath();calibFxCtx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);calibFxCtx.fill();
    calibFxCtx.restore();
  }
  for(let i=calibFloaties.length-1;i>=0;i--){
    const f=calibFloaties[i];
    f.x+=f.vx;f.y+=f.vy;f.life-=f.decay;
    if(f.life<=0){calibFloaties.splice(i,1);continue;}
    if(f.type==='heart') drawHeart(calibFxCtx,f.x,f.y,f.size*f.life,f.life*0.9);
    else drawMiniStar(calibFxCtx,f.x,f.y,f.size*f.life*0.5,'#ffd700');
  }
}
function drawCalibStars(t){
  if(!calibStarCtx) return;
  const W=calibStarCvs.width,H=calibStarCvs.height;
  calibStarCtx.clearRect(0,0,W,H);
  calibStars.forEach(s=>{
    s.tw+=0.04;
    const a=0.3+0.7*Math.abs(Math.sin(s.tw+s.x*0.01));
    calibStarCtx.beginPath();calibStarCtx.arc(s.x,s.y,s.r,0,Math.PI*2);
    calibStarCtx.fillStyle=`rgba(200,190,255,${a})`;calibStarCtx.fill();
  });
}

function updateSparkles(ctx){
  for(let i=VAL_PARTICLES.length-1;i>=0;i--){
    const p=VAL_PARTICLES[i];
    p.x+=p.vx;p.y+=p.vy;p.vy+=0.12;p.life-=0.035;
    if(p.life<=0){VAL_PARTICLES.splice(i,1);continue;}
    ctx.save();ctx.globalAlpha=p.life;ctx.fillStyle=`hsl(${p.hue},100%,65%)`;
    ctx.beginPath();ctx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);ctx.fill();ctx.restore();
  }
}
function drawStar(ctx,x,y,radius,twinklePhase,entranceProgress){
  const r=radius*entranceProgress;
  const innerR=r*0.4,points=5;
  const twinkle=1+Math.sin(twinklePhase*8)*0.08*entranceProgress;
  const rr=r*twinkle;
  const glowSize=rr*(1.8+Math.sin(twinklePhase*6)*0.2);
  const grad=ctx.createRadialGradient(x,y,0,x,y,glowSize);
  grad.addColorStop(0,`rgba(255,220,50,${0.35*entranceProgress})`);
  grad.addColorStop(1,'rgba(255,220,50,0)');
  ctx.beginPath();ctx.arc(x,y,glowSize,0,Math.PI*2);ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();
  for(let i=0;i<points*2;i++){
    const angle=(i*Math.PI/points)-Math.PI/2;
    const rad=i%2===0?rr:innerR;
    i===0?ctx.moveTo(x+Math.cos(angle)*rad,y+Math.sin(angle)*rad):ctx.lineTo(x+Math.cos(angle)*rad,y+Math.sin(angle)*rad);
  }
  ctx.closePath();
  const sg=ctx.createRadialGradient(x,y-rr*0.2,0,x,y,rr);
  sg.addColorStop(0,'#fff9c4');sg.addColorStop(0.4,'#ffd700');sg.addColorStop(1,'#ff9f00');
  ctx.fillStyle=sg;ctx.shadowColor='#ffd700';ctx.shadowBlur=20*entranceProgress;ctx.fill();ctx.shadowBlur=0;
  if(entranceProgress>0.8){
    ctx.beginPath();ctx.arc(x-rr*0.2,y-rr*0.25,rr*0.18,0,Math.PI*2);
    ctx.fillStyle=`rgba(255,255,255,${0.6*(entranceProgress-0.8)*5})`;ctx.fill();
  }
}

let _valLastDetectTs = -1;

function startValidation(){
  valPoints=[];
  const W=window.innerWidth,H=window.innerHeight;
  const safeVX=Math.max(80,W*.16),safeVY=Math.max(80,H*.16);
  valPoints=[{x:W/2,y:H/2},{x:safeVX,y:safeVY},{x:W-safeVX,y:H-safeVY},{x:W/2,y:H-safeVY}];
  valIdx=0;valSamples=[];VAL_PARTICLES.length=0;
  _lastVideoTime=-1;
  _valLastDetectTs=-1;
  prevGaze=null;prevGazeTime=null;
  const overlay=document.getElementById('val-overlay');
  if(overlay) overlay.style.display='block';
  const instr=document.getElementById('val-instruction');
  if(instr) instr.style.opacity='1';
  const badge=document.getElementById('val-badge');
  if(badge) badge.style.display='none';
  const tot=document.getElementById('val-badge-tot');
  if(tot) tot.textContent=valPoints.length;
  playChime(528,0.1,0.6);
  setTimeout(()=>{
    if(instr) instr.style.opacity='0';
    setTimeout(()=>{
      if(instr) instr.style.display='none';
      if(badge) badge.style.display='block';
      runStarDot();
    },400);
  },VAL_INTRO_MS);
}

function runStarDot(){
  if(valIdx>=valPoints.length){finishValidation();return;}
  const pt=valPoints[valIdx];
  const valCanvas=document.getElementById('val-canvas');
  if(!valCanvas) return;

  const dpr=window.devicePixelRatio||1;
  valCanvas.width=Math.round(window.innerWidth*dpr);
  valCanvas.height=Math.round(window.innerHeight*dpr);
  valCanvas.style.width=window.innerWidth+'px';
  valCanvas.style.height=window.innerHeight+'px';
  const vCtx=valCanvas.getContext('2d');
  vCtx.setTransform(dpr,0,0,dpr,0,0);

  const numEl=document.getElementById('val-badge-num');
  if(numEl) numEl.textContent=valIdx+1;
  const notes=[523,659,784,880];
  playChime(notes[valIdx%notes.length],0.12,0.5);
  const collected=[];
  const ENTRANCE_MS=450;
  valStart=performance.now();
  let sparkled=false,inGap=true;
  const gapEnd=valStart+VAL_GAP_MS;

  function frame(){
    const now=performance.now();
    vCtx.clearRect(0,0,valCanvas.width,valCanvas.height);
    if(inGap){
      updateSparkles(vCtx);
      if(now>=gapEnd) inGap=false;
      valRaf=requestAnimationFrame(frame);return;
    }
    const starElapsed=now-gapEnd;
    const starProgress=Math.min(starElapsed/VAL_DWELL_MS,1);
    const entPct=Math.min(starElapsed/ENTRANCE_MS,1);
    let entrance;
    if(entPct<1){const t=entPct;entrance=1-Math.pow(1-t,3)*Math.cos(t*Math.PI*2.5);entrance=Math.min(entrance,1.12);}
    else{entrance=1;}
    if(!sparkled&&entPct>=0.9){spawnSparkles(vCtx,pt.x,pt.y);sparkled=true;}
    updateSparkles(vCtx);
    drawStar(vCtx,pt.x,pt.y,VAL_STAR_RADIUS,starElapsed*0.001,Math.min(entrance,1));

    if(starProgress>=VAL_SAMPLE_START&&gazeModel){
      const nowTs=performance.now();
      if(webcam.readyState>=2&&faceLandmarker&&nowTs-_valLastDetectTs>=33){
        _valLastDetectTs=nowTs;
        try{
          const res=faceLandmarker.detectForVideo(webcam,mpNow());
          if(res.faceLandmarks&&res.faceLandmarks.length>0){
            const lm=res.faceLandmarks[0];
            const mat=(res.facialTransformationMatrixes?.length>0)?res.facialTransformationMatrixes[0]:null;
            const feat=extractFeatures(lm,mat);
            if(feat[7]>=0.20){
              // [FIX-V1] Use polyX/polyY (7 elems) matching trainModel
              // [FIX-N1] Model outputs normalised [0,1] — scale back to pixels for affine fitting
              const _W = window.innerWidth, _H = window.innerHeight;
              collected.push({
                px: polyX(feat).reduce((s,v,i)=>s+v*gazeModel.wx[i],0) * _W,
                py: polyY(feat).reduce((s,v,i)=>s+v*gazeModel.wy[i],0) * _H
              });
            }
          }
        }catch(e){ console.warn('[Val] detect error:',e); }
      }
    }
    if(starProgress<1){valRaf=requestAnimationFrame(frame);}
    else{
      if(collected.length>=3){
        const mxs=collected.map(p=>p.px).sort((a,b)=>a-b);
        const mys=collected.map(p=>p.py).sort((a,b)=>a-b);
        const mid=Math.floor(mxs.length/2);
        valSamples.push({px:mxs[mid],py:mys[mid],tx:pt.x,ty:pt.y});
      }
      valIdx++;runStarDot();
    }
  }
  valRaf=requestAnimationFrame(frame);
}

function finishValidation(){
  cancelAnimationFrame(valRaf);

  if(valSamples.length>0 && gazeModel){
    valSamples.forEach((vs,i)=>{
      const errX=Math.round(vs.px-vs.tx),errY=Math.round(vs.py-vs.ty);
      const dist=Math.round(Math.hypot(errX,errY));
    });
  }
  const overlay=document.getElementById('val-overlay');
  if(overlay) overlay.style.display='none';

  if(valSamples.length===0){
    console.warn('[Val] No validation samples — skipping affine correction');
    affineBias={dx:0,dy:0,sx:1,sy:1};
    phase='stimulus'; showScreen('stimulus');
    const hChild=document.getElementById('h-child');if(hChild) hChild.textContent=META.pid;
    const hGroup=document.getElementById('h-group');if(hGroup) hGroup.textContent=META.group;
    startRecording(); return;
  }

  if(valSamples.length>=2){
    affineBias=computeAffineCorrection(valSamples);
    if(affineBias.sy>=1.59) console.warn(`[DIAG] ⚠️ sy hit Y clamp (1.6) — pitch variance too low. Run fullscreen.`);
    else if(affineBias.sy>=1.29) console.warn(`[DIAG] ⚠️ sy=${affineBias.sy.toFixed(2)} — moderate Y correction applied.`);
    if(Math.abs(affineBias.dy)>150) console.warn(`[DIAG] ⚠️ dy=${affineBias.dy.toFixed(0)}px — check fullscreen + distance.`);

    const dxBad = Math.abs(affineBias.dx)>500;
    const sxBad = affineBias.sx>2.4 || affineBias.sx<0.3;
    if(dxBad||sxBad){
      console.warn(`[Val] Catastrophic calibration — retry`);
      affineBias={dx:0,dy:0,sx:1,sy:1};
      calibSamples=[];gazeModel=null;
      const card=document.getElementById('calib-card');
      const h2=card?.querySelector('h2'); const p=card?.querySelector('p');
      if(h2) h2.textContent='🐾 Let\'s try again!';
      if(p) p.innerHTML='Validation showed the tracker was very inaccurate.<br>Please recalibrate carefully.<br><br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, look directly at each creature.';
      const startBtn=document.getElementById('calib-start-btn');
      if(startBtn) startBtn.textContent='🐾 Try Again!';
      showScreen('calib');
      const overlayCalib=document.getElementById('calib-overlay');
      if(overlayCalib) overlayCalib.style.display='flex';
      phase='calib-ready'; return;
    }
    if(Math.abs(affineBias.dx)>200||affineBias.sx>1.4||affineBias.sx<0.65){
      console.warn(`[Val] Moderate offset — proceeding with correction`);
    }
  }

  phase='stimulus'; showScreen('stimulus');
  const hChild=document.getElementById('h-child');if(hChild) hChild.textContent=META.pid;
  const hGroup=document.getElementById('h-group');if(hGroup) hGroup.textContent=META.group;
  startRecording();
}

// ─── SACCADE CLASSIFICATION ──────────────────────────────────────────────────
function classifyGaze(gaze,currentTime){
  if(!prevGaze||!prevGazeTime){prevGaze=gaze;prevGazeTime=currentTime;return'Fixation';}
  const dt=currentTime-prevGazeTime;
  if(dt===0) return'Fixation';
  const dist=Math.hypot(gaze.x-prevGaze.x,gaze.y-prevGaze.y);
  const vel=dist/dt;
  // [FIX-V8] SMI-matched: borderline velocity (0.85-1.0 px/ms) → unclassified '-'
  // SMI reference: 10.7% saccade, 72% fixation, rest unclassified/blink
  const cat = vel > 1.0 ? 'Saccade' : (vel > 0.85 ? '-' : 'Fixation');
  prevGaze=gaze;prevGazeTime=currentTime;
  return cat;
}


// ─── RECORDING ───────────────────────────────────────────────────────────────
function startRecording(){
  sessionStart=Date.now();recordedFrames=[];totalF=0;trackedF=0;
  _lastVT=-1; // reset frame throttle so first stimulus frame isn't skipped
  _lastKnownPupilDiamR=3.5;_lastKnownPupilDiamL=3.5;
  prevGaze=null;prevGazeTime=null; // reset saccade state
  timerInt=setInterval(()=>{
    const s=Math.floor((Date.now()-sessionStart)/1000);
    const timerEl=document.getElementById('h-timer');
    if(timerEl) timerEl.textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  },500);
  if(videoBlob){
    stimVideo.src=videoBlob;stimVideo.muted=true;
    stimVideo.play().catch(()=>showNoVideo());
    stimVideo.onerror=showNoVideo;
    stimVideo.onended=()=>endSession();
    const soundBtn=document.getElementById('sound-btn');
    if(soundBtn) soundBtn.style.display='block';
  } else {showNoVideo();}
}
function showNoVideo(){
  const noVideo=document.getElementById('no-video');
  if(noVideo) noVideo.style.display='flex';
  const soundBtn=document.getElementById('sound-btn');
  if(soundBtn) soundBtn.style.display='none';
}
document.getElementById('stim-file-input')?.addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  META.stimulus=f.name;
  stimVideo.src=URL.createObjectURL(f);stimVideo.muted=true;
  const noVideo=document.getElementById('no-video');
  if(noVideo) noVideo.style.display='none';
  stimVideo.play().catch(()=>{});
  stimVideo.onended=()=>endSession();
  const soundBtn=document.getElementById('sound-btn');
  if(soundBtn) soundBtn.style.display='block';
});
document.getElementById('sound-btn')?.addEventListener('click',()=>{
  stimVideo.muted=false;stimVideo.play().catch(()=>{});
  const soundBtn=document.getElementById('sound-btn');
  if(soundBtn) soundBtn.style.display='none';
});

// ─── MAIN PROCESSING LOOP ────────────────────────────────────────────────────
let _calibProcTs = -1;

function processingLoop(){
  if(phase==='done') return;

  if(phase==='calib-run'){
    if(webcam.readyState>=2&&faceLandmarker){
      const now=performance.now();
      if(now-_calibProcTs<33){procRaf=requestAnimationFrame(processingLoop);return;}
      _calibProcTs=now;
      try{
        const res=faceLandmarker.detectForVideo(webcam,mpNow());
        const hasFace=!!(res.faceLandmarks&&res.faceLandmarks.length>0);
        calibFacePresent=hasFace;
        if(hasFace){
          const lm=res.faceLandmarks[0];
          const mat=(res.facialTransformationMatrixes?.length>0)?res.facialTransformationMatrixes[0]:null;
          const feat=extractFeatures(lm,mat);
          _calibLastFeat=feat;
          const ear=feat[7];

          if(_earCalibSamples.length<60&&ear>0.15){
            _earCalibSamples.push(ear);
            if(_earCalibSamples.length===60){
              const sorted=[..._earCalibSamples].sort((a,b)=>a-b);
              _earThreshold=Math.max(0.12,Math.min(0.26,sorted[36]*0.60));
            }
          }
          const isBlink=ear<_earThreshold;
          if(!isBlink){
            const rawGaze=gazeModel?predictGaze(feat,gazeModel):estimateGazeFromIris(feat);
            _calibCurrentGaze=rawGaze; _calibLastGaze=rawGaze; _calibLastGazeTs=performance.now();
          } else {
            const bridgeAge=performance.now()-_calibLastGazeTs;
            _calibCurrentGaze=bridgeAge<CALIB_IRIS_BRIDGE_MS?_calibLastGaze:null;
          }

          if((calibState==='sampling'||calibState==='holding')&&!isBlink){
            const pt=calibPoints[calibIdx];
            if(pt){
              calibSamples.push({feat,sx:pt.x,sy:pt.y});
              _calibPointSamples.push({feat,sx:pt.x,sy:pt.y});            }
          }
        } else {
          const bridgeAge=performance.now()-_calibLastGazeTs;
          _calibCurrentGaze=bridgeAge<CALIB_IRIS_BRIDGE_MS?_calibLastGaze:null;
        }
      }catch(e){ console.error('[Calib] detectForVideo error:',e); }
    }
    procRaf=requestAnimationFrame(processingLoop);
    return;
  }

  if(phase==='validation'){procRaf=requestAnimationFrame(processingLoop);return;}

  // ── STIMULUS ──
  if(webcam.readyState>=2&&faceLandmarker){
    const _stimNow = performance.now();
    if(_stimNow - _lastVT < 16.5){procRaf=requestAnimationFrame(processingLoop);return;}
    _lastVT = _stimNow;
    const res=faceLandmarker.detectForVideo(webcam,mpNow());
    const hasFace=!!(res.faceLandmarks&&res.faceLandmarks.length>0);
    const mat=(res.facialTransformationMatrixes?.length>0)?res.facialTransformationMatrixes[0]:null;
    calibFacePresent=hasFace;
    if(phase==='stimulus'){
      const hFace=document.getElementById('h-face');
      if(hFace){hFace.textContent=hasFace?'Yes':'No';hFace.className=hasFace?'ok':'bad';}
      const stFace=document.getElementById('st-face');
      if(stFace){stFace.textContent=hasFace?'Yes':'No';stFace.className='sv '+(hasFace?'ok':'bad');}
    }
    if(hasFace){
      const lm=res.faceLandmarks[0];
      const feat=extractFeatures(lm,mat);
      const ear=feat[7];
      const isBlink=ear<_earThreshold;
      const nowMs=Date.now()-sessionStart;
      const headPos={x:(lm[33].x+lm[263].x)/2,y:(lm[33].y+lm[263].y)/2};

      // [FIX-V9] Pupil position scaled to SMI camera space (1280×1024)
      let leftPupilX=null,leftPupilY=null,rightPupilX=null,rightPupilY=null;
      try{
        let sumLx=0,sumLy=0,sumRx=0,sumRy=0;
        LEFT_IRIS.forEach(i=>{sumLx+=lm[i].x;sumLy+=lm[i].y;});
        RIGHT_IRIS.forEach(i=>{sumRx+=lm[i].x;sumRy+=lm[i].y;});
        leftPupilX  = (sumLx/LEFT_IRIS.length)  * SMI_CAM_W;
        leftPupilY  = (sumLy/LEFT_IRIS.length)  * SMI_CAM_H;
        rightPupilX = (sumRx/RIGHT_IRIS.length) * SMI_CAM_W;
        rightPupilY = (sumRy/RIGHT_IRIS.length) * SMI_CAM_H;
      }catch(e){}

      if(phase==='stimulus'){
        totalF++;
        let gaze=null;
        if(!isBlink){
          if(gazeModel&&!calibSkipActive) gaze=predictGaze(feat,gazeModel);
        }

        const dispX = feat._3d ? feat._3d.vergenceDisp : 11.5;
        const dispY = dispX * 0.2;
        const gazeXL = gaze ? gaze.x + dispX : NaN;
        const gazeYL = gaze ? gaze.y + dispY : NaN;

        // [FIX-V3] Pupil size: SMI formula = diam_mm × 3.73
        const pupilDiamR = feat._3d ? feat._3d.pupilDiamR : Math.max(0,(feat[7]-0.20)*6.5+2.8);
        const pupilDiamL = feat._3d ? feat._3d.pupilDiamL : pupilDiamR - 0.08;

        // [FIX-V2] Track last known pupil diameter (preserved during blinks)
        if(!isBlink && pupilDiamR>0) _lastKnownPupilDiamR = pupilDiamR;
        if(!isBlink && pupilDiamL>0) _lastKnownPupilDiamL = pupilDiamL;

        // [FIX-V3] SMI-matched: PupilSize_px = diam_mm × 3.73
        const pupilSizePxR = isBlink ? 0 : Math.max(0, pupilDiamR * SMI_PUPIL_PX_PER_MM);
        const pupilSizePxL = isBlink ? 0 : Math.max(0, pupilDiamL * SMI_PUPIL_PX_PER_MM);

        const catGroup = 'Eye';
        const catRight = isBlink ? 'Blink' : (gaze ? classifyGaze(gaze,nowMs) : 'Fixation');
        const catLeft  = catRight;

        const frameData={
          t:nowMs, tracked:gaze?1:0,
          gazeX:gaze?.x??NaN, gazeY:gaze?.y??NaN,
          gazeXL, gazeYL,
          leftPupilX, leftPupilY, rightPupilX, rightPupilY,
          pupilSizePxR, pupilSizePxL,
          // [FIX-V2] Store last-known diameter for blink rows
          pupilDiamR: isBlink ? _lastKnownPupilDiamR : pupilDiamR,
          pupilDiamL: isBlink ? _lastKnownPupilDiamL : Math.max(0,pupilDiamL),
          eyeRX: feat._3d?.eyeRX ?? null,
          eyeRY: feat._3d?.eyeRY ?? null,
          eyeRZ: feat._3d?.eyeRZ ?? null,
          eyeLX: feat._3d?.eyeLX ?? null,
          eyeLY: feat._3d?.eyeLY ?? null,
          eyeLZ: feat._3d?.eyeLZ ?? null,
          gazeVX: feat._3d?.gazeVX ?? feat[0],
          gazeVY: feat._3d?.gazeVY ?? feat[1],
          gazeVZ: feat._3d?.gazeVZ ?? feat[2],
          catGroup, catRight, catLeft,
          category: catRight,
          isBlink,
          feat,
        };
        if(gaze){
          trackedF++;
          gazeCtx.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
          const stGaze=document.getElementById('st-gaze');
          if(stGaze){stGaze.textContent='Tracking';stGaze.className='sv ok';}
        } else {
          if(isBlink){prevGaze=null;prevGazeTime=null;}
          gazeCtx.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
          const stGaze=document.getElementById('st-gaze');
          if(stGaze){stGaze.textContent=isBlink?'Blink':'Lost';stGaze.className='sv bad';}
        }
        recordedFrames.push(frameData);
        if(recordedFrames.length>1){
          const prevT=recordedFrames[recordedFrames.length-2].t;
          const dt=frameData.t-prevT;        }        const stFrames=document.getElementById('st-frames');
        if(stFrames) stFrames.textContent=recordedFrames.length;
        const stTrack=document.getElementById('st-track');
        if(stTrack) stTrack.textContent=Math.round(trackedF/totalF*100)+'%';
        if(recordedFrames.length>10){
          const ys=recordedFrames.filter(f=>f.tracked).map(f=>f.gazeY);
          if(ys.length>1){
            const my=ys.reduce((a,b)=>a+b,0)/ys.length;
            const sy=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);
            const el=document.getElementById('st-ystd');
            if(el){el.textContent=sy.toFixed(0)+'px';el.className='sv '+(sy>30?'ok':'bad');}
          }
        }
      }
    } else if(phase==='stimulus'){
      totalF++;
      const nowMs=Date.now()-sessionStart;
      recordedFrames.push({
        t:nowMs,tracked:0,gazeX:NaN,gazeY:NaN,gazeXL:NaN,gazeYL:NaN,
        leftPupilX:null,leftPupilY:null,rightPupilX:null,rightPupilY:null,
        pupilSizePxR:0,pupilSizePxL:0,
        // [FIX-V2] Use last-known diameter for face-off frames too
        pupilDiamR:_lastKnownPupilDiamR,pupilDiamL:_lastKnownPupilDiamL,
        eyeRX:null,eyeRY:null,eyeRZ:null,eyeLX:null,eyeLY:null,eyeLZ:null,
        gazeVX:null,gazeVY:null,gazeVZ:null,
        catGroup:'Eye',catRight:'Blink',catLeft:'Blink',
        category:'Blink',isBlink:true,feat:null
      });
      const stTrack=document.getElementById('st-track');
      if(stTrack&&totalF>0) stTrack.textContent=Math.round(trackedF/totalF*100)+'%';
    }
  }
  procRaf=requestAnimationFrame(processingLoop);
}

// ─── CSV (SMI RED FORMAT — v16 SMI-matched) ──────────────────────────────────
// [FIX-V7] Removed 'Port Status' and "groupe d'enfants" — not in SMI format
// Column count now matches SMI RED exactly (57 columns)
const CSV_HDR=[
  'Unnamed: 0','RecordingTime [ms]','Time of Day [h:m:s:ms]',
  'Trial','Stimulus','Export Start Trial Time [ms]','Export End Trial Time [ms]',
  'Participant','Color','Tracking Ratio [%]',
  'Category Group','Category Right','Category Left',
  'Index Right','Index Left',
  'Pupil Size Right X [px]','Pupil Size Right Y [px]','Pupil Diameter Right [mm]',
  'Pupil Size Left X [px]','Pupil Size Left Y [px]','Pupil Diameter Left [mm]',
  'Point of Regard Right X [px]','Point of Regard Right Y [px]',
  'Point of Regard Left X [px]','Point of Regard Left Y [px]',
  'AOI Name Right','AOI Name Left',
  'Gaze Vector Right X','Gaze Vector Right Y','Gaze Vector Right Z',
  'Gaze Vector Left X','Gaze Vector Left Y','Gaze Vector Left Z',
  'Eye Position Right X [mm]','Eye Position Right Y [mm]','Eye Position Right Z [mm]',
  'Eye Position Left X [mm]','Eye Position Left Y [mm]','Eye Position Left Z [mm]',
  'Pupil Position Right X [px]','Pupil Position Right Y [px]',
  'Pupil Position Left X [px]','Pupil Position Left Y [px]',
  'Annotation Name','Annotation Description','Annotation Tags',
  'Mouse Position X [px]','Mouse Position Y [px]',
  'Scroll Direction X','Scroll Direction Y','Content',
  'AOI Group Right','AOI Scope Right','AOI Order Right',
  'AOI Group Left','AOI Scope Left','AOI Order Binocular'
].join(',');

function buildCSV(){
  const biasMeta=`# GazeTrack v17.1 | bias_dx=${affineBias.dx.toFixed(2)} bias_dy=${affineBias.dy.toFixed(2)} bias_sx=${affineBias.sx.toFixed(4)} bias_sy=${affineBias.sy.toFixed(4)} val_samples=${valSamples.length} calib_samples=${calibSamples.length} viewport=${window.innerWidth}x${window.innerHeight}`;

  const totalDuration=recordedFrames.length>0?recordedFrames[recordedFrames.length-1].t:0;
  const trackingRatio=totalF>0?(trackedF/totalF*100):0;
  const colorMap={ASD:'DarkViolet',TD:'SteelBlue',other:'Gray'};
  const color=colorMap[META.group]||'Gray';
  const fn=(val,d=4)=>(val!==null&&val!==undefined&&!isNaN(val))?Number(val).toFixed(d):'-';
  const fn1=(val)=>fn(val,1);

  const pad=(n,w=2)=>String(Math.floor(n)).padStart(w,'0');

  // [FIX-V6] Separator row at trial start (SMI Category Group=Information)
  const firstTime = recordedFrames.length>0 ? recordedFrames[0].t : 0;
  const firstAbsTime = sessionStart + firstTime;
  const fd = new Date(firstAbsTime);
  const firstTod = `${pad(fd.getHours())}:${pad(fd.getMinutes())}:${pad(fd.getSeconds())}:${pad(fd.getMilliseconds(),3)}`;
  const sepRow = [
    0, firstTime.toFixed(3), firstTod,
    'Trial001', META.stimulus||'-', 0, totalDuration.toFixed(3),
    META.pid, color, trackingRatio.toFixed(4),
    'Information','Separator','Separator', 1, 1,
    '-','-','-','-','-','-',
    '-','-','-','-',
    '-','-',
    '-','-','-','-','-','-',
    '-','-','-','-','-','-',
    '-','-','-','-',
    '-','-','-','-','-','-','-',
    (META.stimulus||'-').toLowerCase(),
    '-','-','-','-','-','-'
  ];

  const lines=[biasMeta, CSV_HDR, sepRow.join(',')];

  recordedFrames.forEach((f,i)=>{
    const absTime=sessionStart+f.t;
    const d=new Date(absTime);
    const tod=`${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}:${pad(d.getMilliseconds(),3)}`;

    const isBlink = f.isBlink || f.category==='Blink';

    // [FIX-V2] SMI outputs 0.0000 for PoR during blinks, not '-'
    const gxR = isBlink ? '0.0000' : fn(f.gazeX,4);
    const gyR = isBlink ? '0.0000' : fn(f.gazeY,4);
    const gxL = isBlink ? '0.0000' : fn(f.gazeXL,4);
    const gyL = isBlink ? '0.0000' : fn(f.gazeYL,4);

    // [FIX-V2] Pupil size 0 during blinks; diameter preserved (last known value already in f.pupilDiamR)
    // [FIX-V3] Pupil size in px = diam_mm × 3.73 (SMI formula)
    const psRx = isBlink ? '0.0000' : fn(f.pupilSizePxR,4);
    const psRy = psRx;
    const psLx = isBlink ? '0.0000' : fn(f.pupilSizePxL,4);
    const psLy = psLx;
    // Diameter: always output (SMI keeps diameter during blinks from last known)
    const pdR = fn(f.pupilDiamR,4);
    const pdL = fn(f.pupilDiamL,4);

    const gvRx = fn(f.gazeVX,4);
    const gvRy = fn(f.gazeVY,4);
    const gvRz = fn(f.gazeVZ,4);
    const gvLx = gvRx; const gvLy = gvRy; const gvLz = gvRz;

    const epRx = fn(f.eyeRX,4); const epRy = fn(f.eyeRY,4); const epRz = fn(f.eyeRZ,4);
    const epLx = fn(f.eyeLX,4); const epLy = fn(f.eyeLY,4); const epLz = fn(f.eyeLZ,4);

    const ppRx = fn1(f.rightPupilX); const ppRy = fn1(f.rightPupilY);
    const ppLx = fn1(f.leftPupilX);  const ppLy = fn1(f.leftPupilY);

    const catG = f.catGroup || 'Eye';
    const catR = f.catRight  || f.category || '-';
    const catL = f.catLeft   || catR;

    // [FIX-V7] Row has exactly 57 fields — no Port Status, no groupe d'enfants
    const row=[
      i+1, f.t.toFixed(3), tod, 'Trial001', META.stimulus||'-', 0, totalDuration.toFixed(3),
      META.pid, color, trackingRatio.toFixed(4),
      catG, catR, catL, i+2, i+2,
      psRx, psRy, pdR, psLx, psLy, pdL,
      gxR, gyR, gxL, gyL,
      '-', '-',
      gvRx, gvRy, gvRz, gvLx, gvLy, gvLz,
      epRx, epRy, epRz, epLx, epLy, epLz,
      ppRx, ppRy, ppLx, ppLy,
      '-','-','-','-','-','-','-',
      META.stimulus||'-','-','-','-','-','-','-'
    ];
    lines.push(row.map(v=>v===undefined?'-':v).join(','));
  });
  return lines.join('\n');
}

function downloadCSV(){
  if(!csvData) return;
  const ts=new Date().toISOString().replace(/[:.]/g,'-');
  const fn=`gaze_${META.pid}_${META.group}_${ts}.csv`;
  const url=URL.createObjectURL(new Blob([csvData],{type:'text/csv'}));
  Object.assign(document.createElement('a'),{href:url,download:fn}).click();
  URL.revokeObjectURL(url);
}

// ─── CALIBRATION MAIN ────────────────────────────────────────────────────────
function startCalib() {
  calibPoints  = buildCalibPoints();
  calibSamples = [];
  calibIdx     = 0;
  doneCalibPoints = new Set();
  calibParticles  = [];
  calibFloaties   = [];
  creatureEls     = [];
  calibState   = 'idle';
  calibFacePresent = false;
  _calibCurrentGaze = null;
  _calibLastGaze    = null;
  _calibLastGazeTs  = 0;
  _calibLastFeat    = null;
  _calibHoldStart   = null;
  _calibHoldLostAt  = null;
  _calibSampling    = false;
  _calibSparkled    = false;
  _calibLoopT       = 0;
  _calibDwellAccum  = 0;
  _calibLastFrameTs = 0;
  _calibProcTs      = -1;
  _calibTargetY     = -1;

  const _lastLm = _calibLastFeat;
  const iodCheck = _lastLm ? _lastLm[8] : 0;
  if (iodCheck > 0.20) {
    const card = document.getElementById('calib-card');
    if (card) {
      const h2 = card.querySelector('h2'); if (h2) h2.textContent = '📏 Move back a little!';
      const p  = card.querySelector('p');
      if (p) p.innerHTML = 'You\'re sitting too close to the screen.<br><br>Ask the child to move back until their face is about <strong style="color:var(--accent)">arm\'s length</strong> away, then try again.';
    }
    const startBtn = document.getElementById('calib-start-btn'); if (startBtn) startBtn.textContent = '🐾 Try Again!';
    document.getElementById('calib-overlay')?.setAttribute('style','display:flex');
    phase = 'calib-ready'; return;
  }
  _earCalibSamples = []; _earThreshold = 0.20;
  affineBias = {dx:0, dy:0, sx:1, sy:1}; // reset so retries don't inherit bad correction
  initCalibScene();
  updateCalibHUD();

  const overlay=document.getElementById('calib-overlay');
  if(overlay) overlay.style.display='none';

  const dpr=window.devicePixelRatio||1;
  if(calibCanvas){
    calibCanvas.width=Math.round(window.innerWidth*dpr);
    calibCanvas.height=Math.round(window.innerHeight*dpr);
    calibCanvas.style.width=window.innerWidth+'px';
    calibCanvas.style.height=window.innerHeight+'px';
    calibCtx.setTransform(dpr,0,0,dpr,0,0);
  }

  setCalibBanner('🌟 Welcome, Star Keeper! Help the magical creatures eat — just look at them!');
  ensureDebugDot();
  setTimeout(()=>advanceCalibPoint(), 1200);
  if(!calibRaf) runCalibLoop();
}

function advanceCalibPoint() {
  if (calibIdx >= CALIB_TOTAL_PTS) { finaliseCalib(); return; }

  clearTimeout(_calibSkipTimer);
  _calibHoldStart    = null;
  _calibHoldLostAt   = null;
  _calibSampling     = false;
  _calibPointSamples = [];
  _calibSparkled     = false;
  _calibDwellAccum   = 0;
  _calibLastFrameTs  = 0;
  calibState         = 'gap';

  updateCalibHUD();

  const skipBtn = document.getElementById('calib-skip-btn');
  if (skipBtn) skipBtn.style.display = calibFailCount >= 1 ? 'inline-block' : 'none';

  const myIdx = calibIdx;

  setTimeout(() => {
    if (calibIdx !== myIdx) return;
    showCalibCreature(myIdx);
    calibState = 'showing';

    const cr  = creatureEls[myIdx];
    const pt  = calibPoints[myIdx];
    const isBottomCorner = pt && pt.isCorner && pt.y > window.innerHeight * 0.5;
    setCalibBanner(isBottomCorner
      ? `👀 Look at ${cr?.def?.name || 'the creature'} in the corner!`
      : `👀 Look at ${cr?.def?.name || 'the creature'}!`);

    const skipMs = isBottomCorner ? 18000 : CALIB_FORCE_SKIP_MS;
    _calibSkipTimer = setTimeout(() => {
      if (calibIdx !== myIdx) return;
      if (calibState === 'done-pt') return;
      console.warn(`Force-skipping point ${myIdx}`);
      calibIdx++;
      advanceCalibPoint();
    }, skipMs);
  }, CALIB_GAP_MS);
}

function showCalibCreature(idx) {
  const calibScreen=document.getElementById('s-calib');
  if(!calibScreen) return;

  const oldEl=document.getElementById('calib-creature-'+idx);
  if(oldEl) oldEl.remove();

  const pt=calibPoints[idx];
  const def=CREATURE_DEFS[idx%CREATURE_DEFS.length];
  const SIZE=160;

  const wrap=document.createElement('div');
  wrap.id='calib-creature-'+idx;
  wrap.className='calib-creature-wrap';
  wrap.style.left=pt.x+'px';
  wrap.style.top=pt.y+'px';

  const cvs=document.createElement('canvas');
  cvs.width=SIZE;cvs.height=SIZE;
  cvs.style.cssText=`width:${SIZE}px;height:${SIZE}px;display:block;`;
  wrap.appendChild(cvs);
  calibScreen.appendChild(wrap);

  creatureEls[idx]={wrap,cvs,ctx:cvs.getContext('2d'),size:SIZE,def,idx};

  requestAnimationFrame(()=>{wrap.classList.add('visible');});
  playAnimalJingle(idx);

  const dirs=['the centre','the top-left','the top-right','the bottom-right','the bottom-left'];
  const msgs=[
    `👀 ${def.name} is in ${dirs[idx]}! They want ${def.hunger}!`,
    `🌟 Look at ${def.name} in ${dirs[idx]}! Feed them ${def.hunger}!`,
    `✨ ${def.name} is waiting in ${dirs[idx]}! Give them ${def.hunger}!`,
    `💫 ${def.name} is hungry in ${dirs[idx]}! They love ${def.hunger}!`,
    `🎯 Almost done! ${def.name} in ${dirs[idx]} wants ${def.hunger}!`,
  ];
  setCalibBanner(msgs[idx%msgs.length]);
}

let _calibDwellAccum  = 0;
let _calibLastFrameTs = 0;

function runCalibLoop() {
  calibRaf = requestAnimationFrame(() => {
    if (phase !== 'calib-run') { calibRaf = null; return; }

    _calibLoopT += 0.016;
    const now = performance.now();
    const dt  = _calibLastFrameTs > 0 ? Math.min(now - _calibLastFrameTs, 50) : 16;
    _calibLastFrameTs = now;

    drawCalibStars(_calibLoopT);
    updateCalibFX();

    if (calibCtx) {
      const dpr = window.devicePixelRatio || 1;
      calibCtx.clearRect(0, 0, calibCanvas.width/dpr, calibCanvas.height/dpr);
    }

    const activeGaze = _calibCurrentGaze || _calibLastGaze;
    updateDebugOverlay(activeGaze, _calibLastFeat, calibIdx, calibPoints);

    creatureEls.forEach((cr, i) => {
      if (!cr || !cr.ctx) return;
      const pt = calibPoints[i];
      if (!pt) return;
      const isDone = doneCalibPoints.has(i);
      const radius = getCalibGazeRadius(pt);

      if (i === calibIdx) _calibTargetY = pt.y;

      const gazeNear = activeGaze
        ? (gazeModel
            ? Math.hypot(activeGaze.x - pt.x, activeGaze.y - pt.y) < radius
            : Math.abs(activeGaze.x - pt.x) < 220)
        : false;

      let holdPct = 0;
      const dwellActive = (i === calibIdx && !isDone
        && calibState !== 'gap' && calibState !== 'done-pt' && calibState !== 'idle');

      if (dwellActive) {
        if (calibState === 'showing') calibState = 'holding';

        _calibDwellAccum = Math.min(_calibDwellAccum + dt, CALIB_DWELL_REQUIRED_MS);
        holdPct = _calibDwellAccum / CALIB_DWELL_REQUIRED_MS;

        if (holdPct >= 0.20 && !_calibSampling) {
          _calibSampling      = true;
          _calibSamplingStart = now;
          _calibPointSamples  = [];
          calibState          = 'sampling';
          setCalibBanner(`😋 ${cr.def.name} loves it! Keep looking!`);
        }

        if (_calibDwellAccum >= CALIB_DWELL_REQUIRED_MS && !_calibSparkled) {
          _calibSparkled = true;
          addCalibBurst(pt.x, pt.y, cr.def.color, 36);
          addCalibHearts(pt.x, pt.y, 8);
          addCalibStars(pt.x, pt.y, 10);
          playHappyJingle(i);
          startConfettiLight();
          calibState = 'done-pt';
          clearTimeout(_calibSkipTimer);
          doneCalibPoints.add(i);
          if (cr.wrap) cr.wrap.classList.add('done');
          updateCalibHUD();
          const cheers = [
            `🎉 ${cr.def.name} is SO happy! Yummy ${cr.def.hunger}!`,
            `✨ Amazing! ${cr.def.name} loves you!`,
            `🌟 Incredible! ${cr.def.name} is doing a happy dance!`,
            `💖 ${cr.def.name} is full! You're the best Star Keeper!`,
            `🎊 ALL FED! You saved the magical forest!`,
          ];
          setCalibBanner(cheers[i]);
          setTimeout(() => { calibIdx++; advanceCalibPoint(); }, 900);
        }
      }

      const isHappy = gazeNear && !isDone;
      drawCreatureOnCanvas(cr.ctx, cr.def, cr.size, _calibLoopT, gazeNear, holdPct, isHappy, isDone);
    });

    runCalibLoop();
  });
}

function finaliseCalib() {
  const rafId=calibRaf;
  calibRaf=null;
  if(rafId) cancelAnimationFrame(rafId);
  removeDebugDot();

  if(calibSamples.length<MIN_SAMPLES){
    calibFailCount++;
    if(calibSamples.length>=10){
      gazeModel=trainModel(calibSamples);
      if(gazeModel){ startConfettiBig(); phase='validation'; startValidation(); return; }
    }
    const card=document.getElementById('calib-card');
    const h2=card?.querySelector('h2');
    const p=card?.querySelector('p');
    const startBtn=document.getElementById('calib-start-btn');
    const skipBtn=document.getElementById('calib-skip-btn');
    const overlay=document.getElementById('calib-overlay');
    if(h2) h2.textContent='🐾 Let\'s try again!';
    if(p){
      const tip=calibSamples.length===0
        ?'Child may have looked away. Remind them to watch the animals!'
        :calibSamples.length<10
          ?`Only ${calibSamples.length} samples. Try again!`
          :`${calibSamples.length} samples (need ${MIN_SAMPLES}). Try better lighting or move closer.`;
      p.innerHTML=`${tip}<br><br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, say <strong style="color:#fff">"Look at the animal!"</strong>`;
    }
    if(startBtn) startBtn.textContent='🐾 Try Again!';
    if(skipBtn&&calibFailCount>=1) skipBtn.style.display='inline-block';
    if(overlay) overlay.style.display='flex';
    showScreen('calib');
    calibSamples=[];
    phase='calib-ready';
    return;
  }

  gazeModel=trainModel(calibSamples);
  if(!gazeModel) calibSkipActive=true;
  startConfettiBig();
  phase='validation';
  startValidation();
}

// ─── CALIB BUTTONS ───────────────────────────────────────────────────────────
document.getElementById('calib-skip-btn')?.addEventListener('click',()=>{
  calibSkipActive=true; calibState='idle';
  cancelAnimationFrame(calibRaf); calibRaf=null;
  if(calibCtx) calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
  phase='validation'; startValidation();
});
document.getElementById('calib-start-btn')?.addEventListener('click',()=>{
  const overlay=document.getElementById('calib-overlay');
  if(overlay) overlay.style.display='none';
  calibSamples=[];
  phase='calib-run';
  // [FIX-A] Force fullscreen before calibration — viewport must be maximised
  // dy=-183px in testing was caused entirely by running in a browser toolbar window
  if(!document.fullscreenElement){
    document.documentElement.requestFullscreen().catch(()=>{}).finally(()=>{ setTimeout(startCalib, 150); });
  } else {
    startCalib();
  }
});

// ══════════════════════════════════════════════════════════════════════════════
//  STAR VALIDATION
let _valLastDetectTs = -1;

function endSession(){
  if(phase==='done') return;
  phase='done';


  clearInterval(timerInt);
  cancelAnimationFrame(procRaf);
  cancelAnimationFrame(calibRaf);
  cancelAnimationFrame(valRaf);
  stimVideo.pause();
  csvData=buildCSV();

  const pct2=totalF>0?Math.round(trackedF/totalF*100):0;
  const dur=Math.round((Date.now()-sessionStart)/1000);
  const ys=recordedFrames.filter(f=>f.tracked).map(f=>f.gazeY);
  let ystd=0;
  if(ys.length>1){const my=ys.reduce((a,b)=>a+b,0)/ys.length;ystd=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);}
  const biasOk=Math.abs(affineBias.dx)>5||Math.abs(affineBias.dy)>5;
  const biasLabel=biasOk?`${affineBias.dx>0?'+':''}${affineBias.dx.toFixed(0)},${affineBias.dy>0?'+':''}${affineBias.dy.toFixed(0)}px`:'Minimal';
  const fixCount=recordedFrames.filter(f=>f.category==='Fixation').length;
  const sacCount=recordedFrames.filter(f=>f.category==='Saccade').length;
  const statsEl=document.getElementById('done-stats');
  if(statsEl){
    statsEl.innerHTML=`
      <div class="done-stat"><div class="n">${recordedFrames.length}</div><div class="l">FRAMES</div></div>
      <div class="done-stat"><div class="n">${pct2}%</div><div class="l">TRACKED</div></div>
      <div class="done-stat"><div class="n">${dur}s</div><div class="l">DURATION</div></div>
      <div class="done-stat"><div class="n" style="color:${ystd>30?'var(--accent)':'var(--warn)'}">${ystd.toFixed(0)}px</div><div class="l">Y STD</div></div>
      <div class="done-stat"><div class="n" style="color:var(--accent)">${fixCount}</div><div class="l">FIXATIONS</div></div>
      <div class="done-stat"><div class="n" style="color:var(--gold)">${sacCount}</div><div class="l">SACCADES</div></div>
      <div class="done-stat"><div class="n" style="color:var(--accent);font-size:13px">${biasLabel}</div><div class="l">BIAS CORR</div></div>
    `;
  }
  showScreen('done');
  const ts2=new Date().toISOString().replace(/[:.]/g,'-');
  setTimeout(()=>uploadToMongo(csvData,`gaze_${META.pid}_${META.group}_${ts2}.csv`),600);
}

document.getElementById('end-btn')?.addEventListener('click',endSession);
document.getElementById('btn-dl')?.addEventListener('click',()=>downloadCSV());
document.getElementById('btn-restart')?.addEventListener('click',()=>location.reload());

// ─── MONGODB UPLOAD ──────────────────────────────────────────────────────────
function driveSetStatus(icon,msg,color){
  const el=document.getElementById('drive-status');if(!el)return;
  const iconEl=document.getElementById('drive-icon');if(iconEl)iconEl.textContent=icon;
  const msgEl=document.getElementById('drive-msg');if(msgEl)msgEl.textContent=msg;
  el.style.borderColor=color||'var(--border)';
}
async function uploadToMongo(csvText,filename){
  driveSetStatus('☁️','Saving to database...','var(--border)');
  try{
    const resp=await fetch(MONGO_API_URL,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({filename,pid:META.pid,age:META.age,group:META.group,
        clinician:META.clinician,location:META.location,notes:META.notes,
        timestamp:new Date().toISOString(),csv:csvText})
    });
    if(!resp.ok) throw new Error('Server '+resp.status);
    driveSetStatus('✅','Saved to database!','rgba(0,229,176,0.4)');
    const btn=document.getElementById('btn-dl');
    if(btn){btn.textContent='✅ Saved - click to download locally';btn.style.opacity='1';btn.style.pointerEvents='auto';btn.onclick=()=>downloadCSV();}
  }catch(err){
    driveSetStatus('❌','Database save failed - downloading locally','rgba(255,92,58,0.4)');
    downloadCSV();
  }
}

// ─── DEBUG & CLEANUP ──────────────────────────────────────────────────────────
window.addEventListener('beforeunload',()=>{
  if(sessionStream) sessionStream.getTracks().forEach(t=>t.stop());
  if(camStream)     camStream.getTracks().forEach(t=>t.stop());
});
window.addEventListener('resize',()=>{
  if(phase==='calib-ready') calibPoints=buildCalibPoints();
});

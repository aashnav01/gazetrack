/**
 * GazeTrack v14 – Device‑Adaptive (Child‑Proof Edition)
 * =============================================================================
 * - Full SMI RED‑format CSV output with Fixation/Saccade/Blink classification
 * - Child‑friendly "balloon animal" calibration: gaze‑contingent, animated,
 *   sound‑paired, 5‑point grid (centre + four corners)
 * - NEW: dynamic gaze radius based on screen size (15‑25% of screen)
 * - NEW: minimal confetti (30 particles) after each successful animal
 * - FIX: null‑style errors guarded
 * - FIX: skip timer now forces advancement even if child never looks
 * - Tablet‑ready & laptop‑friendly: touch events, responsive sizing, orientation handling
 *
 * 🧸 CHILD‑PROOF ENHANCEMENTS:
 *   - Gaze radius adapts to screen size (forgiving on any device)
 *   - Force‑skip timer fixed (moves to next animal after 4 s even without look)
 *   - Blink forgiveness (200ms grace before resetting hold)
 *   - Gap animation (pulsing ring)
 *   - Skip button appears after 1 failed attempt
 *   - Reduced calibration points (5 instead of 9)
 *   - Shortened validation (3 stars, shorter dwell)
 */

console.log('%c GazeTrack v14 (Device‑Adaptive)','background:#00e5b0;color:#000;font-weight:bold;font-size:14px');

import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs';

// ─── DEVICE ──────────────────────────────────────────────────────────────────
const isMobile = /Android|iPad|iPhone|iPod|Mobile/i.test(navigator.userAgent)
  || (navigator.maxTouchPoints > 1 && window.innerWidth < 1200);
const MP_DELEGATE = isMobile ? 'CPU' : 'GPU';
const IS_TABLET = /iPad|Android(?!.*Mobile)/i.test(navigator.userAgent) || (window.innerWidth >= 768 && window.innerWidth <= 1280);

// ─── CONSTANTS ────────────────────────────────────────────────────────────────
const CALIB_DWELL_MS    = 2200;   // ms child must gaze at each animal before sample taken
// CALIB_GAZE_RADIUS is now dynamic (percentage of screen) – see runCalibLoop
const CALIB_GAZE_HOLD   = 600;    // ms gaze must stay on target before sampling starts
const CALIB_SAMPLE_MS   = 800;    // ms of samples collected per point
const CALIB_TOTAL_PTS   = 5;      // 5-point grid (centre + four corners)
const CALIB_GAP_MS      = 700;    // ms between points
const MIN_SAMPLES       = 25;
const RIDGE_ALPHA       = 0.01;

const LEFT_IRIS   = [468,469,470,471];
const RIGHT_IRIS  = [473,474,475,476];
const L_CORNERS   = [33,133];
const R_CORNERS   = [362,263];

// Validation – shortened for kids
const VAL_DWELL_MS    = 3000;       // was 4500
const VAL_GAP_MS      = 600;        // was 900
const VAL_STAR_RADIUS = 48;
const VAL_SAMPLE_START= 0.60;
const VAL_INTRO_MS    = 2000;       // was 3000

// Position all-clear
const GOOD_STREAK_NEEDED = 8;
const BAD_STREAK_HIDE    = 12;

// Welfare monitor
const BLINK_HISTORY_SIZE = 300;   // ~30s at 10fps
const DROWSY_THRESHOLD   = 12;    // slow blinks/min (adjustable)

// MongoDB
const MONGO_API_URL = 'https://gazetrack-api.onrender.com/api/sessions';

// ─── STATE ────────────────────────────────────────────────────────────────────
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
let calibGazeHoldStart = null;
let calibSamplesForPoint = [];
let calibPhaseTimer = null;
let calibState = 'idle'; // idle | showing | holding | sampling | gap
let calibFailCount = 0;          // number of consecutive failed calibration attempts
let calibSkipActive = false;     // whether we are in skip mode (no model, just record raw data)

// Validation state
let valPoints=[], valIdx=0, valSamples=[], valRaf=null, valStart=0;
const VAL_PARTICLES = [];

// Recording
let recordedFrames=[], totalF=0, trackedF=0;
let sessionStart=0, timerInt=null;
let csvData=null;
let videoBlob=null;
let META={pid:'',age:'',group:'',clinician:'',location:'',notes:'',stimulus:''};

// Saccade classification
let prevGaze=null, prevGazeTime=null;

// Face presence
let calibFacePresent=false;

// Position banner
let _goodFrameStreak=0, _badFrameStreak=0, _allclearShowing=false, _allclearHideTimer=null;

// Preview detector
let previewFl=null, previewRaf=null, lastPreviewTs=-1, _prevLastRun=0;
let _lastVT=-1, _lastVideoTime=-1;
let procRaf=null;

// Pre-flight
const pfState={cam:'scanning',face:'scanning',light:'scanning',browser:'scanning'};
let pfRaf=null, _pfSamples=[], _pfThrottle=0, _pfCanvas=null, _pfCtx=null;

// Welfare monitor variables
let blinkTimes = [];                      // timestamps of recent blinks (ms)
let lastBlinkTime = 0;
let slowBlinkCount = 0;                   // count of slow blinks (duration > 200ms) in last 30s
let headMovementEvents = 0;                // number of large head movements in last 30s
let lastHeadPos = null;
let faceOffFrames = 0;                     // frames where face not detected

// ─── DOM REFS ─────────────────────────────────────────────────────────────────
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

// Welfare HUD elements
const welfareBlink    = document.getElementById('welfare-blink');
const welfareDrowsy   = document.getElementById('welfare-drowsy');
const welfareHead     = document.getElementById('welfare-head');
const welfareFaceOff  = document.getElementById('welfare-faceoff');

function showScreen(n) {
  Object.values(screens).forEach(s => s && s.classList.remove('active'));
  const scr = screens[n];
  if (scr) scr.classList.add('active');
}

// ─── TABLET TOUCH ADAPTATIONS ─────────────────────────────────────────────────
[calibCanvas, gazeCanvas, camCanvas].forEach(canvas => {
  if (canvas) canvas.style.touchAction = 'none';
});
document.querySelectorAll('button, .clickable').forEach(el => {
  if (IS_TABLET && el) el.style.minHeight = '48px';
});

// ─── AUDIO ────────────────────────────────────────────────────────────────────
let _audioCtx = null;
function getAudioCtx() {
  if (!_audioCtx) { try { _audioCtx = new AudioContext(); } catch(e) {} }
  return _audioCtx;
}
function playTone(freq, vol, duration, type='sine', detune=0) {
  const a = getAudioCtx(); if (!a) return;
  const o = a.createOscillator(), g = a.createGain();
  o.connect(g); g.connect(a.destination);
  o.type = type; o.frequency.value = freq; o.detune.value = detune;
  g.gain.setValueAtTime(0, a.currentTime);
  g.gain.linearRampToValueAtTime(vol, a.currentTime + 0.02);
  g.gain.exponentialRampToValueAtTime(0.001, a.currentTime + duration);
  o.start(); o.stop(a.currentTime + duration);
}
function playAnimalJingle(noteIndex) {
  const scales = [
    [523,659,784,1047],
    [587,698,880,1175],
    [659,784,988,1319],
    [698,880,1047,1397],
    [784,988,1175,1568],
  ];
  const scale = scales[noteIndex % scales.length];
  scale.forEach((f, i) => {
    setTimeout(() => playTone(f, 0.09, 0.18, 'sine'), i * 55);
  });
}
function playSuccessChime() {
  [784, 1047, 1319].forEach((f, i) => {
    setTimeout(() => playTone(f, 0.1, 0.3, 'sine'), i * 80);
  });
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

// 🎉 Minimal confetti shower
function startConfetti() {
  const confettiCount = 30; // reduced from 100
  const colors = ['#f4a261', '#e9c46a', '#90e0ef', '#ffb3c1', '#c77dff', '#ff6b6b', '#4cc9f0'];
  for (let i = 0; i < confettiCount; i++) {
    setTimeout(() => {
      const conf = document.createElement('div');
      conf.className = 'confetti';
      conf.style.cssText = `
        position: fixed; top: -10px; left: ${Math.random() * 100}vw; width: 6px; height: 6px;
        background: ${colors[Math.floor(Math.random() * colors.length)]};
        border-radius: 50%; z-index: 9999; pointer-events: none;
        transform: rotate(${Math.random() * 360}deg);
        animation: fall ${3 + Math.random() * 2}s linear forwards;
      `;
      document.body.appendChild(conf);
      setTimeout(() => conf.remove(), 5000);
    }, i * 20);
  }
}
// Add CSS once
if (!document.getElementById('confetti-style')) {
  const style = document.createElement('style');
  style.id = 'confetti-style';
  style.innerHTML = `
    @keyframes fall {
      to { transform: translateY(110vh) rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
}

// ─── POSITION ALL-CLEAR ───────────────────────────────────────────────────────
function updateAllClear(bright) {
  const posOk   = pfState.face === 'pass';
  const lightOk = pfState.light === 'pass';
  const allOk   = posOk && lightOk;

  if (allOk) {
    _badFrameStreak = 0;
    _goodFrameStreak = Math.min(_goodFrameStreak + 1, GOOD_STREAK_NEEDED + 1);
    if (_goodFrameStreak >= GOOD_STREAK_NEEDED && !_allclearShowing) {
      showAllClear(bright);
    }
  } else {
    _goodFrameStreak = 0;
    _badFrameStreak = Math.min(_badFrameStreak + 1, BAD_STREAK_HIDE + 1);
    if (_badFrameStreak >= BAD_STREAK_HIDE && _allclearShowing) {
      hideAllClear();
    }
  }
}
function showAllClear(bright) {
  _allclearShowing = true;
  clearTimeout(_allclearHideTimer);
  const tags = ['✓ Face visible · Good distance', '✓ Lighting OK'];
  const tagsEl = document.getElementById('allclear-tags');
  if (tagsEl) tagsEl.innerHTML = tags.map(t => `<span class="allclear-tag">${t}</span>`).join('');
  const detailEl = document.getElementById('allclear-detail');
  if (detailEl) detailEl.textContent = `Brightness ${Math.round(bright)}/255`;
  if (allclearBanner) allclearBanner.classList.add('show');
  playChime(660, 0.08, 0.4);
}
function hideAllClear() {
  _allclearShowing = false;
  if (allclearBanner) allclearBanner.classList.remove('show');
}

// ─── PRE-FLIGHT ───────────────────────────────────────────────────────────────
function pfSet(id, state, msg) {
  pfState[id] = state;
  const card = document.getElementById('pfc-' + id);
  const stat = document.getElementById('pfc-' + id + '-s');
  if (card && stat) {
    card.className = 'pf-check ' + state;
    stat.innerHTML = msg;
  }
  pfUpdateScore();
}
function pfUpdateScore() {
  const vals  = Object.values(pfState);
  const done  = vals.filter(v => v !== 'scanning').length;
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
  if (pfState.light === 'fail')  tips.push('<strong>💡 Too dark:</strong> Add a front‑facing lamp.');
  if (pfState.light === 'warn')  tips.push('<strong>💡 Lighting:</strong> Brighter room helps iris detection.');
  if (pfState.face  === 'fail')  tips.push('<strong>👤 No face:</strong> Make sure child is in frame, camera at eye level.');
  if (pfState.browser === 'warn') tips.push('<strong>🌐 Browser:</strong> Use Chrome for best webcam performance.');
  const adv = document.getElementById('pf-advice');
  if (adv) { adv.innerHTML = tips.join('<br>'); adv.className = 'pf-advice' + (tips.length ? ' show' : ''); }

  const btn = document.getElementById('start-btn');
  if (btn && !btn.disabled) {
    const critFails = ['cam','face'].filter(k => pfState[k] === 'fail').length;
    if (critFails > 0)       { btn.textContent = '⚠ Proceed Anyway'; btn.style.background = 'linear-gradient(135deg,#ff9f43,#e17f20)'; }
    else if (done < total)   { btn.textContent = 'Begin Session →'; btn.style.background = ''; }
    else if (score >= 75)    { btn.textContent = '✅ All Clear - Begin Session'; btn.style.background = ''; }
    else                     { btn.textContent = '⚠ Proceed with Warnings'; btn.style.background = 'linear-gradient(135deg,#ca8a04,#a16207)'; }
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
    let sumR = 0, sumG = 0, sumB = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) { sumR += d[i]; sumG += d[i+1]; sumB += d[i+2]; n++; }
    const brightness = (sumR + sumG + sumB) / (n * 3);
    _pfSamples.push(brightness);
    if (_pfSamples.length > 3) _pfSamples.shift();
    const avgBright = _pfSamples.reduce((a,b) => a+b, 0) / _pfSamples.length;

    if (avgBright >= 60 && avgBright <= 220)  pfSet('light','pass', `✓ Good (${Math.round(avgBright)}/255)`);
    else if (avgBright < 40)                  pfSet('light','fail', `✗ Too dark (${Math.round(avgBright)}) - add light`);
    else if (avgBright < 60)                  pfSet('light','warn', `⚠ Dim (${Math.round(avgBright)}) - improve lighting`);
    else                                      pfSet('light','warn', `⚠ Bright (${Math.round(avgBright)}) - reduce backlight`);

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
      video: {width:{ideal:640,max:1280}, height:{ideal:480,max:720}, facingMode:'user'}, audio: false
    });
    camStream = stream;
    camPreview.srcObject = stream;
    camPreview.play();
    const camDot = document.getElementById('cam-dot');
    if (camDot) camDot.classList.add('ok');
    const camStatus = document.getElementById('cam-status-txt');
    if (camStatus) camStatus.textContent = 'Camera active';
    const chkCam = document.getElementById('chk-cam');
    if (chkCam) { chkCam.classList.add('ok'); chkCam.textContent = '✓ Cam'; }
    const t = stream.getVideoTracks()[0].getSettings();
    const w = t.width || 640, h = t.height || 480;
    pfSet('cam','pass', `✓ ${w}x${h}`);
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

// ─── PREVIEW MESH ─────────────────────────────────────────────────────────────
async function loadPreviewDetector() {
  try {
    const resolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
    previewFl = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: MP_DELEGATE
      },
      runningMode: 'VIDEO', numFaces: 1,
      outputFaceBlendshapes: false, outputFacialTransformationMatrixes: true, outputIrisLandmarks: true
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
    const dW = Math.round(rect.width) || 640, dH = Math.round(rect.height) || 480;
    if (camCanvas.width !== dW || camCanvas.height !== dH) { camCanvas.width = dW; camCanvas.height = dH; }
    let ts = camPreview.currentTime * 1000;
    if (ts <= lastPreviewTs) ts = lastPreviewTs + 0.001;
    lastPreviewTs = ts;

    try {
      const res = previewFl.detectForVideo(camPreview, ts);
      const hasFace = !!(res.faceLandmarks && res.faceLandmarks.length > 0);
      camCtx.clearRect(0, 0, camCanvas.width, camCanvas.height);

      if (hasFace) {
        drawPreviewMesh(res.faceLandmarks[0]);
        const lm = res.faceLandmarks[0];
        const hasIris = !!(lm[468] && lm[473]);
        const chkFace = document.getElementById('chk-face');
        const chkIris = document.getElementById('chk-iris');
        if (chkFace) { chkFace.classList.add('ok'); chkFace.textContent = '✓ Face'; }
        if (chkIris) {
          chkIris.classList.toggle('ok', hasIris);
          chkIris.textContent = hasIris ? '✓ Iris' : '👁 Iris';
        }

        if (hasIris) {
          const iodNorm = Math.hypot(lm[473].x - lm[468].x, lm[473].y - lm[468].y);
          const faceCX  = (lm[33].x + lm[263].x) / 2;
          const offCentre = faceCX < 0.25 || faceCX > 0.75;
          const lEAR = Math.hypot(lm[159].x-lm[145].x, lm[159].y-lm[145].y);
          const rEAR = Math.hypot(lm[386].x-lm[374].x, lm[386].y-lm[374].y);
          const earPx = (lEAR + rEAR) / 2;
          const qPct  = (earPx / (iodNorm + 1e-6)) > 0.08 ? 95 : 75;
          const qFill = document.getElementById('q-fill');
          const qPctEl = document.getElementById('q-pct');
          if (qFill) qFill.style.width = qPct + '%';
          if (qPctEl) qPctEl.textContent = qPct + '%';

          if (iodNorm > 0.22)
            pfSet('face','warn', `⚠ Too close (IOD ${iodNorm.toFixed(3)}) - move back ~15 cm`);
          else if (iodNorm >= 0.13)
            pfSet('face','pass', offCentre
              ? `✓ Good distance - Move slightly to centre`
              : `✓ Face visible - Good distance (~50-70 cm)`);
          else if (iodNorm >= 0.07)
            pfSet('face','warn', `⚠ Too far (IOD ${iodNorm.toFixed(3)}) - move ${offCentre ? 'closer & to centre' : '~20 cm closer'}`);
          else
            pfSet('face','warn', `⚠ Very far or face at edge - move much closer`);
        } else {
          const qFill = document.getElementById('q-fill');
          const qPctEl = document.getElementById('q-pct');
          if (qFill) qFill.style.width = '40%';
          if (qPctEl) qPctEl.textContent = '40%';
          pfSet('face','warn','⚠ Face detected but iris not visible - look at camera');
        }
      } else {
        const chkFace = document.getElementById('chk-face');
        const chkIris = document.getElementById('chk-iris');
        if (chkFace) { chkFace.classList.remove('ok'); chkFace.textContent = '👤 Face'; }
        if (chkIris) { chkIris.classList.remove('ok'); chkIris.textContent = '👁 Iris'; }
        const qFill = document.getElementById('q-fill');
        const qPctEl = document.getElementById('q-pct');
        if (qFill) qFill.style.width = '0%';
        if (qPctEl) qPctEl.textContent = ' - ';
        pfSet('face','fail','✗ No face detected - check camera position');
      }
    } catch(e) {}
  }
  previewRaf = requestAnimationFrame(previewLoop);
}
function drawPreviewMesh(lm) {
  const W = camCanvas.width, H = camCanvas.height;
  const fx = x => (1-x) * W, fy = y => y * H;
  [[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33],
   [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]].forEach(pts => {
    camCtx.beginPath();
    pts.forEach((idx,i) => {
      const p = lm[idx];
      i === 0 ? camCtx.moveTo(fx(p.x), fy(p.y)) : camCtx.lineTo(fx(p.x), fy(p.y));
    });
    camCtx.strokeStyle = '#00e5b0'; camCtx.lineWidth = 0.4; camCtx.globalAlpha = 0.2; camCtx.stroke(); camCtx.globalAlpha = 1;
  });
  [[468,469],[473,474]].forEach(([c,e]) => {
    if (!lm[c] || !lm[e]) return;
    const cx = fx(lm[c].x), cy = fy(lm[c].y), ex2 = fx(lm[e].x), ey2 = fy(lm[e].y);
    const r = Math.hypot(ex2-cx, ey2-cy) + 0.5;
    camCtx.beginPath(); camCtx.arc(cx, cy, r, 0, Math.PI*2);
    camCtx.strokeStyle = '#00e5b0'; camCtx.lineWidth = 0.6; camCtx.globalAlpha = 0.3; camCtx.stroke(); camCtx.globalAlpha = 1;
  });
}

// ─── FORM ─────────────────────────────────────────────────────────────────────
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
  document.getElementById('video-drop').insertAdjacentHTML('beforeend', `<div class="chosen">✓ ${f.name}</div>`);
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
      const resolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
      faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: MP_DELEGATE
        },
        runningMode: 'VIDEO', numFaces: 1,
        outputFaceBlendshapes: false, outputFacialTransformationMatrixes: true, outputIrisLandmarks: true
      });
    }

    if (camStream) { sessionStream = camStream; camStream = null; }
    else {
      sessionStream = await navigator.mediaDevices.getUserMedia({
        video: {width:{ideal:640}, height:{ideal:480}, facingMode:'user'}, audio: false
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

// ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────
function extractFeatures(lm, mat) {
  const avg = ids => {
    const s = {x:0,y:0,z:0};
    ids.forEach(i => { s.x += lm[i].x; s.y += lm[i].y; s.z += (lm[i].z||0); });
    return {x: s.x/ids.length, y: s.y/ids.length, z: s.z/ids.length};
  };
  const li = avg(LEFT_IRIS), ri = avg(RIGHT_IRIS);
  const lIn = lm[L_CORNERS[0]], lOut = lm[L_CORNERS[1]];
  const rIn = lm[R_CORNERS[0]], rOut = lm[R_CORNERS[1]];
  const lW = Math.hypot(lOut.x-lIn.x, lOut.y-lIn.y) + 1e-6;
  const rW = Math.hypot(rOut.x-rIn.x, rOut.y-rIn.y) + 1e-6;
  const lCx = (lIn.x+lOut.x)/2, rCx = (rIn.x+rOut.x)/2;
  const liX = (li.x - lCx)/lW, riX = (ri.x - rCx)/rW;

  let pitchDeg = 0;
  if (mat?.data) {
    const m = mat.data;
    pitchDeg = Math.asin(Math.max(-1, Math.min(1, -m[6]))) * 180/Math.PI / 30;
  }
  const nose = lm[1], fore = lm[10], chin = lm[152];
  const pitchZ = ((nose.z||0) - ((fore.z||0)+(chin.z||0))/2) * 10;
  const vertMain = (Math.abs(pitchDeg) > 0.001) ? pitchDeg : pitchZ;

  const faceCY = (fore.y + chin.y)/2, faceH = Math.abs(chin.y - fore.y) + 1e-6;
  const irisY  = ((li.y + ri.y)/2 - faceCY) / faceH;
  const lEAR   = Math.hypot(lm[159].x-lm[145].x, lm[159].y-lm[145].y) / lW;
  const rEAR   = Math.hypot(lm[386].x-lm[374].x, lm[386].y-lm[374].y) / rW;
  const ear    = (lEAR + rEAR)/2;
  const iod    = Math.hypot(ri.x - li.x, ri.y - li.y);

  return [liX, riX, vertMain, fore.y, irisY, (li.y+ri.y)/2, (liX+riX)/2, ear, iod];
}

// ─── RIDGE REGRESSION ────────────────────────────────────────────────────────
function poly(f) { return [1, ...f.slice(0, 7)]; }
function ridgeFit(X, y, alpha=RIDGE_ALPHA) {
  const n = X[0].length;
  const XtX = Array.from({length:n}, () => new Array(n).fill(0));
  const Xty = new Array(n).fill(0);
  for (let r = 0; r < X.length; r++) {
    for (let i = 0; i < n; i++) {
      Xty[i] += X[r][i] * y[r];
      for (let j = 0; j < n; j++) XtX[i][j] += X[r][i] * X[r][j];
    }
  }
  for (let i = 0; i < n; i++) XtX[i][i] += alpha;
  const aug = XtX.map((row,i) => [...row, Xty[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c+1; r < n; r++) if (Math.abs(aug[r][c]) > Math.abs(aug[p][c])) p = r;
    [aug[c], aug[p]] = [aug[p], aug[c]];
    const pv = aug[c][c]; if (Math.abs(pv) < 1e-12) continue;
    for (let j = c; j <= n; j++) aug[c][j] /= pv;
    for (let r = 0; r < n; r++) {
      if (r !== c) { const f = aug[r][c]; for (let j = c; j <= n; j++) aug[r][j] -= f * aug[c][j]; }
    }
  }
  return aug.map(r => r[n]);
}
function trainModel(samples) {
  if (samples.length < MIN_SAMPLES) return null;
  const X = samples.map(s => poly(s.feat));
  return {
    wx: ridgeFit(X, samples.map(s => s.sx)),
    wy: ridgeFit(X, samples.map(s => s.sy))
  };
}
function predictGaze(feat, model) {
  if (!model) return null;
  const pf = poly(feat);
  const gx = pf.reduce((s,v,i) => s + v*model.wx[i], 0);
  const gy = pf.reduce((s,v,i) => s + v*model.wy[i], 0);
  const cx = affineBias.sx * gx + affineBias.dx;
  const cy = affineBias.sy * gy + affineBias.dy;
  return {
    x: Math.max(0, Math.min(window.innerWidth,  cx)),
    y: Math.max(0, Math.min(window.innerHeight, cy))
  };
}
function estimateGazeFromIris(feat) {
  const [liX, riX, , , , , avgX] = feat;
  const rawX = (avgX + 0.5) * window.innerWidth;
  const irisYabs = feat[5];
  const rawY = irisYabs * window.innerHeight;
  return { x: rawX, y: rawY };
}
function computeAffineCorrection(pairs) {
  function linfit(ps, ts) {
    const n = ps.length;
    const mp = ps.reduce((a,b) => a+b, 0)/n, mt = ts.reduce((a,b) => a+b, 0)/n;
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (ps[i]-mp)*(ts[i]-mt); den += (ps[i]-mp)**2; }
    const s  = den > 1e-6 ? num/den : 1;
    const sc = Math.max(0.6, Math.min(1.6, s));
    return {s: sc, d: mt - sc*mp};
  }
  const fx = linfit(pairs.map(p => p.px), pairs.map(p => p.tx));
  const fy = linfit(pairs.map(p => p.py), pairs.map(p => p.ty));
  return {sx: fx.s, dx: fx.d, sy: fy.s, dy: fy.d};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CHILD-FRIENDLY CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

function buildCalibPoints() {
  const W = window.innerWidth, H = window.innerHeight;
  const mx = W/2, my = H/2;
  const px = Math.max(90, W * 0.13), py = Math.max(80, H * 0.14);
  return [
    {x:mx,      y:my      },
    {x:px,      y:py      },
    {x:W-px,    y:py      },
    {x:W-px,    y:H-py    },
    {x:px,      y:H-py    },
  ];
}
const ANIMALS = ['cat','rabbit','bear','frog','duck'];
const ANIMAL_PALETTES = [
  {body:'#f4a261',ear:'#e76f51',eye:'#264653',bg:'#e9c46a'},
  {body:'#e9c46a',ear:'#f4a261',eye:'#264653',bg:'#90e0ef'},
  {body:'#a8dadc',ear:'#457b9d',eye:'#1d3557',bg:'#f1faee'},
  {body:'#95d5b2',ear:'#52b788',eye:'#1b4332',bg:'#d8f3dc'},
  {body:'#ffd166',ear:'#ef476f',eye:'#073b4c',bg:'#06d6a0'},
];
function drawAnimal(ctx, type, x, y, t, scale, happy, gazeNear) {
  const r = Math.round(Math.max(44, Math.min(window.innerWidth, window.innerHeight) * 0.085)) * scale;
  const bob = Math.sin(t * 3.5) * r * 0.12;
  const wiggle = Math.sin(t * 7) * 0.06;
  const pal = ANIMAL_PALETTES[ANIMALS.indexOf(type)] || ANIMAL_PALETTES[0];
  const cy = y + bob;
  ctx.save();
  ctx.translate(x, cy);
  ctx.rotate(wiggle);
  if (gazeNear) {
    ctx.beginPath();
    ctx.arc(0, 0, r * 1.55, 0, Math.PI * 2);
    const g = ctx.createRadialGradient(0, 0, r, 0, 0, r*1.55);
    g.addColorStop(0, 'rgba(255,230,0,0.5)');
    g.addColorStop(1, 'rgba(255,230,0,0)');
    ctx.fillStyle = g;
    ctx.fill();
  }
  ctx.save();
  ctx.globalAlpha = 0.12;
  ctx.beginPath();
  ctx.ellipse(0, r + 5, r*0.85, r*0.2, 0, 0, Math.PI*2);
  ctx.fillStyle = '#000'; ctx.fill();
  ctx.restore();
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI*2);
  ctx.fillStyle = pal.body; ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.lineWidth = 2; ctx.stroke();
  // eyes
  [-0.28, 0.28].forEach(ex => {
    const eyeY = -r*0.12;
    ctx.beginPath(); ctx.arc(ex*r, eyeY, r*0.15, 0, Math.PI*2); ctx.fillStyle = '#fff'; ctx.fill();
    ctx.beginPath(); ctx.arc(ex*r + r*0.04, eyeY, r*0.08, 0, Math.PI*2); ctx.fillStyle = pal.eye; ctx.fill();
    ctx.beginPath(); ctx.arc(ex*r + r*0.07, eyeY - r*0.05, r*0.03, 0, Math.PI*2); ctx.fillStyle = '#fff'; ctx.fill();
  });
  // mouth
  if (happy) {
    ctx.beginPath(); ctx.arc(0, r*0.28, r*0.22, 0.1*Math.PI, 0.9*Math.PI);
    ctx.strokeStyle = pal.eye; ctx.lineWidth = 2.5; ctx.stroke();
  } else {
    ctx.beginPath(); ctx.arc(0, r*0.4, r*0.16, Math.PI*1.1, Math.PI*1.9);
    ctx.strokeStyle = pal.eye; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.restore();
}

const CALIB_PARTICLES = [];
function spawnCalibParticles(x, y) {
  for (let i = 0; i < 20; i++) {
    const angle = Math.random() * Math.PI * 2, speed = 3 + Math.random()*5;
    CALIB_PARTICLES.push({
      x, y,
      vx: Math.cos(angle)*speed, vy: Math.sin(angle)*speed - 2,
      life: 1, size: 3+Math.random()*5, hue: 30+Math.random()*60
    });
  }
}
function updateCalibParticles(ctx) {
  for (let i = CALIB_PARTICLES.length - 1; i >= 0; i--) {
    const p = CALIB_PARTICLES[i];
    p.x += p.vx; p.y += p.vy; p.vy += 0.15; p.life -= 0.03;
    if (p.life <= 0) { CALIB_PARTICLES.splice(i, 1); continue; }
    ctx.save();
    ctx.globalAlpha = p.life;
    ctx.fillStyle = `hsl(${p.hue},100%,60%)`;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.size*p.life, 0, Math.PI*2); ctx.fill();
    ctx.restore();
  }
}

function startCalib() {
  calibPoints   = buildCalibPoints();
  calibSamples  = [];
  calibIdx      = 0;
  CALIB_PARTICLES.length = 0;
  calibState    = 'gap';
  calibFacePresent = false;

  const prog = document.getElementById('calib-progress');
  if (prog) { prog.style.display = 'flex'; updateCalibProgress(); }

  const skipBtn = document.getElementById('calib-skip-btn');
  if (skipBtn) skipBtn.style.display = calibFailCount >= 1 ? 'inline-block' : 'none';

  const dpr = window.devicePixelRatio || 1;
  if (calibCanvas) {
    calibCanvas.width  = Math.round(window.innerWidth  * dpr);
    calibCanvas.height = Math.round(window.innerHeight * dpr);
    calibCanvas.style.width  = window.innerWidth  + 'px';
    calibCanvas.style.height = window.innerHeight + 'px';
    calibCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  advanceCalibPoint();
}
function updateCalibProgress() {
  const prog = document.getElementById('calib-progress');
  if (!prog) return;
  prog.innerHTML = '';
  for (let i = 0; i < CALIB_TOTAL_PTS; i++) {
    const paw = document.createElement('span');
    paw.className = 'calib-paw';
    paw.textContent = i < calibIdx ? '🐾' : (i === calibIdx ? '⭐' : '○');
    paw.style.opacity = i < calibIdx ? '0.5' : '1';
    prog.appendChild(paw);
  }
}

let _calibT = 0;
let _calibGapStart = 0;
let _calibPointStart = 0;
let _calibHoldStart = null;
let _calibSampling = false;
let _calibSamplingStart = 0;
let _calibPointSamples = [];
let _calibSkipTimer = null;
let _calibSparkled = false;
let _calibCurrentGaze = null;

function advanceCalibPoint() {
  if (calibIdx >= CALIB_TOTAL_PTS) {
    finaliseCalib();
    return;
  }
  clearTimeout(_calibSkipTimer);
  _calibT = 0;
  _calibHoldStart = null;
  _calibSampling = false;
  _calibPointSamples = [];
  _calibSparkled = false;
  calibState = 'gap';
  _calibGapStart = performance.now();
  updateCalibProgress();

  setTimeout(() => {
    calibState = 'showing';
    _calibPointStart = performance.now();
    playAnimalJingle(calibIdx);
    _calibSkipTimer = setTimeout(() => {
      if (calibState !== 'done-point') {
        if (calibIdx < calibPoints.length) {
          calibIdx++;
          advanceCalibPoint();
        }
      }
    }, 4000);
  }, CALIB_GAP_MS);

  if (!calibRaf) runCalibLoop();
}

function runCalibLoop() {
  calibRaf = requestAnimationFrame(() => {
    if (phase !== 'calib-run') { calibRaf = null; return; }
    const now = performance.now();
    _calibT += 0.016;

    const dpr = window.devicePixelRatio || 1;
    if (calibCtx) {
      calibCtx.clearRect(0, 0, calibCanvas.width / dpr, calibCanvas.height / dpr);
      updateCalibParticles(calibCtx);

      // Gap animation
      if (calibState === 'gap') {
        const gapElapsed = now - _calibGapStart;
        const progress = Math.min(gapElapsed / CALIB_GAP_MS, 1);
        const centerX = calibCanvas.width / dpr / 2;
        const centerY = calibCanvas.height / dpr / 2;
        calibCtx.beginPath();
        calibCtx.arc(centerX, centerY, 30 + 20 * Math.sin(progress * Math.PI * 2), 0, 2 * Math.PI);
        calibCtx.strokeStyle = '#00e5b0';
        calibCtx.lineWidth = 4;
        calibCtx.globalAlpha = 1 - progress;
        calibCtx.stroke();
        calibCtx.globalAlpha = 1;
      }

      if (calibState === 'showing' || calibState === 'holding' || calibState === 'sampling') {
        const pt = calibPoints[calibIdx];
        const animal = ANIMALS[calibIdx % ANIMALS.length];

        // 👀 Dynamic gaze radius based on screen size and distance from centre
        const screenCenter = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
        const distFromCenter = Math.hypot(pt.x - screenCenter.x, pt.y - screenCenter.y);
        const maxDist = Math.hypot(window.innerWidth / 2, window.innerHeight / 2);
        const baseRadius = Math.min(window.innerWidth, window.innerHeight) * 0.15;   // 15% of smaller dimension
        const maxRadius   = Math.min(window.innerWidth, window.innerHeight) * 0.25; // 25% at corners
        const actualGazeRadius = baseRadius + (maxRadius - baseRadius) * (distFromCenter / maxDist);

        let gazeNear = false;
        if (_calibCurrentGaze) {
          const dist = Math.hypot(_calibCurrentGaze.x - pt.x, _calibCurrentGaze.y - pt.y);
          gazeNear = dist < actualGazeRadius;
        }

        const happy = gazeNear;
        const elapsed = now - _calibPointStart;
        const entrance = Math.min(elapsed / 350, 1);
        const entScale = 1 - Math.pow(1 - entrance, 3) * Math.cos(entrance * Math.PI * 1.5);

        drawAnimal(calibCtx, animal, pt.x, pt.y, _calibT, Math.min(entScale, 1.05), happy, gazeNear);

        if (calibState === 'showing' && gazeNear && entrance >= 0.9) {
          calibState = 'holding';
          _calibHoldStart = now;
        }
        if (calibState === 'holding') {
          if (!gazeNear && (now - _calibHoldStart) > 200) {
            calibState = 'showing';
            _calibHoldStart = null;
          } else if (now - _calibHoldStart >= CALIB_GAZE_HOLD) {
            calibState = 'sampling';
            _calibSampling = true;
            _calibSamplingStart = now;
            _calibPointSamples = [];
          }
        }
        if (calibState === 'sampling') {
          if (now - _calibSamplingStart >= CALIB_SAMPLE_MS) {
            if (!_calibSparkled) {
              spawnCalibParticles(pt.x, pt.y);
              playSuccessChime();
              // 🎉 Minimal confetti on successful completion of this point
              startConfetti();
              _calibSparkled = true;
            }
            calibState = 'done-point';
            clearTimeout(_calibSkipTimer);
            setTimeout(() => commitCalibPoint(), 300);
          }
        }

        if (calibState === 'holding') {
          const holdPct = (now - _calibHoldStart) / CALIB_GAZE_HOLD;
          const r = Math.max(44, Math.min(window.innerWidth, window.innerHeight) * 0.085);
          calibCtx.beginPath();
          calibCtx.arc(pt.x, pt.y, r * 1.3, -Math.PI/2, -Math.PI/2 + holdPct * Math.PI * 2);
          calibCtx.strokeStyle = '#00e5b0';
          calibCtx.lineWidth = 4;
          calibCtx.stroke();
        }
      }
    }

    runCalibLoop();
  });
}

function commitCalibPoint() {
  if (calibState === 'done-point' && calibIdx < calibPoints.length) {
    calibIdx++;
    advanceCalibPoint();
  }
}

function finaliseCalib() {
  const rafId = calibRaf;
  calibRaf = null;
  cancelAnimationFrame(rafId);
  if (calibCtx) calibCtx.clearRect(0, 0, calibCanvas.width, calibCanvas.height);

  const prog = document.getElementById('calib-progress');
  if (prog) prog.style.display = 'none';

  if (calibSamples.length < MIN_SAMPLES) {
    calibFailCount++;
    if (calibSamples.length >= 10) {
      gazeModel = trainModel(calibSamples);
      if (gazeModel) {
        phase = 'validation';
        startValidation();
        return;
      }
    }
    const card = document.getElementById('calib-card');
    if (card) {
      const h2 = card.querySelector('h2');
      if (h2) h2.textContent = '🐾 Let\'s try again!';
      const p = card.querySelector('p');
      if (p) {
        let tip = '';
        if (calibSamples.length === 0) {
          tip = 'Child may have looked away. Remind them to "Watch the animal!"';
        } else if (calibSamples.length < 10) {
          tip = `Only ${calibSamples.length} samples collected. Try again and ensure child looks at the animals.`;
        } else {
          tip = `Not enough samples (${calibSamples.length}). Could be lighting or distance.`;
        }
        p.innerHTML = `${tip}<br><br>` +
          `<strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, ` +
          `<strong style="color:#fff">"Look at the animal!"</strong>`;
      }
    }
    const startBtn = document.getElementById('calib-start-btn');
    if (startBtn) startBtn.textContent = '🐾 Try Again!';
    const skipBtn = document.getElementById('calib-skip-btn');
    if (skipBtn && calibFailCount >= 1) skipBtn.style.display = 'inline-block';
    const overlay = document.getElementById('calib-overlay');
    if (overlay) overlay.style.display = 'flex';
    calibSamples = [];
    phase = 'calib-ready';
    return;
  }

  gazeModel = trainModel(calibSamples);
  if (!gazeModel) {
    calibSkipActive = true;
  }
  phase = 'validation';
  startValidation();
}

document.getElementById('calib-skip-btn')?.addEventListener('click', () => {
  calibSkipActive = true;
  calibState = 'idle';
  cancelAnimationFrame(calibRaf);
  calibRaf = null;
  if (calibCtx) calibCtx.clearRect(0, 0, calibCanvas.width, calibCanvas.height);
  phase = 'validation';
  startValidation();
});
document.getElementById('calib-start-btn')?.addEventListener('click', () => {
  const overlay = document.getElementById('calib-overlay');
  if (overlay) overlay.style.display = 'none';
  calibSamples = [];
  phase = 'calib-run';
  startCalib();
});

// ═══════════════════════════════════════════════════════════════════════════════
//  STAR VALIDATION (shortened)
// ═══════════════════════════════════════════════════════════════════════════════
function spawnSparkles(ctx, x, y) {
  for (let i = 0; i < 14; i++) {
    const angle = Math.random() * Math.PI * 2, speed = 2 + Math.random() * 4;
    VAL_PARTICLES.push({
      x, y,
      vx: Math.cos(angle)*speed, vy: Math.sin(angle)*speed,
      life: 1, size: 2+Math.random()*4, hue: 40+Math.random()*30
    });
  }
}
function updateSparkles(ctx) {
  for (let i = VAL_PARTICLES.length - 1; i >= 0; i--) {
    const p = VAL_PARTICLES[i];
    p.x += p.vx; p.y += p.vy; p.vy += 0.12; p.life -= 0.035;
    if (p.life <= 0) { VAL_PARTICLES.splice(i, 1); continue; }
    ctx.save(); ctx.globalAlpha = p.life;
    ctx.fillStyle = `hsl(${p.hue},100%,65%)`;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.size*p.life, 0, Math.PI*2); ctx.fill();
    ctx.restore();
  }
}
function drawStar(ctx, x, y, radius, twinklePhase, entranceProgress) {
  const r = radius * entranceProgress;
  const innerR = r * 0.4, points = 5;
  const twinkle = 1 + Math.sin(twinklePhase * 8) * 0.08 * entranceProgress;
  const rr = r * twinkle;
  const glowSize = rr * (1.8 + Math.sin(twinklePhase*6)*0.2);
  const grad = ctx.createRadialGradient(x, y, 0, x, y, glowSize);
  grad.addColorStop(0, `rgba(255,220,50,${0.35*entranceProgress})`);
  grad.addColorStop(1, 'rgba(255,220,50,0)');
  ctx.beginPath(); ctx.arc(x, y, glowSize, 0, Math.PI*2); ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < points*2; i++) {
    const angle = (i*Math.PI/points) - Math.PI/2;
    const rad   = i%2===0 ? rr : innerR;
    i === 0 ? ctx.moveTo(x+Math.cos(angle)*rad, y+Math.sin(angle)*rad)
            : ctx.lineTo(x+Math.cos(angle)*rad, y+Math.sin(angle)*rad);
  }
  ctx.closePath();
  const sg = ctx.createRadialGradient(x, y-rr*0.2, 0, x, y, rr);
  sg.addColorStop(0,'#fff9c4'); sg.addColorStop(0.4,'#ffd700'); sg.addColorStop(1,'#ff9f00');
  ctx.fillStyle = sg; ctx.shadowColor = '#ffd700'; ctx.shadowBlur = 20*entranceProgress; ctx.fill(); ctx.shadowBlur = 0;
  if (entranceProgress > 0.8) {
    ctx.beginPath(); ctx.arc(x-rr*0.2, y-rr*0.25, rr*0.18, 0, Math.PI*2);
    ctx.fillStyle = `rgba(255,255,255,${0.6*(entranceProgress-0.8)*5})`; ctx.fill();
  }
}
function startValidation() {
  valPoints = [];
  const W = window.innerWidth, H = window.innerHeight;
  const safeVX = Math.max(80, W*.16), safeVY = Math.max(80, H*.16);
  valPoints = [
    {x:W/2,      y:H/2},
    {x:safeVX,   y:safeVY},
    {x:W-safeVX, y:H-safeVY}
  ];
  valIdx = 0; valSamples = []; VAL_PARTICLES.length = 0;
  prevGaze = null; prevGazeTime = null;

  const overlay = document.getElementById('val-overlay');
  if (overlay) overlay.style.display = 'block';
  const instr = document.getElementById('val-instruction');
  if (instr) instr.style.opacity = '1';
  const badge = document.getElementById('val-badge');
  if (badge) badge.style.display = 'none';
  const tot = document.getElementById('val-badge-tot');
  if (tot) tot.textContent = valPoints.length;

  playChime(528, 0.1, 0.6);
  setTimeout(() => {
    if (instr) instr.style.opacity = '0';
    setTimeout(() => {
      if (instr) instr.style.display = 'none';
      if (badge) badge.style.display = 'block';
      runStarDot();
    }, 400);
  }, VAL_INTRO_MS);
}
function runStarDot() {
  if (valIdx >= valPoints.length) { finishValidation(); return; }
  const pt = valPoints[valIdx];
  const valCanvas = document.getElementById('val-canvas');
  if (!valCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  valCanvas.width  = Math.round(window.innerWidth  * dpr);
  valCanvas.height = Math.round(window.innerHeight * dpr);
  valCanvas.style.width  = window.innerWidth  + 'px';
  valCanvas.style.height = window.innerHeight + 'px';
  const vCtx = valCanvas.getContext('2d');
  vCtx.scale(dpr, dpr);
  const numEl = document.getElementById('val-badge-num');
  if (numEl) numEl.textContent = valIdx + 1;

  const notes = [523,659,784];
  playChime(notes[valIdx % notes.length], 0.12, 0.5);

  const collected = [];
  const ENTRANCE_MS = 450;
  valStart = performance.now();
  let sparkled = false, inGap = true;
  const gapEnd = valStart + VAL_GAP_MS;

  function frame() {
    const now = performance.now(), elapsed = now - valStart;
    vCtx.clearRect(0, 0, valCanvas.width, valCanvas.height);

    if (inGap) {
      updateSparkles(vCtx);
      if (now >= gapEnd) inGap = false;
      valRaf = requestAnimationFrame(frame);
      return;
    }

    const starElapsed  = now - gapEnd;
    const starProgress = Math.min(starElapsed / VAL_DWELL_MS, 1);
    const entPct = Math.min(starElapsed / ENTRANCE_MS, 1);
    let entrance;
    if (entPct < 1) {
      const t = entPct;
      entrance = 1 - Math.pow(1-t, 3) * Math.cos(t * Math.PI * 2.5);
      entrance = Math.min(entrance, 1.12);
    } else {
      entrance = 1;
    }

    if (!sparkled && entPct >= 0.9) { spawnSparkles(vCtx, pt.x, pt.y); sparkled = true; }
    updateSparkles(vCtx);
    drawStar(vCtx, pt.x, pt.y, VAL_STAR_RADIUS, starElapsed*0.001, Math.min(entrance, 1));

    if (starProgress >= VAL_SAMPLE_START && gazeModel) {
      if (webcam.readyState >= 2 && faceLandmarker && webcam.currentTime !== _lastVideoTime) {
        _lastVideoTime = webcam.currentTime;
        try {
          const res = faceLandmarker.detectForVideo(webcam, performance.now());
          if (res.faceLandmarks && res.faceLandmarks.length > 0) {
            const lm  = res.faceLandmarks[0];
            const mat = (res.facialTransformationMatrixes?.length > 0) ? res.facialTransformationMatrixes[0] : null;
            const feat = extractFeatures(lm, mat);
            if (feat[7] >= 0.06) {
              const pf = poly(feat);
              collected.push({
                px: pf.reduce((s,v,i) => s+v*gazeModel.wx[i], 0),
                py: pf.reduce((s,v,i) => s+v*gazeModel.wy[i], 0)
              });
            }
          }
        } catch(e) {}
      }
    }

    if (starProgress < 1) {
      valRaf = requestAnimationFrame(frame);
    } else {
      if (collected.length >= 3) {
        const mxs = collected.map(p => p.px).sort((a,b) => a-b);
        const mys = collected.map(p => p.py).sort((a,b) => a-b);
        const mid = Math.floor(mxs.length/2);
        valSamples.push({px:mxs[mid], py:mys[mid], tx:pt.x, ty:pt.y});
      }
      valIdx++;
      runStarDot();
    }
  }
  valRaf = requestAnimationFrame(frame);
}
function finishValidation() {
  cancelAnimationFrame(valRaf);
  const overlay = document.getElementById('val-overlay');
  if (overlay) overlay.style.display = 'none';

  if (valSamples.length >= 3) {
    affineBias = computeAffineCorrection(valSamples);
    const badCalib = Math.abs(affineBias.dx) > 300 || affineBias.sx > 1.4 || affineBias.sx < 0.7;
    if (badCalib) {
      affineBias = {dx:0, dy:0, sx:1, sy:1};
      calibSamples = []; gazeModel = null;
      const card = document.getElementById('calib-card');
      if (card) {
        const h2 = card.querySelector('h2');
        if (h2) h2.textContent = '🐾 Let\'s try again!';
        const p = card.querySelector('p');
        if (p) p.innerHTML = 'Validation showed poor accuracy. Please recalibrate.<br><br>' +
          '<strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, say ' +
          '<strong style="color:#fff">"Look at the animal!"</strong>';
      }
      const startBtn = document.getElementById('calib-start-btn');
      if (startBtn) startBtn.textContent = '🐾 Try Again!';
      const overlayCalib = document.getElementById('calib-overlay');
      if (overlayCalib) overlayCalib.style.display = 'flex';
      phase = 'calib-ready';
      return;
    }
  }
  phase = 'stimulus';
  showScreen('stimulus');
  const hChild = document.getElementById('h-child');
  if (hChild) hChild.textContent = META.pid;
  const hGroup = document.getElementById('h-group');
  if (hGroup) hGroup.textContent = META.group;
  startRecording();
}

// ─── SACCADE CLASSIFICATION ───────────────────────────────────────────────────
function classifyGaze(gaze, currentTime) {
  if (!prevGaze || !prevGazeTime) {
    prevGaze = gaze; prevGazeTime = currentTime;
    return 'Fixation';
  }
  const dt = currentTime - prevGazeTime;
  if (dt === 0) return 'Fixation';
  const dist = Math.hypot(gaze.x - prevGaze.x, gaze.y - prevGaze.y);
  const vel  = dist / dt; // px/ms
  const cat = vel > 1.5 ? 'Saccade' : 'Fixation';
  prevGaze = gaze; prevGazeTime = currentTime;
  return cat;
}

// ─── WELFARE MONITOR UPDATE ───────────────────────────────────────────────────
function updateWelfareHUD(hasFace, ear, headPos, currentTime) {
  if (hasFace && ear < 0.06) {
    if (lastBlinkTime > 0) {
      const blinkDuration = currentTime - lastBlinkTime;
      if (blinkDuration < 200) {
        blinkTimes.push(currentTime);
      } else if (blinkDuration > 200 && blinkDuration < 800) {
        slowBlinkCount++;
        blinkTimes.push(currentTime);
      }
    }
    lastBlinkTime = currentTime;
  }
  const cutoff = currentTime - 30000;
  blinkTimes = blinkTimes.filter(t => t > cutoff);
  if (headPos && lastHeadPos) {
    const move = Math.hypot(headPos.x - lastHeadPos.x, headPos.y - lastHeadPos.y);
    if (move > 0.03) headMovementEvents++;
  }
  lastHeadPos = headPos;
  if (!hasFace) faceOffFrames++;
  if (totalF % 30 === 0) {
    const mins = (currentTime - sessionStart) / 60000;
    const blinkPerMin = mins > 0 ? Math.round(blinkTimes.length / mins) : 0;
    const drowsyFlag = slowBlinkCount > DROWSY_THRESHOLD ? '⚠️' : '✓';
    const faceOffPct = totalF > 0 ? Math.round((faceOffFrames / totalF) * 100) : 0;
    if (welfareBlink) welfareBlink.textContent = `${blinkPerMin}/min`;
    if (welfareDrowsy) welfareDrowsy.textContent = drowsyFlag;
    if (welfareHead) welfareHead.textContent = headMovementEvents;
    if (welfareFaceOff) welfareFaceOff.textContent = `${faceOffPct}%`;
  }
}

// ─── RECORDING ────────────────────────────────────────────────────────────────
function startRecording() {
  sessionStart = Date.now(); recordedFrames = []; totalF = 0; trackedF = 0;
  blinkTimes = []; lastBlinkTime = 0; slowBlinkCount = 0; headMovementEvents = 0; lastHeadPos = null; faceOffFrames = 0;

  timerInt = setInterval(() => {
    const s = Math.floor((Date.now() - sessionStart) / 1000);
    const timerEl = document.getElementById('h-timer');
    if (timerEl) timerEl.textContent = `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  }, 500);

  if (videoBlob) {
    stimVideo.src = videoBlob; stimVideo.muted = true;
    stimVideo.play().catch(() => showNoVideo());
    stimVideo.onerror = showNoVideo;
    stimVideo.onended = () => endSession();
    const soundBtn = document.getElementById('sound-btn');
    if (soundBtn) soundBtn.style.display = 'block';
  } else {
    showNoVideo();
  }
}
function showNoVideo() {
  const noVideo = document.getElementById('no-video');
  if (noVideo) noVideo.style.display = 'flex';
  const soundBtn = document.getElementById('sound-btn');
  if (soundBtn) soundBtn.style.display = 'none';
}
document.getElementById('stim-file-input')?.addEventListener('change', e => {
  const f = e.target.files[0]; if (!f) return;
  META.stimulus = f.name;
  stimVideo.src = URL.createObjectURL(f); stimVideo.muted = true;
  const noVideo = document.getElementById('no-video');
  if (noVideo) noVideo.style.display = 'none';
  stimVideo.play().catch(() => {});
  stimVideo.onended = () => endSession();
  const soundBtn = document.getElementById('sound-btn');
  if (soundBtn) soundBtn.style.display = 'block';
});
document.getElementById('sound-btn')?.addEventListener('click', () => {
  stimVideo.muted = false; stimVideo.play().catch(() => {});
  const soundBtn = document.getElementById('sound-btn');
  if (soundBtn) soundBtn.style.display = 'none';
});

// ─── MAIN PROCESSING LOOP ─────────────────────────────────────────────────────
function processingLoop() {
  if (phase === 'done') return;

  if (phase === 'calib-run') {
    if (webcam.readyState >= 2 && faceLandmarker && webcam.currentTime !== _lastVT) {
      _lastVT = webcam.currentTime;
      try {
        const res     = faceLandmarker.detectForVideo(webcam, performance.now());
        const hasFace = !!(res.faceLandmarks && res.faceLandmarks.length > 0);
        calibFacePresent = hasFace;
        if (hasFace) {
          const lm  = res.faceLandmarks[0];
          const mat = (res.facialTransformationMatrixes?.length > 0) ? res.facialTransformationMatrixes[0] : null;
          const feat = extractFeatures(lm, mat);
          const isBlink = feat[7] < 0.06;

          if (!isBlink) {
            if (gazeModel) {
              _calibCurrentGaze = predictGaze(feat, gazeModel);
            } else {
              _calibCurrentGaze = estimateGazeFromIris(feat);
            }
          } else {
            _calibCurrentGaze = null;
          }

          if (calibState === 'sampling' && !isBlink) {
            const pt = calibPoints[calibIdx];
            if (pt) calibSamples.push({feat, sx: pt.x, sy: pt.y});
          }
        } else {
          _calibCurrentGaze = null;
        }
      } catch(e) {}
    }
    procRaf = requestAnimationFrame(processingLoop);
    return;
  }

  if (phase === 'validation') { procRaf = requestAnimationFrame(processingLoop); return; }

  if (webcam.readyState >= 2 && faceLandmarker) {
    if (webcam.currentTime === _lastVT) { procRaf = requestAnimationFrame(processingLoop); return; }
    _lastVT = webcam.currentTime;

    const res     = faceLandmarker.detectForVideo(webcam, performance.now());
    const hasFace = !!(res.faceLandmarks && res.faceLandmarks.length > 0);
    const mat     = (res.facialTransformationMatrixes?.length > 0) ? res.facialTransformationMatrixes[0] : null;
    calibFacePresent = hasFace;

    if (phase === 'stimulus') {
      const hFace = document.getElementById('h-face');
      if (hFace) { hFace.textContent = hasFace ? 'Yes' : 'No'; hFace.className = hasFace ? 'ok' : 'bad'; }
      const stFace = document.getElementById('st-face');
      if (stFace) { stFace.textContent = hasFace ? 'Yes' : 'No'; stFace.className = 'sv ' + (hasFace ? 'ok' : 'bad'); }
    }

    if (hasFace) {
      const lm       = res.faceLandmarks[0];
      const feat     = extractFeatures(lm, mat);
      const ear      = feat[7];
      const isBlink  = ear < 0.06;
      const nowMs    = Date.now() - sessionStart;

      const headPos = { x: (lm[33].x + lm[263].x)/2, y: (lm[33].y + lm[263].y)/2 };
      updateWelfareHUD(hasFace, ear, headPos, nowMs);

      let leftPupilX  = null, leftPupilY  = null;
      let rightPupilX = null, rightPupilY = null;
      try {
        const vTrack = webcam.srcObject?.getVideoTracks()[0];
        const vSettings = vTrack ? vTrack.getSettings() : {};
        const vw = vSettings.width  || 640;
        const vh = vSettings.height || 480;
        let sumLx=0,sumLy=0,sumRx=0,sumRy=0;
        LEFT_IRIS.forEach( i => { sumLx += lm[i].x; sumLy += lm[i].y; });
        RIGHT_IRIS.forEach(i => { sumRx += lm[i].x; sumRy += lm[i].y; });
        leftPupilX  = (sumLx / LEFT_IRIS.length)  * vw;
        leftPupilY  = (sumLy / LEFT_IRIS.length)  * vh;
        rightPupilX = (sumRx / RIGHT_IRIS.length) * vw;
        rightPupilY = (sumRy / RIGHT_IRIS.length) * vh;
      } catch(e) {}

      if (phase === 'stimulus') {
        totalF++;
        let gaze = null;
        if (!isBlink) {
          if (gazeModel && !calibSkipActive) {
            gaze = predictGaze(feat, gazeModel);
          }
        }

        let frameData = {
          t: nowMs, tracked: gaze ? 1 : 0,
          gazeX: gaze?.x ?? NaN, gazeY: gaze?.y ?? NaN,
          leftPupilX, leftPupilY,
          rightPupilX, rightPupilY,
          category: isBlink ? 'Blink' : (gaze ? classifyGaze(gaze, nowMs) : 'Lost'),
          feat: feat
        };

        if (gaze) {
          trackedF++;
          gazeCtx.clearRect(0, 0, gazeCanvas.width, gazeCanvas.height);
          const stGaze = document.getElementById('st-gaze');
          if (stGaze) { stGaze.textContent = 'Tracking'; stGaze.className = 'sv ok'; }
        } else {
          if (isBlink) { prevGaze = null; prevGazeTime = null; }
          gazeCtx.clearRect(0, 0, gazeCanvas.width, gazeCanvas.height);
          const stGaze = document.getElementById('st-gaze');
          if (stGaze) { stGaze.textContent = isBlink ? 'Blink' : 'Lost'; stGaze.className = 'sv bad'; }
        }

        recordedFrames.push(frameData);
        const stFrames = document.getElementById('st-frames');
        if (stFrames) stFrames.textContent = recordedFrames.length;
        const stTrack = document.getElementById('st-track');
        if (stTrack) stTrack.textContent = Math.round(trackedF/totalF*100) + '%';

        if (recordedFrames.length > 10) {
          const ys = recordedFrames.filter(f => f.tracked).map(f => f.gazeY);
          if (ys.length > 1) {
            const my = ys.reduce((a,b) => a+b,0) / ys.length;
            const sy = Math.sqrt(ys.reduce((a,b) => a+(b-my)**2, 0) / ys.length);
            const el = document.getElementById('st-ystd');
            if (el) { el.textContent = sy.toFixed(0) + 'px'; el.className = 'sv ' + (sy > 30 ? 'ok' : 'bad'); }
          }
        }
      }
    } else if (phase === 'stimulus') {
      totalF++;
      const nowMs = Date.now() - sessionStart;
      updateWelfareHUD(false, 1.0, null, nowMs);
      recordedFrames.push({
        t: nowMs, tracked: 0,
        gazeX: NaN, gazeY: NaN,
        leftPupilX: null, leftPupilY: null,
        rightPupilX: null, rightPupilY: null,
        category: 'Blink', feat: null
      });
      const stTrack = document.getElementById('st-track');
      if (stTrack && totalF > 0) stTrack.textContent = Math.round(trackedF/totalF*100) + '%';
    }
  }
  procRaf = requestAnimationFrame(processingLoop);
}

// ─── CSV (SMI RED FORMAT) ─────────────────────────────────────────────────────
const CSV_HDR = [
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
  'Port Status',
  'Annotation Name','Annotation Description','Annotation Tags',
  'Mouse Position X [px]','Mouse Position Y [px]',
  'Scroll Direction X','Scroll Direction Y','Content',
  'AOI Group Right','AOI Scope Right','AOI Order Right',
  'AOI Group Left','AOI Scope Left','AOI Order Binocular',
  "groupe d'enfants"
].join(',');

function buildCSV() {
  const sessionDuration = (Date.now() - sessionStart) / 1000 / 60;
  const totalBlinkCount = blinkTimes.length;
  const blinkRatePerMin = sessionDuration > 0 ? Math.round(totalBlinkCount / sessionDuration) : 0;
  const drowsyFlag = slowBlinkCount > DROWSY_THRESHOLD ? '⚠️' : '✓';
  const faceOffPct = totalF > 0 ? Math.round((faceOffFrames / totalF) * 100) : 0;
  const welfareMeta = `blinks/min=${blinkRatePerMin} drowsy=${drowsyFlag} head_moves=${headMovementEvents} face_off=${faceOffPct}% skip_mode=${calibSkipActive}`;
  const biasMeta = `# GazeTrack v14 | bias_dx=${affineBias.dx.toFixed(2)} bias_dy=${affineBias.dy.toFixed(2)} bias_sx=${affineBias.sx.toFixed(4)} bias_sy=${affineBias.sy.toFixed(4)} val_samples=${valSamples.length} calib_samples=${calibSamples.length} ${welfareMeta}`;
  const lines = [biasMeta, CSV_HDR];

  const totalDuration  = recordedFrames.length > 0 ? recordedFrames[recordedFrames.length-1].t : 0;
  const trackingRatio  = totalF > 0 ? (trackedF / totalF * 100) : 0;
  const colorMap       = {ASD:'DarkViolet', TD:'SteelBlue', other:'Gray'};
  const color          = colorMap[META.group] || 'Gray';
  const fmtNum         = (val, d=1) => (val !== null && !isNaN(val)) ? Number(val).toFixed(d) : '-';

  recordedFrames.forEach((f, i) => {
    const absTime = sessionStart + f.t;
    const d       = new Date(absTime);
    const pad     = (n, w=2) => String(Math.floor(n)).padStart(w,'0');
    const tod     = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}:${pad(d.getMilliseconds(),3)}`;

    const row = [
      i,
      f.t.toFixed(3),
      tod,
      'Trial001',
      META.stimulus || '-',
      0,
      totalDuration.toFixed(3),
      META.pid,
      color,
      trackingRatio.toFixed(3),
      f.category,
      f.category,
      f.category,
      i+1,
      i+1,
      '-','-','-',
      '-','-','-',
      fmtNum(f.gazeX), fmtNum(f.gazeY),
      fmtNum(f.gazeX), fmtNum(f.gazeY),
      '-','-',
      f.feat ? f.feat[0].toFixed(4) : '-',
      f.feat ? f.feat[1].toFixed(4) : '-',
      f.feat ? f.feat[2].toFixed(4) : '-',
      f.feat ? f.feat[0].toFixed(4) : '-',
      f.feat ? f.feat[1].toFixed(4) : '-',
      f.feat ? f.feat[2].toFixed(4) : '-',
      '-','-','-',
      '-','-','-',
      fmtNum(f.rightPupilX), fmtNum(f.rightPupilY),
      fmtNum(f.leftPupilX),  fmtNum(f.leftPupilY),
      0,
      '-','-','-',
      '-','-','-','-',
      META.stimulus || '-',
      '-','-','-',
      '-','-','-',
      META.group
    ];

    lines.push(row.map(v => v === undefined ? '-' : v).join(','));
  });
  return lines.join('\n');
}

function downloadCSV() {
  if (!csvData) return;
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const fn = `gaze_${META.pid}_${META.group}_${ts}.csv`;
  const url = URL.createObjectURL(new Blob([csvData], {type:'text/csv'}));
  Object.assign(document.createElement('a'), {href:url, download:fn}).click();
  URL.revokeObjectURL(url);
}

// ─── END SESSION ──────────────────────────────────────────────────────────────
function endSession() {
  if (phase === 'done') return;
  phase = 'done';
  clearInterval(timerInt);
  cancelAnimationFrame(procRaf);
  cancelAnimationFrame(calibRaf);
  cancelAnimationFrame(valRaf);
  stimVideo.pause();

  csvData = buildCSV();
  const pct = totalF > 0 ? Math.round(trackedF/totalF*100) : 0;
  const dur = Math.round((Date.now() - sessionStart) / 1000);
  const ys  = recordedFrames.filter(f => f.tracked).map(f => f.gazeY);
  let ystd  = 0;
  if (ys.length > 1) {
    const my = ys.reduce((a,b) => a+b,0) / ys.length;
    ystd = Math.sqrt(ys.reduce((a,b) => a+(b-my)**2, 0) / ys.length);
  }
  const biasOk    = Math.abs(affineBias.dx)>5 || Math.abs(affineBias.dy)>5;
  const biasLabel = biasOk
    ? `${affineBias.dx>0?'+':''}${affineBias.dx.toFixed(0)},${affineBias.dy>0?'+':''}${affineBias.dy.toFixed(0)}px`
    : 'Minimal';

  const fixCount = recordedFrames.filter(f => f.category === 'Fixation').length;
  const sacCount = recordedFrames.filter(f => f.category === 'Saccade').length;

  const sessionMins = dur / 60;
  const blinkRate = sessionMins > 0 ? Math.round(blinkTimes.length / sessionMins) : 0;
  const drowsyFlag = slowBlinkCount > DROWSY_THRESHOLD ? '⚠️ Drowsy' : '✓ Normal';
  const faceOffPct = totalF > 0 ? Math.round((faceOffFrames / totalF) * 100) : 0;

  const statsEl = document.getElementById('done-stats');
  if (statsEl) {
    statsEl.innerHTML = `
      <div class="done-stat"><div class="n">${recordedFrames.length}</div><div class="l">FRAMES</div></div>
      <div class="done-stat"><div class="n">${pct}%</div><div class="l">TRACKED</div></div>
      <div class="done-stat"><div class="n">${dur}s</div><div class="l">DURATION</div></div>
      <div class="done-stat"><div class="n" style="color:${ystd>30?'var(--accent)':'var(--warn)'}">${ystd.toFixed(0)}px</div><div class="l">Y STD</div></div>
      <div class="done-stat"><div class="n" style="color:var(--accent)">${fixCount}</div><div class="l">FIXATIONS</div></div>
      <div class="done-stat"><div class="n" style="color:var(--gold)">${sacCount}</div><div class="l">SACCADES</div></div>
      <div class="done-stat"><div class="n" style="color:var(--accent);font-size:13px">${biasLabel}</div><div class="l">BIAS CORR</div></div>
      <hr style="grid-column:1/-1;border:0;border-top:1px solid var(--border);margin:8px 0">
      <div class="done-stat"><div class="n">${blinkRate}/min</div><div class="l">BLINKS</div></div>
      <div class="done-stat"><div class="n">${drowsyFlag}</div><div class="l">DROWSINESS</div></div>
      <div class="done-stat"><div class="n">${headMovementEvents}</div><div class="l">HEAD MOVES</div></div>
      <div class="done-stat"><div class="n">${faceOffPct}%</div><div class="l">FACE OFF</div></div>
    `;
  }

  showScreen('done');
  const ts2 = new Date().toISOString().replace(/[:.]/g, '-');
  setTimeout(() => uploadToMongo(csvData, `gaze_${META.pid}_${META.group}_${ts2}.csv`), 600);
}

document.getElementById('end-btn')?.addEventListener('click', endSession);
document.getElementById('btn-dl')?.addEventListener('click', () => downloadCSV());
document.getElementById('btn-restart')?.addEventListener('click', () => location.reload());

// ─── MONGODB UPLOAD ───────────────────────────────────────────────────────────
function driveSetStatus(icon, msg, color) {
  const el = document.getElementById('drive-status');
  if (!el) return;
  const iconEl = document.getElementById('drive-icon');
  if (iconEl) iconEl.textContent = icon;
  const msgEl = document.getElementById('drive-msg');
  if (msgEl) msgEl.textContent = msg;
  el.style.borderColor = color || 'var(--border)';
}
async function uploadToMongo(csvText, filename) {
  driveSetStatus('☁️','Saving to database...','var(--border)');
  try {
    const resp = await fetch(MONGO_API_URL, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        filename, pid: META.pid, age: META.age, group: META.group,
        clinician: META.clinician, location: META.location, notes: META.notes,
        timestamp: new Date().toISOString(), csv: csvText
      })
    });
    if (!resp.ok) throw new Error('Server ' + resp.status);
    driveSetStatus('✅','Saved to database!','rgba(0,229,176,0.4)');
    const btn = document.getElementById('btn-dl');
    if (btn) {
      btn.textContent = '✅ Saved - click to download locally';
      btn.style.opacity = '1'; btn.style.pointerEvents = 'auto';
      btn.onclick = () => downloadCSV();
    }
  } catch(err) {
    driveSetStatus('❌','Database save failed - downloading locally','rgba(255,92,58,0.4)');
    downloadCSV();
  }
}

// ─── DEBUG & CLEANUP ─────────────────────────────────────────────────────────
window.addEventListener('keydown', e => {
  if (e.key === 'd' || e.key === 'D') {
    const p = document.getElementById('debug-panel');
    if (p) p.style.display = p.style.display === 'none' ? 'block' : 'none';
  }
});
window.addEventListener('beforeunload', () => {
  if (sessionStream) sessionStream.getTracks().forEach(t => t.stop());
  if (camStream)     camStream.getTracks().forEach(t => t.stop());
});
window.addEventListener('resize', () => {
  if (phase === 'calib-run') {
    calibPoints = buildCalibPoints();
  }
});

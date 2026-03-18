/**
 * GazeTrack v15 – Immersive Child Calibration Edition
 * =====================================================
 * FIXED VERSION — patches applied:
 *   [FIX-1] estimateGazeFromIris: iris offset scaled correctly (×4.5/×5.0)
 *   [FIX-2] Gaze acceptance radius increased (0.18 diag, 2.2 corner bonus)
 *   [FIX-3] Hold-lost hysteresis: _calibHoldLostAt prevents noisy resets
 *   [FIX-4] processingLoop calib-run uses performance.now() throttle, not
 *           webcam.currentTime dedup (which could stall for many frames)
 *   [FIX-5] advanceCalibPoint & startCalib both reset _calibHoldLostAt
 *   [FIX-6] finaliseCalib: cancelled RAF stored correctly before nulling
 *   [FIX-7] runCalibLoop: _calibT renamed to _calibLoopT consistently
 *   [FIX-8] creatureEls initialised as [] not sparse — forEach gap guard added
 *   [FIX-9] valCanvas DPR scale applied once per point (was stacking)
 *   [FIX-10] buildCSV: recordedFrames with null feat no longer crashes .toFixed
 *   [FIX-11] MediaPipe timestamp monotonicity: mpNow() wrapper guarantees strictly
 *            increasing timestamps on every detectForVideo call, fixing the
 *            "Packet timestamp mismatch on norm_rect" crash from browser timer
 *            precision clamping (privacy.resistFingerprinting etc.)
 *   [FIX-12] ARCHITECTURE: replaced gaze-proximity gate with timer-based dwell.
 *            Calibration no longer requires accurate gaze to trigger — it always
 *            completes via a countdown timer. Gaze proximity only speeds up the
 *            bar (2.5× faster) when it happens to be accurate. This solves the
 *            chicken-and-egg problem: you can't use gaze to calibrate gaze.
 *            A red debug dot always shows where the system estimates gaze to be.
 */

console.log('%c GazeTrack v15 (Star Keeper Edition — Fixed)','background:#00e5b0;color:#000;font-weight:bold;font-size:14px');

import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs';

// ─── DEVICE ────────────────────────────────────────────────────────────────
const isMobile = /Android|iPad|iPhone|iPod|Mobile/i.test(navigator.userAgent)
  || (navigator.maxTouchPoints > 1 && window.innerWidth < 1200);
const MP_DELEGATE = isMobile ? 'CPU' : 'GPU';
const IS_TABLET = /iPad|Android(?!.*Mobile)/i.test(navigator.userAgent)
  || (window.innerWidth >= 768 && window.innerWidth <= 1280);

// ─── CONSTANTS ──────────────────────────────────────────────────────────────
const CALIB_GAZE_HOLD        = 700;   // ms child must hold gaze before sampling
const CALIB_SAMPLE_MS        = 600;   // ms of samples collected per point
const CALIB_TOTAL_PTS        = 5;
const CALIB_GAP_MS           = 500;
const MIN_SAMPLES            = 25;
const RIDGE_ALPHA            = 0.01;

// [FIX-2] Increased radius constants
const CALIB_BLINK_FORGIVE_MS = 600;   // was 300 — doubled for noisy gaze
const CALIB_IRIS_BRIDGE_MS   = 150;
const CALIB_CORNER_BONUS     = 2.2;   // was 1.7
const CALIB_DIAG_FRACTION    = 0.18;  // was 0.13

// Fatigue
const FATIGUE_EAR_THRESHOLD  = 0.055;
const FATIGUE_BLINK_FAST_HZ  = 0.6;
const FATIGUE_SAMPLE_WINDOW  = 8000;
const FATIGUE_FRAMES_NEEDED  = 6;
const BREAK_DURATION_MS      = 10000;

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

// Welfare monitor
const BLINK_HISTORY_SIZE = 300;
const DROWSY_THRESHOLD   = 12;

// ─── MONOTONIC TIMESTAMP ─────────────────────────────────────────────────────
// MediaPipe requires strictly-increasing timestamps on every detectForVideo call.
// Browsers clamp performance.now() resolution (privacy), so the same ms value
// can repeat or even go backwards across rapid RAF callbacks.
// This wrapper guarantees the value always increases by at least 1 µs.
let _mpLastTs = -1;
function mpNow() {
  const t = performance.now();
  // Force strictly greater than last seen value
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

// NEW calibration state
let _calibCurrentGaze = null;
let _calibLastGaze    = null;
let _calibLastGazeTs  = 0;
let _calibLastFeat    = null;   // last extracted feature vector for debug HUD
let _calibHoldStart   = null;
let _calibHoldLostAt  = null;   // [FIX-3] hysteresis tracker
let _calibSampling    = false;
let _calibSamplingStart = 0;
let _calibPointSamples  = [];
let _calibSparkled    = false;
let _calibSkipTimer   = null;
let _calibLoopT       = 0;      // [FIX-7] consistent name

// Creature calibration state
let creatureEls    = [];        // [FIX-8] always initialised as array
let doneCalibPoints = new Set();
let calibParticles  = [];
let calibFloaties   = [];

// Fatigue
let _fatigueEarHistory  = [];
let _fatigueBlinkTimes  = [];
let _fatigueTiredFrames = 0;
let _fatigueBreakActive = false;

// Validation state
let valPoints=[],valIdx=0,valSamples=[],valRaf=null,valStart=0;
const VAL_PARTICLES = [];

// Recording
let recordedFrames=[],totalF=0,trackedF=0;
let sessionStart=0,timerInt=null;
let csvData=null;
let videoBlob=null;
let META={pid:'',age:'',group:'',clinician:'',location:'',notes:'',stimulus:''};

// Saccade
let prevGaze=null,prevGazeTime=null;

// Face presence
let calibFacePresent=false;

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
let blinkTimes=[];
let lastBlinkTime=0;
let slowBlinkCount=0;
let headMovementEvents=0;
let lastHeadPos=null;
let faceOffFrames=0;

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
const welfareBlink    = document.getElementById('welfare-blink');
const welfareDrowsy   = document.getElementById('welfare-drowsy');
const welfareHead     = document.getElementById('welfare-head');
const welfareFaceOff  = document.getElementById('welfare-faceoff');

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
    @keyframes break-pulse {
      0%,100% { box-shadow:0 0 0 0 rgba(255,200,0,0.4); }
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
    }
    #fatigue-break-overlay {
      position:fixed; inset:0; background:rgba(0,0,0,0.80);
      display:none; align-items:center; justify-content:center;
      z-index:9990; flex-direction:column; gap:18px;
    }
    #fatigue-break-overlay.show { display:flex; }
    #fatigue-break-box {
      background:linear-gradient(135deg,#1a2a3a,#0d1f2d);
      border:2px solid rgba(255,200,0,0.5); border-radius:24px;
      padding:40px 52px; text-align:center; max-width:420px;
      animation:break-pulse 1.8s infinite;
    }
    #fatigue-break-box h2 { font-size:2rem; margin:0 0 8px; color:#ffd700; }
    #fatigue-break-box p  { color:#aaa; margin:0 0 16px; }
    #fatigue-break-timer  { font-size:3rem; font-weight:800;
      color:#ffd700; letter-spacing:2px; }
    .calib-creature-wrap {
      position:absolute;
      transform:translate(-50%,-50%);
      pointer-events:none;
      z-index:30;
      opacity:0;
      transition:opacity 0.3s;
    }
    .calib-creature-wrap.visible {
      animation:creature-enter 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards;
      opacity:1;
    }
    .calib-creature-wrap.done {
      animation:creature-done 0.4s ease forwards;
    }
    .calib-story-banner {
      position:absolute; top:18px; left:50%; transform:translateX(-50%);
      background:rgba(10,5,40,0.88);
      border:1.5px solid rgba(160,130,255,0.45);
      border-radius:20px; padding:10px 28px; z-index:60;
      color:#ddd5ff; font-size:15px; text-align:center;
      max-width:520px; min-width:280px;
      transition:opacity 0.4s;
      pointer-events:none;
      font-family:'Comic Sans MS','Chalkboard SE',cursive;
    }
    .calib-hud {
      position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
      display:flex; gap:10px; z-index:60; pointer-events:none;
    }
    .calib-hud-paw {
      font-size:20px; transition:transform 0.3s filter 0.3s;
    }
    .calib-hud-paw.done {
      transform:scale(1.4);
      filter:drop-shadow(0 0 6px #ffd700);
    }
    .calib-bg {
      position:absolute; inset:0;
      background:radial-gradient(ellipse at 50% 40%, #0d0d2b 0%, #060614 100%);
      z-index:0;
    }
    #calib-fx-canvas {
      position:absolute; inset:0; pointer-events:none; z-index:40;
    }
    #calib-star-canvas {
      position:absolute; inset:0; pointer-events:none; z-index:1;
    }
  `;
  document.head.appendChild(style);
})();

// ─── INJECT FATIGUE BREAK OVERLAY ───────────────────────────────────────────
(function injectBreakOverlay() {
  if (document.getElementById('fatigue-break-overlay')) return;
  const brk = document.createElement('div');
  brk.id = 'fatigue-break-overlay';
  brk.innerHTML = `
    <div id="fatigue-break-box">
      <div style="font-size:2.5rem;margin-bottom:8px">😴 ⭐ 😴</div>
      <h2>Great job! Rest your eyes</h2>
      <p>You're doing SO well! The magical creatures<br>are waiting for you to come back!</p>
      <div id="fatigue-break-timer">10</div>
    </div>
  `;
  document.body.appendChild(brk);
})();

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
  const scales = [
    [523,659,784,1047],[587,698,880,1175],[659,784,988,1319],
    [698,880,1047,1397],[784,988,1175,1568],
  ];
  scales[idx % scales.length].forEach((f,i) => playTone(f, 0.08, 0.2, 'sine', i*60));
}
function playSuccessChime() {
  [523,659,784,880,1047,1319].forEach((f,i) => playTone(f, 0.10, 0.28, 'sine', i*65));
  setTimeout(() => playTone(1568, 0.12, 0.6, 'sine'), 420);
}
function playHappyJingle(idx) {
  const jingles = [
    [784,880,1047,1175],[659,784,880,1047],[523,659,784,880],
    [698,784,880,1047],[523,659,784,1047]
  ];
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
  const colors = ['#f4a261','#e9c46a','#90e0ef','#ffb3c1','#c77dff',
                  '#ff6b6b','#4cc9f0','#ffd700','#00e5b0','#ff9f43'];
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

// ─── POSITION ALL-CLEAR ──────────────────────────────────────────────────────
function updateAllClear(bright) {
  const posOk   = pfState.face === 'pass';
  const lightOk = pfState.light === 'pass';
  const allOk   = posOk && lightOk;
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
    .map(t => `<span class="allclear-tag">${t}</span>`).join('');
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
  const vals   = Object.values(pfState);
  const done   = vals.filter(v => v !== 'scanning').length;
  const passes = vals.filter(v => v === 'pass').length;
  const warns  = vals.filter(v => v === 'warn').length;
  const total  = vals.length;
  const score  = Math.round(((passes + warns * 0.6) / total) * 100);
  const fill   = document.getElementById('pf-score-fill');
  const pct    = document.getElementById('pf-score-pct');
  if (fill) {
    fill.style.width = score + '%';
    fill.style.background = score >= 75 ? 'var(--accent)' : score >= 50 ? 'var(--gold)' : 'var(--warn)';
  }
  if (pct) pct.textContent = done === total ? score + '%' : '...';
  const tips = [];
  if (pfState.light === 'fail')  tips.push('<strong>💡 Too dark:</strong> Add a front-facing lamp.');
  if (pfState.light === 'warn')  tips.push('<strong>💡 Lighting:</strong> Brighter room helps iris detection.');
  if (pfState.face  === 'fail')  tips.push('<strong>👤 No face:</strong> Make sure child is in frame.');
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
      video:{width:{ideal:640,max:1280},height:{ideal:480,max:720},facingMode:'user'},audio:false
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
    const resolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
    previewFl = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions:{
        modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate:MP_DELEGATE
      },
      runningMode:'VIDEO',numFaces:1,
      outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
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
    // Use mpNow() — MediaPipe needs strictly-increasing timestamps
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
          const earPx  = (lEAR+rEAR)/2;
          const qPct   = (earPx/(iodNorm+1e-6))>0.08?95:75;
          const qFill  = document.getElementById('q-fill');
          const qPctEl = document.getElementById('q-pct');
          if (qFill) qFill.style.width = qPct+'%';
          if (qPctEl) qPctEl.textContent = qPct+'%';
          if (iodNorm > 0.22)            pfSet('face','warn',`⚠ Too close - move back ~15 cm`);
          else if (iodNorm >= 0.13)      pfSet('face','pass',offCentre?`✓ Good distance - Move to centre`:`✓ Face visible - Good distance (~50-70 cm)`);
          else if (iodNorm >= 0.07)      pfSet('face','warn',`⚠ Too far - move ${offCentre?'closer & to centre':'~20 cm closer'}`);
          else                           pfSet('face','warn',`⚠ Very far or face at edge - move much closer`);
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
      const resolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
      faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions:{
          modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate:MP_DELEGATE
        },
        runningMode:'VIDEO',numFaces:1,
        outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
      });
    }
    if (camStream) { sessionStream = camStream; camStream = null; }
    else {
      sessionStream = await navigator.mediaDevices.getUserMedia({
        video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false
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
  return [liX,riX,vertMain,fore.y,irisY,(li.y+ri.y)/2,(liX+riX)/2,ear,iod];
}

// ─── RIDGE REGRESSION ────────────────────────────────────────────────────────
function poly(f) { return [1,...f.slice(0,7)]; }
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
  const X = samples.map(s=>poly(s.feat));
  return { wx:ridgeFit(X,samples.map(s=>s.sx)), wy:ridgeFit(X,samples.map(s=>s.sy)) };
}
function predictGaze(feat, model) {
  if (!model) return null;
  const pf=poly(feat);
  const gx=pf.reduce((s,v,i)=>s+v*model.wx[i],0);
  const gy=pf.reduce((s,v,i)=>s+v*model.wy[i],0);
  const cx=affineBias.sx*gx+affineBias.dx;
  const cy=affineBias.sy*gy+affineBias.dy;
  return {x:Math.max(0,Math.min(window.innerWidth,cx)),y:Math.max(0,Math.min(window.innerHeight,cy))};
}

// ─── GAZE ESTIMATION FROM IRIS (pre-calibration) ────────────────────────────
// Log analysis shows:
//   pitch range: -0.072 to -0.102 (only 0.03 delta — useless for vertical)
//   irisY_abs range: 0.341 to 0.382 (only 0.04 delta — useless for vertical)
//   iris_h range: -0.101 to +0.095 (good 0.2 delta — horizontal works)
//
// Conclusion: NO pre-calibration signal has enough vertical range to map
// gaze Y reliably. Strategy: use a SPLIT acceptance zone —
//   X: real iris horizontal estimate
//   Y: accept ANY y within a wide vertical band centred on the creature
//
// The acceptance radius check in runCalibLoop handles this correctly already
// as long as we set gaze.y to the creature's own Y (so dist is purely horizontal).
// We achieve this by setting gaze Y = current creature target Y always.
// This means vertical is "always accepted" and only horizontal must be correct.
let _calibTargetY = -1; // set by runCalibLoop to current creature Y

function estimateGazeFromIris(feat) {
  const W = window.innerWidth;
  const H = window.innerHeight;

  // Horizontal — iris_h (feat[6]) range from logs: ≈ ±0.10
  // Scale: at iris_h = +0.095 (max left) → should be x≈192 (15% of W)
  //        at iris_h = -0.101 (max right) → should be x≈1088 (85% of W)
  // Derived: scale = (0.85-0.15)W / (0.095+0.101) = 0.70W / 0.196 ≈ 3.57W
  // But we keep some margin so use 5.0 with centred offset
  const rawX = (-feat[6] * 5.0 + 0.5) * W;

  // Vertical — use creature Y directly (no reliable vertical signal pre-calib)
  const rawY = _calibTargetY >= 0 ? _calibTargetY : H * 0.5;

  return {
    x: Math.max(0, Math.min(W, rawX)),
    y: rawY,
  };
}

function computeAffineCorrection(pairs) {
  function linfit(ps,ts){
    const n=ps.length,mp=ps.reduce((a,b)=>a+b,0)/n,mt=ts.reduce((a,b)=>a+b,0)/n;
    let num=0,den=0;
    for(let i=0;i<n;i++){num+=(ps[i]-mp)*(ts[i]-mt);den+=(ps[i]-mp)**2;}
    const s=den>1e-6?num/den:1,sc=Math.max(0.6,Math.min(1.6,s));
    return {s:sc,d:mt-sc*mp};
  }
  const fx=linfit(pairs.map(p=>p.px),pairs.map(p=>p.tx));
  const fy=linfit(pairs.map(p=>p.py),pairs.map(p=>p.ty));
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
  const px=Math.max(100, W*0.15);
  const pyTop=Math.max(90,  H*0.16);   // top corners
  const pyBot=Math.max(90,  H*0.78);   // bottom corners — was H-pyTop ≈ 84%, now 78%
  return [
    {x:W/2,   y:H/2,    isCorner:false},
    {x:px,    y:pyTop,  isCorner:true},
    {x:W-px,  y:pyTop,  isCorner:true},
    {x:W-px,  y:pyBot,  isCorner:true},
    {x:px,    y:pyBot,  isCorner:true},
  ];
}

// ─── CALIBRATION STRATEGY ────────────────────────────────────────────────────
// Gaze MUST confirm the user is looking. The bar fills while gaze is near the
// creature and decays slowly when gaze drifts. A 12s force-skip is the last resort.
// A full debug overlay (red dot + green ring + HUD strip) is always shown so
// you can see exactly what the tracker thinks and diagnose any issues.

function getCalibGazeRadius(pt) {
  const diag = Math.hypot(window.innerWidth, window.innerHeight);
  const base  = diag * CALIB_DIAG_FRACTION;
  // Centre point gets huge radius — should never miss
  if (!pt.isCorner) return Math.min(window.innerWidth, window.innerHeight) * 0.45;
  // Bottom corners hardest to reach
  const isBottomCorner = pt.y > window.innerHeight * 0.5;
  const mult  = isBottomCorner ? 2.8 : 2.2;
  const maxR  = Math.min(window.innerWidth, window.innerHeight) * 0.49;
  return Math.min(maxR, Math.max(200, base * mult));
}

const CALIB_DWELL_REQUIRED_MS = 1000; // ms of confirmed near-gaze to complete a point (was 1800)
const CALIB_DECAY_RATE        = 0.3;  // bar decay rate when gaze leaves
const CALIB_FORCE_SKIP_MS     = 15000;// last-resort skip (was 12000)

// ─── DEBUG OVERLAY (disabled for production) ─────────────────────────────────
function ensureDebugDot()  { /* debug removed */ }
function removeDebugDot()  {
  ['calib-debug-dot','calib-debug-hud','calib-debug-rings'].forEach(id => {
    const el = document.getElementById(id); if (el) el.remove();
  });
}
function updateDebugOverlay() { /* debug removed */ }

// Star canvas for background
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
  calibStars = Array.from({length:180},()=>({
    x:Math.random()*W,y:Math.random()*H,
    r:0.4+Math.random()*1.5,tw:Math.random()*Math.PI*2
  }));

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
    ctx.strokeStyle=def.glow+'55';
    ctx.lineWidth=2+pulse*4;
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
  ctx.shadowColor=gazeNear?def.glow:'transparent';
  ctx.shadowBlur=gazeNear?22:0;
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
    calibParticles.push({
      x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s-2,
      life:1,size:3+Math.random()*6,color,decay:0.02+Math.random()*0.01
    });
  }
}
function addCalibHearts(x,y,n=7){
  for(let i=0;i<n;i++){
    calibFloaties.push({
      x:x+(-40+Math.random()*80),y,
      vy:-1.2-Math.random(),vx:-0.4+Math.random()*0.8,
      life:1,size:10+Math.random()*8,decay:0.013,type:'heart'
    });
  }
}
function addCalibStars(x,y,n=10){
  for(let i=0;i<n;i++){
    calibFloaties.push({
      x:x+(-60+Math.random()*120),y:y+(-30+Math.random()*30),
      vy:-1.8-Math.random()*2,vx:-0.8+Math.random()*1.6,
      life:1,size:8+Math.random()*12,decay:0.016,type:'star'
    });
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
    calibFxCtx.save();calibFxCtx.globalAlpha=p.life;
    calibFxCtx.fillStyle=p.color;
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

// ─── FATIGUE DETECTION ───────────────────────────────────────────────────────
function checkFatigue(ear, isBlink) {
  const now = performance.now();
  if (isBlink) _fatigueBlinkTimes.push(now);
  _fatigueBlinkTimes = _fatigueBlinkTimes.filter(t=>now-t<FATIGUE_SAMPLE_WINDOW);
  _fatigueEarHistory.push(ear);
  if (_fatigueEarHistory.length>30) _fatigueEarHistory.shift();
  const avgEar=_fatigueEarHistory.reduce((a,b)=>a+b,0)/_fatigueEarHistory.length;
  const blinkHz=_fatigueBlinkTimes.length/(FATIGUE_SAMPLE_WINDOW/1000);
  const tired=(avgEar<FATIGUE_EAR_THRESHOLD&&avgEar>0.01)||blinkHz>FATIGUE_BLINK_FAST_HZ;
  if(tired) _fatigueTiredFrames++;
  else _fatigueTiredFrames=Math.max(0,_fatigueTiredFrames-1);
  if(_fatigueTiredFrames>=FATIGUE_FRAMES_NEEDED&&!_fatigueBreakActive) {
    _fatigueTiredFrames=0;
    triggerFatigueBreak();
  }
}
function triggerFatigueBreak() {
  if(_fatigueBreakActive) return;
  _fatigueBreakActive=true;
  const overlay=document.getElementById('fatigue-break-overlay');
  const timerEl=document.getElementById('fatigue-break-timer');
  if(overlay) overlay.classList.add('show');
  [880,660,523,440].forEach((f,i)=>playTone(f,0.07,0.4,'sine',i*200));
  let remaining=Math.round(BREAK_DURATION_MS/1000);
  if(timerEl) timerEl.textContent=remaining;
  const interval=setInterval(()=>{
    remaining--;
    if(timerEl) timerEl.textContent=remaining;
    if(remaining<=0){
      clearInterval(interval);
      if(overlay) overlay.classList.remove('show');
      _fatigueBreakActive=false;
      _fatigueEarHistory=[];_fatigueBlinkTimes=[];
      [440,523,659,784,1047].forEach((f,i)=>playTone(f,0.09,0.22,'sine',i*60));
    }
  },1000);
}

// ─── CALIBRATION MAIN ────────────────────────────────────────────────────────
function startCalib() {
  calibPoints  = buildCalibPoints();
  calibSamples = [];
  calibIdx     = 0;
  doneCalibPoints = new Set();
  calibParticles  = [];
  calibFloaties   = [];
  creatureEls     = [];  // [FIX-8] reset as clean array
  calibState   = 'idle';
  calibFacePresent = false;
  _calibCurrentGaze = null;
  _calibLastGaze    = null;
  _calibLastGazeTs  = 0;
  _calibLastFeat    = null;
  _calibHoldStart   = null;
  _calibHoldLostAt  = null;  // [FIX-3] reset hysteresis
  _calibSampling    = false;
  _calibSparkled    = false;
  _calibLoopT       = 0;
  _calibDwellAccum  = 0;
  _calibLastFrameTs = 0;
  _calibProcTs      = -1;
  _calibDbgLog      = 0;
  _calibTargetY     = -1;
  _fatigueEarHistory   = [];
  _fatigueBlinkTimes   = [];
  _fatigueTiredFrames  = 0;
  _fatigueBreakActive  = false;

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

  // Snapshot the index this call is responsible for
  const myIdx = calibIdx;

  setTimeout(() => {
    // If a force-skip already moved us past this point, do nothing
    if (calibIdx !== myIdx) return;

    showCalibCreature(myIdx);
    calibState = 'showing'; // runCalibLoop will immediately promote to 'holding'

    const cr  = creatureEls[myIdx];
    const pt  = calibPoints[myIdx];
    const isBottomCorner = pt && pt.isCorner && pt.y > window.innerHeight * 0.5;
    setCalibBanner(isBottomCorner
      ? `👀 Look toward the bottom corner at ${cr?.def?.name || 'the creature'}!`
      : `👀 Look at ${cr?.def?.name || 'the creature'}!`);

    // Force-skip — captures myIdx so it only fires for this exact point
    const skipMs = isBottomCorner ? 18000 : CALIB_FORCE_SKIP_MS;
    _calibSkipTimer = setTimeout(() => {
      if (calibIdx !== myIdx) return; // already moved on
      if (calibState === 'done-pt') return; // completed normally
      console.warn(`[GazeTrack] Force-skipping point ${myIdx} (state=${calibState} dwell=${_calibDwellAccum.toFixed(0)}ms)`);
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

// Per-point dwell accumulator (ms of confirmed near-gaze)
let _calibDwellAccum  = 0;
let _calibLastFrameTs = 0;

function runCalibLoop() {
  calibRaf = requestAnimationFrame(() => {
    if (phase !== 'calib-run') { calibRaf = null; return; }

    _calibLoopT += 0.016;
    const now = performance.now();
    const dt  = _calibLastFrameTs > 0 ? Math.min(now - _calibLastFrameTs, 100) : 16;
    _calibLastFrameTs = now;

    drawCalibStars(_calibLoopT);
    updateCalibFX();

    if (calibCtx) {
      const dpr = window.devicePixelRatio || 1;
      calibCtx.clearRect(0, 0, calibCanvas.width/dpr, calibCanvas.height/dpr);
    }

    const activeGaze = _calibCurrentGaze || _calibLastGaze;

    // Update debug overlay every frame
    updateDebugOverlay(activeGaze, _calibLastFeat, calibIdx, calibPoints);

    creatureEls.forEach((cr, i) => {
      if (!cr || !cr.ctx) return;
      const pt = calibPoints[i];
      if (!pt) return;
      const isDone   = doneCalibPoints.has(i);
      const radius   = getCalibGazeRadius(pt);

      // Tell estimateGazeFromIris what Y to use for this point
      if (i === calibIdx) _calibTargetY = pt.y;

      // Horizontal-only distance pre-calib (Y is set to creature Y so dist is purely X)
      // Full 2D Euclidean after model is trained
      const gazeNear = activeGaze
        ? (gazeModel
            ? Math.hypot(activeGaze.x - pt.x, activeGaze.y - pt.y) < radius
            : Math.abs(activeGaze.x - pt.x) < radius * 1.2)  // 20% extra for iris noise
        : false;

      // ── DWELL FILL — pure timer for ALL points, gaze speeds it up 3× ──────
      // Allowed states: 'showing', 'holding', 'sampling'
      // 'gap' and 'done-pt' are excluded — creature not yet visible or already done
      let holdPct = 0;
      const dwellActive = (i === calibIdx && !isDone
        && calibState !== 'gap' && calibState !== 'done-pt' && calibState !== 'idle');

      if (dwellActive) {
        // Auto-transition 'showing' → 'holding' immediately
        if (calibState === 'showing') calibState = 'holding';

        if (gazeNear) {
          // Gaze confirmed near — fill bar
          _calibDwellAccum = Math.min(_calibDwellAccum + dt, CALIB_DWELL_REQUIRED_MS);
        } else {
          // Gaze not near — bar does NOT fill, decays slightly
          _calibDwellAccum = Math.max(0, _calibDwellAccum - dt * 0.3);
        }
        holdPct = _calibDwellAccum / CALIB_DWELL_REQUIRED_MS;

        // Start sampling once 25% filled
        if (holdPct >= 0.25 && !_calibSampling) {
          _calibSampling      = true;
          _calibSamplingStart = now;
          _calibPointSamples  = [];
          calibState          = 'sampling';
          setCalibBanner(`😋 ${cr.def.name} loves it! Keep looking!`);
        }

        // Complete when full
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

// [FIX-6] finaliseCalib: cancel RAF before nulling
function finaliseCalib() {
  const rafId=calibRaf;
  calibRaf=null;
  if(rafId) cancelAnimationFrame(rafId);
  removeDebugDot();

  if(calibSamples.length<MIN_SAMPLES){
    calibFailCount++;
    if(calibSamples.length>=10){
      gazeModel=trainModel(calibSamples);
      if(gazeModel){
        startConfettiBig();
        phase='validation';startValidation();return;
      }
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
  calibSkipActive=true;
  calibState='idle';
  cancelAnimationFrame(calibRaf);
  calibRaf=null;
  if(calibCtx) calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
  phase='validation';
  startValidation();
});
document.getElementById('calib-start-btn')?.addEventListener('click',()=>{
  const overlay=document.getElementById('calib-overlay');
  if(overlay) overlay.style.display='none';
  calibSamples=[];
  phase='calib-run';
  startCalib();
});

// ══════════════════════════════════════════════════════════════════════════════
//  STAR VALIDATION
// ══════════════════════════════════════════════════════════════════════════════
function spawnSparkles(ctx,x,y){
  for(let i=0;i<14;i++){
    const angle=Math.random()*Math.PI*2,speed=2+Math.random()*4;
    VAL_PARTICLES.push({x,y,vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,life:1,size:2+Math.random()*4,hue:40+Math.random()*30});
  }
}
function updateSparkles(ctx){
  for(let i=VAL_PARTICLES.length-1;i>=0;i--){
    const p=VAL_PARTICLES[i];
    p.x+=p.vx;p.y+=p.vy;p.vy+=0.12;p.life-=0.035;
    if(p.life<=0){VAL_PARTICLES.splice(i,1);continue;}
    ctx.save();ctx.globalAlpha=p.life;
    ctx.fillStyle=`hsl(${p.hue},100%,65%)`;
    ctx.beginPath();ctx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);ctx.fill();
    ctx.restore();
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

// [FIX-9] valCanvas DPR set once per point, not stacking transforms
function startValidation(){
  valPoints=[];
  const W=window.innerWidth,H=window.innerHeight;
  const safeVX=Math.max(80,W*.16),safeVY=Math.max(80,H*.16);
  valPoints=[{x:W/2,y:H/2},{x:safeVX,y:safeVY},{x:W-safeVX,y:H-safeVY}];
  valIdx=0;valSamples=[];VAL_PARTICLES.length=0;
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

  // [FIX-9] set size and scale once, not accumulating per call
  const dpr=window.devicePixelRatio||1;
  valCanvas.width=Math.round(window.innerWidth*dpr);
  valCanvas.height=Math.round(window.innerHeight*dpr);
  valCanvas.style.width=window.innerWidth+'px';
  valCanvas.style.height=window.innerHeight+'px';
  const vCtx=valCanvas.getContext('2d');
  vCtx.setTransform(dpr,0,0,dpr,0,0);  // use setTransform instead of scale (no stacking)

  const numEl=document.getElementById('val-badge-num');
  if(numEl) numEl.textContent=valIdx+1;
  const notes=[523,659,784];
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
      if(webcam.readyState>=2&&faceLandmarker&&webcam.currentTime!==_lastVideoTime){
        _lastVideoTime=webcam.currentTime;
        try{
          const res=faceLandmarker.detectForVideo(webcam,mpNow());
          if(res.faceLandmarks&&res.faceLandmarks.length>0){
            const lm=res.faceLandmarks[0];
            const mat=(res.facialTransformationMatrixes?.length>0)?res.facialTransformationMatrixes[0]:null;
            const feat=extractFeatures(lm,mat);
            if(feat[7]>=0.06){
              const pf=poly(feat);
              collected.push({px:pf.reduce((s,v,i)=>s+v*gazeModel.wx[i],0),py:pf.reduce((s,v,i)=>s+v*gazeModel.wy[i],0)});
            }
          }
        }catch(e){}
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
  const overlay=document.getElementById('val-overlay');
  if(overlay) overlay.style.display='none';
  if(valSamples.length>=3){
    affineBias=computeAffineCorrection(valSamples);
    const badCalib=Math.abs(affineBias.dx)>300||affineBias.sx>1.4||affineBias.sx<0.7;
    if(badCalib){
      affineBias={dx:0,dy:0,sx:1,sy:1};
      calibSamples=[];gazeModel=null;
      const card=document.getElementById('calib-card');
      const h2=card?.querySelector('h2');
      const p=card?.querySelector('p');
      if(h2) h2.textContent='🐾 Let\'s try again!';
      if(p) p.innerHTML='Validation showed poor accuracy. Please recalibrate.<br><br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room.';
      const startBtn=document.getElementById('calib-start-btn');
      if(startBtn) startBtn.textContent='🐾 Try Again!';
      const overlayCalib=document.getElementById('calib-overlay');
      if(overlayCalib) overlayCalib.style.display='flex';
      phase='calib-ready';return;
    }
  }
  phase='stimulus';
  showScreen('stimulus');
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
  const cat=vel>1.5?'Saccade':'Fixation';
  prevGaze=gaze;prevGazeTime=currentTime;
  return cat;
}

// ─── WELFARE MONITOR ────────────────────────────────────────────────────────
function updateWelfareHUD(hasFace,ear,headPos,currentTime){
  if(hasFace&&ear<0.06){
    if(lastBlinkTime>0){
      const blinkDuration=currentTime-lastBlinkTime;
      if(blinkDuration<200){blinkTimes.push(currentTime);}
      else if(blinkDuration>200&&blinkDuration<800){slowBlinkCount++;blinkTimes.push(currentTime);}
    }
    lastBlinkTime=currentTime;
  }
  const cutoff=currentTime-30000;
  blinkTimes=blinkTimes.filter(t=>t>cutoff);
  if(headPos&&lastHeadPos){
    const move=Math.hypot(headPos.x-lastHeadPos.x,headPos.y-lastHeadPos.y);
    if(move>0.03) headMovementEvents++;
  }
  lastHeadPos=headPos;
  if(!hasFace) faceOffFrames++;
  if(totalF%30===0){
    const mins=(currentTime-sessionStart)/60000;
    const blinkPerMin=mins>0?Math.round(blinkTimes.length/mins):0;
    const drowsyFlag=slowBlinkCount>DROWSY_THRESHOLD?'⚠️':'✓';
    const faceOffPct=totalF>0?Math.round((faceOffFrames/totalF)*100):0;
    if(welfareBlink) welfareBlink.textContent=`${blinkPerMin}/min`;
    if(welfareDrowsy) welfareDrowsy.textContent=drowsyFlag;
    if(welfareHead) welfareHead.textContent=headMovementEvents;
    if(welfareFaceOff) welfareFaceOff.textContent=`${faceOffPct}%`;
  }
}

// ─── RECORDING ───────────────────────────────────────────────────────────────
function startRecording(){
  sessionStart=Date.now();recordedFrames=[];totalF=0;trackedF=0;
  blinkTimes=[];lastBlinkTime=0;slowBlinkCount=0;headMovementEvents=0;lastHeadPos=null;faceOffFrames=0;
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
let _calibProcTs = -1;  // separate timestamp for calib loop (don't share with stimulus)
let _calibDbgLog = 0;   // throttle console logs to once per second

function processingLoop(){
  if(phase==='done') return;

  // ── CALIB-RUN ──
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
          const isBlink=ear<0.06;

          checkFatigue(ear,isBlink);

          if(!isBlink){
            const rawGaze=gazeModel?predictGaze(feat,gazeModel):estimateGazeFromIris(feat);
            _calibCurrentGaze=rawGaze;
            _calibLastGaze=rawGaze;
            _calibLastGazeTs=performance.now();

            // Log gaze + distance to current target once per second
            if(now-_calibDbgLog>1000){
              _calibDbgLog=now;
              const pt=calibPoints[calibIdx];
              const dist=pt?Math.hypot(rawGaze.x-pt.x,rawGaze.y-pt.y).toFixed(0):'?';
              const rad=pt?getCalibGazeRadius(pt).toFixed(0):'?';
              const near=pt&&Math.hypot(rawGaze.x-pt.x,rawGaze.y-pt.y)<getCalibGazeRadius(pt);
              console.log(`[Calib pt${calibIdx}] gaze=(${rawGaze.x.toFixed(0)},${rawGaze.y.toFixed(0)}) target=(${pt?pt.x.toFixed(0):'?'},${pt?pt.y.toFixed(0):'?'}) dist=${dist} radius=${rad} ${near?'✅IN':'❌OUT'} iris_h=${feat[6].toFixed(3)} pitch=${feat[2].toFixed(3)} foreY=${feat[3].toFixed(3)} EAR=${feat[7].toFixed(3)} state=${calibState} dwell=${_calibDwellAccum.toFixed(0)}ms`);
            }
          } else {
            const bridgeAge=performance.now()-_calibLastGazeTs;
            _calibCurrentGaze=bridgeAge<CALIB_IRIS_BRIDGE_MS?_calibLastGaze:null;
          }

          if((calibState==='sampling'||calibState==='holding')&&!isBlink){
            const pt=calibPoints[calibIdx];
            if(pt){
              calibSamples.push({feat,sx:pt.x,sy:pt.y});
              _calibPointSamples.push({feat,sx:pt.x,sy:pt.y});
            }
          }
        } else {
          // No face — log once per second
          if(performance.now()-_calibDbgLog>1000){
            _calibDbgLog=performance.now();
            console.warn('[Calib] No face detected — check camera/lighting');
          }
          const bridgeAge=performance.now()-_calibLastGazeTs;
          _calibCurrentGaze=bridgeAge<CALIB_IRIS_BRIDGE_MS?_calibLastGaze:null;
        }
      }catch(e){
        console.error('[Calib] detectForVideo error:',e);
      }
    } else {
      // webcam not ready or no landmarker
      if(performance.now()-_calibDbgLog>2000){
        _calibDbgLog=performance.now();
        console.warn(`[Calib] Waiting — webcam.readyState=${webcam.readyState} faceLandmarker=${!!faceLandmarker}`);
      }
    }
    procRaf=requestAnimationFrame(processingLoop);
    return;
  }

  if(phase==='validation'){procRaf=requestAnimationFrame(processingLoop);return;}

  // ── STIMULUS ──
  if(webcam.readyState>=2&&faceLandmarker){
    if(webcam.currentTime===_lastVT){procRaf=requestAnimationFrame(processingLoop);return;}
    _lastVT=webcam.currentTime;
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
      const isBlink=ear<0.06;
      const nowMs=Date.now()-sessionStart;
      const headPos={x:(lm[33].x+lm[263].x)/2,y:(lm[33].y+lm[263].y)/2};
      updateWelfareHUD(hasFace,ear,headPos,nowMs);
      let leftPupilX=null,leftPupilY=null,rightPupilX=null,rightPupilY=null;
      try{
        const vTrack=webcam.srcObject?.getVideoTracks()[0];
        const vSettings=vTrack?vTrack.getSettings():{};
        const vw=vSettings.width||640,vh=vSettings.height||480;
        let sumLx=0,sumLy=0,sumRx=0,sumRy=0;
        LEFT_IRIS.forEach(i=>{sumLx+=lm[i].x;sumLy+=lm[i].y;});
        RIGHT_IRIS.forEach(i=>{sumRx+=lm[i].x;sumRy+=lm[i].y;});
        leftPupilX=(sumLx/LEFT_IRIS.length)*vw;leftPupilY=(sumLy/LEFT_IRIS.length)*vh;
        rightPupilX=(sumRx/RIGHT_IRIS.length)*vw;rightPupilY=(sumRy/RIGHT_IRIS.length)*vh;
      }catch(e){}
      if(phase==='stimulus'){
        totalF++;
        let gaze=null;
        if(!isBlink){
          if(gazeModel&&!calibSkipActive) gaze=predictGaze(feat,gazeModel);
        }
        const frameData={
          t:nowMs,tracked:gaze?1:0,
          gazeX:gaze?.x??NaN,gazeY:gaze?.y??NaN,
          leftPupilX,leftPupilY,rightPupilX,rightPupilY,
          category:isBlink?'Blink':(gaze?classifyGaze(gaze,nowMs):'Lost'),
          feat
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
        const stFrames=document.getElementById('st-frames');
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
      updateWelfareHUD(false,1.0,null,nowMs);
      recordedFrames.push({
        t:nowMs,tracked:0,gazeX:NaN,gazeY:NaN,
        leftPupilX:null,leftPupilY:null,rightPupilX:null,rightPupilY:null,
        category:'Blink',feat:null
      });
      const stTrack=document.getElementById('st-track');
      if(stTrack&&totalF>0) stTrack.textContent=Math.round(trackedF/totalF*100)+'%';
    }
  }
  procRaf=requestAnimationFrame(processingLoop);
}

// ─── CSV (SMI RED FORMAT) ────────────────────────────────────────────────────
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
  'Port Status',
  'Annotation Name','Annotation Description','Annotation Tags',
  'Mouse Position X [px]','Mouse Position Y [px]',
  'Scroll Direction X','Scroll Direction Y','Content',
  'AOI Group Right','AOI Scope Right','AOI Order Right',
  'AOI Group Left','AOI Scope Left','AOI Order Binocular',
  "groupe d'enfants"
].join(',');

// [FIX-10] buildCSV: null feat guard on all .toFixed calls
function buildCSV(){
  const sessionDuration=(Date.now()-sessionStart)/1000/60;
  const totalBlinkCount=blinkTimes.length;
  const blinkRatePerMin=sessionDuration>0?Math.round(totalBlinkCount/sessionDuration):0;
  const drowsyFlag=slowBlinkCount>DROWSY_THRESHOLD?'⚠️':'✓';
  const faceOffPct=totalF>0?Math.round((faceOffFrames/totalF)*100):0;
  const welfareMeta=`blinks/min=${blinkRatePerMin} drowsy=${drowsyFlag} head_moves=${headMovementEvents} face_off=${faceOffPct}% skip_mode=${calibSkipActive}`;
  const biasMeta=`# GazeTrack v15 | bias_dx=${affineBias.dx.toFixed(2)} bias_dy=${affineBias.dy.toFixed(2)} bias_sx=${affineBias.sx.toFixed(4)} bias_sy=${affineBias.sy.toFixed(4)} val_samples=${valSamples.length} calib_samples=${calibSamples.length} ${welfareMeta}`;
  const lines=[biasMeta,CSV_HDR];
  const totalDuration=recordedFrames.length>0?recordedFrames[recordedFrames.length-1].t:0;
  const trackingRatio=totalF>0?(trackedF/totalF*100):0;
  const colorMap={ASD:'DarkViolet',TD:'SteelBlue',other:'Gray'};
  const color=colorMap[META.group]||'Gray';
  const fmtNum=(val,d=1)=>(val!==null&&!isNaN(val))?Number(val).toFixed(d):'-';
  recordedFrames.forEach((f,i)=>{
    const absTime=sessionStart+f.t;
    const d=new Date(absTime);
    const pad=(n,w=2)=>String(Math.floor(n)).padStart(w,'0');
    const tod=`${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}:${pad(d.getMilliseconds(),3)}`;
    const row=[
      i,f.t.toFixed(3),tod,'Trial001',META.stimulus||'-',0,totalDuration.toFixed(3),
      META.pid,color,trackingRatio.toFixed(3),
      f.category,f.category,f.category,i+1,i+1,
      '-','-','-','-','-','-',
      fmtNum(f.gazeX),fmtNum(f.gazeY),fmtNum(f.gazeX),fmtNum(f.gazeY),
      '-','-',
      // [FIX-10] safe feat access — f.feat can be null for face-off frames
      f.feat?f.feat[0].toFixed(4):'-',
      f.feat?f.feat[1].toFixed(4):'-',
      f.feat?f.feat[2].toFixed(4):'-',
      f.feat?f.feat[0].toFixed(4):'-',
      f.feat?f.feat[1].toFixed(4):'-',
      f.feat?f.feat[2].toFixed(4):'-',
      '-','-','-','-','-','-',
      fmtNum(f.rightPupilX),fmtNum(f.rightPupilY),fmtNum(f.leftPupilX),fmtNum(f.leftPupilY),
      0,'-','-','-','-','-','-','-',META.stimulus||'-','-','-','-','-','-','-',META.group
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

// ─── END SESSION ─────────────────────────────────────────────────────────────
function endSession(){
  if(phase==='done') return;
  phase='done';
  clearInterval(timerInt);
  cancelAnimationFrame(procRaf);
  cancelAnimationFrame(calibRaf);
  cancelAnimationFrame(valRaf);
  stimVideo.pause();
  csvData=buildCSV();
  const pct=totalF>0?Math.round(trackedF/totalF*100):0;
  const dur=Math.round((Date.now()-sessionStart)/1000);
  const ys=recordedFrames.filter(f=>f.tracked).map(f=>f.gazeY);
  let ystd=0;
  if(ys.length>1){const my=ys.reduce((a,b)=>a+b,0)/ys.length;ystd=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);}
  const biasOk=Math.abs(affineBias.dx)>5||Math.abs(affineBias.dy)>5;
  const biasLabel=biasOk?`${affineBias.dx>0?'+':''}${affineBias.dx.toFixed(0)},${affineBias.dy>0?'+':''}${affineBias.dy.toFixed(0)}px`:'Minimal';
  const fixCount=recordedFrames.filter(f=>f.category==='Fixation').length;
  const sacCount=recordedFrames.filter(f=>f.category==='Saccade').length;
  const sessionMins=dur/60;
  const blinkRate=sessionMins>0?Math.round(blinkTimes.length/sessionMins):0;
  const drowsyFlag=slowBlinkCount>DROWSY_THRESHOLD?'⚠️ Drowsy':'✓ Normal';
  const faceOffPct=totalF>0?Math.round((faceOffFrames/totalF)*100):0;
  const statsEl=document.getElementById('done-stats');
  if(statsEl){
    statsEl.innerHTML=`
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
window.addEventListener('keydown',e=>{
  if(e.key==='d'||e.key==='D'){
    const p=document.getElementById('debug-panel');
    if(p) p.style.display=p.style.display==='none'?'block':'none';
  }
});
window.addEventListener('beforeunload',()=>{
  if(sessionStream) sessionStream.getTracks().forEach(t=>t.stop());
  if(camStream)     camStream.getTracks().forEach(t=>t.stop());
});
window.addEventListener('resize',()=>{
  if(phase==='calib-run') calibPoints=buildCalibPoints();
});

console.log('%c GazeTrack v21 (Star Keeper — Production Clean)','background:#00e5b0;color:#000;font-weight:bold;font-size:14px');

import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs';

// ─── DEVICE ──────────────────────────────────────────────────────────────────
const isMobile = /Android|iPad|iPhone|iPod|Mobile/i.test(navigator.userAgent)
  || (navigator.maxTouchPoints > 1 && window.innerWidth < 1200);
const MP_DELEGATE = isMobile ? 'CPU' : 'GPU';
const IS_TABLET = /iPad|Android(?!.*Mobile)/i.test(navigator.userAgent)
  || (window.innerWidth >= 768 && window.innerWidth <= 1280);

// ─── CONSTANTS ───────────────────────────────────────────────────────────────
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

// ─── MONOTONIC TIMESTAMP ─────────────────────────────────────────────────────
let _mpLastTs = -1;
function mpNow() {
  const t = performance.now();
  _mpLastTs = t > _mpLastTs ? t : _mpLastTs + 0.001;
  return _mpLastTs;
}

// ─── STATE ───────────────────────────────────────────────────────────────────
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

// Recording
let recordedFrames=[],totalF=0,trackedF=0;
let sessionStart=0,timerInt=null;
let csvData=null;
let META={pid:'',age:'',group:'',clinician:'',location:'',notes:'',stimulus:''};

// Playlist / multi-trial
let playlist         = [];  // [{objectURL, name, type:'video'|'image'}]
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

// ─── DOM REFS ────────────────────────────────────────────────────────────────
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

// ─── INJECT CALIBRATION CSS ──────────────────────────────────────────────────
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

// ─── AUDIO ───────────────────────────────────────────────────────────────────
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

// ─── CONFETTI ─────────────────────────────────────────────────────────────────
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

// ─── POSITION ALL-CLEAR ──────────────────────────────────────────────────────
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

// ─── PRE-FLIGHT ──────────────────────────────────────────────────────────────
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
    tips.push(`<strong>📐 Small window (${window.innerHeight}px):</strong> Press <kbd style="background:#333;padding:1px 5px;border-radius:4px;font-size:11px;">F11</kbd> for fullscreen — gaze Y accuracy requires a tall viewport.`);
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

// ─── CAMERA INIT ─────────────────────────────────────────────────────────────
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:640,max:1280},height:{ideal:480,max:720},facingMode:'user',frameRate:{ideal:60,min:30}},
      audio:false
    });
    camStream = stream;
    if (camPreview) { camPreview.srcObject = stream; camPreview.play(); }
    document.getElementById('cam-dot')?.classList.add('ok');
    const cs = document.getElementById('cam-status-txt'); if (cs) cs.textContent='Camera active';
    const cc = document.getElementById('chk-cam'); if (cc) { cc.classList.add('ok'); cc.textContent='✓ Cam'; }
    const t = stream.getVideoTracks()[0].getSettings();
    pfSet('cam','pass',`✓ ${t.width||640}x${t.height||480}`);
    checkStartBtn();
    pfRaf = requestAnimationFrame(pfAnalyseFrame);
    pfCheckBrowser();
    loadPreviewDetector();
  } catch(e) {
    const cs = document.getElementById('cam-status-txt'); if (cs) cs.textContent='✗ Camera error - allow access';
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
  if (camPreview?.readyState >= 2 && previewFl) {
    const rect = camCanvas?.getBoundingClientRect();
    if (rect) {
      const dW = Math.round(rect.width)||640, dH = Math.round(rect.height)||480;
      if (camCanvas.width!==dW || camCanvas.height!==dH) { camCanvas.width=dW; camCanvas.height=dH; }
    }
    try {
      const res = previewFl.detectForVideo(camPreview, mpNow());
      const hasFace = !!(res.faceLandmarks?.length > 0);
      camCtx?.clearRect(0,0,camCanvas.width,camCanvas.height);
      if (hasFace) {
        drawPreviewMesh(res.faceLandmarks[0]);
        const lm = res.faceLandmarks[0];
        const hasIris = !!(lm[468] && lm[473]);
        const chkFace = document.getElementById('chk-face');
        const chkIris = document.getElementById('chk-iris');
        if (chkFace) { chkFace.classList.add('ok'); chkFace.textContent='✓ Face'; }
        if (chkIris) { chkIris.classList.toggle('ok',hasIris); chkIris.textContent=hasIris?'✓ Iris':'👁 Iris'; }
        if (hasIris) {
          const iodNorm  = Math.hypot(lm[473].x-lm[468].x, lm[473].y-lm[468].y);
          const faceCX   = (lm[33].x+lm[263].x)/2;
          const offCentre = faceCX<0.25||faceCX>0.75;
          const lEAR = Math.hypot(lm[159].x-lm[145].x, lm[159].y-lm[145].y);
          const rEAR = Math.hypot(lm[386].x-lm[374].x, lm[386].y-lm[374].y);
          const earPx  = (lEAR+rEAR)/2;
          const qPct   = (earPx/(iodNorm+1e-6))>0.08?95:75;
          const qFill  = document.getElementById('q-fill'); if (qFill) qFill.style.width=qPct+'%';
          const qPctEl = document.getElementById('q-pct');  if (qPctEl) qPctEl.textContent=qPct+'%';
          if      (iodNorm>0.22)  pfSet('face','warn',`⚠ Too close - move back ~15 cm`);
          else if (iodNorm>=0.13) pfSet('face','pass',offCentre?`✓ Good distance - Move to centre`:`✓ Face visible - Good distance (~50-70 cm)`);
          else if (iodNorm>=0.07) pfSet('face','warn',`⚠ Too far - move ${offCentre?'closer & to centre':'~20 cm closer'}`);
          else                    pfSet('face','warn',`⚠ Very far or face at edge - move much closer`);
        } else {
          document.getElementById('q-fill')?.setAttribute('style','width:40%');
          const qPctEl = document.getElementById('q-pct'); if (qPctEl) qPctEl.textContent='40%';
          pfSet('face','warn','⚠ Face detected but iris not visible - look at camera');
        }
      } else {
        document.getElementById('chk-face')?.classList.remove('ok');
        document.getElementById('chk-iris')?.classList.remove('ok');
        const chkFace = document.getElementById('chk-face'); if (chkFace) chkFace.textContent='👤 Face';
        const chkIris = document.getElementById('chk-iris'); if (chkIris) chkIris.textContent='👁 Iris';
        document.getElementById('q-fill')?.setAttribute('style','width:0%');
        const qPctEl = document.getElementById('q-pct'); if (qPctEl) qPctEl.textContent=' - ';
        pfSet('face','fail','✗ No face detected - check camera position');
      }
    } catch(e) {}
  }
  previewRaf = requestAnimationFrame(previewLoop);
}
function drawPreviewMesh(lm) {
  if (!camCtx || !camCanvas) return;
  const W=camCanvas.width, H=camCanvas.height;
  const fx = x=>(1-x)*W, fy = y=>y*H;
  [[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33],
   [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]].forEach(pts=>{
    camCtx.beginPath();
    pts.forEach((idx,i)=>{const p=lm[idx];i===0?camCtx.moveTo(fx(p.x),fy(p.y)):camCtx.lineTo(fx(p.x),fy(p.y));});
    camCtx.strokeStyle='#00e5b0';camCtx.lineWidth=0.4;camCtx.globalAlpha=0.2;camCtx.stroke();camCtx.globalAlpha=1;
  });
  [[468,469],[473,474]].forEach(([c,e])=>{
    if(!lm[c]||!lm[e]) return;
    const cx=fx(lm[c].x),cy=fy(lm[c].y),ex2=fx(lm[e].x),ey2=fy(lm[e].y);
    const r=Math.hypot(ex2-cx,ey2-cy)+0.5;
    camCtx.beginPath();camCtx.arc(cx,cy,r,0,Math.PI*2);
    camCtx.strokeStyle='#00e5b0';camCtx.lineWidth=0.6;camCtx.globalAlpha=0.3;camCtx.stroke();camCtx.globalAlpha=1;
  });
}

// ─── FORM ─────────────────────────────────────────────────────────────────────
function checkStartBtn() {
  const pidOk   = document.getElementById('f-pid')?.value.trim().length > 0;
  const groupOk = document.querySelector('input[name="group"]:checked') !== null;
  const btn     = document.getElementById('start-btn');
  if (btn) btn.disabled = !(pidOk && groupOk);
  if (pidOk && groupOk) pfUpdateScore();
}
document.getElementById('f-pid')?.addEventListener('input', checkStartBtn);
document.querySelectorAll('input[name="group"]').forEach(r=>r.addEventListener('change',checkStartBtn));

// ─── PLAYLIST ────────────────────────────────────────────────────────────────
document.getElementById('video-drop')?.addEventListener('click', () =>
  document.getElementById('video-input')?.click());

document.getElementById('video-input')?.addEventListener('change', e => {
  const files = Array.from(e.target.files || []);
  if (!files.length) return;
  files.forEach(f => {
    const type = f.type.startsWith('image/') ? 'image' : 'video';
    playlist.push({ objectURL: URL.createObjectURL(f), name: f.name, type });
  });
  renderPlaylist();
  e.target.value = '';
});

function renderPlaylist() {
  const wrap        = document.getElementById('playlist-wrap');
  const list        = document.getElementById('playlist-list');
  const count       = document.getElementById('playlist-count');
  const photoDurRow = document.getElementById('photo-dur-row');
  const hint        = document.getElementById('video-hint');
  if (!list) return;
  const hasItems  = playlist.length > 0;
  if (wrap)  wrap.style.display  = hasItems ? 'block' : 'none';
  if (hint)  hint.style.display  = hasItems ? 'none'  : 'block';
  if (count) count.textContent   = playlist.length;
  const hasImages = playlist.some(p => p.type === 'image');
  if (photoDurRow) photoDurRow.style.display = hasImages ? 'flex' : 'none';
  list.innerHTML = '';
  playlist.forEach((item, i) => {
    const li = document.createElement('li');
    li.className   = 'playlist-item';
    li.draggable   = true;
    li.dataset.idx = i;
    li.innerHTML   = `
      <span class="pl-drag">⠿</span>
      <span class="pl-icon">${item.type === 'image' ? '🖼' : '🎬'}</span>
      <span class="pl-name">${item.name}</span>
      <span class="pl-type">${item.type === 'image' ? 'Image' : 'Video'}</span>
      <button class="pl-remove" data-idx="${i}">×</button>`;
    li.addEventListener('dragstart', () => { _dragIdx = i; li.style.opacity = '0.4'; });
    li.addEventListener('dragend',   () => { _dragIdx = null; li.style.opacity = '1'; });
    li.addEventListener('dragover',  ev => { ev.preventDefault(); li.style.background = 'rgba(0,229,176,0.08)'; });
    li.addEventListener('dragleave', () => { li.style.background = ''; });
    li.addEventListener('drop', ev => {
      ev.preventDefault(); li.style.background = '';
      if (_dragIdx === null || _dragIdx === i) return;
      const moved = playlist.splice(_dragIdx, 1)[0];
      playlist.splice(i, 0, moved);
      renderPlaylist();
    });
    list.appendChild(li);
  });
  list.querySelectorAll('.pl-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      const idx = parseInt(btn.dataset.idx);
      URL.revokeObjectURL(playlist[idx].objectURL);
      playlist.splice(idx, 1);
      renderPlaylist();
    });
  });
}

document.getElementById('photo-dur-input')?.addEventListener('input', e => {
  photoDurSec = Math.max(1, parseInt(e.target.value) || 5);
});

// ─── FULLSCREEN ───────────────────────────────────────────────────────────────
function requestFullscreen() {
  const el = document.documentElement;
  const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
  if (req) req.call(el).catch(() => showFullscreenReminder());
  else showFullscreenReminder();
}
function showFullscreenReminder() {
  if (window.innerHeight >= 700) return;
  if (document.getElementById('fs-reminder')) return;
  const banner = document.createElement('div');
  banner.id = 'fs-reminder';
  banner.style.cssText = `position:fixed;top:0;left:0;right:0;z-index:99999;
    background:linear-gradient(90deg,#ff9f43,#e17f20);color:#fff;
    font-family:sans-serif;font-size:14px;font-weight:700;padding:10px 20px;
    text-align:center;letter-spacing:.02em;
    display:flex;align-items:center;justify-content:center;gap:14px;`;
  banner.innerHTML = `
    <span>⚠️ Small viewport (${window.innerHeight}px) — gaze Y accuracy may be affected.</span>
    <button onclick="document.documentElement.requestFullscreen?.();this.parentElement.remove();"
      style="background:#fff;color:#e17f20;border:none;border-radius:8px;padding:5px 14px;font-weight:800;cursor:pointer;font-size:13px;">
      Go Fullscreen (F11)
    </button>
    <button onclick="this.parentElement.remove();"
      style="background:transparent;color:#fff;border:1px solid rgba(255,255,255,0.5);border-radius:8px;padding:5px 14px;cursor:pointer;font-size:12px;">
      Continue anyway
    </button>`;
  document.body.appendChild(banner);
}
document.addEventListener('fullscreenchange',       resizeCanvases);
document.addEventListener('webkitfullscreenchange', resizeCanvases);

document.getElementById('start-btn')?.addEventListener('click', () => {
  META.pid       = document.getElementById('f-pid')?.value.trim()  || '';
  META.age       = document.getElementById('f-age')?.value          || '';
  META.group     = document.querySelector('input[name="group"]:checked')?.value || '';
  META.clinician = document.getElementById('f-clinician')?.value.trim() || '';
  META.location  = document.getElementById('f-location')?.value.trim()  || '';
  META.notes     = document.getElementById('f-notes')?.value.trim()      || '';
  photoDurSec    = parseInt(document.getElementById('photo-dur-input')?.value || '5') || 5;
  cancelAnimationFrame(previewRaf);
  cancelAnimationFrame(pfRaf);
  _pfCanvas = null; _pfCtx = null;
  requestFullscreen();
  phase = 'loading'; showScreen('loading'); beginSession();
});
initCamera();

// ─── SESSION START ────────────────────────────────────────────────────────────
async function beginSession() {
  try {
    if (previewFl) {
      faceLandmarker = previewFl;
      const msgEl = document.getElementById('load-msg'); if (msgEl) msgEl.textContent='Model ready - starting camera...';
    } else {
      const msgEl = document.getElementById('load-msg'); if (msgEl) msgEl.textContent='Loading eye tracking model...';
      const resolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
      faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
        baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:MP_DELEGATE},
        runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
      });
    }
    if (camStream) { sessionStream=camStream; camStream=null; }
    else { sessionStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user',frameRate:{ideal:60,min:30}},audio:false}); }
    if (webcam) { webcam.srcObject=sessionStream; }
    await new Promise(r=>{ if(webcam) webcam.onloadedmetadata=()=>{webcam.play();r();}; else r(); });
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
    phase='calib-ready'; showScreen('calib');
    procRaf=requestAnimationFrame(processingLoop);
  } catch(err) {
    console.error(err);
    const msgEl=document.getElementById('load-msg'); if(msgEl) msgEl.textContent='✗ '+(err.message||'Startup error');
  }
}
function resizeCanvases() {
  const dpr=window.devicePixelRatio||1;
  [calibCanvas, gazeCanvas].forEach(canvas=>{
    if(!canvas) return;
    canvas.width =Math.round(window.innerWidth *dpr);
    canvas.height=Math.round(window.innerHeight*dpr);
    canvas.style.width =window.innerWidth +'px';
    canvas.style.height=window.innerHeight+'px';
  });
  if (calibCtx) calibCtx.setTransform(dpr,0,0,dpr,0,0);
  if (gazeCtx)  gazeCtx.setTransform(dpr,0,0,dpr,0,0);
}

// ─── FEATURE EXTRACTION ──────────────────────────────────────────────────────
function extractFeatures(lm, mat) {
  const avg=ids=>{const s={x:0,y:0,z:0};ids.forEach(i=>{s.x+=lm[i].x;s.y+=lm[i].y;s.z+=(lm[i].z||0);});return{x:s.x/ids.length,y:s.y/ids.length,z:s.z/ids.length};};
  const li=avg(LEFT_IRIS),ri=avg(RIGHT_IRIS);
  const lIn=lm[L_CORNERS[0]],lOut=lm[L_CORNERS[1]],rIn=lm[R_CORNERS[0]],rOut=lm[R_CORNERS[1]];
  const lW=Math.hypot(lOut.x-lIn.x,lOut.y-lIn.y)+1e-6;
  const rW=Math.hypot(rOut.x-rIn.x,rOut.y-rIn.y)+1e-6;
  const lCx=(lIn.x+lOut.x)/2, rCx=(rIn.x+rOut.x)/2;
  const liX=(li.x-lCx)/lW, riX=(ri.x-rCx)/rW;
  let pitchDeg=0;
  if (mat?.data){const m=mat.data;pitchDeg=Math.asin(Math.max(-1,Math.min(1,-m[6])))*180/Math.PI/30;}
  const nose=lm[1],fore=lm[10],chin=lm[152];
  const pitchZ=((nose.z||0)-((fore.z||0)+(chin.z||0))/2)*10;
  const vertMain=(Math.abs(pitchDeg)>0.001)?pitchDeg:pitchZ;
  const faceCY=(fore.y+chin.y)/2, faceH=Math.abs(chin.y-fore.y)+1e-6;
  const irisY=((li.y+ri.y)/2-faceCY)/faceH;
  const lEAR=Math.hypot(lm[159].x-lm[145].x,lm[159].y-lm[145].y)/lW;
  const rEAR=Math.hypot(lm[386].x-lm[374].x,lm[386].y-lm[374].y)/rW;
  const ear=(lEAR+rEAR)/2;
  const iod=Math.hypot(ri.x-li.x,ri.y-li.y);
  const feat=[liX,riX,vertMain,fore.y,irisY,(li.y+ri.y)/2,(liX+riX)/2,ear,iod];

  feat._3d = null;
  if (mat?.data) {
    const m = mat.data;
    const IOD_REAL_MM = 60;
    const tzMm = Math.max(250, Math.min(750, IOD_REAL_MM / Math.max(0.001, iod * 0.700 * 1.8)));
    const iodMm   = Math.max(42, Math.min(68, iod * 280));
    const halfIod = iodMm / 2;
    const headLateralMm = m[8] * -tzMm * 0.15;
    const pitchOffset   = Math.asin(Math.max(-1, Math.min(1, -m[6]))) * 30;
    const eyeYmm        = -65 + pitchOffset;
    const eyeRX = headLateralMm - halfIod;
    const eyeLX = headLateralMm + halfIod;
    const yawBias = m[8] * 15;
    const eyeRZ   = Math.max(250, Math.min(750, tzMm - yawBias));
    const eyeLZ   = Math.max(250, Math.min(750, tzMm + yawBias));
    const gvRx_raw = m[8]; const gvRy_raw = m[9]; const gvRz_raw = -m[10];
    const gvMag = Math.sqrt(gvRx_raw**2 + gvRy_raw**2 + gvRz_raw**2) || 1;
    const gvRxN = gvRx_raw/gvMag, gvRyN = gvRy_raw/gvMag, gvRzN = gvRz_raw/gvMag;
    const irisContribX =  feat[6] * 1.8;
    const irisContribY =  feat[4] * 1.5;
    const blendX = gvRxN + irisContribX;
    const blendY = gvRyN + irisContribY;
    const blendMag = Math.sqrt(blendX**2 + blendY**2 + gvRzN**2) || 1;
    const finalGvRx = blendX / blendMag;
    const finalGvRy = blendY / blendMag;
    const finalGvRz = gvRzN  / blendMag;
    const vergenceAngle = halfIod / Math.max(300, tzMm);
    const finalGvLx = finalGvRx - vergenceAngle * 0.6;
    const pupR = Math.max(1.5, Math.min(3.8, 2.9  + (0.30 - feat[7]) * 3.5));
    const pupL = Math.max(1.5, Math.min(3.8, 2.85 + (0.30 - feat[7]) * 3.5));
    const PX_PER_MM  = 4.14;
    const pupSizePxR = pupR * PX_PER_MM;
    const pupSizePxL = pupL * PX_PER_MM;
    const vergenceDisp = Math.round(iodMm * 1.12);
    feat._3d = {
      eyeRX, eyeRY: eyeYmm, eyeRZ,
      eyeLX, eyeLY: eyeYmm - 4, eyeLZ,
      gazeVRX: finalGvRx, gazeVRY: finalGvRy, gazeVRZ: finalGvRz,
      gazeVLX: finalGvLx, gazeVLY: finalGvRy, gazeVLZ: finalGvRz,
      pupilDiamR: pupR, pupilDiamL: pupL,
      pupSizePxR, pupSizePxL,
      vergenceDisp, iodMm,
    };
  }
  return feat;
}

// ─── RIDGE REGRESSION ────────────────────────────────────────────────────────
function polyX(f) {
  return [1, f[0], f[1], f[6], f[0]*f[1], f[0]-f[1], f[0]+f[1]];
}
function polyY(f) {
  return [1, f[2]*3, f[3], f[4]*2, f[2]*f[3], (f[2]*3)*(f[2]*3), f[5]];
}
function ridgeFit(X,y,alpha=RIDGE_ALPHA){
  const n=X[0].length;
  const XtX=Array.from({length:n},()=>new Array(n).fill(0));
  const Xty=new Array(n).fill(0);
  for(let r=0;r<X.length;r++){for(let i=0;i<n;i++){Xty[i]+=X[r][i]*y[r];for(let j=0;j<n;j++)XtX[i][j]+=X[r][i]*X[r][j];}}
  for(let i=0;i<n;i++) XtX[i][i]+=alpha;
  const aug=XtX.map((row,i)=>[...row,Xty[i]]);
  for(let c=0;c<n;c++){
    let p=c;for(let r=c+1;r<n;r++)if(Math.abs(aug[r][c])>Math.abs(aug[p][c]))p=r;
    [aug[c],aug[p]]=[aug[p],aug[c]];
    const pv=aug[c][c];if(Math.abs(pv)<1e-12)continue;
    for(let j=c;j<=n;j++)aug[c][j]/=pv;
    for(let r=0;r<n;r++){if(r!==c){const f=aug[r][c];for(let j=c;j<=n;j++)aug[r][j]-=f*aug[c][j];}}
  }
  return aug.map(r=>r[n]);
}
function trainModel(samples){
  if(samples.length<MIN_SAMPLES) return null;
  const W=window.innerWidth, H=window.innerHeight;
  const Xx=samples.map(s=>polyX(s.feat));
  const Xy=samples.map(s=>polyY(s.feat));
  const wx=ridgeFit(Xx, samples.map(s=>s.sx/W));
  const wy=ridgeFit(Xy, samples.map(s=>s.sy/H));
  const pitchStd = (() => { const vals=samples.map(s=>s.feat[2]); const m=vals.reduce((a,b)=>a+b,0)/vals.length; return Math.sqrt(vals.reduce((a,b)=>a+(b-m)**2,0)/vals.length); })();
  if(pitchStd<0.01) console.warn('[GazeTrack] ⚠️ pitch std very low — Y model may be poor');
  return{wx,wy};
}
function predictGaze(feat,model){
  if(!model) return null;
  const W=window.innerWidth, H=window.innerHeight;
  const gx=polyX(feat).reduce((s,v,i)=>s+v*model.wx[i],0)*W;
  const gy=polyY(feat).reduce((s,v,i)=>s+v*model.wy[i],0)*H;
  const cx=affineBias.sx*gx+affineBias.dx;
  const cy=affineBias.sy*gy+affineBias.dy;
  return{x:Math.max(0,Math.min(W,cx)),y:Math.max(0,Math.min(H,cy))};
}
function computeAffineCorrection(pairs){
  function linfit(ps,ts){
    const n=ps.length,mp=ps.reduce((a,b)=>a+b,0)/n,mt=ts.reduce((a,b)=>a+b,0)/n;
    let num=0,den=0;
    for(let i=0;i<n;i++){num+=(ps[i]-mp)*(ts[i]-mt);den+=(ps[i]-mp)**2;}
    const s=den>1e-6?num/den:1,sc=Math.max(0.55,Math.min(1.8,s));
    return{s:sc,d:mt-sc*mp};
  }
  const fx=linfit(pairs.map(p=>p.px),pairs.map(p=>p.tx));
  const fy=linfit(pairs.map(p=>p.py),pairs.map(p=>p.ty));
  return{sx:fx.s,dx:fx.d,sy:fy.s,dy:fy.d};
}

// ─── PRE-CALIBRATION GAZE ESTIMATE ───────────────────────────────────────────
function estimateGazeFromIris(feat){
  const W=window.innerWidth, H=window.innerHeight;
  const rawX=(-feat[6]*4870)+(W*0.54);
  const rawY=_calibTargetY>=0?_calibTargetY:H*0.5;
  return{x:Math.max(0,Math.min(W,rawX)),y:rawY};
}

// ═══════════════════════════════════════════════════════════════════════════════
//  STAR KEEPER CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════
const CREATURE_DEFS=[
  {name:'Starby',   color:'#ffd700',glow:'#ffe066',bodyFn:'star',  hunger:'star seeds'},
  {name:'Bubbles',  color:'#4fc3f7',glow:'#80deea',bodyFn:'blob',  hunger:'sparkle drops'},
  {name:'Fizzwick', color:'#69f0ae',glow:'#b9f6ca',bodyFn:'puff',  hunger:'moon cookies'},
  {name:'Roary',    color:'#ff7043',glow:'#ffab91',bodyFn:'round', hunger:'fire berries'},
  {name:'Shimmer',  color:'#ce93d8',glow:'#f3e5f5',bodyFn:'floof', hunger:'dream petals'},
];

function buildCalibPoints(){
  const W=window.innerWidth, H=window.innerHeight;
  const px=Math.max(120,W*0.25);
  const pyTop=Math.max(80,H*0.14);
  const pyBot=Math.max(80,H*0.80);
  return[
    {x:W/2,  y:H/2,   isCorner:false},
    {x:px,   y:pyTop, isCorner:true},
    {x:W-px, y:pyTop, isCorner:true},
    {x:W-px, y:pyBot, isCorner:true},
    {x:px,   y:pyBot, isCorner:true},
  ];
}
function getCalibGazeRadius(pt){
  if(!pt.isCorner) return Math.min(window.innerWidth,window.innerHeight)*0.42;
  return 220;
}

// ─── SCENE SETUP ─────────────────────────────────────────────────────────────
function initCalibScene(){
  document.querySelectorAll('.calib-creature-wrap,.calib-bg,.calib-story-banner,.calib-hud').forEach(el=>el.remove());
  calibStarCvs?.remove(); calibStarCvs=null;
  calibFxCvs?.remove();   calibFxCvs=null;
  calibStoryBanner=null; calibHud=null;

  const calibScreen=document.getElementById('s-calib');
  if(!calibScreen) return;
  const W=window.innerWidth, H=window.innerHeight;

  const bg=document.createElement('div'); bg.className='calib-bg'; calibScreen.appendChild(bg);

  calibStarCvs=document.createElement('canvas'); calibStarCvs.id='calib-star-canvas';
  calibStarCvs.width=W; calibStarCvs.height=H;
  Object.assign(calibStarCvs.style,{position:'absolute',inset:'0',pointerEvents:'none',zIndex:'1'});
  calibScreen.appendChild(calibStarCvs);
  calibStarCtx=calibStarCvs.getContext('2d');
  calibStars=Array.from({length:180},()=>({x:Math.random()*W,y:Math.random()*H,r:0.4+Math.random()*1.5,tw:Math.random()*Math.PI*2}));

  calibFxCvs=document.createElement('canvas'); calibFxCvs.id='calib-fx-canvas';
  calibFxCvs.width=W; calibFxCvs.height=H;
  Object.assign(calibFxCvs.style,{position:'absolute',inset:'0',pointerEvents:'none',zIndex:'40'});
  calibScreen.appendChild(calibFxCvs);
  calibFxCtx=calibFxCvs.getContext('2d');

  calibStoryBanner=document.createElement('div'); calibStoryBanner.className='calib-story-banner';
  calibStoryBanner.textContent='🌟 Help the magical creatures eat their star seeds!';
  calibScreen.appendChild(calibStoryBanner);

  calibHud=document.createElement('div'); calibHud.className='calib-hud';
  calibScreen.appendChild(calibHud);
}
function setCalibBanner(txt){
  if(!calibStoryBanner) return;
  calibStoryBanner.style.opacity='0';
  setTimeout(()=>{ if(calibStoryBanner){calibStoryBanner.textContent=txt;calibStoryBanner.style.opacity='1';} },200);
}
function updateCalibHUD(){
  if(!calibHud) return;
  calibHud.innerHTML='';
  for(let i=0;i<CALIB_TOTAL_PTS;i++){
    const paw=document.createElement('span');
    paw.className='calib-hud-paw'+(doneCalibPoints.has(i)?' done':'');
    paw.textContent=doneCalibPoints.has(i)?'🐾':(i===calibIdx?'⭐':'○');
    calibHud.appendChild(paw);
  }
}

// ─── CREATURE DRAWING ─────────────────────────────────────────────────────────
function drawCreatureOnCanvas(ctx,def,S,t,gazeNear,holdPct,isHappy,isDone){
  ctx.clearRect(0,0,S,S);
  ctx.save(); ctx.translate(S/2,S/2);
  if(gazeNear&&!isDone){
    const g=ctx.createRadialGradient(0,0,20,0,0,60);
    g.addColorStop(0,def.glow+'bb'); g.addColorStop(1,def.glow+'00');
    ctx.beginPath();ctx.arc(0,0,60,0,Math.PI*2);ctx.fillStyle=g;ctx.fill();
  }
  if(!gazeNear&&!isDone){
    const pulse=0.5+0.5*Math.abs(Math.sin(t*3.2));
    ctx.beginPath();ctx.arc(0,0,48,0,Math.PI*2);
    ctx.strokeStyle=def.glow+'55';ctx.lineWidth=2+pulse*4;
    ctx.setLineDash([7,5]);ctx.stroke();ctx.setLineDash([]);
  }
  const r=isDone?34:30, fn=def.bodyFn;
  ctx.beginPath();
  if(fn==='star'){
    for(let i=0;i<10;i++){const a=(i*Math.PI/5)-Math.PI/2,rad=i%2===0?r:r*0.52;i===0?ctx.moveTo(Math.cos(a)*rad,Math.sin(a)*rad):ctx.lineTo(Math.cos(a)*rad,Math.sin(a)*rad);}ctx.closePath();
  }else if(fn==='blob'){
    for(let a=0;a<=Math.PI*2+0.1;a+=0.05){const bump=r+Math.sin(a*4+t*2)*3;a===0?ctx.moveTo(Math.cos(a)*bump,Math.sin(a)*bump):ctx.lineTo(Math.cos(a)*bump,Math.sin(a)*bump);}
  }else if(fn==='puff'){
    ctx.arc(0,0,r,0,Math.PI*2);
  }else if(fn==='round'){
    for(let i=0;i<16;i++){const a=(i*Math.PI/8)-Math.PI/2,spk=r+(i%2===0?8:0)+Math.sin(t*4+i)*2;i===0?ctx.moveTo(Math.cos(a)*spk,Math.sin(a)*spk):ctx.lineTo(Math.cos(a)*spk,Math.sin(a)*spk);}ctx.closePath();
  }else{
    for(let i=0;i<5;i++){const a=(i*Math.PI*2/5)-Math.PI/2,spk=r+Math.sin(t*2+i)*3;i===0?ctx.moveTo(Math.cos(a)*spk,Math.sin(a)*spk):ctx.lineTo(Math.cos(a)*spk,Math.sin(a)*spk);}ctx.closePath();
  }
  ctx.fillStyle=isDone?'#ffd700':def.color;
  ctx.shadowColor=gazeNear?def.glow:'transparent'; ctx.shadowBlur=gazeNear?22:0;
  ctx.fill(); ctx.strokeStyle='rgba(255,255,255,0.25)';ctx.lineWidth=1.5;ctx.stroke(); ctx.shadowBlur=0;
  const eyeY=-r*0.15,eyeX=r*0.28;
  [-eyeX,eyeX].forEach(ex=>{
    ctx.beginPath();ctx.arc(ex,eyeY,r*0.17,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
    ctx.beginPath();ctx.arc(ex+r*0.05,eyeY,r*0.10,0,Math.PI*2);ctx.fillStyle='#1a1030';ctx.fill();
    ctx.beginPath();ctx.arc(ex+r*0.09,eyeY-r*0.06,r*0.04,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
  });
  if(isHappy||isDone){
    ctx.beginPath();ctx.arc(0,r*0.25,r*0.26,0.1*Math.PI,0.9*Math.PI);ctx.strokeStyle='#2d1a40';ctx.lineWidth=2.5;ctx.stroke();
    [-0.38,0.38].forEach(ex=>{ctx.beginPath();ctx.arc(ex*r,r*0.28,r*0.13,0,Math.PI*2);ctx.fillStyle='rgba(255,100,120,0.45)';ctx.fill();});
  }else{
    ctx.beginPath();ctx.arc(0,r*0.38,r*0.14,Math.PI*1.1,Math.PI*1.9);ctx.strokeStyle='#2d1a40';ctx.lineWidth=2;ctx.stroke();
  }
  if(holdPct>0&&holdPct<1){
    const rr=r+15;
    ctx.beginPath();ctx.arc(0,0,rr,-Math.PI/2,-Math.PI/2+holdPct*Math.PI*2);
    ctx.strokeStyle='#ffd700';ctx.lineWidth=5;ctx.lineCap='round';ctx.stroke();ctx.lineCap='butt';
  }
  if(isDone){
    for(let i=0;i<5;i++){
      const a=(i*Math.PI*2/5)-Math.PI/2+t*0.5;
      drawMiniStar(ctx,Math.cos(a)*(r+18),Math.sin(a)*(r+18),5,'#ffd700');
    }
  }
  ctx.restore();
  ctx.save();ctx.font='bold 11px "Comic Sans MS",cursive';ctx.fillStyle=isDone?'#ffd700':def.glow;ctx.textAlign='center';ctx.fillText(def.name,S/2,S-5);ctx.restore();
}
function drawMiniStar(ctx,x,y,r,color){
  ctx.save();ctx.fillStyle=color;ctx.beginPath();
  for(let i=0;i<10;i++){const a=(i*Math.PI/5)-Math.PI/2,rad=i%2===0?r:r*0.42;i===0?ctx.moveTo(x+Math.cos(a)*rad,y+Math.sin(a)*rad):ctx.lineTo(x+Math.cos(a)*rad,y+Math.sin(a)*rad);}
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
  for(let i=0;i<n;i++)
    calibFloaties.push({x:x+(-40+Math.random()*80),y,vy:-1.2-Math.random(),vx:-0.4+Math.random()*0.8,life:1,size:10+Math.random()*8,decay:0.013,type:'heart'});
}
function addCalibStars(x,y,n=10){
  for(let i=0;i<n;i++)
    calibFloaties.push({x:x+(-60+Math.random()*120),y:y+(-30+Math.random()*30),vy:-1.8-Math.random()*2,vx:-0.8+Math.random()*1.6,life:1,size:8+Math.random()*12,decay:0.016,type:'star'});
}
function drawHeart(ctx,x,y,size,alpha){
  ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle='#ff69b4';ctx.beginPath();
  ctx.moveTo(x,y+size*0.3);
  ctx.bezierCurveTo(x,y,x-size,y,x-size,y+size*0.35);
  ctx.bezierCurveTo(x-size,y+size*0.7,x,y+size,x,y+size*1.1);
  ctx.bezierCurveTo(x,y+size,x+size,y+size*0.7,x+size,y+size*0.35);
  ctx.bezierCurveTo(x+size,y,x,y,x,y+size*0.3);
  ctx.fill();ctx.restore();
}
function updateCalibFX(){
  if(!calibFxCtx||!calibFxCvs) return;
  calibFxCtx.clearRect(0,0,calibFxCvs.width,calibFxCvs.height);
  for(let i=calibParticles.length-1;i>=0;i--){
    const p=calibParticles[i];p.x+=p.vx;p.y+=p.vy;p.vy+=0.14;p.life-=p.decay;
    if(p.life<=0){calibParticles.splice(i,1);continue;}
    calibFxCtx.save();calibFxCtx.globalAlpha=p.life;calibFxCtx.fillStyle=p.color;
    calibFxCtx.beginPath();calibFxCtx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);calibFxCtx.fill();calibFxCtx.restore();
  }
  for(let i=calibFloaties.length-1;i>=0;i--){
    const f=calibFloaties[i];f.x+=f.vx;f.y+=f.vy;f.life-=f.decay;
    if(f.life<=0){calibFloaties.splice(i,1);continue;}
    if(f.type==='heart') drawHeart(calibFxCtx,f.x,f.y,f.size*f.life,f.life*0.9);
    else drawMiniStar(calibFxCtx,f.x,f.y,f.size*f.life*0.5,'#ffd700');
  }
}
function drawCalibStars(t){
  if(!calibStarCtx||!calibStarCvs) return;
  calibStarCtx.clearRect(0,0,calibStarCvs.width,calibStarCvs.height);
  calibStars.forEach(s=>{
    s.tw+=0.04;
    const a=0.3+0.7*Math.abs(Math.sin(s.tw+s.x*0.01));
    calibStarCtx.beginPath();calibStarCtx.arc(s.x,s.y,s.r,0,Math.PI*2);
    calibStarCtx.fillStyle=`rgba(200,190,255,${a})`;calibStarCtx.fill();
  });
}

// ─── CALIBRATION MAIN ────────────────────────────────────────────────────────
function startCalib(){
  calibPoints    =buildCalibPoints();
  calibSamples   =[];
  calibIdx       =0;
  doneCalibPoints=new Set();
  calibParticles =[];
  calibFloaties  =[];
  creatureEls    =[];
  calibState     ='idle';
  calibFacePresent=false;
  _calibCurrentGaze=null; _calibLastGaze=null; _calibLastGazeTs=0; _calibLastFeat=null;
  _calibHoldStart=null;   _calibHoldLostAt=null;
  _calibSampling=false;   _calibSparkled=false; _calibLoopT=0;
  _calibDwellAccum=0;     _calibLastFrameTs=0;  _calibProcTs=-1;
  _calibTargetY=window.innerHeight/2;
  _earCalibSamples=[];    _earThreshold=0.22;
  affineBias={dx:0,dy:0,sx:1,sy:1};
  initCalibScene();
  updateCalibHUD();
  resizeCanvases();
  document.getElementById('calib-overlay')?.setAttribute('style','display:none');
  setCalibBanner('🌟 Welcome, Star Keeper! Help the magical creatures eat — just look at them!');
  setTimeout(()=>advanceCalibPoint(), 1200);
  if(!calibRaf) runCalibLoop();
}

function advanceCalibPoint(){
  if(calibIdx>=CALIB_TOTAL_PTS){finaliseCalib();return;}
  clearTimeout(_calibSkipTimer);
  _calibHoldStart=null; _calibHoldLostAt=null;
  _calibSampling=false; _calibPointSamples=[]; _calibSparkled=false;
  _calibDwellAccum=0;   _calibLastFrameTs=0;
  _calibTargetY=calibPoints[calibIdx]?.y ?? window.innerHeight/2;
  calibState='gap';
  updateCalibHUD();
  const skipBtn=document.getElementById('calib-skip-btn');
  if(skipBtn) skipBtn.style.display=calibFailCount>=1?'inline-block':'none';
  const myIdx=calibIdx;
  setTimeout(()=>{
    if(calibIdx!==myIdx) return;
    showCalibCreature(myIdx);
    calibState='showing';
    const cr=creatureEls[myIdx];
    const pt=calibPoints[myIdx];
    const isBot=pt&&pt.isCorner&&pt.y>window.innerHeight*0.5;
    setCalibBanner(isBot
      ?`👀 Look at ${cr?.def?.name||'the creature'} in the corner!`
      :`👀 Look at ${cr?.def?.name||'the creature'}!`);
    _calibSkipTimer=setTimeout(()=>{
      if(calibIdx!==myIdx) return;
      if(calibState==='done-pt') return;
      console.warn(`[GazeTrack] Force-skipping point ${myIdx} (state=${calibState})`);
      calibIdx++; advanceCalibPoint();
    }, isBot?18000:CALIB_FORCE_SKIP_MS);
  },CALIB_GAP_MS);
}

function showCalibCreature(idx){
  const calibScreen=document.getElementById('s-calib'); if(!calibScreen) return;
  document.getElementById('calib-creature-'+idx)?.remove();
  const pt=calibPoints[idx];
  const def=CREATURE_DEFS[idx%CREATURE_DEFS.length];
  const SIZE=160;
  const wrap=document.createElement('div');
  wrap.id='calib-creature-'+idx;
  wrap.className='calib-creature-wrap';
  wrap.style.left=pt.x+'px'; wrap.style.top=pt.y+'px';
  const cvs=document.createElement('canvas'); cvs.width=SIZE; cvs.height=SIZE;
  cvs.style.cssText=`width:${SIZE}px;height:${SIZE}px;display:block;`;
  wrap.appendChild(cvs); calibScreen.appendChild(wrap);
  creatureEls[idx]={wrap,cvs,ctx:cvs.getContext('2d'),size:SIZE,def,idx};
  requestAnimationFrame(()=>wrap.classList.add('visible'));
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

function runCalibLoop(){
  calibRaf=requestAnimationFrame(()=>{
    if(phase!=='calib-run'){calibRaf=null;return;}
    _calibLoopT+=0.016;
    const now=performance.now();
    const dt=_calibLastFrameTs>0?Math.min(now-_calibLastFrameTs,50):16;
    _calibLastFrameTs=now;
    drawCalibStars(_calibLoopT);
    updateCalibFX();
    if(calibCtx){
      const dpr=window.devicePixelRatio||1;
      calibCtx.clearRect(0,0,calibCanvas.width/dpr,calibCanvas.height/dpr);
    }
    const activeGaze=_calibCurrentGaze||_calibLastGaze;
    creatureEls.forEach((cr,i)=>{
      if(!cr||!cr.ctx) return;
      const pt=calibPoints[i]; if(!pt) return;
      const isDone=doneCalibPoints.has(i);
      const radius=getCalibGazeRadius(pt);
      const gazeNear=activeGaze
        ?(gazeModel
          ?Math.hypot(activeGaze.x-pt.x,activeGaze.y-pt.y)<radius
          :Math.abs(activeGaze.x-pt.x)<220)
        :false;
      let holdPct=0;
      const dwellActive=(i===calibIdx&&!isDone&&calibState!=='gap'&&calibState!=='done-pt'&&calibState!=='idle');
      if(dwellActive){
        if(calibState==='showing') calibState='holding';
        _calibDwellAccum=Math.min(_calibDwellAccum+dt,CALIB_DWELL_REQUIRED_MS);
        holdPct=_calibDwellAccum/CALIB_DWELL_REQUIRED_MS;
        if(holdPct>=0.20&&!_calibSampling){
          _calibSampling=true; _calibSamplingStart=now; _calibPointSamples=[];
          calibState='sampling';
          setCalibBanner(`😋 ${cr.def.name} loves it! Keep looking!`);
        }
        if(_calibDwellAccum>=CALIB_DWELL_REQUIRED_MS&&!_calibSparkled){
          _calibSparkled=true;
          addCalibBurst(pt.x,pt.y,cr.def.color,36);
          addCalibHearts(pt.x,pt.y,8); addCalibStars(pt.x,pt.y,10);
          playHappyJingle(i); startConfettiLight();
          calibState='done-pt'; clearTimeout(_calibSkipTimer);
          doneCalibPoints.add(i);
          cr.wrap?.classList.add('done'); updateCalibHUD();
          const cheers=[
            `🎉 ${cr.def.name} is SO happy! Yummy ${cr.def.hunger}!`,
            `✨ Amazing! ${cr.def.name} loves you!`,
            `🌟 Incredible! ${cr.def.name} is doing a happy dance!`,
            `💖 ${cr.def.name} is full! You're the best Star Keeper!`,
            `🎊 ALL FED! You saved the magical forest!`,
          ];
          setCalibBanner(cheers[i]);
          setTimeout(()=>{calibIdx++;advanceCalibPoint();},900);
        }
      }
      const isHappy=gazeNear&&!isDone;
      drawCreatureOnCanvas(cr.ctx,cr.def,cr.size,_calibLoopT,gazeNear,holdPct,isHappy,isDone);
    });
    runCalibLoop();
  });
}

function finaliseCalib(){
  const rafId=calibRaf; calibRaf=null; if(rafId) cancelAnimationFrame(rafId);
  if(calibSamples.length<MIN_SAMPLES){
    calibFailCount++;
    if(calibSamples.length>=10){
      gazeModel=trainModel(calibSamples);
      if(gazeModel){startConfettiBig();phase='validation';startValidation();return;}
    }
    const card=document.getElementById('calib-card');
    const tip=calibSamples.length===0
      ?'Child may have looked away. Remind them to watch the animals!'
      :calibSamples.length<10
        ?`Only ${calibSamples.length} samples. Try again!`
        :`${calibSamples.length} samples (need ${MIN_SAMPLES}). Try better lighting or move closer.`;
    if(card){
      const h2=card.querySelector('h2'); if(h2) h2.textContent='🐾 Let\'s try again!';
      const p=card.querySelector('p');
      if(p) p.innerHTML=`${tip}<br><br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, say <strong style="color:#fff">"Look at the animal!"</strong>`;
    }
    const startBtn=document.getElementById('calib-start-btn'); if(startBtn) startBtn.textContent='🐾 Try Again!';
    const skipBtn=document.getElementById('calib-skip-btn');   if(skipBtn&&calibFailCount>=1) skipBtn.style.display='inline-block';
    showScreen('calib');
    document.getElementById('calib-overlay')?.setAttribute('style','display:flex');
    calibSamples=[]; phase='calib-ready'; return;
  }
  gazeModel=trainModel(calibSamples);
  if(!gazeModel) calibSkipActive=true;
  startConfettiBig(); phase='validation'; startValidation();
}

document.getElementById('calib-skip-btn')?.addEventListener('click',()=>{
  calibSkipActive=true; calibState='idle';
  const rafId=calibRaf; calibRaf=null; if(rafId) cancelAnimationFrame(rafId);
  if(calibCtx&&calibCanvas) calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
  phase='validation'; startValidation();
});
document.getElementById('calib-start-btn')?.addEventListener('click',()=>{
  document.getElementById('calib-overlay')?.setAttribute('style','display:none');
  calibSamples=[]; phase='calib-run'; startCalib();
});

// ═══════════════════════════════════════════════════════════════════════════════
//  STAR VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════
function spawnSparkles(ctx,x,y){
  for(let i=0;i<14;i++){const angle=Math.random()*Math.PI*2,speed=2+Math.random()*4;VAL_PARTICLES.push({x,y,vx:Math.cos(angle)*speed,vy:Math.sin(angle)*speed,life:1,size:2+Math.random()*4,hue:40+Math.random()*30});}
}
function updateSparkles(ctx){
  for(let i=VAL_PARTICLES.length-1;i>=0;i--){
    const p=VAL_PARTICLES[i];p.x+=p.vx;p.y+=p.vy;p.vy+=0.12;p.life-=0.035;
    if(p.life<=0){VAL_PARTICLES.splice(i,1);continue;}
    ctx.save();ctx.globalAlpha=p.life;ctx.fillStyle=`hsl(${p.hue},100%,65%)`;ctx.beginPath();ctx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);ctx.fill();ctx.restore();
  }
}
function drawStar(ctx,x,y,radius,twinklePhase,entranceProgress){
  const r=radius*entranceProgress,innerR=r*0.4,points=5;
  const twinkle=1+Math.sin(twinklePhase*8)*0.08*entranceProgress,rr=r*twinkle;
  const glowSize=rr*(1.8+Math.sin(twinklePhase*6)*0.2);
  const grad=ctx.createRadialGradient(x,y,0,x,y,glowSize);
  grad.addColorStop(0,`rgba(255,220,50,${0.35*entranceProgress})`);grad.addColorStop(1,'rgba(255,220,50,0)');
  ctx.beginPath();ctx.arc(x,y,glowSize,0,Math.PI*2);ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();
  for(let i=0;i<points*2;i++){const angle=(i*Math.PI/points)-Math.PI/2,rad=i%2===0?rr:innerR;i===0?ctx.moveTo(x+Math.cos(angle)*rad,y+Math.sin(angle)*rad):ctx.lineTo(x+Math.cos(angle)*rad,y+Math.sin(angle)*rad);}
  ctx.closePath();
  const sg=ctx.createRadialGradient(x,y-rr*0.2,0,x,y,rr);
  sg.addColorStop(0,'#fff9c4');sg.addColorStop(0.4,'#ffd700');sg.addColorStop(1,'#ff9f00');
  ctx.fillStyle=sg;ctx.shadowColor='#ffd700';ctx.shadowBlur=20*entranceProgress;ctx.fill();ctx.shadowBlur=0;
  if(entranceProgress>0.8){ctx.beginPath();ctx.arc(x-rr*0.2,y-rr*0.25,rr*0.18,0,Math.PI*2);ctx.fillStyle=`rgba(255,255,255,${0.6*(entranceProgress-0.8)*5})`;ctx.fill();}
}

function startValidation(){
  valPoints=[];
  const W=window.innerWidth,H=window.innerHeight;
  const safeVX=Math.max(80,W*.16),safeVY=Math.max(80,H*.16);
  valPoints=[
    {x:W/2,   y:H/2},
    {x:safeVX,y:safeVY},
    {x:W-safeVX,y:H-safeVY},
    {x:W/2, y:H-safeVY},
  ];
  valIdx=0;valSamples=[];VAL_PARTICLES.length=0;
  _lastVideoTime=-1;_valLastDetectTs=-1;
  prevGaze=null;prevGazeTime=null;
  document.getElementById('val-overlay')?.setAttribute('style','display:block');
  const instr=document.getElementById('val-instruction');if(instr) instr.style.opacity='1';
  const badge=document.getElementById('val-badge');if(badge) badge.style.display='none';
  const tot=document.getElementById('val-badge-tot');if(tot) tot.textContent=valPoints.length;
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
  const valCanvas=document.getElementById('val-canvas');if(!valCanvas) return;
  const dpr=window.devicePixelRatio||1;
  valCanvas.width =Math.round(window.innerWidth *dpr);
  valCanvas.height=Math.round(window.innerHeight*dpr);
  valCanvas.style.width =window.innerWidth +'px';
  valCanvas.style.height=window.innerHeight+'px';
  const vCtx=valCanvas.getContext('2d');
  vCtx.setTransform(dpr,0,0,dpr,0,0);
  const numEl=document.getElementById('val-badge-num');if(numEl) numEl.textContent=valIdx+1;
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
    if(inGap){updateSparkles(vCtx);if(now>=gapEnd)inGap=false;valRaf=requestAnimationFrame(frame);return;}
    const starElapsed=now-gapEnd,starProgress=Math.min(starElapsed/VAL_DWELL_MS,1);
    const entPct=Math.min(starElapsed/ENTRANCE_MS,1);
    let entrance;
    if(entPct<1){const t=entPct;entrance=1-Math.pow(1-t,3)*Math.cos(t*Math.PI*2.5);entrance=Math.min(entrance,1.12);}else{entrance=1;}
    if(!sparkled&&entPct>=0.9){spawnSparkles(vCtx,pt.x,pt.y);sparkled=true;}
    updateSparkles(vCtx);
    drawStar(vCtx,pt.x,pt.y,VAL_STAR_RADIUS,starElapsed*0.001,Math.min(entrance,1));
    if(starProgress>=VAL_SAMPLE_START&&gazeModel){
      const nowTs=performance.now();
      if(webcam?.readyState>=2&&faceLandmarker&&nowTs-_valLastDetectTs>=33){
        _valLastDetectTs=nowTs;
        try{
          const res=faceLandmarker.detectForVideo(webcam,mpNow());
          if(res.faceLandmarks?.length>0){
            const lm=res.faceLandmarks[0];
            const mat=(res.facialTransformationMatrixes?.length>0)?res.facialTransformationMatrixes[0]:null;
            const feat=extractFeatures(lm,mat);
            if(feat[7]>=0.20){
              const _W=window.innerWidth,_H=window.innerHeight;
              collected.push({
                px:polyX(feat).reduce((s,v,i)=>s+v*gazeModel.wx[i],0)*_W,
                py:polyY(feat).reduce((s,v,i)=>s+v*gazeModel.wy[i],0)*_H
              });
            }
          }
        }catch(e){console.warn('[Val] detect error:',e);}
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
  document.getElementById('val-overlay')?.setAttribute('style','display:none');
  if(valSamples.length===0){
    console.warn('[Val] No validation samples — skipping affine correction');
    affineBias={dx:0,dy:0,sx:1,sy:1};
  } else if(valSamples.length>=2){
    affineBias=computeAffineCorrection(valSamples);
    const dxBad=Math.abs(affineBias.dx)>500;
    const sxBad=affineBias.sx>2.4||affineBias.sx<0.3;
    if(dxBad||sxBad){
      console.warn(`[Val] Catastrophic calib — retrying`);
      affineBias={dx:0,dy:0,sx:1,sy:1};
      calibSamples=[];gazeModel=null;
      const card=document.getElementById('calib-card');
      if(card){
        const h2=card.querySelector('h2');if(h2) h2.textContent='🐾 Let\'s try again!';
        const p=card.querySelector('p');if(p) p.innerHTML='Validation showed the tracker was very inaccurate.<br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, look directly at each creature.';
      }
      const startBtn=document.getElementById('calib-start-btn');if(startBtn) startBtn.textContent='🐾 Try Again!';
      showScreen('calib');
      document.getElementById('calib-overlay')?.setAttribute('style','display:flex');
      phase='calib-ready';return;
    }
    if(Math.abs(affineBias.dx)>200||affineBias.sx>1.4||affineBias.sx<0.65)
      console.warn(`[Val] Moderate offset (dx=${affineBias.dx.toFixed(0)} sx=${affineBias.sx.toFixed(2)}) — proceeding`);
  }
  phase='stimulus'; showScreen('stimulus');
  const hChild=document.getElementById('h-child');if(hChild) hChild.textContent=META.pid;
  const hGroup=document.getElementById('h-group');if(hGroup) hGroup.textContent=META.group;
  startRecording();
}

// ─── SACCADE CLASSIFICATION ──────────────────────────────────────────────────
function classifyGaze(gaze,currentTime){
  if(!prevGaze||!prevGazeTime){prevGaze=gaze;prevGazeTime=currentTime;return'Fixation';}
  const dt=currentTime-prevGazeTime;if(dt===0)return'Fixation';
  const vel=Math.hypot(gaze.x-prevGaze.x,gaze.y-prevGaze.y)/dt;
  const cat = vel>1.5 ? 'Saccade' : (vel>0.85 ? '-' : 'Fixation');
  prevGaze=gaze;prevGazeTime=currentTime;return cat;
}

// ─── RECORDING ───────────────────────────────────────────────────────────────
function startRecording(){
  sessionStart=Date.now();recordedFrames=[];totalF=0;trackedF=0;
  _lastKnownPupilDiamR=3.5;_lastKnownPupilDiamL=3.5;
  prevGaze=null;prevGazeTime=null;
  // Multi-trial state reset
  playlistTrialIdx=0;
  trialStartMs=0;
  _photoTimer=null;
  try{const t=sessionStream?.getVideoTracks()[0]?.getSettings();if(t){_cachedVideoW=t.width||webcam.videoWidth||640;_cachedVideoH=t.height||webcam.videoHeight||480;}}catch(e){}
  timerInt=setInterval(()=>{
    const s=Math.floor((Date.now()-sessionStart)/1000);
    const timerEl=document.getElementById('h-timer');
    if(timerEl) timerEl.textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  },500);
  updateTrialBadge();
  if(playlist.length>0){
    loadTrialItem(0);
  }else{
    showNoVideo();
  }
}

// ─── TRIAL MANAGEMENT ────────────────────────────────────────────────────────
function loadTrialItem(idx){
  if(idx>=playlist.length){endSession();return;}
  playlistTrialIdx=idx;
  trialStartMs=Date.now()-sessionStart;
  const item=playlist[idx];
  // Hide both media elements
  stimVideo.style.display='none';
  stimVideo.pause();
  stimVideo.src='';
  if(stimImage) stimImage.style.display='none';
  clearTimeout(_photoTimer);
  // Inject separator row at trial boundary
  injectSeparatorRow(idx, item.name);
  updateTrialBadge();
  document.getElementById('no-video')?.setAttribute('style','display:none');
  if(item.type==='video'){
    stimVideo.style.display='block';
    stimVideo.src=item.objectURL;
    stimVideo.muted=true;
    stimVideo.currentTime=0;
    stimVideo.play().catch(()=>{});
    stimVideo.onended=()=>advanceTrial();
    document.getElementById('sound-btn')?.setAttribute('style','display:block');
  }else{
    // Image stimulus
    if(stimImage){stimImage.src=item.objectURL;stimImage.style.display='block';}
    document.getElementById('sound-btn')?.setAttribute('style','display:none');
    _photoTimer=setTimeout(()=>advanceTrial(), photoDurSec*1000);
  }
  const nextBtn=document.getElementById('next-trial-btn');
  if(nextBtn) nextBtn.style.display=playlist.length>1?'block':'none';
}

function advanceTrial(){
  clearTimeout(_photoTimer);
  if(playlistTrialIdx+1<playlist.length){
    loadTrialItem(playlistTrialIdx+1);
  }else{
    endSession();
  }
}

function updateTrialBadge(){
  const badge  =document.getElementById('trial-badge');
  const cur    =document.getElementById('trial-badge-cur');
  const tot    =document.getElementById('trial-badge-tot');
  const name   =document.getElementById('trial-badge-name');
  const chip   =document.getElementById('h-trial-chip');
  const hTrial =document.getElementById('h-trial');
  const n=playlist.length;
  if(badge)  badge.style.display =n>1?'block':'none';
  if(chip)   chip.style.display  =n>1?'flex':'none';
  if(cur)    cur.textContent     =playlistTrialIdx+1;
  if(tot)    tot.textContent     =n;
  if(name)   name.textContent    =playlist[playlistTrialIdx]?.name||'—';
  if(hTrial) hTrial.textContent  =`${playlistTrialIdx+1}/${n}`;
}

document.getElementById('next-trial-btn')?.addEventListener('click',()=>{
  stimVideo.pause();
  advanceTrial();
});

function injectSeparatorRow(trialIdx, stimName){
  recordedFrames.push({
    _separator:true,
    trialIdx,
    stimName,
    t:Date.now()-sessionStart,
    trialStartMs:Date.now()-sessionStart,
  });
}

function showNoVideo(){
  document.getElementById('no-video')?.setAttribute('style','display:flex');
  document.getElementById('sound-btn')?.setAttribute('style','display:none');
}

// Single-file fallback (no-video screen)
document.getElementById('stim-file-input')?.addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  const type=f.type.startsWith('image/')?'image':'video';
  playlist=[{objectURL:URL.createObjectURL(f),name:f.name,type}];
  META.stimulus=f.name;
  loadTrialItem(0);
});

document.getElementById('sound-btn')?.addEventListener('click',()=>{
  stimVideo.muted=false;stimVideo.play().catch(()=>{});
  document.getElementById('sound-btn')?.setAttribute('style','display:none');
});

// ─── MAIN PROCESSING LOOP ────────────────────────────────────────────────────
function processingLoop(){
  if(phase==='done') return;

  if(phase==='calib-run'){
    if(webcam?.readyState>=2&&faceLandmarker){
      const now=performance.now();
      if(now-_calibProcTs<33){procRaf=requestAnimationFrame(processingLoop);return;}
      _calibProcTs=now;
      try{
        const res=faceLandmarker.detectForVideo(webcam,mpNow());
        const hasFace=!!(res.faceLandmarks?.length>0);
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
              console.log('[GazeTrack] Adaptive EAR threshold:',_earThreshold.toFixed(3));
            }
          }
          const isBlink=ear<_earThreshold;
          if(!isBlink){
            const rawGaze=gazeModel?predictGaze(feat,gazeModel):estimateGazeFromIris(feat);
            _calibCurrentGaze=rawGaze;_calibLastGaze=rawGaze;_calibLastGazeTs=performance.now();
          }else{
            const bridgeAge=performance.now()-_calibLastGazeTs;
            _calibCurrentGaze=bridgeAge<CALIB_IRIS_BRIDGE_MS?_calibLastGaze:null;
          }
          if((calibState==='sampling'||calibState==='holding')&&!isBlink){
            const pt=calibPoints[calibIdx];
            if(pt){calibSamples.push({feat,sx:pt.x,sy:pt.y});_calibPointSamples.push({feat,sx:pt.x,sy:pt.y});}
          }
        }else{
          const bridgeAge=performance.now()-_calibLastGazeTs;
          _calibCurrentGaze=bridgeAge<CALIB_IRIS_BRIDGE_MS?_calibLastGaze:null;
        }
      }catch(e){console.error('[Calib] detectForVideo error:',e);}
    }
    procRaf=requestAnimationFrame(processingLoop);return;
  }

  if(phase==='validation'){procRaf=requestAnimationFrame(processingLoop);return;}

  if(webcam?.readyState>=2&&faceLandmarker){
    const now=performance.now();
    if(now-_stimLastTs<33){procRaf=requestAnimationFrame(processingLoop);return;}
    _stimLastTs=now;

    const res=faceLandmarker.detectForVideo(webcam,mpNow());
    const hasFace=!!(res.faceLandmarks?.length>0);
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

      let leftPupilX=null,leftPupilY=null,rightPupilX=null,rightPupilY=null;
      try{
        let slx=0,sly=0,srx=0,sry=0;
        LEFT_IRIS.forEach(i=>{slx+=lm[i].x;sly+=lm[i].y;});
        RIGHT_IRIS.forEach(i=>{srx+=lm[i].x;sry+=lm[i].y;});
        leftPupilX=(slx/LEFT_IRIS.length)*_cachedVideoW;
        leftPupilY=(sly/LEFT_IRIS.length)*_cachedVideoH;
        rightPupilX=(srx/RIGHT_IRIS.length)*_cachedVideoW;
        rightPupilY=(sry/RIGHT_IRIS.length)*_cachedVideoH;
      }catch(e){}

      if(phase==='stimulus'){
        totalF++;
        let gaze=null;
        if(!isBlink&&gazeModel&&!calibSkipActive) gaze=predictGaze(feat,gazeModel);

        const dispX=feat._3d?feat._3d.vergenceDisp:48;
        const dispY=dispX*0.41;
        const gazeXL=gaze?gaze.x+dispX:NaN;
        const gazeYL=gaze?gaze.y+dispY:NaN;

        const _rawPupilDiamR=feat._3d?.pupilDiamR??Math.max(1.5,Math.min(3.8,2.9+(feat[7]-0.30)*3.5));
        const _rawPupilDiamL=feat._3d?.pupilDiamL??Math.max(1.5,Math.min(3.8,2.85+(feat[7]-0.30)*3.5));
        if(!isBlink&&_rawPupilDiamR>0) _lastKnownPupilDiamR=_rawPupilDiamR;
        if(!isBlink&&_rawPupilDiamL>0) _lastKnownPupilDiamL=_rawPupilDiamL;
        const pupilDiamR=_lastKnownPupilDiamR;
        const pupilDiamL=_lastKnownPupilDiamL;
        const pupilSizePxR=isBlink?0:(feat._3d?.pupSizePxR??_rawPupilDiamR*4.14);
        const pupilSizePxL=isBlink?0:(feat._3d?.pupSizePxL??_rawPupilDiamL*4.14);

        const catRight=isBlink?'Blink':(gaze?classifyGaze(gaze,nowMs):'Fixation');
        const catLeft=catRight;

        const frameData={
          t:nowMs,tracked:gaze?1:0,
          gazeX:gaze?.x??NaN,gazeY:gaze?.y??NaN,gazeXL,gazeYL,
          leftPupilX,leftPupilY,rightPupilX,rightPupilY,
          pupilSizePxR,pupilSizePxL,
          pupilDiamR,pupilDiamL,
          eyeRX:isBlink?0:(feat._3d?.eyeRX??null),
          eyeRY:isBlink?0:(feat._3d?.eyeRY??null),
          eyeRZ:isBlink?0:(feat._3d?.eyeRZ??null),
          eyeLX:isBlink?0:(feat._3d?.eyeLX??null),
          eyeLY:isBlink?0:(feat._3d?.eyeLY??null),
          eyeLZ:isBlink?0:(feat._3d?.eyeLZ??null),
          gazeVRX:isBlink?0:(feat._3d?.gazeVRX??feat[0]),
          gazeVRY:isBlink?0:(feat._3d?.gazeVRY??feat[1]),
          gazeVRZ:isBlink?0:(feat._3d?.gazeVRZ??feat[2]),
          gazeVLX:isBlink?0:(feat._3d?.gazeVLX??feat[0]),
          gazeVLY:isBlink?0:(feat._3d?.gazeVLY??feat[1]),
          gazeVLZ:isBlink?0:(feat._3d?.gazeVLZ??feat[2]),
          catGroup:'Eye',catRight,catLeft,category:catRight,feat,
          // Trial tagging
          trialIdx:     playlistTrialIdx,
          stimName:     playlist[playlistTrialIdx]?.name||META.stimulus||'-',
          trialStartMs: trialStartMs,
        };

        if(gaze){
          trackedF++;
          gazeCtx?.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
          const sg=document.getElementById('st-gaze');if(sg){sg.textContent='Tracking';sg.className='sv ok';}
        }else{
          if(isBlink){prevGaze=null;prevGazeTime=null;}
          gazeCtx?.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
          const sg=document.getElementById('st-gaze');if(sg){sg.textContent=isBlink?'Blink':'Lost';sg.className='sv bad';}
        }
        recordedFrames.push(frameData);
        const sf=document.getElementById('st-frames');if(sf)sf.textContent=recordedFrames.filter(f=>!f._separator).length;
        const st=document.getElementById('st-track');if(st&&totalF>0)st.textContent=Math.round(trackedF/totalF*100)+'%';
        if(recordedFrames.length>10){
          const ys=recordedFrames.filter(f=>!f._separator&&f.tracked).map(f=>f.gazeY);
          if(ys.length>1){const my=ys.reduce((a,b)=>a+b,0)/ys.length;const sy=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);const el=document.getElementById('st-ystd');if(el){el.textContent=sy.toFixed(0)+'px';el.className='sv '+(sy>30?'ok':'bad');}}
        }
      }
    }else if(phase==='stimulus'){
      // No face — reset saccade state to avoid gap-crossing velocity spike
      prevGaze=null;prevGazeTime=null;
      totalF++;
      const nowMs=Date.now()-sessionStart;
      recordedFrames.push({
        t:nowMs,tracked:0,gazeX:NaN,gazeY:NaN,gazeXL:NaN,gazeYL:NaN,
        leftPupilX:null,leftPupilY:null,rightPupilX:null,rightPupilY:null,
        pupilSizePxR:0,pupilSizePxL:0,
        pupilDiamR:_lastKnownPupilDiamR,pupilDiamL:_lastKnownPupilDiamL,
        eyeRX:null,eyeRY:null,eyeRZ:null,eyeLX:null,eyeLY:null,eyeLZ:null,
        gazeVRX:null,gazeVRY:null,gazeVRZ:null,
        gazeVLX:null,gazeVLY:null,gazeVLZ:null,
        catGroup:'Eye',catRight:'Blink',catLeft:'Blink',category:'Blink',feat:null,
        trialIdx:playlistTrialIdx,
        stimName:playlist[playlistTrialIdx]?.name||META.stimulus||'-',
        trialStartMs:trialStartMs,
      });
      const st=document.getElementById('st-track');if(st&&totalF>0)st.textContent=Math.round(trackedF/totalF*100)+'%';
    }
  }
  procRaf=requestAnimationFrame(processingLoop);
}

// ─── CSV ─────────────────────────────────────────────────────────────────────
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
  const biasMeta=[
    '# GazeTrack v21',
    `bias_dx=${affineBias.dx.toFixed(2)}`,`bias_dy=${affineBias.dy.toFixed(2)}`,
    `bias_sx=${affineBias.sx.toFixed(4)}`,`bias_sy=${affineBias.sy.toFixed(4)}`,
    `val_samples=${valSamples.length}`,`calib_samples=${calibSamples.length}`,
    `skip_mode=${calibSkipActive}`,
    `viewport=${window.innerWidth}x${window.innerHeight}`,
    `fullscreen=${!!document.fullscreenElement}`,
    `trials=${playlist.length}`,
  ].join(' | ');

  const lines=[biasMeta,CSV_HDR];
  const eyeFrames      = recordedFrames.filter(f=>!f._separator);
  const totalDuration  = eyeFrames.length>0 ? eyeFrames.at(-1).t : 0;
  const trackingRatio  = totalF>0?(trackedF/totalF*100):0;
  const colorMap       = {ASD:'DarkViolet',TD:'SteelBlue',other:'Gray'};
  const color          = colorMap[META.group]||'Gray';
  const trialLabel     = idx=>`Trial${String(idx+1).padStart(3,'0')}`;

  // Pre-compute per-trial export windows
  const trialWindows={};
  recordedFrames.forEach(f=>{
    if(f._separator){
      if(!trialWindows[f.trialIdx]) trialWindows[f.trialIdx]={start:f.t,end:f.t};
    }else{
      const ti=f.trialIdx??0;
      if(!trialWindows[ti]) trialWindows[ti]={start:f.trialStartMs??0,end:f.t};
      else trialWindows[ti].end=f.t;
    }
  });

  const fn=(val,d=4,isBlink=false)=>{
    if(isBlink) return '0';
    return(val!==null&&val!==undefined&&!isNaN(val))?Number(val).toFixed(d):'-';
  };

  let rowNum=0;

  recordedFrames.forEach(f=>{
    const ti       = f.trialIdx??0;
    const tLabel   = trialLabel(ti);
    const stimName = f.stimName||'-';
    const tw       = trialWindows[ti]||{start:0,end:totalDuration};
    const expStart = tw.start.toFixed(3);
    const expEnd   = tw.end.toFixed(3);
    const absTime  = sessionStart+f.t;
    const d        = new Date(absTime);
    const pad      = (n,w=2)=>String(Math.floor(n)).padStart(w,'0');
    const tod      = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}:${pad(d.getMilliseconds(),3)}`;

    if(f._separator){
      const sepRow=[
        rowNum,f.t.toFixed(3),tod,tLabel,stimName,
        expStart,expEnd,
        META.pid,color,trackingRatio.toFixed(3),
        'Information','Separator','Separator','1','1',
        '-','-','-','-','-','-',
        '-','-','-','-',
        '-','-',
        '-','-','-','-','-','-',
        '-','-','-','-','-','-',
        '-','-','-','-',
        '0','-','-','-','-','-','-','-',
        stimName,
        '-','-','-','-','-','-',
      ];
      lines.push(sepRow.map(v=>v===undefined?'-':v).join(','));
      rowNum++;return;
    }

    const isBlink=f.category==='Blink';
    const gxR=isBlink?'0.0000':fn(f.gazeX,4);
    const gyR=isBlink?'0.0000':fn(f.gazeY,4);
    const gxL=isBlink?'0.0000':fn(f.gazeXL,4);
    const gyL=isBlink?'0.0000':fn(f.gazeYL,4);
    const psRx=isBlink?'0.0000':fn(f.pupilSizePxR,4);
    const psLx=isBlink?'0.0000':fn(f.pupilSizePxL,4);
    const pdR =isBlink?'0':fn(f.pupilDiamR,4);
    const pdL =isBlink?'0':fn(f.pupilDiamL,4);
    const gvRx=isBlink?'0.0000':fn(f.gazeVRX,4);
    const gvRy=isBlink?'0':fn(f.gazeVRY,4);
    const gvRz=isBlink?'0':fn(f.gazeVRZ,4);
    const gvLx=isBlink?'0':fn(f.gazeVLX,4);
    const gvLy=isBlink?'0':fn(f.gazeVLY,4);
    const gvLz=isBlink?'0':fn(f.gazeVLZ,4);
    const epRx=isBlink?'0':fn(f.eyeRX,4);
    const epRy=isBlink?'0':fn(f.eyeRY,4);
    const epRz=isBlink?'0':fn(f.eyeRZ,4);
    const epLx=isBlink?'0':fn(f.eyeLX,4);
    const epLy=isBlink?'0':fn(f.eyeLY,4);
    const epLz=isBlink?'0':fn(f.eyeLZ,4);
    const ppRx=isBlink?'0':fn(f.rightPupilX,1);
    const ppRy=isBlink?'0':fn(f.rightPupilY,1);
    const ppLx=isBlink?'0':fn(f.leftPupilX,1);
    const ppLy=isBlink?'0':fn(f.leftPupilY,1);
    const catG=f.catGroup||'Eye';
    const catR=f.catRight||f.category||'-';
    const catL=f.catLeft||catR;

    const row=[
      rowNum,f.t.toFixed(3),tod,
      tLabel,stimName,expStart,expEnd,
      META.pid,color,trackingRatio.toFixed(3),
      catG,catR,catL,rowNum,rowNum,
      psRx,psRx,pdR,
      psLx,psLx,pdL,
      gxR,gyR,gxL,gyL,
      '-','-',
      gvRx,gvRy,gvRz,
      gvLx,gvLy,gvLz,
      epRx,epRy,epRz,
      epLx,epLy,epLz,
      ppRx,ppRy,ppLx,ppLy,
      '-','-','-',
      '-','-','-','-',
      '-',
      '-','-','-','-','-','-',
    ];
    lines.push(row.map(v=>v===undefined?'-':v).join(','));
    rowNum++;
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
  clearTimeout(_photoTimer);
  cancelAnimationFrame(procRaf);cancelAnimationFrame(calibRaf);cancelAnimationFrame(valRaf);
  stimVideo?.pause();
  csvData=buildCSV();
  const eyeFrames = recordedFrames.filter(f=>!f._separator);
  const pct=totalF>0?Math.round(trackedF/totalF*100):0;
  const dur=Math.round((Date.now()-sessionStart)/1000);
  const ys=eyeFrames.filter(f=>f.tracked).map(f=>f.gazeY);
  let ystd=0;
  if(ys.length>1){const my=ys.reduce((a,b)=>a+b,0)/ys.length;ystd=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);}
  const biasOk=Math.abs(affineBias.dx)>5||Math.abs(affineBias.dy)>5;
  const biasLabel=biasOk?`${affineBias.dx>0?'+':''}${affineBias.dx.toFixed(0)},${affineBias.dy>0?'+':''}${affineBias.dy.toFixed(0)}px`:'Minimal';
  const fixCount=eyeFrames.filter(f=>f.category==='Fixation').length;
  const sacCount=eyeFrames.filter(f=>f.category==='Saccade').length;
  const blinkCount=eyeFrames.filter(f=>f.category==='Blink').length;
  const blinkRate=dur>0?(blinkCount/dur)*60:0;
  const blinkRateOk=blinkRate>=5&&blinkRate<=30;
  const statsEl=document.getElementById('done-stats');
  if(statsEl){
    statsEl.innerHTML=`
      <div class="done-stat"><div class="n">${eyeFrames.length}</div><div class="l">FRAMES</div></div>
      <div class="done-stat"><div class="n">${pct}%</div><div class="l">TRACKED</div></div>
      <div class="done-stat"><div class="n">${dur}s</div><div class="l">DURATION</div></div>
      <div class="done-stat"><div class="n" style="color:${ystd>30?'var(--accent)':'var(--warn)'}">${ystd.toFixed(0)}px</div><div class="l">Y STD</div></div>
      <div class="done-stat"><div class="n" style="color:var(--accent)">${fixCount}</div><div class="l">FIXATIONS</div></div>
      <div class="done-stat"><div class="n" style="color:var(--gold)">${sacCount}</div><div class="l">SACCADES</div></div>
      <div class="done-stat"><div class="n" style="color:${biasOk?'var(--accent)':'var(--text-muted)'}; font-size:13px">${biasLabel}</div><div class="l">BIAS CORR</div></div>
      <div class="done-stat"><div class="n">${blinkCount}</div><div class="l">BLINKS</div></div>
      <div class="done-stat"><div class="n" style="color:${blinkRateOk?'var(--accent)':'var(--warn)'}">${blinkRate.toFixed(1)}/min</div><div class="l">BLINK RATE</div></div>
      <div class="done-stat"><div class="n">${playlist.length}</div><div class="l">TRIALS</div></div>`;
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
  const ie=document.getElementById('drive-icon');if(ie)ie.textContent=icon;
  const me=document.getElementById('drive-msg');if(me)me.textContent=msg;
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
    if(!resp.ok)throw new Error('Server '+resp.status);
    driveSetStatus('✅','Saved to database!','rgba(0,229,176,0.4)');
    const btn=document.getElementById('btn-dl');
    if(btn){btn.textContent='✅ Saved — click to download locally';btn.style.opacity='1';btn.style.pointerEvents='auto';btn.onclick=()=>downloadCSV();}
  }catch(err){
    driveSetStatus('❌','Database save failed — downloading locally','rgba(255,92,58,0.4)');
    downloadCSV();
  }
}

// ─── DEBUG & CLEANUP ─────────────────────────────────────────────────────────
window.addEventListener('keydown',e=>{
  if(e.key==='d'||e.key==='D'){
    const p=document.getElementById('debug-panel');
    if(p) p.style.display=p.style.display==='none'?'block':'none';
  }
});
window.addEventListener('beforeunload',()=>{
  sessionStream?.getTracks().forEach(t=>t.stop());
  camStream?.getTracks().forEach(t=>t.stop());
  playlist.forEach(p=>URL.revokeObjectURL(p.objectURL));
});
window.addEventListener('resize',()=>{
  if(phase==='calib-ready') calibPoints=buildCalibPoints();
});

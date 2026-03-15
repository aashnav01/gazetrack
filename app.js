console.log('%c GazeTrack v12  -  live position all-clear + child-friendly star validation ','background:#00e5b0;color:#000;font-weight:bold;font-size:13px');
import { FaceLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs';

const isMobile = /Android|iPad|iPhone|iPod|Mobile/i.test(navigator.userAgent)
  || (navigator.maxTouchPoints > 1 && window.innerWidth < 1200);
const MP_DELEGATE = isMobile ? 'CPU' : 'GPU';

// CONFIG
const CALIB_MS    = 20000;
const MIN_SAMPLES = 30;
const RIDGE_ALPHA = 0.01;
const LEFT_IRIS   = [468,469,470,471];
const RIGHT_IRIS  = [473,474,475,476];
const L_CORNERS   = [33,133];
const R_CORNERS   = [362,263];

//  STAR VALIDATION CONFIG 
// Longer dwell so children have time to locate and fixate each star
// Star validation constants (superseded values kept for reference)
const VAL_GAP_MS      = 500;    // gap between stars (ms)
const VAL_STAR_RADIUS = 36;     // star radius (px)
const VAL_INTRO_MS    = 1800;   // intro screen duration (ms)

// -- Audio ------------------------------------------------------------------
let _audioCtx=null;
function getAudioCtx(){if(!_audioCtx){try{_audioCtx=new AudioContext();}catch(e){}}return _audioCtx;}

const calibSound=(()=>{return()=>{const a=getAudioCtx();if(!a)return;const o=a.createOscillator(),g=a.createGain();o.connect(g);g.connect(a.destination);o.frequency.value=880;g.gain.setValueAtTime(0,a.currentTime);g.gain.linearRampToValueAtTime(0.15,a.currentTime+0.01);g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+0.3);o.start();o.stop(a.currentTime+0.3);};})();

function playChime(freq,vol,duration){
  const a=getAudioCtx();if(!a)return;
  const o=a.createOscillator(),g=a.createGain();
  o.connect(g);g.connect(a.destination);
  o.type='sine';
  o.frequency.setValueAtTime(freq*0.8,a.currentTime);
  o.frequency.linearRampToValueAtTime(freq,a.currentTime+0.1);
  g.gain.setValueAtTime(0,a.currentTime);
  g.gain.linearRampToValueAtTime(vol,a.currentTime+0.05);
  g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+duration);
  o.start();o.stop(a.currentTime+duration);
}


// -- PEEK-A-BOO CALIBRATION CONFIG -------------------------------------------
const CALIB_ATTRACT_MS     = 700;   // box jiggles - draws eye to location
const CALIB_POPUP_MS       = 350;   // spring-in animation
const CALIB_STILL_MS       = 1800;  // character holds still - sample window
const CALIB_BYE_MS         = 250;   // wave goodbye
const CALIB_GAP_MS         = 400;   // blank between points - eyes settle
const CALIB_MIN_PT_SAMPLES = 4;     // min gaze samples per point before advancing
const CALIB_CHAR_R         = 42;    // character radius (px)

// STATE
let phase = 'intake';
let faceLandmarker = null;
let camStream = null;
let sessionStream = null;
let gazeModel = null;
let calibSamples = [];
let valPoints=[], valIdx=0, valSamples=[], valRaf=null, valStart=0;
let affineBias = {dx:0,dy:0,sx:1,sy:1};
let calibStart=0, calibSoundPlayed=false;
let calibPath=[], calibRaf=null, procRaf=null;
let recordedFrames=[], totalF=0, trackedF=0;
let sessionStart=0, timerInt=null;
let csvData=null;
let calibFacePresent=false;
let videoBlob=null;
let META={pid:'',age:'',group:'',clinician:'',location:'',notes:''};

//  POSITION ALL-CLEAR STATE 
// Tracks consecutive good-position frames to debounce noise
let _goodFrameStreak    = 0;
let _badFrameStreak     = 0;
let _allclearShowing    = false;
let _allclearHideTimer  = null;
const GOOD_STREAK_NEEDED = 8;   // ~1.6s of stable good position before showing
const BAD_STREAK_HIDE    = 12;  // ~2.4s of bad before hiding again

// DOM
const screens={
  intake:   document.getElementById('s-intake'),
  loading:  document.getElementById('s-loading'),
  calib:    document.getElementById('s-calib'),
  stimulus: document.getElementById('s-stimulus'),
  done:     document.getElementById('s-done'),
};
const camPreview  = document.getElementById('cam-preview');
const camCanvas   = document.getElementById('cam-canvas');
const camCtx      = camCanvas.getContext('2d');
const calibCanvas = document.getElementById('calib-canvas');
const calibCtx    = calibCanvas.getContext('2d');
const gazeCanvas  = document.getElementById('gaze-canvas');
const gazeCtx     = gazeCanvas.getContext('2d');
const webcam      = document.getElementById('webcam');
const stimVideo   = document.getElementById('stim-video');
const allclearBanner = document.getElementById('position-allclear');

function showScreen(n){
  Object.values(screens).forEach(s=>s.classList.remove('active'));
  screens[n].classList.add('active');
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  POSITION ALL-CLEAR LOGIC
//  
//  Runs every preflight frame. Requires that:
//   - face detected at good distance (MediaPipe IOD check in previewLoop)
//   - lighting is good (brightness pixel check)
//   - both true for GOOD_STREAK_NEEDED consecutive frames
//
//  When the banner appears it also announces specific checks that
//  passed so the parent knows exactly what improved.
//  When position degrades again it hides after BAD_STREAK_HIDE frames
//  to avoid flickering during small head movements.
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function updateAllClear(bright){
  // Read directly from pfState  -  no re-computation, single source of truth
  const posOk   = pfState.face === 'pass';
  const lightOk = pfState.light === 'pass';
  const allOk   = posOk && lightOk;

  if(allOk){
    _badFrameStreak = 0;
    _goodFrameStreak = Math.min(_goodFrameStreak + 1, GOOD_STREAK_NEEDED + 1);
    if(_goodFrameStreak >= GOOD_STREAK_NEEDED && !_allclearShowing){
      showAllClear(bright);
    }
  } else {
    _goodFrameStreak = 0;
    _badFrameStreak = Math.min(_badFrameStreak + 1, BAD_STREAK_HIDE + 1);
    if(_badFrameStreak >= BAD_STREAK_HIDE && _allclearShowing){
      hideAllClear();
    }
  }
}

function showAllClear(bright){
  _allclearShowing = true;
  clearTimeout(_allclearHideTimer);
  const tags = ['\u2713 Face visible . Good distance', '\u2713 Lighting OK'];
  document.getElementById('allclear-tags').innerHTML =
    tags.map(t=>`<span class="allclear-tag">${t}</span>`).join('');
  document.getElementById('allclear-detail').textContent =
    `Brightness ${Math.round(bright)}/255`;
  allclearBanner.classList.add('show');
  playChime(660, 0.08, 0.4);
}

function hideAllClear(){
  _allclearShowing = false;
  allclearBanner.classList.remove('show');
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  PRE-FLIGHT  -  pixel analysis, runs during intake
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
const pfState={cam:'scanning',face:'scanning',light:'scanning',browser:'scanning'};
let pfRaf=null;
let _pfSamples=[], _pfThrottle=0;
let _pfCanvas=null, _pfCtx=null;

function pfSet(id,state,msg){
  pfState[id]=state;
  const card=document.getElementById('pfc-'+id);
  const stat=document.getElementById('pfc-'+id+'-s');
  if(!card||!stat)return;
  card.className='pf-check '+state;
  stat.innerHTML=msg;
  pfUpdateScore();
}

function pfUpdateScore(){
  const vals=Object.values(pfState);
  const done=vals.filter(v=>v!=='scanning').length;
  const passes=vals.filter(v=>v==='pass').length;
  const warns=vals.filter(v=>v==='warn').length;
  const total=vals.length;
  const score=Math.round(((passes+warns*0.6)/total)*100);
  const fill=document.getElementById('pf-score-fill');
  const pct=document.getElementById('pf-score-pct');
  if(fill){fill.style.width=score+'%';fill.style.background=score>=75?'var(--accent)':score>=50?'var(--gold)':'var(--warn)';}
  if(pct) pct.textContent=done===total?score+'%':'...';
  const tips=[];
  if(pfState.light==='fail')  tips.push('<strong>\u{1F4A1} Too dark:</strong> Add a front-facing lamp.');
  if(pfState.light==='warn')  tips.push('<strong>\u{1F4A1} Lighting:</strong> Brighter room helps iris detection.');
  if(pfState.face==='fail')   tips.push('<strong>\u{1F464} No face:</strong> Make sure child is in frame, camera at eye level.');
  if(pfState.browser==='warn')tips.push('<strong>\u{1F310} Browser:</strong> Use Chrome for best webcam performance.');
  const adv=document.getElementById('pf-advice');
  if(adv){adv.innerHTML=tips.join('<br>');adv.className='pf-advice'+(tips.length?' show':'');}
  // Button label only  -  enabled/disabled is managed solely by checkStartBtn
  const btn=document.getElementById('start-btn');
  if(!btn.disabled){
    const critFails=['cam','face'].filter(k=>pfState[k]==='fail').length;
    if(critFails>0){btn.textContent='\u26A0\uFE0F Proceed Anyway';btn.style.background='linear-gradient(135deg,#ff9f43,#e17f20)';}
    else if(done<total){btn.textContent='Begin Session \u2192';btn.style.background='';}
    else if(score>=75){btn.textContent='\u2705 All Clear  -  Begin Session';btn.style.background='';}
    else{btn.textContent='\u26A0\uFE0F Proceed with Warnings';btn.style.background='linear-gradient(135deg,#ca8a04,#a16207)';}
  }
}

function pfAnalyseFrame(){
  if(phase!=='intake'){return;}
  const now=performance.now();
  if(now-_pfThrottle<200){pfRaf=requestAnimationFrame(pfAnalyseFrame);return;}
  _pfThrottle=now;
  if(!camPreview||camPreview.readyState<2){pfRaf=requestAnimationFrame(pfAnalyseFrame);return;}
  if(!_pfCanvas){_pfCanvas=document.createElement('canvas');_pfCanvas.width=80;_pfCanvas.height=60;_pfCtx=_pfCanvas.getContext('2d',{willReadFrequently:true});}
  try{
    _pfCtx.drawImage(camPreview,0,0,80,60);
    const d=_pfCtx.getImageData(0,0,80,60).data;
    let sumR=0,sumG=0,sumB=0,n=0;
    for(let i=0;i<d.length;i+=4){
      sumR+=d[i];sumG+=d[i+1];sumB+=d[i+2];n++;
    }
    const brightness=(sumR+sumG+sumB)/(n*3);
    _pfSamples.push(brightness);
    if(_pfSamples.length>3) _pfSamples.shift();
    const avgBright=_pfSamples.reduce((a,b)=>a+b,0)/_pfSamples.length;

    // Lighting
    if(avgBright>=60&&avgBright<=220)    pfSet('light','pass',`\u2713 Good (${Math.round(avgBright)}/255)`);
    else if(avgBright<40)                pfSet('light','fail',`\u2717 Too dark (${Math.round(avgBright)})  -  add light`);
    else if(avgBright<60)                pfSet('light','warn',` Dim (${Math.round(avgBright)})  -  improve lighting`);
    else                                 pfSet('light','warn',` Bright (${Math.round(avgBright)})  -  reduce backlight`);

    //  Live all-clear update (lighting only  -  position driven by MediaPipe in previewLoop) 
    updateAllClear(avgBright);

  }catch(e){}
  pfRaf=requestAnimationFrame(pfAnalyseFrame);
}

function pfCheckBrowser(){
  const ua=navigator.userAgent;
  const isChrome=/Chrome/.test(ua)&&!/Edg/.test(ua)&&!/OPR/.test(ua);
  const isEdge=/Edg/.test(ua);
  const isFirefox=/Firefox/.test(ua);
  if(isChrome)       pfSet('browser','pass','\u2713 Chrome  -  optimal');
  else if(isEdge)    pfSet('browser','pass','\u2713 Edge  -  good');
  else if(isFirefox) pfSet('browser','warn',' Firefox  -  use Chrome for best results');
  else               pfSet('browser','warn',' Use Chrome for best results');
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  CAMERA INIT
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
async function initCamera(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:640,max:1280},height:{ideal:480,max:720},facingMode:'user'},audio:false
    });
    camStream=stream;
    camPreview.srcObject=stream;
    camPreview.play();
    document.getElementById('cam-dot').classList.add('ok');
    document.getElementById('cam-status-txt').textContent='Camera active';
    document.getElementById('chk-cam').classList.add('ok');
    document.getElementById('chk-cam').textContent='\u2713 Cam';
    const t=stream.getVideoTracks()[0].getSettings();
    const w=t.width||640, h=t.height||480;
    pfSet('cam','pass',`\u2713 ${w}x${h}`);
    checkStartBtn();
    pfRaf=requestAnimationFrame(pfAnalyseFrame);
    pfCheckBrowser();
    loadPreviewDetector();
  }catch(e){
    document.getElementById('cam-status-txt').textContent='\u2717 Camera error  -  allow access';
    pfSet('cam','fail','\u2717 Camera denied or not found');
    pfSet('face','fail','\u2717 No camera');
    pfSet('light','fail','\u2717 No camera');
  }
}

// Preview mesh
let previewFl=null, previewRaf=null, lastPreviewTs=-1, _prevLastRun=0;
async function loadPreviewDetector(){
  try{
    const resolver=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
    previewFl=await FaceLandmarker.createFromOptions(resolver,{
      baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:MP_DELEGATE},
      runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
    });
    previewLoop();
  }catch(e){}
}

function previewLoop(){
  if(phase!=='intake'){return;}
  const now=performance.now();
  if(now-_prevLastRun<125){previewRaf=requestAnimationFrame(previewLoop);return;}
  _prevLastRun=now;
  if(camPreview.readyState>=2&&previewFl){
    const rect=camCanvas.getBoundingClientRect();
    const dW=Math.round(rect.width)||640, dH=Math.round(rect.height)||480;
    if(camCanvas.width!==dW||camCanvas.height!==dH){camCanvas.width=dW;camCanvas.height=dH;}
    let ts=camPreview.currentTime*1000;
    if(ts<=lastPreviewTs) ts=lastPreviewTs+0.001;
    lastPreviewTs=ts;
    try{
      const res=previewFl.detectForVideo(camPreview,ts);
      const hasFace=!!(res.faceLandmarks&&res.faceLandmarks.length>0);
      camCtx.clearRect(0,0,camCanvas.width,camCanvas.height);
      if(hasFace){
        drawPreviewMesh(res.faceLandmarks[0]);
        const lm=res.faceLandmarks[0];
        const hasIris=!!(lm[468]&&lm[473]);
        document.getElementById('chk-face').classList.add('ok');
        document.getElementById('chk-face').textContent='\u2713 Face';
        document.getElementById('chk-iris').classList.toggle('ok',hasIris);
        document.getElementById('chk-iris').textContent=hasIris?'\u2713 Iris':'\u{1F441} Iris';

        //  MediaPipe-driven Position & Distance check 
        // iodNorm = inter-ocular distance in normalised 0-1 space
        // Reliable regardless of lighting/skin tone  -  pure geometry
        if(hasIris){
          const iodNorm=Math.hypot(lm[473].x-lm[468].x,lm[473].y-lm[468].y);
          const faceCX=(lm[33].x+lm[263].x)/2; // horizontal face centre (0=left,1=right)
          const offCentre=faceCX<0.25||faceCX>0.75;
          const lEAR=Math.hypot(lm[159].x-lm[145].x,lm[159].y-lm[145].y);
          const rEAR=Math.hypot(lm[386].x-lm[374].x,lm[386].y-lm[374].y);
          const earPx=(lEAR+rEAR)/2;
          const qPct=(earPx/(iodNorm+1e-6))>0.08?95:75;
          document.getElementById('q-fill').style.width=qPct+'%';
          document.getElementById('q-pct').textContent=qPct+'%';
          // iodNorm thresholds (calibrated for typical laptop/tablet webcam FOV):
          // >0.22 = very close (<35cm), 0.13-0.22 = good (40-70cm),
          // 0.07-0.13 = far (70-100cm), <0.07 = very far / small face
          if(iodNorm>0.22)
            pfSet('face','warn',` Too close (IOD ${iodNorm.toFixed(3)})  -  move back ~15 cm`);
          else if(iodNorm>=0.13)
            pfSet('face','pass', offCentre
              ? `\u2713 Good distance . Move slightly to centre`
              : `\u2713 Face visible . Good distance (~50-70 cm)`);
          else if(iodNorm>=0.07)
            pfSet('face','warn',` Too far (IOD ${iodNorm.toFixed(3)})  -  move ${offCentre?'closer & to centre':'~20 cm closer'}`);
          else
            pfSet('face','warn',` Very far or face at edge  -  move much closer`);
        } else {
          // Face found but no iris  -  can still give useful feedback
          document.getElementById('q-fill').style.width='40%';
          document.getElementById('q-pct').textContent='40%';
          pfSet('face','warn',' Face detected but iris not visible  -  look at camera');
        }
      }else{
        document.getElementById('chk-face').classList.remove('ok');document.getElementById('chk-face').textContent='\u{1F464} Face';
        document.getElementById('chk-iris').classList.remove('ok');document.getElementById('chk-iris').textContent='\\u{1F441} Iris';
        document.getElementById('q-fill').style.width='0%';document.getElementById('q-pct').textContent=' - ';
        pfSet('face','fail','\u2717 No face detected  -  check camera position');
      }
    }catch(e){}
  }
  previewRaf=requestAnimationFrame(previewLoop);
}

function drawPreviewMesh(lm){
  const W=camCanvas.width,H=camCanvas.height;
  const fx=x=>(1-x)*W, fy=y=>y*H;
  [[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33],[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]].forEach(pts=>{
    camCtx.beginPath();pts.forEach((idx,i)=>{const p=lm[idx];i===0?camCtx.moveTo(fx(p.x),fy(p.y)):camCtx.lineTo(fx(p.x),fy(p.y));});
    camCtx.strokeStyle='#00e5b0';camCtx.lineWidth=1.5;camCtx.globalAlpha=0.7;camCtx.stroke();camCtx.globalAlpha=1;
  });
  [[468,469],[473,474]].forEach(([c,e])=>{
    if(!lm[c]||!lm[e])return;
    const cx=fx(lm[c].x),cy=fy(lm[c].y),ex2=fx(lm[e].x),ey2=fy(lm[e].y);
    const r=Math.hypot(ex2-cx,ey2-cy)+1;
    camCtx.beginPath();camCtx.arc(cx,cy,r,0,Math.PI*2);camCtx.strokeStyle='#00e5b0';camCtx.lineWidth=2;camCtx.globalAlpha=0.9;camCtx.stroke();camCtx.globalAlpha=1;
    camCtx.beginPath();camCtx.arc(cx,cy,2,0,Math.PI*2);camCtx.fillStyle='#00e5b0';camCtx.fill();
  });
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  FORM
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function checkStartBtn(){
  const pidOk=document.getElementById('f-pid').value.trim().length>0;
  const groupOk=document.querySelector('input[name="group"]:checked')!==null;
  const btn=document.getElementById('start-btn');
  btn.disabled=!(pidOk&&groupOk);
  if(pidOk&&groupOk) pfUpdateScore();
}
document.getElementById('f-pid').addEventListener('input',checkStartBtn);
document.querySelectorAll('input[name="group"]').forEach(r=>r.addEventListener('change',checkStartBtn));
document.getElementById('video-drop').addEventListener('click',()=>document.getElementById('video-input').click());
document.getElementById('video-input').addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  videoBlob=URL.createObjectURL(f);
  document.getElementById('video-hint').style.display='none';
  document.getElementById('video-drop').insertAdjacentHTML('beforeend',`<div class="chosen">\u2713 ${f.name}</div>`);
});

document.getElementById('start-btn').addEventListener('click',()=>{
  META.pid=document.getElementById('f-pid').value.trim();
  META.age=document.getElementById('f-age').value;
  META.group=document.querySelector('input[name="group"]:checked')?.value||'';
  META.clinician=document.getElementById('f-clinician').value.trim();
  META.location=document.getElementById('f-location').value.trim();
  META.notes=document.getElementById('f-notes').value.trim();
  cancelAnimationFrame(previewRaf);
  cancelAnimationFrame(pfRaf);
  phase='loading';
  showScreen('loading');
  beginSession();
});

initCamera();

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  SESSION START
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
async function beginSession(){
  try{
    if(previewFl){
      faceLandmarker=previewFl;
      document.getElementById('load-msg').textContent='Model ready  -  starting camera...';
    }else{
      document.getElementById('load-msg').textContent='Loading eye tracking model...';
      const resolver=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
      faceLandmarker=await FaceLandmarker.createFromOptions(resolver,{
        baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:MP_DELEGATE},
        runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:true,outputIrisLandmarks:true
      });
    }
    if(camStream){sessionStream=camStream;camStream=null;}
    else{sessionStream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});}
    webcam.srcObject=sessionStream;
    await new Promise(r=>{webcam.onloadedmetadata=()=>{webcam.play();r();};});
    resizeCanvases();
    window.addEventListener('resize',resizeCanvases);
    phase='calib-ready';
    showScreen('calib');
    procRaf=requestAnimationFrame(processingLoop);
  }catch(err){
    console.error(err);
    document.getElementById('load-msg').textContent='\u274C '+(err.message||'Startup error');
  }
}

function resizeCanvases(){
  calibCanvas.width=gazeCanvas.width=window.innerWidth;
  calibCanvas.height=gazeCanvas.height=window.innerHeight;
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  FEATURE EXTRACTION
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function extractFeatures(lm,mat){
  const avg=ids=>{const s={x:0,y:0,z:0};ids.forEach(i=>{s.x+=lm[i].x;s.y+=lm[i].y;s.z+=(lm[i].z||0);});return{x:s.x/ids.length,y:s.y/ids.length,z:s.z/ids.length};};
  const li=avg(LEFT_IRIS),ri=avg(RIGHT_IRIS);
  const lIn=lm[L_CORNERS[0]],lOut=lm[L_CORNERS[1]];
  const rIn=lm[R_CORNERS[0]],rOut=lm[R_CORNERS[1]];
  const lW=Math.hypot(lOut.x-lIn.x,lOut.y-lIn.y)+1e-6;
  const rW=Math.hypot(rOut.x-rIn.x,rOut.y-rIn.y)+1e-6;
  const lCx=(lIn.x+lOut.x)/2, rCx=(rIn.x+rOut.x)/2;
  const liX=(li.x-lCx)/lW, riX=(ri.x-rCx)/rW, avgX=(liX+riX)/2;
  let pitchDeg=0;
  if(mat?.data){const m=mat.data;pitchDeg=Math.asin(Math.max(-1,Math.min(1,-m[6])))*180/Math.PI/30;}
  const nose=lm[1],fore=lm[10],chin=lm[152];
  const pitchZ=((nose.z||0)-((fore.z||0)+(chin.z||0))/2)*10;
  const vertMain=(Math.abs(pitchDeg)>0.001)?pitchDeg:pitchZ;
  const foreheadY=fore.y;
  const faceCY=(fore.y+chin.y)/2, faceH=Math.abs(chin.y-fore.y)+1e-6;
  const irisY=((li.y+ri.y)/2-faceCY)/faceH;
  const lEAR=Math.hypot(lm[159].x-lm[145].x,lm[159].y-lm[145].y)/lW;
  const rEAR=Math.hypot(lm[386].x-lm[374].x,lm[386].y-lm[374].y)/rW;
  const ear=(lEAR+rEAR)/2;
  const iod=Math.hypot(ri.x-li.x,ri.y-li.y);
  return[liX,riX,vertMain,foreheadY,irisY,(li.y+ri.y)/2,avgX,ear,iod];
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  RIDGE REGRESSION
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function poly(f){return[1,...f.slice(0,7)];}
function ridgeFit(X,y,alpha=RIDGE_ALPHA){
  const n=X[0].length;
  const XtX=Array.from({length:n},()=>new Array(n).fill(0));
  const Xty=new Array(n).fill(0);
  for(let r=0;r<X.length;r++){for(let i=0;i<n;i++){Xty[i]+=X[r][i]*y[r];for(let j=0;j<n;j++)XtX[i][j]+=X[r][i]*X[r][j];}}
  for(let i=0;i<n;i++)XtX[i][i]+=alpha;
  const aug=XtX.map((row,i)=>[...row,Xty[i]]);
  for(let c=0;c<n;c++){
    let p=c;for(let r=c+1;r<n;r++)if(Math.abs(aug[r][c])>Math.abs(aug[p][c]))p=r;
    [aug[c],aug[p]]=[aug[p],aug[c]];
    const pv=aug[c][c];if(Math.abs(pv)<1e-12)continue;
    for(let j=c;j<=n;j++)aug[c][j]/=pv;
    for(let r=0;r<n;r++)if(r!==c){const f=aug[r][c];for(let j=c;j<=n;j++)aug[r][j]-=f*aug[c][j];}
  }
  return aug.map(r=>r[n]);
}
function trainModel(samples){
  if(samples.length<MIN_SAMPLES)return null;
  const X=samples.map(s=>poly(s.feat));
  return{wx:ridgeFit(X,samples.map(s=>s.sx)),wy:ridgeFit(X,samples.map(s=>s.sy))};
}
function predictGaze(feat,model){
  if(!model)return null;
  const pf=poly(feat);
  const gx=pf.reduce((s,v,i)=>s+v*model.wx[i],0);
  const gy=pf.reduce((s,v,i)=>s+v*model.wy[i],0);
  const cx=affineBias.sx*gx+affineBias.dx;
  const cy=affineBias.sy*gy+affineBias.dy;
  return{x:Math.max(0,Math.min(window.innerWidth,cx)),y:Math.max(0,Math.min(window.innerHeight,cy))};
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  AFFINE BIAS
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function computeAffineCorrection(pairs){
  function linfit(ps,ts){
    const n=ps.length,mp=ps.reduce((a,b)=>a+b,0)/n,mt=ts.reduce((a,b)=>a+b,0)/n;
    let num=0,den=0;
    for(let i=0;i<n;i++){num+=(ps[i]-mp)*(ts[i]-mt);den+=(ps[i]-mp)**2;}
    const s=den>1e-6?num/den:1;
    const sc=Math.max(0.6,Math.min(1.6,s));
    return{s:sc,d:mt-sc*mp};
  }
  const fx=linfit(pairs.map(p=>p.px),pairs.map(p=>p.tx));
  const fy=linfit(pairs.map(p=>p.py),pairs.map(p=>p.ty));
  return{sx:fx.s,dx:fx.d,sy:fy.s,dy:fy.d};
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  CALIBRATION

// 
//  CALIBRATION  -  Peek-a-Boo Still-Point Approach
//  
//  RESEARCH BASIS:
//  - Children need a STILL target to fixate  -  a moving target produces
//    smooth-pursuit data, not fixation data (unusable for calibration).
//  - Audio + visual onset together draw attention most reliably in ASD.
//  - Use tighter 12% inset grid  -  avoid extreme corners for young eyes.
//  - Experimenter-paced calibration outperforms fixed timers.
//
//  PHASE FLOW PER POINT:
//  1. ATTRACT (700ms): Surprise box jiggles + attention chime at target.
//  2. POPUP (350ms): Character springs out with bounce animation.
//  3. STILL (1800ms): Character holds still  -  gaze samples collected.
//     Clinician can hold SPACEBAR to extend this window if needed.
//  4. BYE (250ms): Character waves + box closes  -  transition cue.
//  5. GAP (400ms): Blank screen  -  eyes settle before next point.
//
//  9 points on tighter 12% inset grid  -  avoids extreme corners.
// 

// Peek-a-boo calibration state
let peekPoints = [];
let peekIdx = 0;
let peekPhase = 'idle';
let peekPhaseStart = 0;
let peekHoldExtend = false;
let peekSamplesThisPoint = 0;
let peekRaf = null;
const CAT_COLS=[{body:'#F8D7E3',stripe:'#E8A0B8'},{body:'#D7EAF8',stripe:'#90BDE0'},{body:'#E8D7F8',stripe:'#C090E0'},{body:'#D7F8E3',stripe:'#80D4A0'},{body:'#F8F0D7',stripe:'#E0C878'},{body:'#F8D7D7',stripe:'#E09090'}];
let bci=0;

function buildPeekPoints(){
  const W=window.innerWidth, H=window.innerHeight;
  const ix=Math.round(W*0.12), iy=Math.round(H*0.12);
  return[
    {x:W/2,   y:H/2},    // centre first  -  easiest, builds trust
    {x:W/2,   y:iy},     // top mid
    {x:W-ix,  y:iy},     // top right
    {x:W-ix,  y:H/2},    // mid right
    {x:W-ix,  y:H-iy},   // bottom right
    {x:W/2,   y:H-iy},   // bottom mid
    {x:ix,    y:H-iy},   // bottom left
    {x:ix,    y:H/2},    // mid left
    {x:ix,    y:iy},     // top left
  ];
}

function _rrect(ctx,x,y,w,h,r){
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
}

function drawBox(ctx,x,y,t){
  const jig=Math.sin(t*Math.PI*8)*6*(1-t);
  const bx=x+jig, by=y+jig*0.5, s=28;
  ctx.save(); ctx.globalAlpha=0.1;
  ctx.beginPath(); ctx.ellipse(bx,by+s+4,s*0.9,s*0.18,0,0,Math.PI*2);
  ctx.fillStyle='#000'; ctx.fill(); ctx.restore();
  ctx.save();
  ctx.fillStyle='#e8a020'; ctx.strokeStyle='#c07010'; ctx.lineWidth=2;
  ctx.beginPath(); _rrect(ctx,bx-s,by-s*0.6,s*2,s*1.6,6); ctx.fill(); ctx.stroke();
  ctx.fillStyle='#f0b830';
  ctx.beginPath(); _rrect(ctx,bx-s-3,by-s*0.6-s*0.45,s*2+6,s*0.45,4); ctx.fill(); ctx.stroke();
  ctx.fillStyle='#fff'; ctx.font='bold 20px sans-serif';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('?',bx,by+s*0.2); ctx.restore();
}

function drawPeekChar(ctx,x,y,scale,wavePct){
  const r=CALIB_CHAR_R*Math.max(0,scale);
  const cy=y+CALIB_CHAR_R*(1-scale)*0.5;
  const {body,stripe}=CAT_COLS[bci];
  ctx.save(); ctx.globalAlpha=0.1;
  ctx.beginPath(); ctx.ellipse(x,cy+r+3,r*0.85,r*0.18,0,0,Math.PI*2);
  ctx.fillStyle='#000'; ctx.fill(); ctx.restore();
  [[-r*0.42,-r*1.1],[r*0.42,-r*1.1]].forEach(([dx])=>{
    const ex=x+dx,ey=cy-r;
    ctx.save(); ctx.translate(ex,ey); ctx.rotate(dx<0?-0.12:0.12);
    ctx.beginPath(); ctx.ellipse(0,-r*0.72,10,r*0.72,0,0,Math.PI*2); ctx.fillStyle=body; ctx.fill();
    ctx.beginPath(); ctx.ellipse(0,-r*0.72,6,r*0.55,0,0,Math.PI*2); ctx.fillStyle=stripe; ctx.fill();
    ctx.restore();
  });
  ctx.beginPath(); ctx.arc(x,cy,r,0,Math.PI*2); ctx.fillStyle=body; ctx.fill();
  ctx.beginPath(); ctx.ellipse(x,cy+r*0.15,r*0.6,r*0.45,0,0,Math.PI*2);
  ctx.fillStyle='#fff'; ctx.globalAlpha=0.7; ctx.fill(); ctx.globalAlpha=1;
  ctx.beginPath(); ctx.arc(x-r*0.28,cy-r*0.15,r*0.1,0,Math.PI*2); ctx.fillStyle='#1a1a2e'; ctx.fill();
  ctx.beginPath(); ctx.arc(x+r*0.28,cy-r*0.15,r*0.1,0,Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(x,cy+r*0.12,r*0.2,0,Math.PI); ctx.strokeStyle='#1a1a2e'; ctx.lineWidth=2*scale; ctx.stroke();
  if(wavePct>0){
    const wAngle=-Math.PI/4-Math.sin(wavePct*Math.PI*3)*0.7;
    ctx.save(); ctx.translate(x+r*0.7,cy+r*0.1); ctx.rotate(wAngle);
    ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(r*0.7,0);
    ctx.strokeStyle=body; ctx.lineWidth=7*scale; ctx.lineCap='round'; ctx.stroke(); ctx.restore();
  }
  ctx.beginPath();
  ctx.moveTo(x-r*0.35,cy+r*0.05); ctx.lineTo(x-r*0.75,cy-r*0.1);
  ctx.moveTo(x-r*0.35,cy+r*0.15); ctx.lineTo(x-r*0.75,cy+r*0.15);
  ctx.moveTo(x+r*0.35,cy+r*0.05); ctx.lineTo(x+r*0.75,cy-r*0.1);
  ctx.moveTo(x+r*0.35,cy+r*0.15); ctx.lineTo(x+r*0.75,cy+r*0.15);
  ctx.strokeStyle=stripe; ctx.lineWidth=1.5*scale; ctx.stroke();
}

function drawCalibBadge(ctx,ptIdx,total,samples,ph){
  const W=calibCanvas.width;
  const ok=samples>=CALIB_MIN_PT_SAMPLES;
  ctx.save();
  ctx.fillStyle=ok?'rgba(0,180,100,0.88)':'rgba(20,20,40,0.78)';
  ctx.beginPath(); _rrect(ctx,W-194,14,178,ph==='still'?62:46,10); ctx.fill();
  ctx.fillStyle='#fff'; ctx.font='bold 14px sans-serif';
  ctx.textAlign='left'; ctx.textBaseline='top';
  ctx.fillText('Point '+(ptIdx+1)+' / '+total,W-182,22);
  ctx.font='12px sans-serif';
  ctx.fillText('Samples: '+samples+(ok?'  \u2713 Good':'  collecting\u2026'),W-182,40);
  if(ph==='still'){
    ctx.fillStyle='rgba(255,255,255,0.55)'; ctx.font='11px sans-serif';
    ctx.fillText('Hold SPACE to extend',W-182,56);
  }
  ctx.restore();
}

function startPeekCalib(){
  peekPoints=buildPeekPoints(); peekIdx=0;
  bci=Math.floor(Math.random()*CAT_COLS.length);
  calibSamples=[];
  window.addEventListener('keydown',_peekSpaceDown);
  window.addEventListener('keyup',_peekSpaceUp);
  runPeekPoint();
}
function _peekSpaceDown(e){if(e.code==='Space')peekHoldExtend=true;}
function _peekSpaceUp(e){if(e.code==='Space')peekHoldExtend=false;}

function runPeekPoint(){
  if(peekIdx>=peekPoints.length){finalisePeekCalib();return;}
  const pt=peekPoints[peekIdx];
  peekPhase='attract'; peekPhaseStart=performance.now(); peekSamplesThisPoint=0;
  const notes=[440,523,587,659,698,784,880,988,1047];
  playChime(notes[peekIdx%notes.length],0.14,0.5);

  function frame(){
    if(phase!=='calib-run'){cancelAnimationFrame(peekRaf);return;}
    const now=performance.now(), elapsed=now-peekPhaseStart;
    calibCanvas.width=window.innerWidth; calibCanvas.height=window.innerHeight;
    calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
    // Soft spotlight
    const grad=calibCtx.createRadialGradient(pt.x,pt.y,10,pt.x,pt.y,110);
    grad.addColorStop(0,'rgba(255,255,200,0.11)'); grad.addColorStop(1,'rgba(255,255,200,0)');
    calibCtx.beginPath(); calibCtx.arc(pt.x,pt.y,110,0,Math.PI*2);
    calibCtx.fillStyle=grad; calibCtx.fill();

    if(peekPhase==='attract'){
      drawBox(calibCtx,pt.x,pt.y,elapsed/CALIB_ATTRACT_MS);
      if(elapsed>=CALIB_ATTRACT_MS){peekPhase='popup';peekPhaseStart=now;playChime(880,0.12,0.28);}
    } else if(peekPhase==='popup'){
      const t=Math.min(elapsed/CALIB_POPUP_MS,1);
      const sc=1-Math.pow(1-t,3)*Math.cos(t*Math.PI*2.2);
      drawPeekChar(calibCtx,pt.x,pt.y,Math.min(sc,1.08),0);
      if(elapsed>=CALIB_POPUP_MS){peekPhase='still';peekPhaseStart=now;}
    } else if(peekPhase==='still'){
      drawPeekChar(calibCtx,pt.x,pt.y,1,0);
      if(elapsed>=CALIB_STILL_MS&&!peekHoldExtend){peekPhase='bye';peekPhaseStart=now;playChime(660,0.08,0.28);}
    } else if(peekPhase==='bye'){
      const wp=Math.min(elapsed/CALIB_BYE_MS,1);
      drawPeekChar(calibCtx,pt.x,pt.y,1-wp*0.5,wp);
      if(elapsed>=CALIB_BYE_MS){peekPhase='gap';peekPhaseStart=now;}
    } else if(peekPhase==='gap'){
      calibCtx.globalAlpha=Math.max(0,1-elapsed/CALIB_GAP_MS);
      drawCalibBadge(calibCtx,peekIdx,peekPoints.length,peekSamplesThisPoint,'gap');
      calibCtx.globalAlpha=1;
      if(elapsed>=CALIB_GAP_MS){peekIdx++;runPeekPoint();return;}
    }
    if(peekPhase!=='gap') drawCalibBadge(calibCtx,peekIdx,peekPoints.length,peekSamplesThisPoint,peekPhase);
    peekRaf=requestAnimationFrame(frame);
  }
  peekRaf=requestAnimationFrame(frame);
}

function finalisePeekCalib(){
  cancelAnimationFrame(peekRaf);
  window.removeEventListener('keydown',_peekSpaceDown);
  window.removeEventListener('keyup',_peekSpaceUp);
  calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
  gazeModel=trainModel(calibSamples);
  if(!gazeModel){
    const card=document.getElementById('calib-card');
    card.querySelector('h2').textContent='\u26a0\ufe0f Calibration incomplete';
    card.querySelector('p').textContent='Only '+calibSamples.length+' samples (need '+MIN_SAMPLES+'). Please retry.';
    document.getElementById('calib-start-btn').textContent='\u21ba Try Again';
    document.getElementById('calib-overlay').style.display='flex';
    calibSamples=[];phase='calib-ready';return;
  }
  phase='validation';
  startValidation();
}


// 
//  STAR VALIDATION  -  Quality-Gated
//  
//  Same star visual, but now the star waits for sufficient clean
//  gaze samples before advancing (or advances on max timeout).
//  This prevents a corrupt validation pass when the child looks away.
//
//  VAL_MIN_GOOD_SAMPLES: star stays until this many samples collected
//  VAL_MAX_DWELL_MS:     hard timeout so it never blocks forever
// 

const VAL_MIN_GOOD_SAMPLES = 6;   // must collect this many before advancing
const VAL_MAX_DWELL_MS     = 4000; // hard cap per star (was 2800ms fixed)

// Particle system
const VAL_PARTICLES=[];
function spawnSparkles(ctx,x,y){
  for(let i=0;i<14;i++){
    const a=Math.random()*Math.PI*2,sp=2+Math.random()*4;
    VAL_PARTICLES.push({x,y,vx:Math.cos(a)*sp,vy:Math.sin(a)*sp,life:1,size:2+Math.random()*4,hue:40+Math.random()*30});
  }
}
function updateSparkles(ctx){
  for(let i=VAL_PARTICLES.length-1;i>=0;i--){
    const p=VAL_PARTICLES[i];
    p.x+=p.vx;p.y+=p.vy;p.vy+=0.12;p.life-=0.035;
    if(p.life<=0){VAL_PARTICLES.splice(i,1);continue;}
    ctx.save();ctx.globalAlpha=p.life;
    ctx.fillStyle='hsl('+p.hue+',100%,65%)';
    ctx.beginPath();ctx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);ctx.fill();ctx.restore();
  }
}

function drawStar(ctx,x,y,radius,phase,entrance){
  const r=radius*entrance;
  const innerR=r*0.4, points=5;
  const twinkle=1+Math.sin(phase*8)*0.08*entrance;
  const rr=r*twinkle;
  const glowSize=rr*(1.8+Math.sin(phase*6)*0.2);
  const grad=ctx.createRadialGradient(x,y,0,x,y,glowSize);
  grad.addColorStop(0,'rgba(255,220,50,'+(0.35*entrance)+')');
  grad.addColorStop(1,'rgba(255,220,50,0)');
  ctx.beginPath();ctx.arc(x,y,glowSize,0,Math.PI*2);ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();
  for(let i=0;i<points*2;i++){
    const angle=(i*Math.PI/points)-Math.PI/2;
    const rad=i%2===0?rr:innerR;
    const px=x+Math.cos(angle)*rad, py=y+Math.sin(angle)*rad;
    i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
  }
  ctx.closePath();
  const sg=ctx.createRadialGradient(x,y-rr*0.2,0,x,y,rr);
  sg.addColorStop(0,'#fff9c4');sg.addColorStop(0.4,'#ffd700');sg.addColorStop(1,'#ff9f00');
  ctx.fillStyle=sg;ctx.shadowColor='#ffd700';ctx.shadowBlur=20*entrance;ctx.fill();ctx.shadowBlur=0;
  if(entrance>0.8){
    ctx.beginPath();ctx.arc(x-rr*0.2,y-rr*0.25,rr*0.18,0,Math.PI*2);
    ctx.fillStyle='rgba(255,255,255,'+(0.6*(entrance-0.8)*5)+')';ctx.fill();
  }
}

let _lastVideoTime=-1;
function startValidation(){
  valPoints=[];
  const W=window.innerWidth,H=window.innerHeight;
  const mx=W*.1,my=H*.1;
  valPoints=[{x:W/2,y:H/2},{x:mx,y:my},{x:W-mx,y:my},{x:W-mx,y:H-my},{x:mx,y:H-my}];
  valIdx=0;valSamples=[];VAL_PARTICLES.length=0;
  document.getElementById('val-overlay').style.display='block';
  document.getElementById('val-instruction').style.opacity='1';
  document.getElementById('val-badge').style.display='none';
  document.getElementById('val-badge-tot').textContent=valPoints.length;
  playChime(528,0.1,0.6);
  setTimeout(()=>{
    document.getElementById('val-instruction').style.opacity='0';
    setTimeout(()=>{
      document.getElementById('val-instruction').style.display='none';
      document.getElementById('val-badge').style.display='block';
      runStarDot();
    },400);
  },VAL_INTRO_MS);
}

function runStarDot(){
  if(valIdx>=valPoints.length){finishValidation();return;}
  const pt=valPoints[valIdx];
  const valCanvas=document.getElementById('val-canvas');
  valCanvas.width=window.innerWidth; valCanvas.height=window.innerHeight;
  const vCtx=valCanvas.getContext('2d');
  document.getElementById('val-badge-num').textContent=valIdx+1;
  const notes=[523,659,784,880,1047];
  playChime(notes[valIdx%notes.length],0.12,0.5);

  const collected=[];
  const ENTRANCE_MS=220;
  valStart=performance.now();
  let sparkled=false;
  let inGap=true;
  const gapEnd=valStart+VAL_GAP_MS;

  function frame(){
    const now=performance.now(), elapsed=now-valStart;
    vCtx.clearRect(0,0,valCanvas.width,valCanvas.height);

    if(inGap){
      updateSparkles(vCtx);
      if(now>=gapEnd) inGap=false;
      valRaf=requestAnimationFrame(frame); return;
    }

    const starElapsed=now-gapEnd;
    const entrance=Math.min(starElapsed/ENTRANCE_MS,1);
    let entranceScaled;
    if(entrance<1){
      const t=entrance;
      entranceScaled=1-Math.pow(1-t,3)*Math.cos(t*Math.PI*2.5);
      entranceScaled=Math.min(entranceScaled,1.12);
    } else { entranceScaled=1; }

    if(!sparkled&&entrance>=0.9){spawnSparkles(vCtx,pt.x,pt.y);sparkled=true;}
    updateSparkles(vCtx);
    drawStar(vCtx,pt.x,pt.y,VAL_STAR_RADIUS,starElapsed*0.001,Math.min(entranceScaled,1));

    // Collect gaze samples
    if(starElapsed>=ENTRANCE_MS*0.8&&gazeModel){
      if(webcam.readyState>=2&&faceLandmarker&&webcam.currentTime!==_lastVideoTime){
        _lastVideoTime=webcam.currentTime;
        try{
          const res=faceLandmarker.detectForVideo(webcam,performance.now());
          if(res.faceLandmarks&&res.faceLandmarks.length>0){
            const lm=res.faceLandmarks[0];
            const mat=(res.facialTransformationMatrixes?.length>0)?res.facialTransformationMatrixes[0]:null;
            const feat=extractFeatures(lm,mat);
            if(feat[7]>=0.06){
              const pf=poly(feat);
              collected.push({
                px:pf.reduce((s,v,i)=>s+v*gazeModel.wx[i],0),
                py:pf.reduce((s,v,i)=>s+v*gazeModel.wy[i],0)
              });
            }
          }
        }catch(e){}
      }
    }

    // Advance when min samples collected OR hard timeout reached
    const enoughSamples = collected.length >= VAL_MIN_GOOD_SAMPLES;
    const timedOut = starElapsed >= VAL_MAX_DWELL_MS;

    if(enoughSamples||timedOut){
      if(collected.length>=3){
        const mxs=collected.map(p=>p.px).sort((a,b)=>a-b);
        const mys=collected.map(p=>p.py).sort((a,b)=>a-b);
        const mid=Math.floor(mxs.length/2);
        valSamples.push({px:mxs[mid],py:mys[mid],tx:pt.x,ty:pt.y});
      }
      valIdx++;
      runStarDot(); return;
    }
    valRaf=requestAnimationFrame(frame);
  }
  valRaf=requestAnimationFrame(frame);
}

function finishValidation(){
  cancelAnimationFrame(valRaf);
  document.getElementById('val-overlay').style.display='none';
  if(valSamples.length>=3){
    affineBias=computeAffineCorrection(valSamples);
    const badCalib=Math.abs(affineBias.dx)>300||affineBias.sx>1.4||affineBias.sx<0.7;
    if(badCalib){
      affineBias={dx:0,dy:0,sx:1,sy:1};
      calibSamples=[];gazeModel=null;
      const card=document.getElementById('calib-card');
      card.querySelector('h2').textContent='\ud83d\udc31 Let\'s try again!';
      card.querySelector('p').innerHTML='Child may have looked away \u2014 that\'s OK!<br><br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, say: <strong style="color:#fff">"Watch the kitty!"</strong>';
      document.getElementById('calib-start-btn').textContent='\ud83d\udc31 Play Again!';
      document.getElementById('calib-overlay').style.display='flex';
      phase='calib-ready';return;
    }
  }
  phase='stimulus';
  showScreen('stimulus');
  document.getElementById('h-child').textContent=META.pid;
  document.getElementById('h-group').textContent=META.group;
  startRecording();
}

document.getElementById('calib-start-btn').addEventListener('click',()=>{
  document.getElementById('calib-overlay').style.display='none';
  calibSamples=[];phase='calib-run';startPeekCalib();
});

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  RECORDING
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function startRecording(){
  sessionStart=Date.now();recordedFrames=[];totalF=0;trackedF=0;
  timerInt=setInterval(()=>{
    const s=Math.floor((Date.now()-sessionStart)/1000);
    document.getElementById('h-timer').textContent=`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  },500);
  if(videoBlob){
    stimVideo.src=videoBlob;stimVideo.muted=true;
    stimVideo.play().catch(()=>showNoVideo());
    stimVideo.onerror=showNoVideo;
    stimVideo.onended=()=>endSession();
    document.getElementById('sound-btn').style.display='block';
  }else{showNoVideo();}
}

function showNoVideo(){
  document.getElementById('no-video').style.display='flex';
  document.getElementById('sound-btn').style.display='none';
}

document.getElementById('stim-file-input').addEventListener('change',e=>{
  const f=e.target.files[0];if(!f)return;
  stimVideo.src=URL.createObjectURL(f);stimVideo.muted=true;
  document.getElementById('no-video').style.display='none';
  stimVideo.play().catch(()=>{});
  stimVideo.onended=()=>endSession();
  document.getElementById('sound-btn').style.display='block';
});

document.getElementById('sound-btn').addEventListener('click',()=>{
  stimVideo.muted=false;stimVideo.play().catch(()=>{});
  document.getElementById('sound-btn').style.display='none';
});

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  MAIN PROCESSING LOOP
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
let _lastVT=-1;
function processingLoop(){
  if(phase==='done')return;
  if(phase==='validation'){procRaf=requestAnimationFrame(processingLoop);return;}
  if(webcam.readyState>=2&&faceLandmarker){
    if(webcam.currentTime===_lastVT){procRaf=requestAnimationFrame(processingLoop);return;}
    _lastVT=webcam.currentTime;
    const res=faceLandmarker.detectForVideo(webcam,performance.now());
    const hasFace=!!(res.faceLandmarks&&res.faceLandmarks.length>0);
    const mat=(res.facialTransformationMatrixes&&res.facialTransformationMatrixes.length>0)?res.facialTransformationMatrixes[0]:null;
    calibFacePresent=hasFace;
    const dbgPanel=document.getElementById('debug-panel');
    if(hasFace&&dbgPanel.style.display!=='none'){
      const lm=res.faceLandmarks[0];const f=extractFeatures(lm,mat);
      document.getElementById('d-vert').textContent=`pitch[2]: ${f[2].toFixed(4)}`;
      document.getElementById('d-foreY').textContent=`foreheadY[3]: ${f[3].toFixed(4)}`;
      document.getElementById('d-irisY').textContent=`irisYabs[5]: ${f[5].toFixed(4)}`;
      document.getElementById('d-liX').textContent=`liX[0]: ${f[0].toFixed(4)}  riX[1]: ${f[1].toFixed(4)}`;
      document.getElementById('d-pitch-src').textContent=`src:${mat?.data?'matrix':'z-coord'} EAR:${f[7].toFixed(3)} IOD:${f[8].toFixed(3)}`;
      document.getElementById('d-bias').textContent=`bias dx=${affineBias.dx.toFixed(1)} dy=${affineBias.dy.toFixed(1)} sx=${affineBias.sx.toFixed(3)} sy=${affineBias.sy.toFixed(3)}`;
      if(gazeModel){const g=predictGaze(f,gazeModel);if(g){document.getElementById('d-py').textContent=`\u2192 pred Y: ${g.y.toFixed(0)}px`;document.getElementById('d-px').textContent=`\u2192 pred X: ${g.x.toFixed(0)}px`;}}
    }
    if(phase==='stimulus'){
      document.getElementById('h-face').textContent=hasFace?'Yes':'No';
      document.getElementById('h-face').className=hasFace?'ok':'bad';
      document.getElementById('st-face').textContent=hasFace?'Yes':'No';
      document.getElementById('st-face').className='sv '+(hasFace?'ok':'bad');
    }
    if(hasFace){
      const lm=res.faceLandmarks[0];
      const features=extractFeatures(lm,mat);
      const isBlink=features[7]<0.06;
      if(phase==='calib-run'){
        // Only collect during STILL phase - character is motionless, fixation stable
        if(peekPhase==='still'&&peekIdx<peekPoints.length&&!isBlink){
          const pt=peekPoints[peekIdx];
          calibSamples.push({feat:features,sx:pt.x,sy:pt.y});
          peekSamplesThisPoint++;
        }
      }
      if(phase==='stimulus'&&gazeModel){
        const gaze=isBlink?null:predictGaze(features,gazeModel);
        totalF++;
        if(gaze){
          trackedF++;
          recordedFrames.push({t:Date.now()-sessionStart,x:gaze.x,y:gaze.y,tracked:1,feat:features});
          gazeCtx.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
          document.getElementById('st-gaze').textContent='Tracking';document.getElementById('st-gaze').className='sv ok';
        }else{
          recordedFrames.push({t:Date.now()-sessionStart,x:NaN,y:NaN,tracked:0,feat:null});
          document.getElementById('st-gaze').textContent='Lost';document.getElementById('st-gaze').className='sv bad';
          gazeCtx.clearRect(0,0,gazeCanvas.width,gazeCanvas.height);
        }
        document.getElementById('st-frames').textContent=recordedFrames.length;
        document.getElementById('st-track').textContent=Math.round(trackedF/totalF*100)+'%';
        if(recordedFrames.length>10){
          const ys=recordedFrames.filter(f=>f.tracked).map(f=>f.y);
          if(ys.length>1){const my=ys.reduce((a,b)=>a+b,0)/ys.length;const sy=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);const el=document.getElementById('st-ystd');el.textContent=sy.toFixed(0)+'px';el.className='sv '+(sy>30?'ok':'bad');}
        }
      }
    }else if(phase==='stimulus'){
      totalF++;recordedFrames.push({t:Date.now()-sessionStart,x:NaN,y:NaN,tracked:0,feat:null});
      if(totalF>0)document.getElementById('st-track').textContent=Math.round(trackedF/totalF*100)+'%';
    }
  }
  procRaf=requestAnimationFrame(processingLoop);
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  CSV
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
const CSV_HDR=['Unnamed: 0','RecordingTime [ms]','Time of Day [h:m:s:ms]','Trial','Stimulus','Export Start Trial Time [ms]','Export End Trial Time [ms]','Participant','Color','Tracking Ratio [%]','Index Right','Pupil Size Right X [px]','Pupil Size Right Y [px]','Point of Regard Right X [px]','Point of Regard Right Y [px]','Gaze Vector Right X','Gaze Vector Right Y','Gaze Vector Right Z','Eye Position Right X [mm]','Eye Position Right Y [mm]','Eye Position Right Z [mm]','Pupil Position Right X [px]','Pupil Position Right Y [px]','Port Status','Annotation Name','Annotation Description','Annotation Tags','Mouse Position X [px]','Mouse Position Y [px]','Scroll Direction X','Scroll Direction Y','Content'].join(',');

function buildCSV(){
  const biasMeta=`# GazeTrack v12 | bias_dx=${affineBias.dx.toFixed(2)} bias_dy=${affineBias.dy.toFixed(2)} bias_sx=${affineBias.sx.toFixed(4)} bias_sy=${affineBias.sy.toFixed(4)} val_samples=${valSamples.length}`;
  const lines=[biasMeta,CSV_HDR];
  recordedFrames.forEach((f,i)=>{
    const d=new Date(sessionStart+f.t);
    const pad=(n,w=2)=>String(Math.floor(n)).padStart(w,'0');
    const tod=`${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}:${pad(d.getMilliseconds(),3)}`;
    const tr=f.tracked===1;
    lines.push([i,f.t.toFixed(3),tod,'Trial001','webcam_stimulus','0','-',META.pid,'-',tr?'100':'0',tr?'1':'0','-','-',tr?f.x.toFixed(1):'-',tr?f.y.toFixed(1):'-',tr&&f.feat?f.feat[0].toFixed(4):'-',tr&&f.feat?f.feat[1].toFixed(4):'-',tr&&f.feat?f.feat[2].toFixed(4):'-','-','-','-','-','-','0','-','-','-','-','-','-','-','-'].join(','));
  });
  return lines.join('\n');
}

function downloadCSV(){
  if(!csvData)return;
  const ts=new Date().toISOString().replace(/[:.]/g,'-');
  const fn=`gaze_${META.pid}_${META.group}_${ts}.csv`;
  const url=URL.createObjectURL(new Blob([csvData],{type:'text/csv'}));
  Object.assign(document.createElement('a'),{href:url,download:fn}).click();
  URL.revokeObjectURL(url);
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  END SESSION
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
function endSession(){
  if(phase==='done')return;
  phase='done';
  clearInterval(timerInt);
  cancelAnimationFrame(procRaf);cancelAnimationFrame(calibRaf);cancelAnimationFrame(valRaf);
  stimVideo.pause();
  csvData=buildCSV();
  const pct=totalF>0?Math.round(trackedF/totalF*100):0;
  const dur=Math.round((Date.now()-sessionStart)/1000);
  const ys=recordedFrames.filter(f=>f.tracked).map(f=>f.y);
  let ystd=0;
  if(ys.length>1){const my=ys.reduce((a,b)=>a+b,0)/ys.length;ystd=Math.sqrt(ys.reduce((a,b)=>a+(b-my)**2,0)/ys.length);}
  const biasOk=Math.abs(affineBias.dx)>5||Math.abs(affineBias.dy)>5;
  const biasLabel=biasOk?`${affineBias.dx>0?'+':''}${affineBias.dx.toFixed(0)},${affineBias.dy>0?'+':''}${affineBias.dy.toFixed(0)}px`:'Minimal';
  document.getElementById('done-stats').innerHTML=`
    <div class="done-stat"><div class="n">${recordedFrames.length}</div><div class="l">FRAMES</div></div>
    <div class="done-stat"><div class="n">${pct}%</div><div class="l">TRACKED</div></div>
    <div class="done-stat"><div class="n">${dur}s</div><div class="l">DURATION</div></div>
    <div class="done-stat"><div class="n" style="color:${ystd>30?'var(--accent)':'var(--warn)'}">${ystd.toFixed(0)}px</div><div class="l">Y STD</div></div>
    <div class="done-stat"><div class="n" style="color:var(--accent);font-size:13px">${biasLabel}</div><div class="l">BIAS CORR</div></div>`;
  showScreen('done');
  const ts2=new Date().toISOString().replace(/[:.]/g,'-');
  setTimeout(()=>uploadToDrive(csvData,`gaze_${META.pid}_${META.group}_${ts2}.csv`),600);
}

document.getElementById('end-btn').addEventListener('click',endSession);
document.getElementById('btn-dl').addEventListener('click',()=>{
  if(_driveFileId) window.open('https://drive.google.com/file/d/'+_driveFileId+'/view','_blank');
  else if(_driveFolderId) window.open('https://drive.google.com/drive/folders/'+_driveFolderId,'_blank');
  else window.open('https://drive.google.com','_blank');
});
document.getElementById('btn-restart').addEventListener('click',()=>location.reload());

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  GOOGLE DRIVE
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
const GDRIVE_CLIENT_ID='864707039212-vosjr7obitpbcd7hjol8d2cvq5d6aj7u.apps.googleusercontent.com';
const GDRIVE_SCOPE='https://www.googleapis.com/auth/drive.file';
const GDRIVE_FOLDER='GazeTrack Sessions';

function driveSetStatus(icon,msg,color){
  const el=document.getElementById('drive-status');if(!el)return;
  document.getElementById('drive-icon').textContent=icon;
  document.getElementById('drive-msg').textContent=msg;
  el.style.borderColor=color||'var(--border)';
}

let _driveFileId=null, _driveFolderId=null;

async function uploadToDrive(csvText,filename){
  driveSetStatus('\u2601\uFE0F','Saving to Google Drive...','var(--border)');
  try{
    const token=await getGoogleToken();
    _driveFolderId=await findOrCreateFolder(GDRIVE_FOLDER,token);
    driveSetStatus('\u2B06\uFE0F','Uploading...','rgba(0,229,176,0.4)');
    const result=await uploadFile(csvText,filename,_driveFolderId,token);
    _driveFileId=result.id||null;
    driveSetStatus('\u2705','Saved! Click "Open in Drive" to view','rgba(0,229,176,0.4)');
    const btn=document.getElementById('btn-dl');
    btn.textContent='\u2601\ufe0f Open in Drive';
    btn.style.opacity='1';
    btn.style.pointerEvents='auto';
  }catch(err){
    if(err.message&&err.message.startsWith('WRONG_ACCOUNT:')){
      const used=err.message.split(':')[1];
      driveSetStatus('\u26d4','Wrong account: '+used+'  -  must use '+AUTHORISED_EMAIL,'rgba(255,92,58,0.4)');
      const btn=document.getElementById('btn-dl');
      btn.textContent='\u21ba Retry Upload';btn.style.opacity='1';btn.style.pointerEvents='auto';
      btn.onclick=()=>{ btn.textContent='\u23f3 Saving...';btn.style.opacity='.45';btn.style.pointerEvents='none'; uploadToDrive(csvData,btn._filename); };
      btn._filename='gaze_'+META.pid+'_'+META.group+'_'+new Date().toISOString().replace(/[:.]/g,'-')+'.csv';
    } else {
      driveSetStatus('\u274c','Drive failed  -  downloading locally instead','rgba(255,92,58,0.4)');
      downloadCSV();
    }
  }
}

//  AUTHORISED ACCOUNT  -  change this line to update 
const AUTHORISED_EMAIL = 'aashna.v01@gmail.com';

function getGoogleToken(){
  return new Promise((resolve,reject)=>{
    if(!window.google){reject(new Error('Google Identity Services not loaded'));return;}
    const client=window.google.accounts.oauth2.initTokenClient({
      client_id:GDRIVE_CLIENT_ID,
      scope:GDRIVE_SCOPE+' https://www.googleapis.com/auth/userinfo.email',
      prompt:'',
      hint:AUTHORISED_EMAIL,
      callback:async (resp)=>{
        if(resp.error){reject(new Error(resp.error));return;}
        // Verify the signed-in account matches the authorised email
        try{
          const info=await fetch('https://www.googleapis.com/oauth2/v3/userinfo',
            {headers:{Authorization:'Bearer '+resp.access_token}});
          const {email}=await info.json();
          if(email.toLowerCase()!==AUTHORISED_EMAIL.toLowerCase()){
            reject(new Error('WRONG_ACCOUNT:'+email));
            return;
          }
        }catch(e){/* if check fails, allow through  -  better to save than lose data */}
        resolve(resp.access_token);
      }
    });
    client.requestAccessToken();
  });
}

async function findOrCreateFolder(name,token){
  const q=encodeURIComponent(`name='${name}' and mimeType='application/vnd.google-apps.folder' and trashed=false`);
  const res=await fetch(`https://www.googleapis.com/drive/v3/files?q=${q}&fields=files(id,name)`,{headers:{Authorization:`Bearer ${token}`}});
  const data=await res.json();
  if(data.files&&data.files.length>0)return data.files[0].id;
  const create=await fetch('https://www.googleapis.com/drive/v3/files',{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:JSON.stringify({name,mimeType:'application/vnd.google-apps.folder'})});
  return(await create.json()).id;
}

async function uploadFile(csvText,filename,folderId,token){
  const boundary='-------GazeTrackBoundary';
  const body=`\r\n--${boundary}\r\nContent-Type: application/json\r\n\r\n${JSON.stringify({name:filename,parents:[folderId]})}\r\n--${boundary}\r\nContent-Type: text/csv\r\n\r\n${csvText}\r\n--${boundary}--`;
  const resp=await fetch('https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart',{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':`multipart/related; boundary="${boundary}"`},body});
  if(!resp.ok)throw new Error(`Upload failed: ${resp.status}`);
  return resp.json();
}

// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
//  DEBUG & CLEANUP
// ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
window.addEventListener('keydown',e=>{
  if(e.key==='d'||e.key==='D'){const p=document.getElementById('debug-panel');p.style.display=p.style.display==='none'?'block':'none';}
});
window.addEventListener('beforeunload',()=>{
  if(sessionStream)sessionStream.getTracks().forEach(t=>t.stop());
  if(camStream)camStream.getTracks().forEach(t=>t.stop());
});

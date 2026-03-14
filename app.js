console.log('%c GazeTrack v12 – live position all-clear + child-friendly star validation ','background:#00e5b0;color:#000;font-weight:bold;font-size:13px');
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

// ── STAR VALIDATION CONFIG ──
// Longer dwell so children have time to locate and fixate each star
const VAL_DWELL_MS    = 2800;   // was 1200ms – 2.3× longer
const VAL_GAP_MS      = 600;    // pause between stars so eyes settle
const VAL_STAR_RADIUS = 36;     // larger target (was 18px)
const VAL_SAMPLE_START= 0.45;   // start collecting after 45% of dwell
const VAL_INTRO_MS    = 2000;   // "Find the star!" shown before first star

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

// ── POSITION ALL-CLEAR STATE ──
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

// ════════════════════════════════════════════════════════════
//  POSITION ALL-CLEAR LOGIC
//  ─────────────────────────────────────────────────────────
//  Runs every preflight frame. Requires that:
//   • face detected at good distance (MediaPipe IOD check in previewLoop)
//   • lighting is good (brightness pixel check)
//   • both true for GOOD_STREAK_NEEDED consecutive frames
//
//  When the banner appears it also announces specific checks that
//  passed so the parent knows exactly what improved.
//  When position degrades again it hides after BAD_STREAK_HIDE frames
//  to avoid flickering during small head movements.
// ════════════════════════════════════════════════════════════
function updateAllClear(bright){
  // Read directly from pfState – no re-computation, single source of truth
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
  const tags = ['✓ Face visible · Good distance', '✓ Lighting OK'];
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

// ════════════════════════════════════════════════════════════
//  PRE-FLIGHT – pixel analysis, runs during intake
// ════════════════════════════════════════════════════════════
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
  if(pct) pct.textContent=done===total?score+'%':'…';
  const tips=[];
  if(pfState.light==='fail')  tips.push('<strong>💡 Too dark:</strong> Add a front-facing lamp.');
  if(pfState.light==='warn')  tips.push('<strong>💡 Lighting:</strong> Brighter room helps iris detection.');
  if(pfState.face==='fail')   tips.push('<strong>👤 No face:</strong> Make sure child is in frame, camera at eye level.');
  if(pfState.browser==='warn')tips.push('<strong>🌐 Browser:</strong> Use Chrome for best webcam performance.');
  const adv=document.getElementById('pf-advice');
  if(adv){adv.innerHTML=tips.join('<br>');adv.className='pf-advice'+(tips.length?' show':'');}
  // Button label only – enabled/disabled is managed solely by checkStartBtn
  const btn=document.getElementById('start-btn');
  if(!btn.disabled){
    const critFails=['cam','face'].filter(k=>pfState[k]==='fail').length;
    if(critFails>0){btn.textContent='⚠️ Proceed Anyway';btn.style.background='linear-gradient(135deg,#ff9f43,#e17f20)';}
    else if(done<total){btn.textContent='Begin Session →';btn.style.background='';}
    else if(score>=75){btn.textContent='✅ All Clear – Begin Session';btn.style.background='';}
    else{btn.textContent='⚠️ Proceed with Warnings';btn.style.background='linear-gradient(135deg,#ca8a04,#a16207)';}
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
    if(avgBright>=60&&avgBright<=220)    pfSet('light','pass',`✓ Good (${Math.round(avgBright)}/255)`);
    else if(avgBright<40)                pfSet('light','fail',`✗ Too dark (${Math.round(avgBright)}) – add light`);
    else if(avgBright<60)                pfSet('light','warn',`⚠ Dim (${Math.round(avgBright)}) – improve lighting`);
    else                                 pfSet('light','warn',`⚠ Bright (${Math.round(avgBright)}) – reduce backlight`);

    // ── Live all-clear update (lighting only – position driven by MediaPipe in previewLoop) ──
    updateAllClear(avgBright);

  }catch(e){}
  pfRaf=requestAnimationFrame(pfAnalyseFrame);
}

function pfCheckBrowser(){
  const ua=navigator.userAgent;
  const isChrome=/Chrome/.test(ua)&&!/Edg/.test(ua)&&!/OPR/.test(ua);
  const isEdge=/Edg/.test(ua);
  const isFirefox=/Firefox/.test(ua);
  if(isChrome)       pfSet('browser','pass','✓ Chrome – optimal');
  else if(isEdge)    pfSet('browser','pass','✓ Edge – good');
  else if(isFirefox) pfSet('browser','warn','⚠ Firefox – use Chrome for best results');
  else               pfSet('browser','warn','⚠ Use Chrome for best results');
}

// ════════════════════════════════════════════════════════════
//  CAMERA INIT
// ════════════════════════════════════════════════════════════
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
    document.getElementById('chk-cam').textContent='✓ Cam';
    const t=stream.getVideoTracks()[0].getSettings();
    const w=t.width||640, h=t.height||480;
    pfSet('cam','pass',`✓ ${w}×${h}`);
    checkStartBtn();
    pfRaf=requestAnimationFrame(pfAnalyseFrame);
    pfCheckBrowser();
    loadPreviewDetector();
  }catch(e){
    document.getElementById('cam-status-txt').textContent='✗ Camera error – allow access';
    pfSet('cam','fail','✗ Camera denied or not found');
    pfSet('face','fail','✗ No camera');
    pfSet('light','fail','✗ No camera');
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
        document.getElementById('chk-face').textContent='✓ Face';
        document.getElementById('chk-iris').classList.toggle('ok',hasIris);
        document.getElementById('chk-iris').textContent=hasIris?'✓ Iris':'👁 Iris';

        // ── MediaPipe-driven Position & Distance check ──
        // iodNorm = inter-ocular distance in normalised 0-1 space
        // Reliable regardless of lighting/skin tone – pure geometry
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
            pfSet('face','warn',`⚠ Too close (IOD ${iodNorm.toFixed(3)}) – move back ~15 cm`);
          else if(iodNorm>=0.13)
            pfSet('face','pass', offCentre
              ? `✓ Good distance · Move slightly to centre`
              : `✓ Face visible · Good distance (~50—70 cm)`);
          else if(iodNorm>=0.07)
            pfSet('face','warn',`⚠ Too far (IOD ${iodNorm.toFixed(3)}) – move ${offCentre?'closer & to centre':'~20 cm closer'}`);
          else
            pfSet('face','warn',`⚠ Very far or face at edge – move much closer`);
        } else {
          // Face found but no iris – can still give useful feedback
          document.getElementById('q-fill').style.width='40%';
          document.getElementById('q-pct').textContent='40%';
          pfSet('face','warn','⚠ Face detected but iris not visible – look at camera');
        }
      }else{
        document.getElementById('chk-face').classList.remove('ok');document.getElementById('chk-face').textContent='👤 Face';
        document.getElementById('chk-iris').classList.remove('ok');document.getElementById('chk-iris').textContent='👁 Iris';
        document.getElementById('q-fill').style.width='0%';document.getElementById('q-pct').textContent='–';
        pfSet('face','fail','✗ No face detected – check camera position');
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

// ════════════════════════════════════════════════════════════
//  FORM
// ════════════════════════════════════════════════════════════
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
  document.getElementById('video-drop').insertAdjacentHTML('beforeend',`<div class="chosen">✓ ${f.name}</div>`);
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

// ════════════════════════════════════════════════════════════
//  SESSION START
// ════════════════════════════════════════════════════════════
async function beginSession(){
  try{
    if(previewFl){
      faceLandmarker=previewFl;
      document.getElementById('load-msg').textContent='Model ready – starting camera…';
    }else{
      document.getElementById('load-msg').textContent='Loading eye tracking model…';
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
    document.getElementById('load-msg').textContent='❌ '+(err.message||'Startup error');
  }
}

function resizeCanvases(){
  calibCanvas.width=gazeCanvas.width=window.innerWidth;
  calibCanvas.height=gazeCanvas.height=window.innerHeight;
}

// ════════════════════════════════════════════════════════════
//  FEATURE EXTRACTION
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
//  RIDGE REGRESSION
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
//  AFFINE BIAS
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
//  CALIBRATION
// ════════════════════════════════════════════════════════════
function buildCalibPath(){
  const W=window.innerWidth,H=window.innerHeight;
  const mx=W/2,my=H/2,px=W*.08,py=H*.08;
  const wpts=[[mx,py,3],[W-px,py,3],[W-px,my,2],[W-px,H-py,3],[mx,H-py,3],[px,H-py,3],[px,my,2],[px,py,3],[mx,py,2],[mx,my,2],[W-px,my,1],[mx,H-py,2],[px,my,1],[mx,py,2],[mx,my,1]];
  const totalW=wpts.reduce((s,w)=>s+w[2],0),totalSteps=900,pts=[];
  for(let wi=0;wi<wpts.length-1;wi++){
    const[x0,y0,w0]=wpts[wi],[x1,y1]=wpts[wi+1];
    const steps=Math.round((w0/totalW)*totalSteps);
    for(let s=0;s<steps;s++){const t=s/Math.max(steps-1,1),st=t*t*(3-2*t);pts.push([x0+(x1-x0)*st,y0+(y1-y0)*st]);}
  }
  while(pts.length<totalSteps)pts.push(wpts[wpts.length-1].slice(0,2));
  return pts.slice(0,totalSteps);
}

const CAT_COLS=[{body:'#F8D7E3',stripe:'#E8A0B8'},{body:'#D7EAF8',stripe:'#90BDE0'},{body:'#E8D7F8',stripe:'#C090E0'},{body:'#D7F8E3',stripe:'#80D4A0'},{body:'#F8F0D7',stripe:'#E0C878'},{body:'#F8D7D7',stripe:'#E09090'}];
let bci=0,bct=0;

function drawCat(x,y,t,happy){
  const ctx=calibCtx,r=36,bob=Math.sin(t*4)*5,cy=y+bob;
  const{body,stripe}=CAT_COLS[bci];
  ctx.save();ctx.globalAlpha=0.13;ctx.beginPath();ctx.ellipse(x,cy+r+4,r*0.85,r*0.22,0,0,Math.PI*2);ctx.fillStyle='#000';ctx.fill();ctx.restore();
  [[-r*0.42,-r*1.1],[r*0.42,-r*1.1]].forEach(([dx])=>{
    const ex=x+dx,ey=cy-r;
    ctx.save();ctx.translate(ex,ey);ctx.rotate(dx<0?-0.12:0.12);
    ctx.beginPath();ctx.ellipse(0,-r*0.72,10,r*0.72,0,0,Math.PI*2);ctx.fillStyle=body;ctx.fill();
    ctx.beginPath();ctx.ellipse(0,-r*0.72,6,r*0.55,0,0,Math.PI*2);ctx.fillStyle=stripe;ctx.fill();
    ctx.restore();
  });
  ctx.beginPath();ctx.arc(x,cy,r,0,Math.PI*2);ctx.fillStyle=body;ctx.fill();
  ctx.beginPath();ctx.ellipse(x,cy+r*0.15,r*0.6,r*0.45,0,0,Math.PI*2);ctx.fillStyle='#fff';ctx.globalAlpha=0.7;ctx.fill();ctx.globalAlpha=1;
  if(happy){
    ctx.beginPath();ctx.arc(x-r*0.28,cy-r*0.15,3,0,Math.PI*2);ctx.fillStyle='#1a1a2e';ctx.fill();
    ctx.beginPath();ctx.arc(x+r*0.28,cy-r*0.15,3,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(x,cy+r*0.1,r*0.2,0,Math.PI);ctx.strokeStyle='#1a1a2e';ctx.lineWidth=2;ctx.stroke();
  }else{
    ctx.beginPath();ctx.arc(x-r*0.28,cy-r*0.1,3,0,Math.PI*2);ctx.fillStyle='#1a1a2e';ctx.fill();
    ctx.beginPath();ctx.arc(x+r*0.28,cy-r*0.1,3,0,Math.PI*2);ctx.fill();
    ctx.beginPath();ctx.arc(x,cy+r*0.25,r*0.18,Math.PI,0);ctx.strokeStyle='#1a1a2e';ctx.lineWidth=2;ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(x-r*0.35,cy+r*0.05);ctx.lineTo(x-r*0.75,cy-r*0.1);
  ctx.moveTo(x-r*0.35,cy+r*0.15);ctx.lineTo(x-r*0.75,cy+r*0.15);
  ctx.moveTo(x+r*0.35,cy+r*0.05);ctx.lineTo(x+r*0.75,cy-r*0.1);
  ctx.moveTo(x+r*0.35,cy+r*0.15);ctx.lineTo(x+r*0.75,cy+r*0.15);
  ctx.strokeStyle=stripe;ctx.lineWidth=1.5;ctx.stroke();
}

// ── Shared audio context ──
let _audioCtx=null;
function getAudioCtx(){if(!_audioCtx){try{_audioCtx=new AudioContext();}catch(e){}}return _audioCtx;}

const calibSound=(()=>{return()=>{const a=getAudioCtx();if(!a)return;const o=a.createOscillator(),g=a.createGain();o.connect(g);g.connect(a.destination);o.frequency.value=880;g.gain.setValueAtTime(0,a.currentTime);g.gain.linearRampToValueAtTime(0.15,a.currentTime+0.01);g.gain.exponentialRampToValueAtTime(0.001,a.currentTime+0.3);o.start();o.stop(a.currentTime+0.3);};})();

// ── Star sound – rising chime ──
function playChime(freq, vol, duration){
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

function startCalibAnim(){
  calibPath=buildCalibPath();
  calibStart=performance.now();
  bci=Math.floor(Math.random()*CAT_COLS.length);
  bct=0;
  const loop=()=>{
    if(phase!=='calib-run')return;
    const elapsed=performance.now()-calibStart;
    const pct=Math.min(elapsed/CALIB_MS,1);
    calibCanvas.width=window.innerWidth;calibCanvas.height=window.innerHeight;
    calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
    const idx=Math.floor(pct*(calibPath.length-1));
    if(calibPath[idx]){
      const[cx,cy]=calibPath[idx];
      bct+=0.016;
      const happy=calibFacePresent;
      if(!happy&&!calibSoundPlayed){calibSound();calibSoundPlayed=true;}
      else if(happy) calibSoundPlayed=false;
      drawCat(cx,cy,bct,happy);
    }
    if(pct>=1){finaliseCalib();return;}
    calibRaf=requestAnimationFrame(loop);
  };
  calibRaf=requestAnimationFrame(loop);
}

function finaliseCalib(){
  cancelAnimationFrame(calibRaf);
  calibCtx.clearRect(0,0,calibCanvas.width,calibCanvas.height);
  gazeModel=trainModel(calibSamples);
  if(!gazeModel){
    const card=document.getElementById('calib-card');
    card.querySelector('h2').textContent='⚠️ Calibration incomplete';
    card.querySelector('p').textContent=`Only ${calibSamples.length} samples (need ${MIN_SAMPLES}). Please retry.`;
    document.getElementById('calib-start-btn').textContent='↺ Retry';
    document.getElementById('calib-overlay').style.display='flex';
    calibSamples=[];phase='calib-ready';return;
  }
  phase='validation';
  startValidation();
}

// ════════════════════════════════════════════════════════════
//  STAR VALIDATION – child-friendly redesign
//  ─────────────────────────────────────────────────────────
//  Key changes vs original 1200ms dot approach:
//  • 2800ms dwell (2.3× longer) – children need time to locate target
//  • 600ms gap between stars – eyes need to settle after each saccade
//  • 36px radius stars – much larger, easier to see peripherally
//  • Springy entrance animation (scale 0→1 in 200ms with overshoot)
//  • Sparkle particle burst on arrival – draws attention naturally
//  • Twinkling shimmer during dwell – keeps child engaged on target
//  • Rising chime sound on each star – audio cue aids attention
//  • "Find the Star!" intro screen – parent instruction before first star
//  • Star count badge – "Star 2 of 5" – progress visible to clinician
//  • Collection window starts at 45% of dwell (1260ms in) – skip
//    the initial saccade movement time, collect stable fixation only
// ════════════════════════════════════════════════════════════

// Particle system for sparkle bursts
const VAL_PARTICLES = [];
function spawnSparkles(ctx, x, y){
  for(let i=0;i<14;i++){
    const angle=Math.random()*Math.PI*2;
    const speed=2+Math.random()*4;
    VAL_PARTICLES.push({
      x,y,
      vx:Math.cos(angle)*speed,
      vy:Math.sin(angle)*speed,
      life:1,
      size:2+Math.random()*4,
      hue:40+Math.random()*30
    });
  }
}
function updateSparkles(ctx){
  for(let i=VAL_PARTICLES.length-1;i>=0;i--){
    const p=VAL_PARTICLES[i];
    p.x+=p.vx;p.y+=p.vy;p.vy+=0.12;p.life-=0.035;
    if(p.life<=0){VAL_PARTICLES.splice(i,1);continue;}
    ctx.save();
    ctx.globalAlpha=p.life;
    ctx.fillStyle=`hsl(${p.hue},100%,65%)`;
    ctx.beginPath();ctx.arc(p.x,p.y,p.size*p.life,0,Math.PI*2);ctx.fill();
    ctx.restore();
  }
}

// Draw a twinkling star shape
function drawStar(ctx, x, y, radius, twinklePhase, entranceProgress){
  const r = radius * entranceProgress; // scale up on entrance
  const innerR = r * 0.4;
  const points = 5;
  const twinkle = 1 + Math.sin(twinklePhase * 8) * 0.08 * entranceProgress;
  const rr = r * twinkle;

  // Glow
  const glowSize = rr * (1.8 + Math.sin(twinklePhase*6)*0.2);
  const grad = ctx.createRadialGradient(x, y, 0, x, y, glowSize);
  grad.addColorStop(0, `rgba(255,220,50,${0.35*entranceProgress})`);
  grad.addColorStop(1, 'rgba(255,220,50,0)');
  ctx.beginPath();ctx.arc(x,y,glowSize,0,Math.PI*2);ctx.fillStyle=grad;ctx.fill();

  // Star shape
  ctx.beginPath();
  for(let i=0;i<points*2;i++){
    const angle = (i*Math.PI/points) - Math.PI/2;
    const rad = i%2===0 ? rr : innerR;
    const px = x + Math.cos(angle)*rad;
    const py = y + Math.sin(angle)*rad;
    i===0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
  }
  ctx.closePath();
  const starGrad = ctx.createRadialGradient(x, y-rr*0.2, 0, x, y, rr);
  starGrad.addColorStop(0,'#fff9c4');
  starGrad.addColorStop(0.4,'#ffd700');
  starGrad.addColorStop(1,'#ff9f00');
  ctx.fillStyle=starGrad;
  ctx.shadowColor='#ffd700';
  ctx.shadowBlur=20*entranceProgress;
  ctx.fill();
  ctx.shadowBlur=0;

  // Shimmer highlight
  if(entranceProgress>0.8){
    ctx.beginPath();
    ctx.arc(x-rr*0.2, y-rr*0.25, rr*0.18, 0, Math.PI*2);
    ctx.fillStyle=`rgba(255,255,255,${0.6*(entranceProgress-0.8)*5})`;
    ctx.fill();
  }
}

let _lastVideoTime=-1;
function startValidation(){
  valPoints=[];
  const W=window.innerWidth,H=window.innerHeight;
  const mx=W*.1,my=H*.1;
  valPoints=[
    {x:W/2,  y:H/2},
    {x:mx,   y:my},
    {x:W-mx, y:my},
    {x:W-mx, y:H-my},
    {x:mx,   y:H-my}
  ];
  valIdx=0;valSamples=[];VAL_PARTICLES.length=0;
  document.getElementById('val-overlay').style.display='block';
  document.getElementById('val-instruction').style.opacity='1';
  document.getElementById('val-badge').style.display='none';
  document.getElementById('val-badge-tot').textContent=valPoints.length;

  // Show intro for VAL_INTRO_MS, then start
  playChime(528, 0.1, 0.6);
  setTimeout(()=>{
    document.getElementById('val-instruction').style.opacity='0';
    setTimeout(()=>{
      document.getElementById('val-instruction').style.display='none';
      document.getElementById('val-badge').style.display='block';
      runStarDot();
    },400);
  }, VAL_INTRO_MS);
}

function runStarDot(){
  if(valIdx>=valPoints.length){finishValidation();return;}

  const pt = valPoints[valIdx];
  const valCanvas = document.getElementById('val-canvas');
  valCanvas.width  = window.innerWidth;
  valCanvas.height = window.innerHeight;
  const vCtx = valCanvas.getContext('2d');
  document.getElementById('val-badge-num').textContent = valIdx+1;

  // Play rising chime – different note for each star so it feels like a game
  const notes = [523, 659, 784, 880, 1047];
  playChime(notes[valIdx % notes.length], 0.12, 0.5);

  const collected = [];
  const ENTRANCE_MS = 220; // springy grow-in duration
  valStart = performance.now();
  let sparkled = false;

  // 600ms gap before star appears (eyes settle)
  let inGap = true;
  const gapEnd = valStart + VAL_GAP_MS;

  function frame(){
    const now = performance.now();
    const elapsed = now - valStart;

    vCtx.clearRect(0, 0, valCanvas.width, valCanvas.height);

    if(inGap){
      // Gap phase – draw sparkles dying out from previous star
      updateSparkles(vCtx);
      if(now >= gapEnd){ inGap=false; }
      valRaf=requestAnimationFrame(frame);
      return;
    }

    // Active star phase
    const starElapsed = now - gapEnd;
    const starProgress = Math.min(starElapsed / VAL_DWELL_MS, 1);

    // Entrance: ease out with slight overshoot (spring)
    let entrance;
    const entPct = Math.min(starElapsed / ENTRANCE_MS, 1);
    if(entPct < 1){
      // Cubic ease-out with bounce
      const t = entPct;
      entrance = 1 - Math.pow(1-t, 3) * Math.cos(t * Math.PI * 2.5);
      entrance = Math.min(entrance, 1.12); // allow slight overshoot
    } else {
      entrance = 1;
    }

    // Sparkle burst once on arrival
    if(!sparkled && entPct >= 0.9){
      spawnSparkles(vCtx, pt.x, pt.y);
      sparkled=true;
    }

    updateSparkles(vCtx);
    drawStar(vCtx, pt.x, pt.y, VAL_STAR_RADIUS, starElapsed*0.001, Math.min(entrance, 1));

    // Collect gaze samples during the stable window (after 45% of dwell)
    if(starProgress >= VAL_SAMPLE_START && gazeModel){
      if(webcam.readyState>=2 && faceLandmarker && webcam.currentTime!==_lastVideoTime){
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

    if(starProgress < 1){
      valRaf=requestAnimationFrame(frame);
    } else {
      // Done with this star
      if(collected.length>=3){
        const mxs=collected.map(p=>p.px).sort((a,b)=>a-b);
        const mys=collected.map(p=>p.py).sort((a,b)=>a-b);
        const mid=Math.floor(mxs.length/2);
        valSamples.push({px:mxs[mid],py:mys[mid],tx:pt.x,ty:pt.y});
      }
      valIdx++;
      runStarDot();
    }
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
      card.querySelector('h2').textContent='⚠️ Let\'s try again!';
      card.querySelector('p').innerHTML='Child may have looked away – that\'s OK!<br><br><strong style="color:var(--accent)">Tip:</strong> Move closer, brighter room, remind: <strong style="color:#fff">"Watch the bunny!"</strong>';
      document.getElementById('calib-start-btn').textContent='⚠️ Play Again!';
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
  calibSamples=[];phase='calib-run';startCalibAnim();
});

// ════════════════════════════════════════════════════════════
//  RECORDING
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
//  MAIN PROCESSING LOOP
// ════════════════════════════════════════════════════════════
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
      if(gazeModel){const g=predictGaze(f,gazeModel);if(g){document.getElementById('d-py').textContent=`→ pred Y: ${g.y.toFixed(0)}px`;document.getElementById('d-px').textContent=`→ pred X: ${g.x.toFixed(0)}px`;}}
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
        const elapsed=performance.now()-calibStart;
        const pct=Math.min(elapsed/CALIB_MS,1);
        const idx=Math.floor(pct*(calibPath.length-1));
        if(calibPath[idx]&&!isBlink) calibSamples.push({feat:features,sx:calibPath[idx][0],sy:calibPath[idx][1]});
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

// ════════════════════════════════════════════════════════════
//  CSV
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
//  END SESSION
// ════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════
//  GOOGLE DRIVE
// ════════════════════════════════════════════════════════════
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
  driveSetStatus('☁️','Saving to Google Drive…','var(--border)');
  try{
    const token=await getGoogleToken();
    _driveFolderId=await findOrCreateFolder(GDRIVE_FOLDER,token);
    driveSetStatus('⬆️','Uploading…','rgba(0,229,176,0.4)');
    const result=await uploadFile(csvText,filename,_driveFolderId,token);
    _driveFileId=result.id||null;
    driveSetStatus('✅','Saved! Click "Open in Drive" to view','rgba(0,229,176,0.4)');
    const btn=document.getElementById('btn-dl');
    btn.textContent='\u2601\ufe0f Open in Drive';
    btn.style.opacity='1';
    btn.style.pointerEvents='auto';
  }catch(err){
    if(err.message&&err.message.startsWith('WRONG_ACCOUNT:')){
      const used=err.message.split(':')[1];
      driveSetStatus('\u26d4','Wrong account: '+used+' – must use '+AUTHORISED_EMAIL,'rgba(255,92,58,0.4)');
      const btn=document.getElementById('btn-dl');
      btn.textContent='\u21ba Retry Upload';btn.style.opacity='1';btn.style.pointerEvents='auto';
      btn.onclick=()=>{ btn.textContent='\u23f3 Saving…';btn.style.opacity='.45';btn.style.pointerEvents='none'; uploadToDrive(csvData,btn._filename); };
      btn._filename='gaze_'+META.pid+'_'+META.group+'_'+new Date().toISOString().replace(/[:.]/g,'-')+'.csv';
    } else {
      driveSetStatus('\u274c','Drive failed – downloading locally instead','rgba(255,92,58,0.4)');
      downloadCSV();
    }
  }
}

// ── AUTHORISED ACCOUNT – change this line to update ──
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
        }catch(e){/* if check fails, allow through – better to save than lose data */}
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

// ════════════════════════════════════════════════════════════
//  DEBUG & CLEANUP
// ════════════════════════════════════════════════════════════
window.addEventListener('keydown',e=>{
  if(e.key==='d'||e.key==='D'){const p=document.getElementById('debug-panel');p.style.display=p.style.display==='none'?'block':'none';}
});
window.addEventListener('beforeunload',()=>{
  if(sessionStream)sessionStream.getTracks().forEach(t=>t.stop());
  if(camStream)camStream.getTracks().forEach(t=>t.stop());
});

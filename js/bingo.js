// wwwroot/js/bingo.js

let libsLoaded = false;
let _videoTrack = null;

async function loadScript(src){
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src; s.async = true;
    s.onload = resolve;
    s.onerror = () => reject(new Error('Failed to load '+src));
    document.head.appendChild(s);
  });
}

async function ensureLibs(){
  if (libsLoaded) return;
  await loadScript('https://docs.opencv.org/4.x/opencv.js');
  await new Promise(res => {
    if (typeof cv !== 'undefined' && cv.Mat) return res();
    cv['onRuntimeInitialized'] = res;
  });
  await loadScript('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js');
  libsLoaded = true;
}

function getOrCreateCanvas(id){
  let c = document.getElementById(id);
  if (!c){
    c = document.createElement('canvas');
    c.id = id;
    c.style.display = 'none';
    document.body.appendChild(c);
  }
  return c;
}

// --- Geometry helpers --------------------------------------------------------

function cropFooterIfNeeded(mat, rows, cols){
  // Many 3x9 tickets have a footer (serial/copyright). Trim a little, conservatively.
  if (rows === 3 && cols === 9){
    const trim = Math.round(mat.rows * 0.09); // conservative (keep more height)
    return mat.roi(new cv.Rect(0, 0, mat.cols, Math.max(1, mat.rows - trim)));
  }
  return mat;
}

function warpLargestQuadOrClone(src){
  const gray = new cv.Mat();
  const blur = new cv.Mat();
  const edges = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, blur, new cv.Size(5,5), 0, 0, cv.BORDER_DEFAULT);
  cv.Canny(blur, edges, 50, 150);
  cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  let best = null, maxArea = 0;
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const peri = cv.arcLength(cnt, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
    if (approx.rows === 4) {
      const area = cv.contourArea(approx);
      if (area > maxArea) { maxArea = area; if (best) best.delete(); best = approx; } else { approx.delete(); }
    } else {
      approx.delete();
    }
  }

  let warped;
  if (best) {
    const pts = [];
    for (let i = 0; i < 4; i++) pts.push({ x: best.intPtr(i,0)[0], y: best.intPtr(i,0)[1] });
    // order: tl, tr, br, bl using sums/diffs
    pts.sort((a,b)=>a.x+a.y - (b.x+b.y));
    const tl = pts[0], br = pts[3];
    const tr = (pts[1].x > pts[2].x) ? pts[1] : pts[2];
    const bl = (pts[1].x > pts[2].x) ? pts[2] : pts[1];

    const widthA = Math.hypot(br.x - bl.x, br.y - bl.y);
    const widthB = Math.hypot(tr.x - tl.x, tr.y - tl.y);
    const maxW = Math.max(widthA, widthB) | 0;
    const heightA = Math.hypot(tr.x - br.x, tr.y - br.y);
    const heightB = Math.hypot(tl.x - bl.x, tl.y - bl.y);
    const maxH = Math.max(heightA, heightB) | 0;

    const srcTri = cv.matFromArray(4,1,cv.CV_32FC2,[tl.x,tl.y,tr.x,tr.y,br.x,br.y,bl.x,bl.y]);
    const dstTri = cv.matFromArray(4,1,cv.CV_32FC2,[0,0,maxW,0,maxW,maxH,0,maxH]);
    const M = cv.getPerspectiveTransform(srcTri, dstTri);
    warped = new cv.Mat();
    cv.warpPerspective(src, warped, M, new cv.Size(maxW, maxH), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

    srcTri.delete(); dstTri.delete(); M.delete(); best.delete();
  } else {
    warped = src.clone();
  }

  gray.delete(); blur.delete(); edges.delete(); contours.delete(); hierarchy.delete();
  return warped;
}

// --- OCR helpers -------------------------------------------------------------

// Column-aware cleanup for weird OCR like "951" -> 51 based on 3x9 column ranges
function chooseNumberForColumn(str, colIdx) {
  if (!str) return null;

  // valid range by column
  const min = colIdx === 0 ? 1 : colIdx * 10;
  const max = colIdx === 8 ? 90 : colIdx * 10 + 9;

  // build candidates from all 1- and 2-digit windows
  const cand = new Set();
  for (let i = 0; i < str.length; i++) {
    const d1 = parseInt(str[i], 10);
    if (!Number.isNaN(d1)) cand.add(d1);
    if (i + 1 < str.length) {
      const d2 = parseInt(str.substring(i, i + 2), 10);
      if (!Number.isNaN(d2)) cand.add(d2);
    }
  }

  // keep only candidates inside the column's range
  const inRange = [...cand].filter(n => n >= min && n <= max);
  if (inRange.length) {
    const center = Math.round((min + max) / 2);
    inRange.sort((a, b) => Math.abs(a - center) - Math.abs(b - center));
    return inRange[0];
  }

  // last small helper: if the last two digits fall inside the range, accept them
  if (str.length >= 2) {
    const last2 = parseInt(str.slice(-2), 10);
    if (!Number.isNaN(last2) && last2 >= min && last2 <= max) return last2;
  }
  return null; // <— never return out-of-range numbers
}

// --- Main segmentation + OCR -------------------------------------------------

async function segmentAndOcr(canvasEl, rows, cols){
  // Normalize canvas to a generous working resolution (helps on small images)
  const MAX_W = 2200; // scale up if smaller
  if (canvasEl.width < MAX_W) {
    const up = document.createElement('canvas');
    const s = MAX_W / canvasEl.width;
    up.width = MAX_W; up.height = Math.round(canvasEl.height * s);
    const uctx = up.getContext('2d');
    uctx.imageSmoothingEnabled = true;
    uctx.imageSmoothingQuality = 'high';
    uctx.drawImage(canvasEl, 0, 0, up.width, up.height);
    canvasEl = up;
  }

  // Read + deskew + trim
  const src = cv.imread(canvasEl);
  const warped = warpLargestQuadOrClone(src); src.delete();
  const gridMat = cropFooterIfNeeded(warped, rows, cols);
  if (gridMat !== warped) warped.delete();

  // Two different binarizations (Otsu + Adaptive) to cover more lighting cases
  const gray = new cv.Mat();
  cv.cvtColor(gridMat, gray, cv.COLOR_RGBA2GRAY);
  const den = new cv.Mat();
  cv.medianBlur(gray, den, 3);

  const bwOtsu = new cv.Mat();
  cv.threshold(den, bwOtsu, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

  const bwAdap = new cv.Mat();
  cv.adaptiveThreshold(den, bwAdap, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 5);

  gray.delete(); den.delete();

  // Compute cell size once
  const cellW = Math.floor(bwOtsu.cols / cols);
  const cellH = Math.floor(bwOtsu.rows / rows);

  // Grid removal that won't eat a thin "1": big kernels (90% cell edge) + conservative pass
  function removeGrid(bw){
    const kh = Math.max(3, Math.floor(cellW * 0.90));
    const kv = Math.max(3, Math.floor(cellH * 0.90));
    const kH = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(kh, 1));
    const kV = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, kv));
    const horiz = new cv.Mat(), vert = new cv.Mat(), lines = new cv.Mat();
    cv.morphologyEx(bw, horiz, cv.MORPH_OPEN, kH);
    cv.morphologyEx(bw, vert,  cv.MORPH_OPEN, kV);
    cv.bitwise_or(horiz, vert, lines);
    const clean = new cv.Mat(); cv.subtract(bw, lines, clean);
    // conservative second pass (70%) in case lines remain
    const kH2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.max(3, Math.floor(cellW*0.7)),1));
    const kV2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, Math.max(3, Math.floor(cellH*0.7))));
    const horiz2 = new cv.Mat(), vert2 = new cv.Mat(), lines2 = new cv.Mat();
    cv.morphologyEx(clean, horiz2, cv.MORPH_OPEN, kH2);
    cv.morphologyEx(clean, vert2,  cv.MORPH_OPEN, kV2);
    cv.bitwise_or(horiz2, vert2, lines2);
    const clean2 = new cv.Mat(); cv.subtract(clean, lines2, clean2);
    // cleanup
    kH.delete(); kV.delete(); horiz.delete(); vert.delete(); lines.delete();
    kH2.delete(); kV2.delete(); horiz2.delete(); vert2.delete(); lines2.delete(); clean.delete();
    return clean2;
  }

  const cleanOtsu = removeGrid(bwOtsu);
  const cleanAdap = removeGrid(bwAdap);

  // Prepare per-cell ROIs from multiple sources and with multiple crops
const padXf = (rows === 3 && cols === 9) ? 0.12 : 0.12;
const padYf = (rows === 3 && cols === 9) ? 0.12 : 0.10;
const EMPTY_INK = 0.012; // ~1.2% of pixels "on" => treat as empty

const cells = []; // [{ variants:[...], col, empty }, ...]
for (let r = 0; r < rows; r++) {
  const row = [];
  for (let c = 0; c < cols; c++) {
    const x = c * cellW, y = r * cellH;

    let padX = Math.floor(cellW * padXf);
    const padY = Math.floor(cellH * padYf);
    if (c === 0 || c === cols - 1) padX = Math.floor(cellW * 0.08);
    const padXWide = Math.max(2, Math.floor(cellW * 0.06));

    const extraY = (rows === 3 && cols === 9 && r === rows - 1) ? -Math.floor(padY * 0.15) : 0;
    const tallUp = (rows === 3 && cols === 9 && r === rows - 1) ? Math.floor(cellH * 0.06) : 0;

    const rectFor = (px, dy = 0, extraH = 0) => {
      const rx = Math.max(0, x + px);
      const ry = Math.max(0, y + padY + extraY + dy);
      const rw = Math.max(1, Math.min(cellW - 2 * px, cleanOtsu.cols - rx));
      const rh = Math.max(1, Math.min(cellH - 2 * padY - extraY - dy + extraH, cleanOtsu.rows - ry));
      return { rx, ry, rw, rh };
    };

    // measure ink on a "clean" binary mat using the base rect
    const base = rectFor(padX);
    const roiCleanO = cleanOtsu.roi(new cv.Rect(base.rx, base.ry, base.rw, base.rh));
    const nz = cv.countNonZero(roiCleanO);
    roiCleanO.delete();
    const inkRatio = nz / (base.rw * base.rh);
    const isEmpty = inkRatio < EMPTY_INK;

    // crop helper (upscale + invert to black-on-white)
    const cropUpscale = (srcMat, rect) => {
      const roi = srcMat.roi(new cv.Rect(rect.rx, rect.ry, rect.rw, rect.rh));
      const scaled = new cv.Mat();
      const target = 160, scaleW = target, scaleH = Math.max(40, Math.round((target * rect.rh) / rect.rw));
      cv.resize(roi, scaled, new cv.Size(scaleW, scaleH), 0, 0, cv.INTER_CUBIC);
      cv.bitwise_not(scaled, scaled);
      const cnv = document.createElement('canvas'); cv.imshow(cnv, scaled);
      roi.delete(); scaled.delete();
      return cnv.toDataURL('image/png');
    };

    const mats = [
      { m: cleanOtsu }, { m: cleanAdap },
      { m: bwOtsu },    { m: bwAdap  },
    ];

    const variants = [];
    const rects = [
      rectFor(padX),                       // normal
      rectFor(padXWide),                   // wide
      ...(tallUp > 0 ? [rectFor(padX, -tallUp, tallUp * 2)] : []) // tall (bottom row)
    ];

    for (const src of mats)
      for (const rt of rects)
        variants.push(cropUpscale(src.m, rt));

    row.push({ variants, col: c, empty: isEmpty });
  }
  cells.push(row);
}

  cleanOtsu.delete(); cleanAdap.delete(); bwOtsu.delete(); bwAdap.delete(); gridMat.delete();

  // ---- OCR with column-aware rules and per-column PSM -----------------------
const worker = await Tesseract.createWorker('eng');
await worker.setParameters({
  tessedit_char_whitelist: '0123456789',
  classify_bln_numeric_mode: '1'
});

const results = [];
for (let r = 0; r < rows; r++) {
  const row = [];
  for (let c = 0; c < cols; c++) {
    const { variants, empty } = cells[r][c];

    if (empty) { // hard empty: no OCR
      row.push(null);
      continue;
    }

    let val = null;
    for (const img of variants) {
      await worker.setParameters({ tessedit_pageseg_mode: (c === 0 ? '10' : '7') });
      const { data } = await worker.recognize(img);
      const raw = (data.text || '').replace(/[^0-9]/g, '');
      val = chooseNumberForColumn(raw, c); // returns null if out-of-range
      if (val != null) break;
    }
    row.push(val);
  }
  results.push(row);
}
await worker.terminate();
return results;

}

// --- Public API --------------------------------------------------------------

window.bingo = {
  startCamera: async function (videoId) {
    const v = document.getElementById(videoId);
    const s = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width:  { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: false
    });
    v.srcObject = s;
    await v.play();

    // store track reference for torch toggle
    _videoTrack = s.getVideoTracks()[0];

    // ensure videoWidth/Height are ready
    if (!v.videoWidth || !v.videoHeight) {
      await new Promise(res => v.onloadedmetadata ? (v.onloadedmetadata = res) : setTimeout(res, 100));
    }
    return true;
  },

  setTorch: async function(on){
    try {
      if (!_videoTrack || !_videoTrack.getCapabilities) return false;
      const caps = _videoTrack.getCapabilities();
      if (!('torch' in caps)) return false;
      await _videoTrack.applyConstraints({ advanced: [{ torch: !!on }] });
      return true;
    } catch { return false; }
  },

  hasTorch: function(){
    try { return !!(_videoTrack && _videoTrack.getCapabilities && ('torch' in _videoTrack.getCapabilities())); }
    catch { return false; }
  },

  stopCamera: function (videoId) {
    const v = document.getElementById(videoId);
    if (!v || !v.srcObject) return;
    v.srcObject.getTracks().forEach(t => t.stop());
    v.srcObject = null;
    _videoTrack = null;
  },

  // Camera → OCR
  captureAndOcr: async function (videoId, canvasId, rows=5, cols=5) {
    try { await ensureLibs(); } catch { alert('Could not load vision libraries.'); return null; }
    const v = document.getElementById(videoId);

    // Let autofocus settle a tick
    await new Promise(r => setTimeout(r, 250));

    const c = getOrCreateCanvas(canvasId);
    c.width = v.videoWidth || v.clientWidth;
    c.height = v.videoHeight || v.clientHeight;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(v, 0, 0, c.width, c.height);

    return await segmentAndOcr(c, rows, cols);
  },

  // File input → OCR
  scanImage: async function(fileInputId, canvasId, rows=5, cols=5){
    try { await ensureLibs(); } catch { alert('Could not load vision libraries.'); return null; }
    const input = document.getElementById(fileInputId);
    if (!input?.files?.length) return null;

    const file = input.files[0];
    const url = URL.createObjectURL(file);
    const img = new Image();
    await new Promise((res,rej)=>{ img.onload=res; img.onerror=rej; img.src=url; });

    const c = getOrCreateCanvas(canvasId);
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);

    return await segmentAndOcr(c, rows, cols);
  },

  scanImages: async function (fileInputId, canvasId, rows = 3, cols = 9, progressRef = null) {
  try { await ensureLibs(); } catch { alert('Could not load vision libraries.'); return []; }

  const input = document.getElementById(fileInputId);
  if (!input?.files?.length) return [];

  const results = [];
  const total = input.files.length;

  for (let i = 0; i < total; i++) {
    const file = input.files[i];
    const url = URL.createObjectURL(file);

    try {
      // progress: BEFORE we process this image
      if (progressRef && progressRef.invokeMethodAsync) {
        try { await progressRef.invokeMethodAsync('OnScanProgress', i + 1, total, file.name || 'image'); } catch {}
      }

      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });

      const c = getOrCreateCanvas(canvasId);
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);

      const grid = await segmentAndOcr(c, rows, cols);
      results.push(grid);
    } catch (e) {
      console.error('scanImages: failed for', file?.name, e);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // clear the picker so selecting the same files again triggers change
  try { input.value = ''; } catch {}

  return results;
}

};

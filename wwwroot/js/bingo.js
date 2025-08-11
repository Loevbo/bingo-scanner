// wwwroot/js/bingo.js

let libsLoaded = false;

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

// For 3×9 tickets, there’s often a footer (serial/copyright) inside the border.
// Trim a bit from the bottom so the last row isn’t polluted.
function cropFooterIfNeeded(mat, rows, cols){
  if (rows === 3 && cols === 9){
    const trim = Math.round(mat.rows * 0.12); // tweak 0.10–0.15 if needed
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
      if (area > maxArea) { maxArea = area; best = approx; }
    } else {
      approx.delete();
    }
  }

  let warped;
  if (best) {
    const pts = [];
    for (let i = 0; i < 4; i++) pts.push({ x: best.intPtr(i,0)[0], y: best.intPtr(i,0)[1] });
    // order points: tl, tr, br, bl using sums/diffs
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
    // Fallback if no quad found
    warped = src.clone();
  }

  gray.delete(); blur.delete(); edges.delete(); contours.delete(); hierarchy.delete();
  return warped;
}

async function segmentAndOcr(canvasEl, rows, cols){
  const src = cv.imread(canvasEl);
  const warped = warpLargestQuadOrClone(src);
  src.delete();

  // Optional footer trim for 3×9
  const gridMat = cropFooterIfNeeded(warped, rows, cols);
  if (gridMat !== warped) warped.delete();

  // Binarize (Otsu) for clearer digits
  const gray = new cv.Mat();
  cv.cvtColor(gridMat, gray, cv.COLOR_RGBA2GRAY);
  const bw = new cv.Mat();
  cv.threshold(gray, bw, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
  gray.delete();

  const cellW = Math.floor(bw.cols / cols);
  const cellH = Math.floor(bw.rows / rows);

  // Tuned padding; smaller for 3×9 (digits are larger & lower)
  const basePadX = rows === 3 && cols === 9 ? 0.10 : 0.12;
  const basePadY = rows === 3 && cols === 9 ? 0.08 : 0.10;

  const images = [];
  for (let r=0; r<rows; r++){
    const rowImgs = [];
    for (let c=0; c<cols; c++){
      const x = c * cellW;
      const y = r * cellH;

      let padX = Math.floor(cellW * basePadX);
      let padY = Math.floor(cellH * basePadY);

      // Nudge last row slightly downward on 3×9 tickets (digits sit lower)
      let ry = y + padY + ((rows === 3 && cols === 9 && r === rows - 1) ? Math.floor(padY * 0.5) : 0);
      let rx = x + padX;

      const rw = Math.max(1, Math.min(cellW - 2*padX, bw.cols - rx));
      const rh = Math.max(1, Math.min(cellH - 2*padY, bw.rows - ry));

      const digit = bw.roi(new cv.Rect(rx, ry, rw, rh));
      const tmp = document.createElement('canvas');
      cv.imshow(tmp, digit);
      rowImgs.push(tmp.toDataURL('image/png'));
      digit.delete();
    }
    images.push(rowImgs);
  }

  bw.delete();
  gridMat.delete();

  const worker = await Tesseract.createWorker('eng');
  const results = [];
  for (let r=0; r<rows; r++){
    const row = [];
    for (let c=0; c<cols; c++){
      const { data } = await worker.recognize(images[r][c], { tessedit_char_whitelist: '0123456789' });
      const txt = (data.text || '').replace(/[^0-9]/g,'');
      row.push(txt ? parseInt(txt, 10) : null);
    }
    results.push(row);
  }
  await worker.terminate();
  return results;
}

window.bingo = {
  startCamera: async function (videoId) {
    const v = document.getElementById(videoId);
    const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    v.srcObject = s;
    await v.play();
    return true;
  },

  stopCamera: function (videoId) {
    const v = document.getElementById(videoId);
    if (!v || !v.srcObject) return;
    v.srcObject.getTracks().forEach(t => t.stop());
    v.srcObject = null;
  },

  // Camera → OCR
  captureAndOcr: async function (videoId, canvasId, rows=5, cols=5) {
    try { await ensureLibs(); } catch { alert('Could not load vision libraries.'); return null; }
    const v = document.getElementById(videoId);
    const c = getOrCreateCanvas(canvasId);
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
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
  }
};

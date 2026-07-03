// ===================== NFL Draft Scouting Model — Dashboard Logic =====================
const D = DASHBOARD_DATA;
const LOWER_BETTER = new Set(["forty", "cone", "shuttle"]);
const METRIC_LABELS = {forty:"40-yd Dash", vertical:"Vertical Jump", bench:"Bench Press", broad:"Broad Jump", cone:"3-Cone Drill", shuttle:"20-yd Shuttle", height:"Height", weight:"Weight"};
const POSITIONS = ["QB","RB","WR","TE","OL","EDGE","LB","CB","S","DL"];

// ---------- Tab navigation ----------
document.querySelectorAll('#tabs button').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    document.querySelectorAll('#tabs button').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('view-'+btn.dataset.view).classList.add('active');
  });
});

// ---------- Populate position selects ----------
['posSelect','predPos'].forEach(id=>{
  const sel = document.getElementById(id);
  POSITIONS.forEach(p=>{ const o=document.createElement('option'); o.value=p; o.textContent=p; sel.appendChild(o); });
});

// ================= PLAYER SEARCH =================
const searchBox = document.getElementById('searchBox');
const searchResults = document.getElementById('searchResults');
searchBox.addEventListener('input', ()=>{
  const q = searchBox.value.trim().toLowerCase();
  if(q.length < 2){ searchResults.style.display='none'; return; }
  const matches = D.players.filter(p=>p.player.toLowerCase().includes(q)).slice(0,12);
  if(matches.length===0){ searchResults.style.display='none'; return; }
  searchResults.innerHTML = matches.map(p=>`<div data-key="${p.player}|${p.year}">${p.player} <span class="muted mono" style="font-size:11px;">${p.pos} &middot; ${p.year} &middot; ${p.school||''}</span></div>`).join('');
  searchResults.style.display='block';
  searchResults.querySelectorAll('div').forEach(div=>{
    div.addEventListener('click', ()=>{ searchResults.style.display='none'; searchBox.value=div.dataset.key.split('|')[0]; renderReport(div.dataset.key); });
  });
});
document.addEventListener('click', (e)=>{ if(!e.target.closest('.search-wrap')) searchResults.style.display='none'; });

function pctileWithinGroup(pos, field, value){
  if(value===null||value===undefined) return null;
  const pool = D.players.filter(p=>p.pos===pos && p[field]!==null && p[field]!==undefined);
  if(pool.length===0) return null;
  let better;
  if(LOWER_BETTER.has(field)) better = pool.filter(p=>p[field] > value).length;
  else better = pool.filter(p=>p[field] < value).length;
  return Math.round(better/pool.length*1000)/10;
}

function gradeFromSAI(sai){
  if(sai===null||sai===undefined) return "N/A";
  if(sai>=95) return "A+ (Elite Freak Athlete)";
  if(sai>=88) return "A (Elite)";
  if(sai>=75) return "B+ (Above Average)";
  if(sai>=55) return "B (Solid)";
  if(sai>=35) return "C (Below Average)";
  if(sai>=15) return "D (Poor)";
  return "F (Well Below Average)";
}

function euclideanComparables(target, k=5){
  const pos = target.pos;
  const pool = D.players.filter(p=>p.pos===pos && p.player+"|"+p.year !== target.player+"|"+target.year);
  const fields = ["height","weight","forty","vertical","bench","broad","cone","shuttle"];
  // position means/stds for standardization (mean-impute missing)
  const stats = {};
  fields.forEach(f=>{
    const vals = D.players.filter(p=>p.pos===pos && p[f]!==null).map(p=>p[f]);
    const mean = vals.reduce((a,b)=>a+b,0)/(vals.length||1);
    const sd = Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/(vals.length||1)) || 1;
    stats[f] = {mean, sd};
  });
  function vec(p){ return fields.map(f=> ((p[f]===null||p[f]===undefined? stats[f].mean : p[f]) - stats[f].mean)/stats[f].sd ); }
  const tvec = vec(target);
  const scored = pool.map(p=>{
    const v = vec(p);
    const dist = Math.sqrt(v.reduce((s,val,i)=>s+(val-tvec[i])**2,0));
    return {p, dist};
  }).sort((a,b)=>a.dist-b.dist).slice(0,k);
  const maxDist = scored.length? scored[scored.length-1].dist : 1;
  return scored.map(s=>({...s.p, similarity: Math.round((1-(s.dist/(maxDist+0.001)))*100*10)/10 + 0 }));
}
// simpler: similarity via inverse distance, normalized so nearest = 100
function comparablesNormalized(target,k=5){
  const pos = target.pos;
  const fields = ["height","weight","forty","vertical","bench","broad","cone","shuttle"];
  const groupPlayers = D.players.filter(p=>p.pos===pos);
  const stats = {};
  fields.forEach(f=>{
    const vals = groupPlayers.filter(p=>p[f]!==null && p[f]!==undefined).map(p=>p[f]);
    const mean = vals.reduce((a,b)=>a+b,0)/(vals.length||1);
    const sd = Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/(vals.length||1)) || 1;
    stats[f]={mean,sd};
  });
  function vec(p){ return fields.map(f=> ((p[f]===null||p[f]===undefined? stats[f].mean : p[f]) - stats[f].mean)/stats[f].sd ); }
  const tvec = vec(target);
  const scored = groupPlayers.filter(p=>!(p.player===target.player && p.year===target.year)).map(p=>{
    const v = vec(p);
    const dist = Math.sqrt(v.reduce((s,val,i)=>s+(val-tvec[i])**2,0));
    return {p,dist};
  }).sort((a,b)=>a.dist-b.dist).slice(0,k);
  const invSims = scored.map(s=>1/(1+s.dist));
  const maxSim = Math.max(...invSims,0.0001);
  return scored.map((s,i)=>({...s.p, similarity: Math.round(invSims[i]/maxSim*1000)/10}));
}

function renderReport(key){
  const target = D.players.find(p=>p.player+"|"+p.year===key);
  if(!target){ document.getElementById('reportArea').innerHTML = '<p class="muted">Player not found.</p>'; return; }
  const fields = ["forty","vertical","broad","cone","shuttle","bench","height","weight"];
  const pctiles = {};
  fields.forEach(f=> pctiles[f] = pctileWithinGroup(target.pos, f, target[f]));
  const strengths = fields.filter(f=>pctiles[f]!==null && pctiles[f]>=80);
  const weaknesses = fields.filter(f=>pctiles[f]!==null && pctiles[f]<=20);
  const comps = comparablesNormalized(target, 5);
  const grade = gradeFromSAI(target.sai);

  const radarFields = ["forty","vertical","broad","cone","shuttle","bench"];
  const radarSize=220, cx=radarSize/2, cy=radarSize/2, R=90;
  const n = radarFields.length;
  function pointFor(i, val){
    const angle = (Math.PI*2*i/n) - Math.PI/2;
    const r = (val/100)*R;
    return [cx + r*Math.cos(angle), cy + r*Math.sin(angle)];
  }
  const playerPts = radarFields.map((f,i)=>pointFor(i, pctiles[f]===null?0:pctiles[f]));
  const elitePts = radarFields.map((f,i)=>pointFor(i, 90));
  const avgPts = radarFields.map((f,i)=>pointFor(i, 50));
  const toPath = pts => pts.map(p=>p.join(',')).join(' ');
  const labelPts = radarFields.map((f,i)=>{ const [x,y] = pointFor(i, 118); return {x,y,label:METRIC_LABELS[f]}; });

  const svgRadar = `<svg width="${radarSize+70}" height="${radarSize+40}" viewBox="0 0 ${radarSize+70} ${radarSize+40}">
    <g transform="translate(20,10)">
    ${[25,50,75,100].map(pct=>`<polygon points="${toPath(radarFields.map((f,i)=>pointFor(i,pct)))}" fill="none" stroke="#2B3438" stroke-width="1"/>`).join('')}
    ${radarFields.map((f,i)=>{const [x,y]=pointFor(i,100); return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#22292c"/>`}).join('')}
    <polygon points="${toPath(elitePts)}" fill="none" stroke="#9AA3A2" stroke-width="1" stroke-dasharray="4,3"/>
    <polygon points="${toPath(avgPts)}" fill="none" stroke="#5c6669" stroke-width="1" stroke-dasharray="1,3"/>
    <polygon points="${toPath(playerPts)}" fill="#D98E2B33" stroke="#D98E2B" stroke-width="2"/>
    ${labelPts.map(p=>`<text x="${p.x}" y="${p.y}" font-size="10" fill="#9AA3A2" text-anchor="middle" font-family="IBM Plex Mono">${p.label}</text>`).join('')}
    </g>
  </svg>`;

  const pred = target.prediction;
  const html = `
  <div class="grid grid-2">
    <div class="card">
      <div style="display:flex;gap:16px;align-items:center;">
        <div class="stamp"><span>${target.sai!==null? Math.round(target.sai): '—'}</span></div>
        <div>
          <div style="font-size:22px;font-weight:700;">${target.player}</div>
          <div class="muted mono" style="font-size:12.5px;">${target.pos} &middot; ${target.school||'—'} &middot; ${target.year}</div>
          <div style="margin-top:6px;"><span class="pill amber">SAI ${target.sai ?? '—'} — ${grade}</span></div>
        </div>
      </div>
      <div class="hash"><span>Actual vs. Model-Expected Draft</span></div>
      <table>
        <tr><td class="muted">Actual</td><td>${target.drafted ? `Round ${target.round}, Pick ${target.pick}` : 'Undrafted'}</td></tr>
        <tr><td class="muted">Model Expected</td><td>${pred? `~Pick ${pred.predicted_pick} (${pred.predicted_round})` : '—'}</td></tr>
      </table>
      <div class="hash"><span>Strengths / Weaknesses</span></div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;">
        ${strengths.length? strengths.map(f=>`<span class="pill turf">${METRIC_LABELS[f]} — ${pctiles[f]}th pct</span>`).join('') : '<span class="muted" style="font-size:12.5px;">No standout metrics</span>'}
        ${weaknesses.map(f=>`<span class="pill reach">${METRIC_LABELS[f]} — ${pctiles[f]}th pct</span>`).join('')}
      </div>
    </div>
    <div class="card">
      <div class="hash" style="margin-top:0;"><span>Athletic Profile vs. ${target.pos} Peers</span></div>
      <div style="display:flex;justify-content:center;">${svgRadar}</div>
      <div class="muted mono" style="font-size:10.5px;text-align:center;">amber = player &middot; dashed = elite (90th pct) &middot; dotted = position average</div>
    </div>
  </div>
  <div class="hash"><span>Closest Historical Comparables (Euclidean, position-normalized)</span></div>
  <div class="card">
    <table><thead><tr><th>Player</th><th>Year</th><th>School</th><th>SAI</th><th>Similarity</th></tr></thead>
    <tbody>${comps.map(c=>`<tr class="row-click" data-key="${c.player}|${c.year}"><td>${c.player}</td><td>${c.year}</td><td>${c.school||''}</td><td>${c.sai??'—'}</td><td>${c.similarity}%</td></tr>`).join('')}</tbody></table>
  </div>`;
  document.getElementById('reportArea').innerHTML = html;
  document.querySelectorAll('#reportArea tr.row-click').forEach(tr=>{
    tr.addEventListener('click', ()=>{ searchBox.value = tr.dataset.key.split('|')[0]; renderReport(tr.dataset.key); window.scrollTo({top:0,behavior:'smooth'}); });
  });
}

// default: show a notable player on load
renderReport(D.players.find(p=>p.player==="Travis Hunter")? "Travis Hunter|2025" : (D.players[0].player+"|"+D.players[0].year));
searchBox.value = "Travis Hunter";

// ================= POSITION EXPLORER =================
function renderPosition(pos){
  const profile = D.profiles[pos];
  const benchDiv = document.getElementById('posBenchmarks');
  if(profile){
    const order = ["40-yd Dash","Vertical Jump","Broad Jump","3-Cone Drill","20-yd Shuttle","Bench Press","Height","Weight"];
    benchDiv.innerHTML = `<div class="muted mono" style="font-size:11.5px;margin-bottom:8px;">n=${profile.n_players} prospects &middot; ${profile.n_drafted} drafted</div>` +
      order.filter(m=>profile.metrics[m]).map(m=>{
        const info = profile.metrics[m];
        return `<div class="bar-row"><div>${m}</div><div class="mono" style="font-size:11.5px;">avg ${info.mean} &nbsp;|&nbsp; elite ${info.elite_pctile_value} &nbsp;|&nbsp; poor ${info.poor_pctile_value}</div><div></div></div>`;
      }).join('');
  } else { benchDiv.innerHTML = '<p class="muted">No data.</p>'; }

  const diDiv = document.getElementById('posDrillImportance');
  const di = D.drill_importance[pos];
  if(di){
    const ranked = di.rf_importance_ranked;
    const maxImp = Math.max(...ranked.map(r=>r[1]));
    diDiv.innerHTML = `<div class="muted mono" style="font-size:11px;margin-bottom:8px;">Random Forest feature importance for predicting draft value, fit within ${pos} only</div>` +
      ranked.map(([drill,imp])=>{
        const corr = di.spearman_vs_draftvalue[drill];
        return `<div class="bar-row"><div>${drill}</div><div class="bar-track"><div class="bar-fill" style="width:${(imp/maxImp*100).toFixed(0)}%;"></div></div><div class="mono" style="font-size:11px;">${(imp*100).toFixed(1)}%</div></div>`;
      }).join('');
  } else { diDiv.innerHTML = '<p class="muted">No data.</p>'; }

  const clusterDiv = document.getElementById('posClusters');
  const cl = D.clusters[pos];
  if(cl){
    clusterDiv.innerHTML = Object.entries(cl.cluster_summary).map(([name,info])=>`
      <div class="card">
        <div style="font-weight:700;font-size:14.5px;margin-bottom:6px;">${name}</div>
        <div class="muted mono" style="font-size:11.5px;line-height:1.7;">
          n = ${info.n}<br>mean 40 = ${info.mean_40 ?? '—'}<br>mean weight = ${info.mean_weight}<br>mean SAI = ${info.mean_SAI ?? '—'}
        </div>
        <div style="margin-top:8px;font-size:12px;">${(info.example_players||[]).slice(0,4).join(', ')}</div>
      </div>`).join('');
  } else { clusterDiv.innerHTML = '<p class="muted">No cluster data.</p>'; }
}
document.getElementById('posSelect').addEventListener('change', e=>renderPosition(e.target.value));
document.getElementById('posSelect').value = "WR";
renderPosition("WR");

// ================= DRAFT PREDICTOR (simplified in-browser approximation) =================
document.getElementById('predPos').value = "WR";
document.getElementById('predictBtn').addEventListener('click', ()=>{
  const pos = document.getElementById('predPos').value;
  const input = {
    height: parseFloat(document.getElementById('predHeight').value) || null,
    weight: parseFloat(document.getElementById('predWeight').value) || null,
    forty: parseFloat(document.getElementById('predForty').value) || null,
    vertical: parseFloat(document.getElementById('predVert').value) || null,
    bench: parseFloat(document.getElementById('predBench').value) || null,
    broad: parseFloat(document.getElementById('predBroad').value) || null,
    cone: parseFloat(document.getElementById('predCone').value) || null,
  };
  const di = D.drill_importance[pos];
  const fieldMap = {height:"Height", weight:"Weight", forty:"40-yd Dash", vertical:"Vertical Jump", bench:"Bench Press", broad:"Broad Jump", cone:"3-Cone Drill"};
  // weighted average percentile using RF importances as weights (simplified proxy for the real model)
  let weightedSum=0, weightTotal=0;
  const details=[];
  Object.entries(fieldMap).forEach(([jsField, pyField])=>{
    const val = input[jsField];
    if(val===null) return;
    const pct = pctileWithinGroup(pos, jsField, val);
    if(pct===null) return;
    const imp = di ? (di.rf_importance[pyField]||0.05) : 0.125;
    weightedSum += pct*imp; weightTotal += imp;
    details.push({field:pyField, pct, imp});
  });
  const overallPct = weightTotal>0 ? weightedSum/weightTotal : 50;
  // map percentile -> approx pick number within this position's real draft-value distribution
  const posValues = D.players.filter(p=>p.pos===pos && p.prediction).map(p=>p.prediction.predicted_pick).sort((a,b)=>a-b);
  const idx = Math.min(posValues.length-1, Math.floor((100-overallPct)/100*posValues.length));
  const approxPick = posValues.length? posValues[idx] : null;
  function roundLabel(v){
    if(v<=32) return "Round 1"; if(v<=64) return "Round 2"; if(v<=105) return "Round 3";
    if(v<=141) return "Round 4"; if(v<=178) return "Round 5"; if(v<=220) return "Round 6";
    if(v<=262) return "Round 7"; return "Undrafted";
  }
  document.getElementById('predictorResult').innerHTML = `
    <div class="card grid grid-2">
      <div>
        <div class="kpi">${approxPick? Math.round(approxPick): '—'}</div>
        <div class="kpi-label">Approx. Expected Pick</div>
        <div style="margin-top:10px;"><span class="pill amber">${approxPick? roundLabel(approxPick): 'Insufficient inputs'}</span></div>
        <p class="muted" style="font-size:12px;margin-top:14px;line-height:1.6;">Overall athletic percentile at ${pos}: <b>${overallPct.toFixed(1)}</b>th, weighted by which drills the trained Random Forest actually found predictive at this position.</p>
      </div>
      <div>
        <div class="hash" style="margin-top:0;"><span>Inputs used</span></div>
        ${details.map(d=>`<div class="bar-row"><div>${d.field}</div><div class="bar-track"><div class="bar-fill amber" style="width:${d.pct}%;"></div></div><div class="mono" style="font-size:11px;">${d.pct}th</div></div>`).join('') || '<p class="muted">Enter at least one measurable.</p>'}
      </div>
    </div>`;
});

// ================= STEALS & REACHES =================
(function(){
  const withPred = D.players.filter(p=>p.drafted && p.prediction && p.prediction.value_delta!==null);
  const steals = [...withPred].sort((a,b)=>b.prediction.value_delta - a.prediction.value_delta).slice(0,20);
  const reaches = [...withPred].sort((a,b)=>a.prediction.value_delta - b.prediction.value_delta).slice(0,20);
  document.getElementById('stealsBody').innerHTML = steals.map(p=>`<tr><td>${p.player}</td><td>${p.pos}</td><td>${p.year}</td><td>Pk ${p.pick}</td><td>~${p.prediction.predicted_pick}</td></tr>`).join('');
  document.getElementById('reachesBody').innerHTML = reaches.map(p=>`<tr><td>${p.player}</td><td>${p.pos}</td><td>${p.year}</td><td>Pk ${p.pick}</td><td>~${p.prediction.predicted_pick}</td></tr>`).join('');
})();

// ================= OUTLIERS =================
(function(){
  const freaks = D.players.filter(p=>p.sai!==null && p.sai>=90 && p.iso_pctile!==null).sort((a,b)=>b.iso_pctile-a.iso_pctile).slice(0,20);
  const worst = D.players.filter(p=>p.drafted && p.sai!==null && p.forty!==null && p.vertical!==null).sort((a,b)=>a.sai-b.sai).slice(0,20);
  document.getElementById('freaksBody').innerHTML = freaks.map(p=>`<tr><td>${p.player}</td><td>${p.pos}</td><td>${p.year}</td><td>${p.sai}</td><td>${p.iso_pctile}%</td></tr>`).join('');
  document.getElementById('worstBody').innerHTML = worst.map(p=>`<tr><td>${p.player}</td><td>${p.pos}</td><td>${p.year}</td><td>Pk ${p.pick??'—'}</td><td>${p.sai}</td></tr>`).join('');
})();

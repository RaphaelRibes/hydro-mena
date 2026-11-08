#!/usr/bin/env python3
"""MENA Studio — a local web GUI for hydro-mena runs.

No dependencies beyond the Python standard library: it serves one page and a
small JSON API over the CSVs/JSON that `hydro-mena run` already writes, so it
can never disagree with the CLI.

    pixi run python gui/app.py --root . --port 8770
    -> http://127.0.0.1:8770

Views: overview cards, RMT threshold scan, Fig-5 module/trait heatmap,
co-occurrence network (force layout), zi-Pi role map, degree distribution,
module sizes. A "new run" form shells out to the hydro-mena binary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "."


# ------------------------------------------------------------------ data layer
def find_runs(root: str) -> list[dict]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "target"]
        if "report.json" in filenames:
            rel = os.path.relpath(dirpath, root)
            try:
                with open(os.path.join(dirpath, "report.json"), encoding="utf-8") as fh:
                    rep = json.load(fh)
                out.append(
                    {
                        "path": rel,
                        "n_otus": rep.get("n_otus"),
                        "n_samples": rep.get("n_samples"),
                        "n_modules": rep.get("modules", {}).get("n_modules"),
                        "st": rep.get("rmt", {}).get("st"),
                        "modularity": rep.get("topology", {}).get("modularity"),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
    out.sort(key=lambda r: r["path"])
    return out


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_run(path: str) -> dict:
    d = os.path.join(ROOT, path)
    with open(os.path.join(d, "report.json"), encoding="utf-8") as fh:
        rep = json.load(fh)
    return {
        "report": rep,
        "nodes": read_csv(os.path.join(d, "nodes.csv")),
        "edges": read_csv(os.path.join(d, "edges.csv")),
        "fig5_r": read_csv(os.path.join(d, "fig5_r.csv")),
        "fig5_p": read_csv(os.path.join(d, "fig5_p.csv")),
        "rmt_scan": read_csv(os.path.join(d, "rmt_scan.csv")),
        "degree_dist": read_csv(os.path.join(d, "degree_dist.csv")),
        "modules": read_csv(os.path.join(d, "modules.csv")),
        "spectrum": read_csv(os.path.join(d, "spectrum.csv")),
    }


# ------------------------------------------------------------------ http layer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - base class signature
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/runs":
            self._send(200, json.dumps(find_runs(ROOT)).encode(), "application/json")
        elif u.path == "/api/run":
            path = q.get("path", [""])[0]
            try:
                self._send(200, json.dumps(load_run(path)).encode(), "application/json")
            except Exception as exc:  # noqa: BLE001 - surface to the UI
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        args = [
            os.environ.get("MENA_BIN", os.path.join(os.getcwd(), "target/release/mena")),
            "run",
            "--abundance", form.get("abundance", [""])[0],
            "--traits", form.get("traits", [""])[0],
            "--out-dir", form.get("out_dir", ["out"])[0],
            "--permutations", form.get("permutations", ["199"])[0],
        ]
        if form.get("st", [""])[0]:
            args += ["--st", form["st"][0]]
        if form.get("rmt_start", [""])[0]:
            args += ["--rmt-start", form["rmt_start"][0]]
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=1800)
            self._send(200, json.dumps({"ok": p.returncode == 0, "log": p.stdout[-4000:] + p.stderr[-4000:]}).encode(), "application/json")
        except Exception as exc:  # noqa: BLE001
            self._send(500, json.dumps({"ok": False, "log": str(exc)}).encode(), "application/json")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>MENA Studio</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ec;--mut:#9aa3b2;--acc:#5aa9e6;--line:#2a2f3a}
*{box-sizing:border-box}body{margin:0;font:13px/1.45 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:15px;margin:0 12px 0 0;font-weight:600}.tag{color:var(--mut);font-size:12px}
select,input,button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 8px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--acc)}
nav{display:flex;gap:4px;padding:8px 16px 0;flex-wrap:wrap}
nav button{border-radius:6px 6px 0 0;border-bottom:none}nav button.on{background:var(--card);color:var(--acc)}
main{padding:14px 16px 40px}section{display:none}section.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:19px;font-variant-numeric:tabular-nums;margin-top:2px}
canvas{background:var(--card);border:1px solid var(--line);border-radius:8px;display:block;max-width:100%}
.row{display:flex;gap:14px;flex-wrap:wrap}.grow{flex:1 1 420px}
table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500}code{color:var(--acc)}
.warn{border-left:3px solid #e0a33a;background:#1d1a12;padding:8px 10px;border-radius:6px;margin-bottom:10px}
.ok{border-left:3px solid #4caf50;background:#121b13;padding:8px 10px;border-radius:6px;margin-bottom:10px}
form{display:grid;gap:8px;max-width:520px}label{color:var(--mut);font-size:12px}
pre{background:#0b0d11;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:280px;overflow:auto;font-size:11px}
</style></head><body>
<header><h1>MENA Studio</h1><span class="tag">run</span>
<select id="runs" style="min-width:280px"></select>
<button id="reload">reload</button><span id="status" class="tag"></span></header>
<nav>
<button data-t="overview" class="on">Overview</button><button data-t="rmt">RMT scan</button>
<button data-t="fig5">Fig-5 heatmap</button><button data-t="network">Network</button>
<button data-t="roles">Roles (zi–Pi)</button><button data-t="degrees">Degrees</button>
<button data-t="modules">Modules</button><button data-t="runnew">New run</button>
</nav>
<main>
<section id="overview" class="on"><div id="banner"></div><div class="cards" id="cards"></div>
<div class="row"><div class="grow"><canvas id="c_sizes" width="620" height="260"></canvas></div>
<div class="grow"><table id="t_mods"></table></div></div></section>
<section id="rmt"><div id="rmt_note"></div><canvas id="c_rmt" width="980" height="380"></canvas></section>
<section id="fig5"><canvas id="c_fig5" width="900" height="420"></canvas></section>
<section id="network"><div class="tag" id="netinfo"></div><canvas id="c_net" width="980" height="640"></canvas></section>
<section id="roles"><canvas id="c_roles" width="760" height="520"></canvas></section>
<section id="degrees"><canvas id="c_deg" width="700" height="440"></canvas></section>
<section id="modules"><canvas id="c_eig" width="900" height="300"></canvas></section>
<section id="runnew"><form id="f"><label>abundance csv<input name="abundance" placeholder="real/abundance_ox.csv"></label>
<label>traits csv<input name="traits" placeholder="real/traits_ox.csv"></label>
<label>output dir<input name="out_dir" value="out_gui"></label>
<label>fixed st (blank = RMT scan)<input name="st" placeholder="0.52"></label>
<label>RMT start<input name="rmt_start" value="0.30"></label>
<label>permutations<input name="permutations" value="199"></label>
<button type="submit">run pipeline</button></form><pre id="log"></pre></section>
</main>
<script>
const $=s=>document.querySelector(s), S={run:null,tab:'overview'};
// CSV-derived identifiers (OTU/module names, run paths) are interpolated into
// markup, so escape them: the data files are external input.
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{S.tab=b.dataset.t;
 document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===S.tab)); draw();});
const fmt=(v,d=2)=>v==null||isNaN(v)?'—':(+v).toFixed(d);
async function loadRuns(sel){const r=await (await fetch('/api/runs')).json();
 $('#runs').innerHTML=r.map(x=>`<option ${x.path===sel?'selected':''} value="${esc(x.path)}">${esc(x.path)} · ${x.n_otus}×${x.n_samples} · st ${fmt(x.st,3)} · M ${fmt(x.modularity)}</option>`).join('');
 if(!sel&&r.length)return loadRun(r[0].path);}
async function loadRun(p){$('#status').textContent='loading…';const r=await (await fetch('/api/run?path='+encodeURIComponent(p))).json();
 if(r.error){$('#status').textContent=r.error;return;} S.run=r;$('#status').textContent=p;draw();}
$('#runs').onchange=e=>loadRun(e.target.value);$('#reload').onclick=()=>loadRuns($('#runs').value);
function draw(){const r=S.run;if(!r)return;const f={overview:overview,rmt:rmtV,fig5:fig5V,network:netV,roles:rolesV,degrees:degV,modules:modV}[S.tab];f&&f(r);}
function cards(r){const t=r.report.topology,m=r.report.modules,x=r.report.rmt;
 const rows=[['OTUs kept',`${x.n_kept} / ${r.report.n_otus}`],['edges',t.edges],['avg degree',fmt(t.avg_k,1)],
 ['GD (path len)',fmt(t.gd)],['avg clustering',fmt(t.avg_cc,3)],['modularity',fmt(t.modularity,3)],
 ['modules',m.n_modules],['density',fmt(t.density,4)],['γ (power law)',fmt(t.gamma)],['R² power law',fmt(t.r2_powerlaw)],
 ['Mantel r',fmt(r.report.mantel.r,3)],['mantel p',fmt(r.report.mantel.p,3)]];
 $('#cards').innerHTML=rows.map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');}
function overview(r){cards(r);const x=r.report.rmt;
 const fixed = r.report.input && r.report.input.fixed_threshold != null;
 // A fixed --st run is a deliberate choice, not a failed RMT scan: reporting it
 // as a fallback failure was misleading.
 $('#banner').innerHTML = fixed
  ? `<div class="ok"><b>Fixed threshold</b> st=${fmt(x.st,3)} — the RMT scan was skipped by request, so no Poisson verdict exists for this graph.</div>`
  : x.converged
   ? `<div class="ok"><b>RMT converged</b> at st=${fmt(x.st,3)} — the spacing GOF was compatible with Poisson.</div>`
   : `<div class="warn"><b>No threshold accepted the Poisson GOF.</b> st=${fmt(x.st,3)} is a diagnostic fallback (closest to critical). Treat the graph as threshold-sensitive; use the RMT tab, or pin <code>--st</code> to a value you can defend.</div>`;
 const mods=r.modules;const c=$('#c_sizes'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 const w=c.width/mods.length;mods.forEach((m,i)=>{const h=+m.size/ Math.max(...mods.map(z=>+z.size))*190;
  g.fillStyle='hsl('+(i*360/Math.max(1,mods.length))+',65%,55%)';g.fillRect(i*w+3,c.height-30-h,w-6,h);
  g.fillStyle='#9aa3b2';g.font='10px sans-serif';g.fillText(m.module,i*w+6,c.height-12);});
 $('#t_mods').innerHTML='<tr><th>module</th><th>OTUs</th><th>var. expl.</th></tr>'+
  mods.map(m=>`<tr><td>${esc(m.module)}</td><td>${esc(m.size)}</td><td>${fmt(m.var_explained,3)}</td></tr>`).join('');}
function axes(g,w,h,pad){g.strokeStyle='#2a2f3a';g.beginPath();g.moveTo(pad,h-pad);g.lineTo(w-8,h-pad);g.moveTo(pad,h-pad);g.lineTo(pad,8);g.stroke();}
function rmtV(r){const s=r.rmt_scan,c=$('#c_rmt'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!s.length){g.fillStyle='#9aa3b2';g.fillText('no scan (fixed --st run)',20,40);return;}
 const pad=52,chis=s.map(p=>+p.chi2),crit=s.map(p=>+p.crit),sts=s.map(p=>+p.st);
 const lo=Math.log10(Math.min(...chis,...crit)*0.8),hi=Math.log10(Math.max(...chis,...crit)*1.25);
 const X=v=>pad+(v-Math.min(...sts))/(Math.max(...sts)-Math.min(...sts)||1)*(c.width-pad-20);
 const Y=v=>c.height-pad-(Math.log10(v)-lo)/(hi-lo)*(c.height-pad-30);
 axes(g,c.width,c.height,pad);
 g.strokeStyle='#e05c5c';g.beginPath();s.forEach((p,i)=>{const x=X(sts[i]),y=Y(crit[i]);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
 g.strokeStyle='#5aa9e6';g.beginPath();s.forEach((p,i)=>{const x=X(sts[i]),y=Y(chis[i]);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
 s.filter(p=>p.poisson==='true').forEach(p=>{g.fillStyle='#4caf50';g.beginPath();g.arc(X(+p.st),Y(+p.chi2),4,0,7);g.fill();});
 g.fillStyle='#e6e8ec';g.font='11px sans-serif';
 g.fillText('χ² observed (blue) vs critical (red), green = Poisson accepted',pad,18);
 g.fillText('st '+fmt(Math.min(...sts),2),c.width-90,c.height-16);
 $('#rmt_note').innerHTML=r.report.rmt.fallback?'<div class="warn">fallback threshold — no acceptance in range</div>':'<div class="ok">converged</div>';}
function fig5V(r){const R=r.fig5_r,P=r.fig5_p,c=$('#c_fig5'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!R.length)return;const cols=Object.keys(R[0]).slice(1),rows=R.length;
 const cw=Math.min(160,(c.width-160)/cols.length),chh=Math.min(52,(c.height-70)/rows);
 const ox=150,oy=40;
 const col=v=>{const t=(v+1)/2;return `rgb(${Math.round(215*t+40*(1-t))},${Math.round(80*t+180*(1-t))},${Math.round(90*t+120*(1-t))})`;};
 R.forEach((row,i)=>cols.forEach((k,j)=>{const v=parseFloat(row[k]);g.fillStyle=col(v);g.fillRect(ox+j*cw,oy+i*chh,cw-2,chh-2);
  g.fillStyle=Math.abs(v)>0.6?'#fff':'#111';g.font='11px sans-serif';g.fillText(v.toFixed(2),ox+j*cw+8,oy+i*chh+chh/2+4);}));
 g.fillStyle='#9aa3b2';g.font='11px sans-serif';
 cols.forEach((k,j)=>g.fillText(k,ox+j*cw+2,oy-8));R.forEach((row,i)=>g.fillText(row.module,6,oy+i*chh+chh/2+4));}
function netV(r){const n=r.nodes,e=r.edges,c=$('#c_net'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!n.length){g.fillStyle='#9aa3b2';g.fillText('nodes.csv missing',20,40);return;}
 const idx=new Map(n.map((d,i)=>[d.OTU,i])),W=c.width,H=c.height,cx=W/2,cy=H/2;
 const pos=n.map((d,i)=>{const a=i/n.length*6.283;const rad=Math.min(W,H)*0.42*Math.sqrt((i%97)/97);return[cx+rad*Math.cos(a),cy+rad*Math.sin(a)];});
 const E=e.map(d=>[idx.get(d.source),idx.get(d.target)]).filter(([a,b])=>a!=null&&b!=null);
 for(let it=0;it<140;it++){const dx=new Float64Array(n.length),dy=new Float64Array(n.length);
  for(let i=0;i<n.length;i++)for(let j=i+1;j<n.length;j++){let ddx=pos[i][0]-pos[j][0],ddy=pos[i][1]-pos[j][1];
   let d2=ddx*ddx+ddy*ddy+0.01,f=900/d2;const L=Math.sqrt(d2);ddx/=L;ddy/=L;
   dx[i]+=ddx*f;dy[i]+=ddy*f;dx[j]-=ddx*f;dy[j]-=ddy*f;}
  for(const [a,b] of E){let ddx=pos[a][0]-pos[b][0],ddy=pos[a][1]-pos[b][1];const d=Math.sqrt(ddx*ddx+ddy*ddy)||1;
   const f=(d-60)*0.012;dx[a]-=ddx/d*f*10;dy[a]-=ddy/d*f*10;dx[b]+=ddx/d*f*10;dy[b]+=ddy/d*f*10;}
  for(let i=0;i<n.length;i++){pos[i][0]+=Math.max(-6,Math.min(6,dx[i]));pos[i][1]+=Math.max(-6,Math.min(6,dy[i]));
   pos[i][0]=Math.max(14,Math.min(W-14,pos[i][0]));pos[i][1]=Math.max(14,Math.min(H-14,pos[i][1]));}}
 g.strokeStyle='rgba(140,150,170,.28)';g.lineWidth=1;g.beginPath();
 for(const [a,b] of E){g.moveTo(pos[a][0],pos[a][1]);g.lineTo(pos[b][0],pos[b][1]);}g.stroke();
 const mods=[...new Set(n.map(d=>+d.module))];
 n.forEach((d,i)=>{const r=3+Math.sqrt(+d.degree)*0.9;g.fillStyle=`hsl(${(+d.module)*360/Math.max(1,mods.length)},70%,58%)`;
  g.beginPath();g.arc(pos[i][0],pos[i][1],r,0,7);g.fill();});
 const hubs=[...n].sort((a,b)=>b.degree-a.degree).slice(0,8);
 g.fillStyle='#e6e8ec';g.font='10px sans-serif';hubs.forEach(d=>{const i=idx.get(d.OTU);g.fillText(d.OTU,pos[i][0]+5,pos[i][1]-5);});
 $('#netinfo').textContent=`${n.length} nodes · ${E.length} edges · colour = module · size = degree`;}
function rolesV(r){const n=r.nodes,c=$('#c_roles'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!n.length)return;const pad=52,zis=n.map(d=>+d.zi),pis=n.map(d=>+d.Pi);
 const zx=Math.max(3,...zis)*1.1,py=Math.max(0.7,...pis)*1.1;
 const X=v=>pad+v/zx*(c.width-pad-20),Y=v=>c.height-pad-v/py*(c.height-pad-30);
 axes(g,c.width,c.height,pad);
 g.setLineDash([4,4]);g.strokeStyle='#5c6472';
 g.beginPath();g.moveTo(X(2.5),Y(0));g.lineTo(X(2.5),Y(py));g.moveTo(X(0),Y(0.62));g.lineTo(X(zx),Y(0.62));g.stroke();g.setLineDash([]);
 n.forEach((d,i)=>{g.fillStyle=`hsl(${(+d.module)*47%360},70%,58%)`;g.beginPath();g.arc(X(+d.zi),Y(+d.Pi),2.6,0,7);g.fill();});
 g.fillStyle='#9aa3b2';g.font='11px sans-serif';
 g.fillText('zi →',c.width-40,c.height-16);g.fillText('Pi',8,20);g.fillText('zi=2.5',X(2.5)+4,Y(py)-6);g.fillText('Pi=0.62',X(0)+4,Y(0.62)-6);}
function degV(r){const d=r.degree_dist,c=$('#c_deg'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!d.length)return;const pad=56,ks=d.map(x=>+x.k).filter(v=>v>0),ps=d.map(x=>+x.pk).filter(v=>v>0);
 const lx=v=>Math.log10(v),lo=lx(Math.min(...ks)),hi=lx(Math.max(...ks)),lp=lx(Math.min(...ps))-0.2,hp=lx(Math.max(...ps))+0.2;
 const X=v=>pad+(lx(v)-lo)/(hi-lo||1)*(c.width-pad-30),Y=v=>c.height-pad-(lx(v)-lp)/(hp-lp)*(c.height-pad-30);
 axes(g,c.width,c.height,pad);g.fillStyle='#5aa9e6';
 d.forEach(x=>{if(+x.k>0&&+x.pk>0){g.beginPath();g.arc(X(+x.k),Y(+x.pk),3.4,0,7);g.fill();}});
 g.fillStyle='#9aa3b2';g.font='11px sans-serif';g.fillText('log k →',c.width-70,c.height-16);g.fillText('log P(k)',8,20);
 g.fillText(`γ = ${fmt(r.report.topology.gamma)}  R² = ${fmt(r.report.topology.r2_powerlaw)}`,pad,20);}
function modV(r){const m=r.modules,c=$('#c_eig'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!m.length)return;const pad=56,w=(c.width-pad-20)/m.length;
 axes(g,c.width,c.height,pad);
 m.forEach((x,i)=>{const h=+x.var_explained*(c.height-pad-30);g.fillStyle=`hsl(${i*360/Math.max(1,m.length)},60%,55%)`;
  g.fillRect(pad+i*w+3,c.height-pad-h,w-6,h);g.fillStyle='#9aa3b2';g.font='10px sans-serif';
  g.fillText(x.module,pad+i*w+4,c.height-pad+12);});
 g.fillStyle='#9aa3b2';g.font='11px sans-serif';g.fillText('PC1 variance explained per module',pad,20);}
$('#f').onsubmit=async e=>{e.preventDefault();$('#log').textContent='running…';
 const b=new URLSearchParams(new FormData(e.target));
 const r=await (await fetch('/api/run',{method:'POST',body:b})).json();
 $('#log').textContent=r.log||'';loadRuns();};
loadRuns();
</script></body></html>"""


def main() -> int:
    global ROOT
    ap = argparse.ArgumentParser(description="MENA Studio — local GUI for hydro-mena runs")
    ap.add_argument("--root", default=".", help="directory searched for run folders (report.json)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    ROOT = os.path.abspath(args.root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    runs = find_runs(ROOT)
    print(f"MENA Studio on http://{args.host}:{args.port}  root={ROOT}  ({len(runs)} runs found)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())

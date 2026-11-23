/* SensorMotion v0.6 — Android browser motion monitor. No model inference. */
(()=>{
'use strict';
const el=id=>document.getElementById(id);
let running=false, records=[], startAt=0, lastEvent=0, recent=[], plotBusy=new Set(), lastPaint=0, accSource='none', gotAcc=0,gotGyro=0, warningTimer=null;
const LIMIT=180000, WINDOW=20, RAD=Math.PI/180;
const finite=n=>typeof n==='number'&&Number.isFinite(n)?n:null;
const axes=obj=>obj&&['x','y','z'].every(k=>finite(obj[k])!==null)?[obj.x,obj.y,obj.z]:null;
const magnitude=a=>a?Math.hypot(...a):null;
const pretty=a=>a?a.map(x=>x.toFixed(3)).join(' / '):'無資料';
function msg(text){el('liveMessage').textContent=text}
function environment(){const secure=window.isSecureContext;const supported='DeviceMotionEvent' in window;el('liveEnvironment').textContent=`${secure?'安全連線 ✓':'非安全連線（Android 需要 HTTPS）'} ｜ ${supported?'瀏覽器支援 DeviceMotionEvent':'瀏覽器不支援 DeviceMotionEvent'} ｜ 建議 Android Chrome，螢幕保持開啟`;el('liveStart').disabled=!secure||!supported;}
function controls(){el('liveStart').disabled=running||!window.isSecureContext||!('DeviceMotionEvent' in window);el('liveStop').disabled=!running;el('liveExport').disabled=records.length===0;el('liveState').textContent=running?'讀取中':'已停止';}
function onMotion(e){if(!running)return;const now=performance.now();const t=(now-startAt)/1000;lastEvent=now;recent.push(now);while(recent.length&&now-recent[0]>5000)recent.shift();
 let acc=axes(e.acceleration), source='linear';if(!acc){acc=axes(e.accelerationIncludingGravity);source=acc?'with_gravity':'none'}
 const rot=e.rotationRate;let gyr=null;if(rot&&[rot.beta,rot.gamma,rot.alpha].every(v=>finite(v)!==null))gyr=[rot.beta*RAD,rot.gamma*RAD,rot.alpha*RAD];
 if(acc)gotAcc++;if(gyr)gotGyro++;if(acc)accSource=source;
 records.push({t,acc,gyr,source,interval:finite(e.interval)});
 if(records.length>=LIMIT){stop();msg('已達 180,000 筆記錄上限，自動停止。請下載 CSV。');return}
 if(now-lastPaint>250){lastPaint=now;render();}
}
function series(which){const arr=records.filter(r=>r.t>=Math.max(0,records[records.length-1].t-WINDOW)&&r[which]);const x=arr.map(r=>r.t),a=arr.map(r=>r[which]);return ['X','Y','Z','Magnitude'].map((name,i)=>({x,y:a.map(v=>i<3?v[i]:magnitude(v)),name,mode:'lines',type:'scatter',line:{width:1.6}}));}
function draw(id,which,unit){const node=el(id);if(typeof Plotly==='undefined'){node.textContent='無法載入 Plotly 圖表；請檢查網路。';return}if(plotBusy.has(id))return;plotBusy.add(id);Promise.resolve(Plotly.react(node,series(which),{margin:{l:52,r:12,t:16,b:48},xaxis:{title:'錄製時間（秒）'},yaxis:{title:unit},legend:{orientation:'h',y:1.2},paper_bgcolor:'#fff',plot_bgcolor:'#fff',uirevision:'live'},{responsive:true,displaylogo:false})).catch(e=>{node.textContent='圖表更新失敗：'+e.message}).finally(()=>plotBusy.delete(id))}
function render(){const last=records[records.length-1];el('liveDuration').textContent=(last?.t||0).toFixed(1)+' 秒';el('liveCounts').textContent=`${records.length.toLocaleString()} / ${gotAcc.toLocaleString()} / ${gotGyro.toLocaleString()}`;const hz=recent.length>=2?(recent.length-1)*1000/(recent[recent.length-1]-recent[0]):0;el('liveHz').textContent=hz?hz.toFixed(1)+' Hz':'— Hz';el('liveAccValues').textContent=pretty(last?.acc);el('liveGyroValues').textContent=pretty(last?.gyr);el('liveAccSource').textContent=accSource==='linear'?'線性加速度（不含重力） · m/s²':accSource==='with_gravity'?'注意：目前使用含重力加速度 · m/s²，與原訓練資料可能不一致':'尚未收到有效加速度資料';if(records.length){draw('liveAccChart','acc','m/s²');draw('liveGyroChart','gyr','rad/s')}controls()}
async function start(){if(running)return;environment();if(!window.isSecureContext){msg('請透過 HTTPS 開啟網站；手機無法直接用電腦的 localhost 連線。');return}if(!('DeviceMotionEvent' in window)){msg('此瀏覽器不支援 DeviceMotionEvent。');return}
 try{if(typeof DeviceMotionEvent.requestPermission==='function'){const permission=await DeviceMotionEvent.requestPermission();if(permission!=='granted'){msg('未取得動作感測器權限。');return}}
 running=true;records=[];recent=[];gotAcc=0;gotGyro=0;accSource='none';startAt=performance.now();lastEvent=0;lastPaint=0;window.addEventListener('devicemotion',onMotion);controls();msg('正在讀取。請輕微移動手機，確認三軸數值與取樣率有變化。');clearInterval(warningTimer);warningTimer=setInterval(()=>{if(running&&performance.now()-startAt>3000&&(!lastEvent||performance.now()-lastEvent>3000))msg('超過 3 秒未收到感測器事件：請檢查 HTTPS、瀏覽器權限、裝置支援與前景狀態。')},1000);
 }catch(e){running=false;controls();msg('無法啟動感測器：'+(e?.message||String(e)))}
}
function stop(){if(!running)return;running=false;window.removeEventListener('devicemotion',onMotion);clearInterval(warningTimer);warningTimer=null;render();msg(`已停止，共 ${records.length.toLocaleString()} 筆事件。可下載 CSV。`)}
function clear(){if(running)stop();records=[];recent=[];gotAcc=0;gotGyro=0;accSource='none';el('liveHz').textContent='— Hz';el('liveDuration').textContent='0.0 秒';el('liveCounts').textContent='0 / 0 / 0';el('liveAccValues').textContent='—';el('liveGyroValues').textContent='—';el('liveAccSource').textContent='等待資料（單位 m/s²）';for(const id of ['liveAccChart','liveGyroChart']){if(typeof Plotly!=='undefined')try{Plotly.purge(el(id))}catch(_){}el(id).replaceChildren()}controls();msg('紀錄已清除。')}
function csvValue(v){return v===null||v===undefined?'':String(v)}
function exportCsv(){if(!records.length)return;const header=['seconds_elapsed','acc_x_m_s2','acc_y_m_s2','acc_z_m_s2','acc_magnitude_m_s2','acc_source','gyro_x_rad_s','gyro_y_rad_s','gyro_z_rad_s','gyro_magnitude_rad_s','event_interval_ms'];const lines=[header.join(',')];for(const r of records){lines.push([r.t.toFixed(4),...(r.acc||[null,null,null]).map(csvValue),csvValue(magnitude(r.acc)),r.source,...(r.gyr||[null,null,null]).map(csvValue),csvValue(magnitude(r.gyr)),csvValue(r.interval)].join(','))}const blob=new Blob(['\ufeff',lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='SensorMotion_live_'+new Date().toISOString().replace(/[:.]/g,'-')+'.csv';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1500)}
function init(){environment();el('liveStart').addEventListener('click',start);el('liveStop').addEventListener('click',stop);el('liveClear').addEventListener('click',clear);el('liveExport').addEventListener('click',exportCsv);document.addEventListener('visibilitychange',()=>{if(document.hidden&&running){stop();msg('網頁已進入背景，已自動停止錄製以避免資料不連續。')}});window.addEventListener('pagehide',()=>{if(running)stop()});controls()}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();

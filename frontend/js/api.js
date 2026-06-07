/**
 * api.js — SSE stream client
 */
const API_BASE = (window.location.hostname==='127.0.0.1'||window.location.hostname==='localhost')
  ? 'http://127.0.0.1:8000' : '';

export async function streamResearch({query,apiKey,onEvent,onDone,onError}) {
  try {
    const response = await fetch(`${API_BASE}/api/v1/research/stream`,{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query,api_key:apiKey||null}),
    });
    if (!response.ok){
      const err=await response.json().catch(()=>({}));
      throw new Error(err.detail||`HTTP ${response.status}`);
    }
    const reader=response.body.getReader();
    const decoder=new TextDecoder();
    let buffer='';
    while(true){
      const{done,value}=await reader.read();
      if(done) break;
      buffer+=decoder.decode(value,{stream:true});
      const lines=buffer.split('\n');
      buffer=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        const raw=line.slice(6).trim();
        if(raw==='[DONE]'){onDone();return;}
        try{onEvent(JSON.parse(raw));}catch{}
      }
    }
    onDone();
  } catch(err){onError(err);}
}

/**
 * main.js — orchestration controller
 */
import { streamResearch } from './api.js';
import {
  showMainPanel, resetUI, startTimer, stopTimer,
  addLog, setNodeState, setLegendActive, activateArrow,
  showLoopIndicator, updateIteration, updateTokens,
  addScoreRing, setRunButton, renderReport,
} from './ui.js';

let finalReportText = '';
let lastNode = null;

window.setQuery = q => {
  document.getElementById('query').value = q;
  document.getElementById('query').focus();
};

window.copyResult = () => {
  navigator.clipboard.writeText(finalReportText).then(()=>{
    const btn = document.querySelector('.result-actions .icon-btn');
    btn.style.color='var(--supervisor)';
    setTimeout(()=>btn.style.color='',1500);
  });
};

window.downloadResult = () => {
  const query = document.getElementById('query').value.trim();
  const blob  = new Blob([finalReportText],{type:'text/markdown'});
  const url   = URL.createObjectURL(blob);
  const a     = Object.assign(document.createElement('a'),{
    href:url, download:'research-'+query.slice(0,30).replace(/\s+/g,'-').toLowerCase()+'.md'
  });
  a.click(); URL.revokeObjectURL(url);
};

function handleEvent(event) {
  const { agent, type, message, thinking, data } = event;

  if (type === 'status') {
    const state = message;
    setNodeState(agent, state, data?.label || '');
    setLegendActive(agent, state === 'active');
    if (state === 'active' && lastNode && lastNode !== agent) {
      activateArrow(lastNode, agent);
    }
    if (state === 'active') lastNode = agent;
  }

  if (type === 'log' || type === 'thought') {
    addLog(agent, message, thinking, type);
  }

  if (type === 'metric' && data) {
    if (data.iteration !== undefined) updateIteration(data.iteration);
    if (data.tokens    !== undefined) updateTokens(data.tokens);
  }

  if (type === 'score' && data) {
    addScoreRing(data.score, data.iteration);
    if (!data.approve) {
      updateIteration(data.iteration, true);
      showLoopIndicator(true, data.iteration + 1);
      addLog('critic',
        `Score ${data.score}/10 below threshold — sending back to Search Agent`,
        false, 'log');
    } else {
      showLoopIndicator(false);
    }
  }

  if (type === 'result' && data) {
    finalReportText = data.report || '';
    renderReport(finalReportText);
  }
}

window.runOrchestration = async function() {
  const query  = document.getElementById('query').value.trim();
  const apiKey = document.getElementById('apikey').value.trim();
  if (!query)  { alert('Please enter a research topic.'); return; }

  lastNode = null;
  setRunButton('loading');
  showMainPanel();
  resetUI();
  startTimer();

  await streamResearch({
    query, apiKey,
    onEvent: handleEvent,
    onDone: () => { stopTimer(); setRunButton('idle'); },
    onError: err => {
      addLog('system', `Error: ${err.message}`);
      stopTimer(); setRunButton('idle');
      ['supervisor','search','critic','writer'].forEach(a=>setLegendActive(a,false));
    },
  });
};

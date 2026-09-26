(() => {
  const STORAGE_KEY = 'omnitransform-ui-v1';
  const CLIENT_KEY = 'omnitransform-client-id';
  const outputDefs = [
    {id:'text', label:'Text', hint:'Plain text / Markdown'},
    {id:'pdf', label:'PDF', hint:'Polished PDF via Typst'},
    {id:'pptx', label:'PPTX', hint:'Editable presentation'},
    {id:'infographic', label:'Infographic', hint:'Factual SVG'},
    {id:'image', label:'Image', hint:'Creative FLUX image'},
  ];
  const clientId = localStorage.getItem(CLIENT_KEY) || crypto.randomUUID();
  localStorage.setItem(CLIENT_KEY, clientId);

  let state = loadState();
  let stagedFiles = [];
  let pollers = new Map();

  const $ = id => document.getElementById(id);
  const els = {
    chatList: $('chatList'), messages: $('messages'), empty: $('emptyState'), title: $('chatTitle'),
    input: $('messageInput'), send: $('sendBtn'), attach: $('attachBtn'), fileInput: $('fileInput'),
    staged: $('stagedFiles'), outputButton: $('outputButton'), outputLabel: $('outputButtonLabel'), outputMenu: $('outputMenu'),
    runtime: $('runtimeStatus'), sidebar: $('sidebar'), backdrop: $('sidebarBackdrop')
  };

  function loadState() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
      return { chats: saved.chats || [], activeChatId: saved.activeChatId || null, outputs: saved.outputs || ['text'] };
    } catch { return { chats: [], activeChatId: null, outputs: ['text'] }; }
  }
  function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
  function activeChat() { return state.chats.find(c => c.id === state.activeChatId) || null; }
  function escapeHtml(s='') { return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
  function fmtBytes(n) { if(n<1024) return `${n} B`; if(n<1048576) return `${(n/1024).toFixed(1)} KB`; return `${(n/1048576).toFixed(1)} MB`; }
  function headers() { return {'X-Client-ID': clientId}; }
  function toast(msg) { const el=document.createElement('div'); el.className='toast'; el.textContent=msg; $('toastHost').appendChild(el); setTimeout(()=>el.remove(),4200); }

  async function api(url, opts={}) {
    opts.headers = {...headers(), ...(opts.headers||{})};
    const res = await fetch(url, opts);
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try { const body=await res.json(); detail=body.detail || detail; } catch {}
      throw new Error(detail);
    }
    return res.json();
  }

  async function bootstrap() {
    try {
      const data = await api('/api/bootstrap');
      if (data.configured) {
        els.runtime.textContent = `Ready · ${data.max_parallel_jobs} parallel job${data.max_parallel_jobs===1?'':'s'}`;
      } else {
        const missing = (data.missing_credentials || []).join(', ');
        els.runtime.textContent = missing ? `Missing: ${missing}` : 'Configuration required';
        toast(missing ? `Render is not exposing: ${missing}` : (data.config_error || 'Backend configuration incomplete'));
      }
    } catch (e) { els.runtime.textContent='Backend unavailable'; toast(e.message); }
  }

  async function createChat(activate=true) {
    try {
      const session = await api('/api/chats', {method:'POST'});
      const chat = { id: session.session_id, title: 'New chat', messages: [], updatedAt: Date.now(), running: 0 };
      state.chats.unshift(chat);
      if (activate) state.activeChatId = chat.id;
      saveState(); render();
      return chat;
    } catch (e) { toast(`Could not create chat: ${e.message}`); return null; }
  }

  function ensureChat() { const c=activeChat(); if(c) return Promise.resolve(c); return createChat(true); }

  function renderOutputs() {
    els.outputMenu.innerHTML = outputDefs.map(o => `
      <label class="output-option">
        <input type="checkbox" data-output="${o.id}" ${state.outputs.includes(o.id)?'checked':''}>
        <span><div class="output-option-title">${o.label}</div><div class="output-option-hint">${o.hint}</div></span>
      </label>`).join('');
    const selected = outputDefs.filter(x => state.outputs.includes(x.id)).map(x=>x.label);
    els.outputLabel.textContent = selected.length === 1 ? selected[0] : `${selected.length} outputs`;
  }

  function renderFiles() {
    els.staged.innerHTML = stagedFiles.map((f,i)=>`<div class="file-chip"><span>${escapeHtml(f.name)} · ${fmtBytes(f.size)}</span><button data-remove-file="${i}">×</button></div>`).join('');
  }

  function renderChats() {
    els.chatList.innerHTML = state.chats.map(c => {
      const hasErr = c.messages.some(m => m.role==='assistant' && m.status==='error');
      const cls = c.running ? 'running' : hasErr ? 'error' : '';
      return `<div class="chat-row ${c.id===state.activeChatId?'active':''}" data-chat="${c.id}">
        <div class="chat-row-title">${escapeHtml(c.title || 'New chat')}</div><div class="job-indicator ${cls}"></div>
      </div>`;
    }).join('');
  }

  function artifactHtml(a, jobId) {
    if (a.status !== 'generated' || !a.download_url) return `<span class="artifact failed">${escapeHtml((a.format||'file').toUpperCase())} · ${escapeHtml(a.status||'failed')}</span>`;
    return `<button class="artifact" data-download-url="${escapeHtml(a.download_url)}" data-download-name="${escapeHtml(a.filename || 'artifact')}"><span>↓</span><span>${escapeHtml(a.filename || (a.format||'file').toUpperCase())}</span></button>`;
  }

  function renderMessage(m) {
    if (m.role === 'user') {
      const files = (m.files||[]).length ? `<div class="user-files">${m.files.map(f=>`<span class="user-file">${escapeHtml(f)}</span>`).join('')}</div>` : '';
      return `<div class="message user"><div><div>${files}</div><div class="user-bubble">${escapeHtml(m.text)}</div></div></div>`;
    }
    let body='';
    if (m.status === 'running' || m.status === 'queued') {
      body = `<div class="loading-row"><span>${escapeHtml(m.stage || 'Working')}</span><span class="loading-dots"><span></span><span></span><span></span></span></div>`;
    } else {
      body = `<div class="assistant-body ${m.status==='error'?'error':''}">${escapeHtml(m.text || 'No response')}</div>`;
      if ((m.artifacts||[]).length) body += `<div class="artifact-grid">${m.artifacts.map(a=>artifactHtml(a,m.jobId)).join('')}</div>`;
      if ((m.warnings||[]).length) body += `<div class="warning-block">${m.warnings.map(escapeHtml).join('<br>')}</div>`;
      if (m.performance?.wall_time_ms) body += `<div class="meta-line">Completed in ${(m.performance.wall_time_ms/1000).toFixed(1)}s${m.verification?.faithfulness_score!=null ? ` · faithfulness ${Number(m.verification.faithfulness_score).toFixed(3)}`:''}</div>`;
    }
    return `<div class="message assistant"><div class="assistant-wrap"><div class="assistant-avatar">O</div><div>${body}</div></div></div>`;
  }

  function renderConversation() {
    const chat = activeChat();
    els.title.textContent = chat?.title || 'New chat';
    els.send.disabled = Boolean(chat?.running);
    els.input.placeholder = chat?.running ? 'This chat is working — open a new chat to continue in parallel' : 'Message OmniTransform';
    if (!chat || !chat.messages.length) { els.empty.style.display='flex'; els.messages.innerHTML=''; return; }
    els.empty.style.display='none';
    els.messages.innerHTML = chat.messages.map(renderMessage).join('');
    requestAnimationFrame(()=>{ document.querySelector('.conversation').scrollTop = document.querySelector('.conversation').scrollHeight; window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'}); });
  }

  function render() { renderChats(); renderOutputs(); renderFiles(); renderConversation(); }

  async function send() {
    const text = els.input.value.trim();
    if (!text) return;
    const chat = await ensureChat(); if (!chat) return;
    const filesNow = [...stagedFiles];
    const outputsNow = [...state.outputs];
    if (!outputsNow.length) { toast('Select at least one output'); return; }

    chat.title = chat.messages.length ? chat.title : text.slice(0,48);
    chat.messages.push({role:'user', text, files:filesNow.map(f=>f.name), at:Date.now()});
    const placeholder = {role:'assistant', text:'', status:'queued', stage:'Queued', artifacts:[], warnings:[], at:Date.now()};
    chat.messages.push(placeholder); chat.running=(chat.running||0)+1; chat.updatedAt=Date.now();
    els.input.value=''; autoResize(); stagedFiles=[]; saveState(); render();

    const fd = new FormData();
    fd.append('query', text); fd.append('formats', JSON.stringify(outputsNow)); fd.append('request_mode','auto');
    filesNow.forEach(f=>fd.append('files', f, f.name));
    try {
      const job = await api(`/api/chats/${chat.id}/jobs`, {method:'POST', body:fd});
      placeholder.jobId = job.job_id; placeholder.status=job.status; placeholder.stage=job.stage; saveState(); render();
      pollJob(chat.id, placeholder, job.job_id);
    } catch(e) {
      placeholder.status='error'; placeholder.text=`Request could not start: ${e.message}`; chat.running=Math.max(0,chat.running-1); saveState(); render();
    }
  }

  function pollJob(chatId, message, jobId) {
    if (pollers.has(jobId)) return;
    const timer = setInterval(async()=>{
      try {
        const job = await api(`/api/jobs/${jobId}`);
        message.status=job.status; message.stage=job.stage;
        if (job.status === 'complete' || job.status === 'complete_with_issues' || job.status === 'error') {
          clearInterval(timer); pollers.delete(jobId);
          const chat=state.chats.find(c=>c.id===chatId);
          if (chat) chat.running=Math.max(0,(chat.running||1)-1);
          const r=job.result||{};
          message.text=r.assistant_text || job.error || 'No result returned.';
          message.artifacts=r.artifacts||[]; message.warnings=r.warnings||[]; message.verification=r.verification||{}; message.performance=r.performance||{};
          saveState(); render();
          if (state.activeChatId !== chatId) toast(`${chat?.title || 'A chat'} finished processing.`);
        } else { saveState(); renderChats(); if(state.activeChatId===chatId) renderConversation(); }
      } catch(e) { clearInterval(timer); pollers.delete(jobId); message.status='error'; message.text=`Lost job status: ${e.message}`; saveState(); render(); }
    }, 1400);
    pollers.set(jobId, timer);
  }

  function resumeJobs() {
    state.chats.forEach(chat => chat.messages.forEach(m => { if(m.role==='assistant' && m.jobId && ['queued','running'].includes(m.status)) pollJob(chat.id,m,m.jobId); }));
  }

  function autoResize() { els.input.style.height='auto'; els.input.style.height=Math.min(els.input.scrollHeight,180)+'px'; }

  $('newChatBtn').onclick = ()=>createChat(true);
  $('newChatTop').onclick = ()=>createChat(true);
  els.attach.onclick = ()=>els.fileInput.click();
  els.fileInput.onchange = () => { stagedFiles.push(...Array.from(els.fileInput.files||[])); els.fileInput.value=''; renderFiles(); };
  els.staged.onclick = e => { const b=e.target.closest('[data-remove-file]'); if(b){ stagedFiles.splice(Number(b.dataset.removeFile),1); renderFiles(); } };
  els.outputButton.onclick = e => { e.stopPropagation(); els.outputMenu.classList.toggle('open'); };
  els.outputMenu.onchange = e => { const cb=e.target.closest('[data-output]'); if(!cb)return; if(cb.checked && !state.outputs.includes(cb.dataset.output))state.outputs.push(cb.dataset.output); if(!cb.checked)state.outputs=state.outputs.filter(x=>x!==cb.dataset.output); if(!state.outputs.length){state.outputs=['text'];} saveState(); renderOutputs(); };
  document.addEventListener('click', e => { if(!e.target.closest('.output-picker-wrap')) els.outputMenu.classList.remove('open'); });
  els.send.onclick=send;
  els.messages.onclick = async e => {
    const button = e.target.closest('[data-download-url]');
    if (!button) return;
    try {
      const res = await fetch(button.dataset.downloadUrl, {headers: headers()});
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href=url; a.download=button.dataset.downloadName || 'artifact'; a.click();
      setTimeout(()=>URL.revokeObjectURL(url), 1000);
    } catch(err) { toast(`Download failed: ${err.message}`); }
  };
  els.input.addEventListener('input', autoResize);
  els.input.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){e.preventDefault(); send();} });
  els.chatList.onclick=e=>{ const row=e.target.closest('[data-chat]'); if(!row)return; state.activeChatId=row.dataset.chat; saveState(); render(); els.sidebar.classList.remove('open'); els.backdrop.classList.remove('show'); };
  $('openSidebar').onclick=()=>{els.sidebar.classList.add('open');els.backdrop.classList.add('show');};
  $('closeSidebar').onclick=()=>{els.sidebar.classList.remove('open');els.backdrop.classList.remove('show');};
  els.backdrop.onclick=$('closeSidebar').onclick;

  render(); bootstrap(); resumeJobs();
})();

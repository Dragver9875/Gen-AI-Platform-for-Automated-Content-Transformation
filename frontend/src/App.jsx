import React, { useState, useRef, useEffect } from 'react';

const USER_ID = 'default_user';

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  // Staged File
  const [stagedFile, setStagedFile] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Pills Selection
  const [activePills, setActivePills] = useState({
    'LinkedIn': true,
    'Exec Summary': true,
    'Advisory': true,
    'Video Script': false,
    'X Thread': false,
    'Slide Deck': false
  });

  // Context Dropdown Controls
  const [audience, setAudience] = useState('Technical');
  const [tone, setTone] = useState('Urgent');
  const [depth, setDepth] = useState('Exhaustive');

  // Load Sessions from PostgreSQL DB on mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Fetch messages whenever active session changes
  useEffect(() => {
    if (activeSessionId) {
      fetchMessages(activeSessionId);
    }
  }, [activeSessionId]);

  // Auto-scroll on message updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  // Helper Toast
  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3000);
  };

  // 1. GET /api/sessions (Fetch from PostgreSQL)
  const fetchSessions = async () => {
    try {
      const res = await fetch(`/api/sessions?user_id=${USER_ID}`);
      if (res.ok) {
        const data = await res.json();
        const loadedSessions = data.sessions || [];
        setSessions(loadedSessions);
        if (loadedSessions.length > 0 && !activeSessionId) {
          setActiveSessionId(loadedSessions[0].id);
        }
      }
    } catch (err) {
      console.warn('Could not fetch sessions from DB server:', err);
    }
  };

  // 2. GET /api/sessions/:id/messages (Fetch messages from PostgreSQL)
  const fetchMessages = async (sessionId) => {
    try {
      const res = await fetch(`/api/sessions/${sessionId}/messages?user_id=${USER_ID}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (err) {
      console.warn(`Could not fetch messages for session ${sessionId}:`, err);
    }
  };

  // 3. Create New Session in PostgreSQL
  const handleNewSession = async () => {
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: USER_ID, title: 'New Context' })
      });
      if (res.ok) {
        const newSession = await res.json();
        setSessions((prev) => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
        setMessages([]);
        setInputText('');
        showToast('New context initialized in Database');
      }
    } catch (err) {
      const newId = String(Date.now());
      const fallbackSession = { id: newId, title: 'New Context', updated_at: new Date().toISOString() };
      setSessions((prev) => [fallbackSession, ...prev]);
      setActiveSessionId(newId);
      setMessages([]);
    }
  };

  // 4. Delete Session from PostgreSQL
  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();
    try {
      await fetch(`/api/sessions/${sessionId}?user_id=${USER_ID}`, { method: 'DELETE' });
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.id !== sessionId);
        setActiveSessionId(remaining[0]?.id || null);
        if (remaining.length === 0) setMessages([]);
      }
      showToast('Session removed from Database');
    } catch (err) {
      console.error('Error deleting session:', err);
    }
  };

  // Toggle Pill
  const togglePill = (pillName) => {
    setActivePills((prev) => ({
      ...prev,
      [pillName]: !prev[pillName]
    }));
  };

  // Handle File Attachment
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      setStagedFile({
        name: file.name,
        size: `${sizeMB} MB`
      });
      showToast(`Attached ${file.name}`);
    }
  };

  // Toggle Runbook Item Checkbox
  const toggleRunbookItem = (msgId, itemIdx) => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id !== msgId || !msg.artifacts?.advisory) return msg;
        const newRunbook = [...msg.artifacts.advisory.runbook];
        newRunbook[itemIdx] = {
          ...newRunbook[itemIdx],
          done: !newRunbook[itemIdx].done
        };
        return {
          ...msg,
          artifacts: {
            ...msg.artifacts,
            advisory: {
              ...msg.artifacts.advisory,
              runbook: newRunbook
            }
          }
        };
      })
    );
  };

  // Switch Artifact Tab
  const setArtifactTab = (msgId, tabKey) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === msgId ? { ...msg, activeTab: tabKey } : msg))
    );
  };

  // 5. Send Prompt & Call Backend Transform Engine (POST /api/transform)
  const handleSend = async () => {
    if (!inputText.trim() && !stagedFile) return;

    const userQuery = inputText.trim() || 'Transform the attached content.';
    const selectedFormats = Object.keys(activePills).filter((k) => activePills[k]);
    const targetFormats = selectedFormats.length > 0 ? selectedFormats : ['Executive Summary'];

    setIsGenerating(true);
    setInputText('');

    try {
      const response = await fetch('/api/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: USER_ID,
          session_id: activeSessionId,
          query: userQuery,
          formats: targetFormats,
          audience,
          tone,
          detail: depth,
          file_name: stagedFile?.name
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.session_id && data.session_id !== activeSessionId) {
          setActiveSessionId(data.session_id);
        }
        setMessages((prev) => [...prev, data.user_message, data.assistant_message]);
        fetchSessions();
      } else {
        throw new Error('API server returned error');
      }
    } catch (err) {
      console.warn('Backend API note:', err);
    } finally {
      setIsGenerating(false);
      setStagedFile(null);
    }
  };

  // Filtered History
  const filteredSessions = sessions.filter((s) =>
    (s.title || '').toLowerCase().includes(searchQuery.toLowerCase())
  );

  const activeSessionObj = sessions.find((s) => s.id === activeSessionId);

  return (
    <div className="flex h-screen w-full bg-white text-slate-900 font-sans antialiased overflow-hidden text-xs">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed top-3 right-3 z-50 bg-slate-900 text-white text-[11px] px-3 py-1.5 rounded-md shadow-md border border-slate-700 flex items-center gap-1.5 animate-bounce">
          <span className="material-symbols-outlined text-[15px] text-emerald-400">check_circle</span>
          <span>{toastMessage}</span>
        </div>
      )}

      {/* LEFT SIDEBAR (Compact width: w-[220px]) */}
      <aside
        className={`${
          sidebarOpen ? 'w-[220px]' : 'w-0 -ml-[220px]'
        } bg-[#F8FAFC] border-r border-slate-200 flex flex-col justify-between transition-all duration-200 z-40 shrink-0 relative`}
      >
        <div className="flex flex-col h-[calc(100%-64px)]">
          {/* Sidebar Brand Header */}
          <div className="h-11 px-3 flex items-center justify-between border-b border-slate-200/80">
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-5 rounded bg-indigo-600 text-white flex items-center justify-center font-bold text-[11px] shadow-xs">
                O
              </div>
              <span className="text-xs text-slate-900 font-bold tracking-tight">OmniTransform</span>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="text-slate-400 hover:text-slate-700 p-1 rounded hover:bg-slate-200/60 transition-colors"
              title="Close Sidebar"
            >
              <span className="material-symbols-outlined text-[16px]">dock_to_right</span>
            </button>
          </div>

          {/* Action & Search */}
          <div className="p-2.5 flex flex-col gap-1.5">
            <button
              onClick={handleNewSession}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-md bg-white border border-slate-200 hover:border-indigo-300 text-slate-800 hover:bg-slate-50 shadow-xs transition-all text-[11px] font-medium"
            >
              <span className="material-symbols-outlined text-indigo-600 text-[15px]">add</span>
              <span>New Context</span>
            </button>

            <div className="relative flex items-center w-full">
              <span className="material-symbols-outlined absolute left-2 text-slate-400 text-[14px] pointer-events-none">
                search
              </span>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search DB contexts..."
                className="w-full bg-white text-slate-800 placeholder:text-slate-400 text-[10px] pl-6 pr-2 py-1 rounded border border-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition-all"
              />
            </div>
          </div>

          {/* PostgreSQL Nav History List */}
          <div className="flex-1 overflow-y-auto px-2.5 space-y-2.5">
            <div className="space-y-0.5">
              <div className="flex items-center justify-between px-1 text-[9px] text-slate-400 font-bold uppercase tracking-wider">
                <span>Database Contexts</span>
                <span className="text-[8px] bg-indigo-50 text-indigo-600 px-1 py-0.2 rounded border border-indigo-100 font-mono">
                  PostgreSQL
                </span>
              </div>
              <nav className="space-y-0.5 pt-1">
                {filteredSessions.length === 0 ? (
                  <p className="text-[10px] text-slate-400 px-2 py-1 italic">No saved contexts found in DB.</p>
                ) : (
                  filteredSessions.map((s) => (
                    <div
                      key={s.id}
                      onClick={() => setActiveSessionId(s.id)}
                      className={`group w-full text-left flex items-center justify-between px-2 py-1 transition-colors truncate rounded text-[11px] cursor-pointer ${
                        s.id === activeSessionId
                          ? 'bg-indigo-50/90 border border-indigo-100 text-indigo-950 font-semibold'
                          : 'text-slate-600 hover:bg-slate-200/50 hover:text-slate-900'
                      }`}
                    >
                      <span className="truncate flex-1">{s.title || 'Untitled Session'}</span>
                      <button
                        onClick={(e) => handleDeleteSession(e, s.id)}
                        className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 p-0.5 transition-opacity"
                        title="Delete Session"
                      >
                        <span className="material-symbols-outlined text-[12px]">delete</span>
                      </button>
                    </div>
                  ))
                )}
              </nav>
            </div>
          </div>
        </div>

        {/* User Profile Footer */}
        <div className="h-14 px-2.5 bg-white border-t border-slate-200 flex flex-col justify-center gap-0.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 min-w-0">
              <div className="w-6 h-6 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-[10px] shrink-0 ring-1 ring-slate-200">
                EV
              </div>
              <div className="flex flex-col min-w-0">
                <span className="text-[10px] font-bold text-slate-900 truncate">Elena Vance</span>
                <span className="text-[9px] text-slate-500 truncate">VP Product Sec</span>
              </div>
            </div>
            <div className="flex items-center gap-0.5">
              <button
                onClick={() => showToast('Settings opened')}
                className="text-slate-400 hover:text-slate-700 p-0.5 rounded hover:bg-slate-100 transition-colors"
                title="Settings"
              >
                <span className="material-symbols-outlined text-[15px]">settings</span>
              </button>
            </div>
          </div>
          <div className="flex items-center justify-between text-slate-500 text-[9px] font-medium truncate">
            <span>Enterprise SecOps</span>
            <span className="material-symbols-outlined text-[11px]">expand_more</span>
          </div>
        </div>
      </aside>

      {/* MAIN VIEWPORT */}
      <div className="flex-1 flex flex-col min-w-0 bg-white h-full relative">
        {/* TOP HEADER */}
        <header className="h-11 bg-white/95 backdrop-blur-md z-30 flex items-center justify-between px-3 md:px-4 border-b border-slate-200 shadow-xs shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="text-slate-500 hover:text-slate-800 p-1 rounded hover:bg-slate-100 transition-colors mr-1"
                title="Open Sidebar"
              >
                <span className="material-symbols-outlined text-[17px]">menu</span>
              </button>
            )}
            <span className="text-xs font-semibold text-slate-900 truncate max-w-md">
              {activeSessionObj?.title || 'Q3 Threat Intel Briefing & Multi-Channel Distribution'}
            </span>
            <div className="hidden sm:flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100 border border-slate-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="text-[9px] text-slate-700 font-medium">PostgreSQL Connected</span>
            </div>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => showToast('Context Branched')}
              className="flex items-center gap-1 px-2 py-1 rounded bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-[10px] transition-colors"
            >
              <span className="material-symbols-outlined text-[13px] text-slate-500">fork_right</span>
              <span className="hidden md:inline">Branch Context</span>
            </button>
            <button
              onClick={() => showToast('Share link copied')}
              className="flex items-center gap-1 px-2 py-1 rounded bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-[10px] transition-colors"
            >
              <span className="material-symbols-outlined text-[13px] text-slate-500">share</span>
              <span className="hidden md:inline">Share</span>
            </button>
            <button
              onClick={() => showToast('Exported artifacts as Markdown')}
              className="flex items-center gap-1 px-2.5 py-1 rounded bg-indigo-600 hover:bg-indigo-700 text-white text-[10px] font-medium transition-colors shadow-xs"
            >
              <span className="material-symbols-outlined text-[13px]">download</span>
              <span>Export All</span>
            </button>
          </div>
        </header>

        {/* CHAT CANVAS (Increased bottom padding pb-64 to completely avoid dock overlap) */}
        <main className="flex-1 overflow-y-auto p-3 md:p-4 pb-64">
          <div className="max-w-[760px] mx-auto flex flex-col gap-3.5">
            {messages.length === 0 ? (
              <div className="py-14 text-center flex flex-col items-center justify-center gap-2">
                <div className="w-9 h-9 rounded-lg bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center text-base font-bold">
                  O
                </div>
                <h2 className="text-sm font-bold text-slate-900">How can OmniTransform assist you today?</h2>
                <p className="text-[11px] text-slate-500 max-w-sm">
                  Enter your prompt in plain English. Your messages and dynamic AI transformations are saved directly to PostgreSQL.
                </p>
              </div>
            ) : (
              messages.map((msg) =>
                msg.sender === 'user' ? (
                  /* USER MESSAGE CARD */
                  <div
                    key={msg.id}
                    className="w-full bg-[#F8FAFC] border border-slate-200/90 rounded-lg p-3 shadow-xs flex flex-col gap-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <div className="w-5 h-5 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-[9px]">
                          EV
                        </div>
                        <span className="text-[11px] font-bold text-slate-900">{msg.user?.name || 'User'}</span>
                        <span className="text-[10px] text-slate-500">{msg.user?.role || ''}</span>
                      </div>
                      <div className="flex items-center gap-1 text-slate-400">
                        <span className="material-symbols-outlined text-[13px]">schedule</span>
                        <span className="text-[10px] text-slate-500">{msg.user?.time || ''}</span>
                      </div>
                    </div>

                    {/* Staged File Chip */}
                    {msg.file && (
                      <div className="flex items-center justify-between bg-white border border-slate-200 px-2.5 py-1 rounded-md shadow-xs">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <div className="w-5 h-5 rounded bg-red-50 text-red-600 border border-red-100 flex items-center justify-center shrink-0">
                            <span className="material-symbols-outlined text-[13px]">picture_as_pdf</span>
                          </div>
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-semibold text-slate-800 truncate">{msg.file.name}</span>
                            <span className="text-[9px] text-slate-400">{msg.file.size} • {msg.file.status}</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 text-emerald-700 shrink-0 bg-emerald-50 border border-emerald-200/70 px-2 py-0.2 rounded-full">
                          <span className="material-symbols-outlined text-[12px]">check_circle</span>
                          <span className="text-[9px] font-medium">Indexed</span>
                        </div>
                      </div>
                    )}

                    {/* User Text */}
                    <p className="text-xs text-slate-800 leading-relaxed font-normal">{msg.text}</p>

                    {/* Formats Badges */}
                    {msg.formats && msg.formats.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1 pt-0.5">
                        {msg.formats.map((fmt) => (
                          <span
                            key={fmt}
                            className="inline-flex items-center gap-1 px-2 py-0.2 rounded-full bg-indigo-50 border border-indigo-200/70 text-indigo-700 text-[9px] font-medium"
                          >
                            <span className="material-symbols-outlined text-[11px]">auto_awesome</span>
                            {fmt}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  /* ASSISTANT RESPONSE CARD */
                  <div
                    key={msg.id}
                    className="w-full bg-white border border-slate-200 rounded-lg p-3 shadow-xs flex flex-col gap-2.5"
                  >
                    {/* Header */}
                    <div className="flex items-center justify-between pb-1 border-b border-slate-100">
                      <div className="flex items-center gap-1.5">
                        <div className="w-5 h-5 rounded bg-indigo-600 text-white flex items-center justify-center font-bold text-[9px] shadow-xs">
                          O
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-xs font-bold text-slate-900">{msg.model || 'OmniTransform v2'}</span>
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.2 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-[9px] font-medium">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                            {msg.latency || '1.8s synthesis'}
                          </span>
                        </div>
                      </div>
                      <span className="text-[9px] text-slate-400">{msg.telemetry || 'PostgreSQL Persisted'}</span>
                    </div>

                    {/* Blurb */}
                    <p className="text-[11px] text-slate-600 leading-relaxed">{msg.blurb}</p>

                    {/* Multi-Tab Artifact Container */}
                    <div className="w-full bg-slate-50/80 border border-slate-200 rounded-md overflow-hidden shadow-xs flex flex-col">
                      {/* Tab Header */}
                      <div className="flex items-center justify-between px-2 pt-1 bg-[#F1F5F9] border-b border-slate-200 overflow-x-auto">
                        <div className="flex items-center gap-1">
                          {Object.keys(msg.artifacts || {}).map((tabKey) => {
                            const art = msg.artifacts[tabKey];
                            const activeTabKey = msg.activeTab || Object.keys(msg.artifacts || {})[0] || 'linkedin';
                            const isActive = activeTabKey === tabKey;
                            return (
                              <button
                                key={tabKey}
                                onClick={() => setArtifactTab(msg.id, tabKey)}
                                className={`flex items-center gap-1 px-2.5 py-1 rounded-t text-[10px] transition-all ${
                                  isActive
                                    ? 'bg-white border-t-2 border-indigo-600 text-slate-900 font-bold shadow-xs'
                                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/50'
                                }`}
                              >
                                <span className={`material-symbols-outlined text-[13px] ${isActive ? 'text-indigo-600' : ''}`}>
                                  {art.icon}
                                </span>
                                <span>{art.title}</span>
                                {art.badge && (
                                  <span className="px-1 py-0.2 rounded bg-slate-200/70 text-slate-600 text-[8px]">
                                    {art.badge}
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>

                        {/* Toolbar */}
                        <div className="flex items-center gap-1 pb-0.5 shrink-0">
                          <button
                            onClick={() => showToast('Copied raw markdown')}
                            className="flex items-center gap-1 px-1.5 py-0.5 rounded text-slate-500 hover:text-slate-900 hover:bg-slate-200/70 text-[9px] transition-colors"
                          >
                            <span className="material-symbols-outlined text-[12px]">content_copy</span>
                            <span>Copy Raw</span>
                          </button>
                          <button
                            onClick={() => showToast('Downloading artifact...')}
                            className="p-0.5 rounded text-slate-500 hover:text-slate-900 hover:bg-slate-200/70 transition-colors"
                            title="Download"
                          >
                            <span className="material-symbols-outlined text-[13px]">download</span>
                          </button>
                        </div>
                      </div>

                      {/* Tab Panels */}
                      <div className="p-3">
                        {(() => {
                          const currentTab = msg.activeTab || Object.keys(msg.artifacts || {})[0] || 'linkedin';

                          return (
                            <>
                              {/* PANEL 1: LINKEDIN POST */}
                              {currentTab === 'linkedin' && msg.artifacts?.linkedin && (
                                <div className="flex flex-col gap-2.5">
                                  <div className="bg-white border border-slate-200 rounded-lg p-3.5 shadow-sm flex flex-col gap-3">
                                    {/* LinkedIn Profile Header */}
                                    <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                                      <div className="flex items-center gap-2.5">
                                        <div className="w-8 h-8 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-xs shadow-xs ring-2 ring-slate-100">
                                          EV
                                        </div>
                                        <div className="flex flex-col">
                                          <div className="flex items-center gap-1">
                                            <span className="text-xs font-bold text-slate-900">Elena Vance</span>
                                            <span className="text-[10px] text-slate-400 font-normal">• 1st</span>
                                          </div>
                                          <span className="text-[10px] text-slate-500 font-medium">VP Product Security & Engineering</span>
                                          <span className="text-[9px] text-slate-400">1h • 🌐 Edited</span>
                                        </div>
                                      </div>
                                      <button className="text-indigo-600 hover:bg-indigo-50 px-2 py-1 rounded-full text-[10px] font-bold transition-colors">
                                        + Follow
                                      </button>
                                    </div>

                                    {/* LinkedIn Post Content */}
                                    <div className="space-y-2 text-xs text-slate-800 leading-relaxed">
                                      <p className="font-bold text-slate-900 text-sm leading-snug">
                                        {msg.artifacts.linkedin.content.headline}
                                      </p>
                                      {msg.artifacts.linkedin.content.body.map((paragraph, i) => (
                                        <p key={i} className="text-slate-700">{paragraph}</p>
                                      ))}
                                      {msg.artifacts.linkedin.content.cta && (
                                        <p className="font-semibold text-slate-900 bg-slate-50 p-2 rounded border-l-2 border-indigo-600 text-[11px]">
                                          {msg.artifacts.linkedin.content.cta}
                                        </p>
                                      )}
                                      <p className="text-indigo-600 text-[11px] font-bold pt-1 tracking-tight">
                                        {msg.artifacts.linkedin.content.hashtags}
                                      </p>
                                    </div>

                                    {/* Social Stats Header */}
                                    <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-[10px] text-slate-500">
                                      <div className="flex items-center gap-1">
                                        <span className="flex -space-x-1">
                                          <span className="w-4 h-4 rounded-full bg-blue-500 text-white flex items-center justify-center text-[8px] font-bold">👍</span>
                                          <span className="w-4 h-4 rounded-full bg-red-500 text-white flex items-center justify-center text-[8px] font-bold">❤️</span>
                                        </span>
                                        <span className="font-semibold text-slate-700">1,842 reactions</span>
                                      </div>
                                      <div className="flex items-center gap-2 text-slate-400">
                                        <span>94 comments</span>
                                        <span>•</span>
                                        <span>31 reposts</span>
                                      </div>
                                    </div>

                                    {/* Authentic LinkedIn Social Actions Bar */}
                                    <div className="grid grid-cols-4 gap-1 pt-1 border-t border-slate-100">
                                      <button onClick={() => showToast('Liked LinkedIn post')} className="flex items-center justify-center gap-1 py-1.5 rounded hover:bg-slate-100 text-slate-600 hover:text-blue-600 font-medium text-[10px] transition-colors">
                                        <span className="material-symbols-outlined text-[15px]">thumb_up</span>
                                        <span>Like</span>
                                      </button>
                                      <button onClick={() => showToast('Opening comments...')} className="flex items-center justify-center gap-1 py-1.5 rounded hover:bg-slate-100 text-slate-600 hover:text-indigo-600 font-medium text-[10px] transition-colors">
                                        <span className="material-symbols-outlined text-[15px]">comment</span>
                                        <span>Comment</span>
                                      </button>
                                      <button onClick={() => showToast('Reposted to feed')} className="flex items-center justify-center gap-1 py-1.5 rounded hover:bg-slate-100 text-slate-600 hover:text-emerald-600 font-medium text-[10px] transition-colors">
                                        <span className="material-symbols-outlined text-[15px]">repeat</span>
                                        <span>Repost</span>
                                      </button>
                                      <button onClick={() => showToast('Post link sent')} className="flex items-center justify-center gap-1 py-1.5 rounded hover:bg-slate-100 text-slate-600 hover:text-purple-600 font-medium text-[10px] transition-colors">
                                        <span className="material-symbols-outlined text-[15px]">send</span>
                                        <span>Send</span>
                                      </button>
                                    </div>
                                  </div>

                                  {/* LinkedIn Publish Action Bar */}
                                  <div className="flex items-center justify-between pt-0.5">
                                    <button
                                      onClick={() => showToast('LinkedIn post text copied!')}
                                      className="flex items-center gap-1 px-2.5 py-1.5 rounded bg-white border border-slate-200 hover:bg-slate-100 text-slate-700 text-[10px] font-medium shadow-xs transition-colors"
                                    >
                                      <span className="material-symbols-outlined text-[13px] text-slate-500">content_copy</span>
                                      <span>Copy Post Text</span>
                                    </button>
                                    <button
                                      onClick={() => showToast('Published directly to LinkedIn Feed!')}
                                      className="flex items-center gap-1 px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-800 text-white text-[10px] font-semibold shadow-xs transition-all"
                                    >
                                      <span className="material-symbols-outlined text-[14px]">rocket_launch</span>
                                      <span>Publish to LinkedIn</span>
                                    </button>
                                  </div>
                                </div>
                              )}

                              {/* PANEL 2: EXECUTIVE SUMMARY */}
                              {currentTab === 'exec' && msg.artifacts?.exec && (
                                <div className="flex flex-col gap-2.5">
                                  <div className="bg-indigo-50/70 border border-indigo-100 p-3 rounded-md flex flex-col gap-1">
                                    <div className="flex items-center gap-1 text-indigo-700">
                                      <span className="material-symbols-outlined text-[15px]">bolt</span>
                                      <span className="text-[10px] font-bold uppercase tracking-wider">Bottom Line Up Front (BLUF)</span>
                                    </div>
                                    <p className="text-xs text-slate-900 font-semibold leading-relaxed">
                                      {msg.artifacts.exec.bluf}
                                    </p>
                                  </div>

                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                    <div className="bg-white border border-slate-200 p-3 rounded-md flex flex-col gap-1 shadow-xs">
                                      <span className="text-xs font-bold text-slate-900 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-[14px] text-indigo-600">insights</span>
                                        <span>Strategic Implications</span>
                                      </span>
                                      <ul className="text-[10px] text-slate-700 space-y-1 list-disc pl-3.5 pt-0.5">
                                        {msg.artifacts.exec.implications.map((imp, i) => (
                                          <li key={i}>{imp}</li>
                                        ))}
                                      </ul>
                                    </div>

                                    <div className="bg-white border border-slate-200 p-3 rounded-md flex flex-col gap-1 shadow-xs">
                                      <span className="text-xs font-bold text-slate-900 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-[14px] text-emerald-600">payments</span>
                                        <span>Budget & Resource Impact</span>
                                      </span>
                                      <ul className="text-[10px] text-slate-700 space-y-1 list-disc pl-3.5 pt-0.5">
                                        {msg.artifacts.exec.budgetImpact.map((b, i) => (
                                          <li key={i}>{b}</li>
                                        ))}
                                      </ul>
                                    </div>
                                  </div>

                                  <div className="flex items-center justify-between pt-0.5">
                                    <button
                                      onClick={() => showToast('Executive briefing copied')}
                                      className="flex items-center gap-1 px-2.5 py-1 rounded bg-white border border-slate-200 hover:bg-slate-100 text-slate-700 text-[10px] font-medium shadow-xs transition-colors"
                                    >
                                      <span className="material-symbols-outlined text-[13px] text-slate-500">content_copy</span>
                                      <span>Copy Executive Briefing</span>
                                    </button>
                                  </div>
                                </div>
                              )}

                              {/* PANEL 3: TECHNICAL ADVISORY */}
                              {currentTab === 'advisory' && msg.artifacts?.advisory && (
                                <div className="flex flex-col gap-2.5">
                                  <div className="flex items-center justify-between bg-red-50 border border-red-200 p-3 rounded-md">
                                    <div className="flex items-center gap-2.5">
                                      <div className="w-8 h-8 rounded bg-red-600 text-white flex items-center justify-center shrink-0 font-bold text-xs shadow-xs">
                                        {msg.artifacts.advisory.cvss}
                                      </div>
                                      <div className="flex flex-col">
                                        <div className="flex items-center gap-1.5">
                                          <span className="text-[11px] font-bold text-red-950 uppercase tracking-wider">CVSS Severity</span>
                                          <span className="px-1.5 py-0.2 rounded bg-white text-red-600 border border-red-200 text-[9px] font-bold">
                                            {msg.artifacts.advisory.cve}
                                          </span>
                                        </div>
                                        <span className="text-xs font-semibold text-red-900">{msg.artifacts.advisory.vulnName}</span>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Affected Infrastructure */}
                                  <div className="bg-white border border-slate-200 p-2.5 rounded-md flex flex-col gap-1 shadow-xs">
                                    <span className="text-[10px] font-bold uppercase text-slate-500">Target Infrastructure Scope</span>
                                    <div className="flex flex-wrap gap-1 pt-0.5">
                                      {msg.artifacts.advisory.assets.map((asset, idx) => (
                                        <span key={idx} className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 text-[9px] font-medium border border-slate-200">
                                          {asset}
                                        </span>
                                      ))}
                                    </div>
                                  </div>

                                  {/* Runbook Checklist */}
                                  <div className="bg-white border border-slate-200 p-3 rounded-md flex flex-col gap-1.5 shadow-xs">
                                    <span className="text-xs font-bold text-slate-900">Actionable Execution Runbook</span>
                                    <div className="flex flex-col gap-1 pt-0.5">
                                      {msg.artifacts.advisory.runbook.map((item, idx) => (
                                        <label
                                          key={idx}
                                          className="flex items-start gap-2 p-1.5 rounded bg-slate-50 hover:bg-slate-100/80 cursor-pointer border border-slate-200/70 transition-colors"
                                        >
                                          <input
                                            type="checkbox"
                                            checked={item.done}
                                            onChange={() => toggleRunbookItem(msg.id, idx)}
                                            className="mt-0.5 w-3.5 h-3.5 rounded text-indigo-600 border-slate-300 focus:ring-indigo-500 cursor-pointer"
                                          />
                                          <div className="flex flex-col">
                                            <span
                                              className={`text-[10px] font-medium ${
                                                item.done ? 'text-slate-400 line-through' : 'text-slate-900'
                                              }`}
                                            >
                                              {item.text}
                                            </span>
                                            <span
                                              className={`text-[9px] ${
                                                item.done ? 'text-emerald-600 font-medium' : 'text-red-600 font-semibold'
                                              }`}
                                            >
                                              {item.note}
                                            </span>
                                          </div>
                                        </label>
                                      ))}
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* PANEL 4: VIDEO SCRIPT */}
                              {currentTab === 'video' && msg.artifacts?.video && (
                                <div className="flex flex-col gap-2.5">
                                  <div className="bg-slate-900 text-white p-3 rounded-md flex flex-col gap-1 shadow-xs">
                                    <span className="text-[9px] uppercase tracking-wider font-bold text-indigo-400">Video Concept</span>
                                    <h3 className="text-xs font-bold text-white">{msg.artifacts.video.content.title}</h3>
                                  </div>
                                  <div className="flex flex-col gap-2">
                                    {msg.artifacts.video.content.scenes.map((sc, idx) => (
                                      <div key={idx} className="bg-white border border-slate-200 p-2.5 rounded-md flex flex-col gap-1 shadow-xs">
                                        <div className="flex items-center justify-between border-b border-slate-100 pb-1">
                                          <span className="text-[10px] font-bold text-indigo-600">Scene {idx + 1}</span>
                                          <span className="px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 text-[9px] font-semibold">{sc.time}</span>
                                        </div>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[10px] pt-1">
                                          <div className="bg-slate-50 p-2 rounded border border-slate-200/70">
                                            <span className="font-bold text-slate-800 block text-[9px] uppercase">📹 Visual Direction</span>
                                            <p className="text-slate-600 text-[10px] pt-0.5">{sc.visual}</p>
                                          </div>
                                          <div className="bg-indigo-50/50 p-2 rounded border border-indigo-100">
                                            <span className="font-bold text-indigo-900 block text-[9px] uppercase">🎙️ Voiceover Audio</span>
                                            <p className="text-slate-800 text-[10px] pt-0.5 italic">{sc.audio}</p>
                                          </div>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* PANEL 5: X / TWITTER THREAD */}
                              {currentTab === 'twitter' && msg.artifacts?.twitter && (
                                <div className="flex flex-col gap-2.5">
                                  <div className="flex flex-col gap-2">
                                    {msg.artifacts.twitter.content.tweets.map((tweet, idx) => (
                                      <div key={idx} className="bg-white border border-slate-200 p-3 rounded-lg shadow-xs flex flex-col gap-1.5">
                                        <div className="flex items-center gap-2">
                                          <div className="w-6 h-6 rounded-full bg-slate-900 text-white flex items-center justify-center font-bold text-[9px]">
                                            EV
                                          </div>
                                          <div className="flex items-center gap-1">
                                            <span className="text-xs font-bold text-slate-900">Elena Vance</span>
                                            <span className="text-[10px] text-slate-400">@ElenaVanceSec • 1h</span>
                                          </div>
                                        </div>
                                        <p className="text-xs text-slate-800 leading-relaxed">{tweet}</p>
                                        <div className="flex items-center gap-4 text-slate-400 text-[10px] pt-1 border-t border-slate-100">
                                          <span className="flex items-center gap-1 hover:text-blue-500 cursor-pointer">💬 14</span>
                                          <span className="flex items-center gap-1 hover:text-emerald-500 cursor-pointer">🔁 42</span>
                                          <span className="flex items-center gap-1 hover:text-red-500 cursor-pointer">❤️ 380</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* PANEL 6: SLIDE DECK */}
                              {currentTab === 'slides' && msg.artifacts?.slides && (
                                <div className="flex flex-col gap-2.5">
                                  <div className="flex flex-col gap-2">
                                    {msg.artifacts.slides.content.slides.map((sl, idx) => (
                                      <div key={idx} className="bg-white border border-slate-200 p-3 rounded-lg shadow-xs flex flex-col gap-2">
                                        <div className="flex items-center justify-between border-b border-slate-100 pb-1">
                                          <span className="text-[10px] font-bold text-slate-500 uppercase">Slide {sl.slideNum || idx + 1}</span>
                                          <span className="text-xs font-bold text-indigo-600">{sl.title}</span>
                                        </div>
                                        <ul className="text-xs text-slate-700 space-y-1 list-disc pl-4">
                                          {sl.bullets.map((b, bIdx) => (
                                            <li key={bIdx}>{b}</li>
                                          ))}
                                        </ul>
                                        {sl.speakerNotes && (
                                          <div className="bg-amber-50 border border-amber-200 p-2 rounded text-[10px] text-amber-900">
                                            <span className="font-bold block text-[9px] uppercase text-amber-800">🗣️ Speaker Notes</span>
                                            <p className="pt-0.5">{sl.speakerNotes}</p>
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    </div>
                  </div>
                )
              )
            )}

            {isGenerating && (
              <div className="w-full bg-white border border-slate-200 rounded-lg p-3 flex items-center gap-2 shadow-xs animate-pulse">
                <div className="w-4 h-4 rounded bg-indigo-600 text-white flex items-center justify-center font-bold text-[9px]">
                  O
                </div>
                <span className="text-[11px] font-medium text-slate-600">Generating AI transformations...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </main>

        {/* FLOATING INGESTION DOCK (Positioned cleanly at bottom without obscuring messages) */}
        <div className="fixed bottom-0 left-0 right-0 z-30 p-2.5 md:p-3 pointer-events-none flex justify-center bg-gradient-to-t from-white via-white/90 to-transparent pt-4">
          <div
            className={`w-full max-w-[760px] pointer-events-auto flex flex-col gap-1 transition-all ${
              sidebarOpen ? 'md:ml-[220px]' : ''
            }`}
          >
            {/* Pill Bar */}
            <div className="w-full flex items-center justify-between px-2.5 py-1 rounded-t-xl bg-white border-t border-x border-slate-200 shadow-xs overflow-x-auto">
              <div className="flex items-center gap-1 shrink-0">
                <span className="text-[10px] text-slate-500 font-bold pr-1">Target Formats:</span>
                {Object.keys(activePills).map((pill) => {
                  const active = activePills[pill];
                  return (
                    <button
                      key={pill}
                      onClick={() => togglePill(pill)}
                      className={`px-2 py-0.2 rounded-full text-[9px] transition-all flex items-center gap-0.5 ${
                        active
                          ? 'bg-indigo-50 border border-indigo-200 text-indigo-700 font-bold'
                          : 'bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-600'
                      }`}
                    >
                      <span className="material-symbols-outlined text-[11px]">
                        {active ? 'check' : 'add'}
                      </span>
                      <span>{pill}</span>
                    </button>
                  );
                })}
              </div>

              {/* Context Dropdowns */}
              <div className="hidden lg:flex items-center gap-1.5 text-slate-500 text-[9px] shrink-0 pl-1">
                <select
                  value={audience}
                  onChange={(e) => setAudience(e.target.value)}
                  className="bg-transparent text-slate-700 font-semibold text-[9px] focus:outline-none cursor-pointer"
                >
                  <option value="Technical">Audience: Technical</option>
                  <option value="Executive">Audience: Executive</option>
                  <option value="General">Audience: General</option>
                </select>
                <span className="text-slate-300">|</span>
                <select
                  value={tone}
                  onChange={(e) => setTone(e.target.value)}
                  className="bg-transparent text-slate-700 font-semibold text-[9px] focus:outline-none cursor-pointer"
                >
                  <option value="Urgent">Tone: Urgent</option>
                  <option value="Professional">Tone: Professional</option>
                  <option value="Casual">Tone: Casual</option>
                </select>
              </div>
            </div>

            {/* Prompt Input Box */}
            <div className="w-full bg-white border border-slate-200 rounded-b-xl p-2.5 shadow-md focus-within:ring-1 focus-within:ring-indigo-500/20 focus-within:border-indigo-300 flex flex-col gap-1.5 transition-all">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder="Paste raw text or enter transformation instructions..."
                rows={2}
                className="w-full bg-transparent text-slate-800 placeholder:text-slate-400 text-xs resize-none border-0 focus:outline-none focus:ring-0 leading-relaxed max-h-32"
              />

              {/* Bottom Controls */}
              <div className="flex items-center justify-between pt-1 border-t border-slate-100">
                <div className="flex items-center gap-1 min-w-0">
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-6 h-6 rounded bg-slate-100 hover:bg-slate-200/80 text-slate-600 hover:text-slate-900 flex items-center justify-center transition-colors shrink-0"
                    title="Attach Document"
                  >
                    <span className="material-symbols-outlined text-[15px]">attach_file</span>
                  </button>

                  {stagedFile && (
                    <div className="flex items-center gap-1 px-1.5 py-0.2 rounded bg-slate-100 border border-slate-200 text-slate-700 text-[9px] shrink-0">
                      <span className="material-symbols-outlined text-indigo-600 text-[13px]">picture_as_pdf</span>
                      <span className="truncate max-w-[110px] font-medium text-slate-800">{stagedFile.name}</span>
                      <button
                        onClick={() => setStagedFile(null)}
                        className="text-slate-400 hover:text-slate-700 ml-0.5 p-0.5 rounded transition-colors"
                        title="Remove"
                      >
                        <span className="material-symbols-outlined text-[11px]">close</span>
                      </button>
                    </div>
                  )}

                  <button
                    onClick={() => showToast('Audio dictation ready')}
                    className="w-6 h-6 rounded bg-slate-100 hover:bg-slate-200/80 text-slate-600 hover:text-slate-900 flex items-center justify-center transition-colors shrink-0"
                    title="Audio Context"
                  >
                    <span className="material-symbols-outlined text-[15px]">mic</span>
                  </button>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[9px] text-slate-400 hidden sm:inline-block">1,420 / 128k</span>
                  <button
                    onClick={handleSend}
                    disabled={isGenerating}
                    className="w-7 h-7 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white flex items-center justify-center shadow-xs transition-all group disabled:opacity-50"
                    title="Send Prompt"
                  >
                    <span className="material-symbols-outlined text-[15px] transition-transform group-hover:scale-110">
                      arrow_upward
                    </span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

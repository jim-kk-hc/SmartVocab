import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type, Modality } from "@google/genai";

// --- Constants & Types ---
const STORAGE_KEY = 'smart_vocab_data';
const STATS_KEY = 'smart_vocab_stats';
const MODEL_TEXT = 'gemini-3-flash-preview';
const MODEL_TTS = 'gemini-2.5-flash-preview-tts';

interface WordCard {
  id: string;
  word: string;
  phonetic: string;
  translation: string;
  examples: string[];
  etymology?: string; 
  derivatives?: { word: string; pos: string; trans: string }[];
  collocations?: { phrase: string; trans: string }[];
  level: number;
  nextReview: number; // Timestamp
  createdAt: number;
  isEnriching?: boolean; 
  enrichStep?: number;
}

interface ActivityLog {
  date: string; // YYYY-MM-DD
  count: number;
}

interface StatsData {
  lastStudyDate: string;
  streak: number;
  totalReviewed: number;
  activityHistory: ActivityLog[];
}

const decodeBase64 = (base64: string) => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};

const decodeAudioData = async (data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> => {
  const dataInt16 = new Int16Array(data.buffer);
  const buffer = ctx.createBuffer(1, dataInt16.length, 24000);
  const channelData = buffer.getChannelData(0);
  for (let i = 0; i < dataInt16.length; i++) {
    channelData[i] = dataInt16[i] / 32768.0;
  }
  return buffer;
};

const App = () => {
  const [vocab, setVocab] = useState<WordCard[]>([]);
  const [stats, setStats] = useState<StatsData>({ 
    lastStudyDate: '', 
    streak: 0, 
    totalReviewed: 0,
    activityHistory: [] 
  });
  const [input, setInput] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [loadingStep, setLoadingStep] = useState<0 | 1 | 2>(0); 
  const [activeTab, setActiveTab] = useState<'study' | 'add' | 'list' | 'stats'>('study');
  const [detailTab, setDetailTab] = useState<'collocations' | 'derivatives' | 'etymology'>('collocations');
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [extraReviewPool, setExtraReviewPool] = useState<WordCard[]>([]);
  const [isExtraReview, setIsExtraReview] = useState(false);
  const [selectedWordId, setSelectedWordId] = useState<string | null>(null);

  const aiRef = useRef(new GoogleGenAI({ apiKey: process.env.API_KEY }));
  const audioCtxRef = useRef<AudioContext | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Computed Stats ---
  const masterCount = useMemo(() => vocab.filter(w => w.level >= 5).length, [vocab]);
  
  const distribution = useMemo(() => {
    const counts = [0, 0, 0, 0, 0, 0, 0];
    vocab.forEach(w => counts[w.level]++);
    const max = Math.max(...counts, 1);
    return counts.map((c, i) => ({ 
      level: i, 
      count: c, 
      ratio: (c / max) * 100,
      label: i === 0 ? '初识' : i === 6 ? '牢记' : `L${i}`
    }));
  }, [vocab]);

  const recentActivity = useMemo(() => {
    const days = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const dateStr = d.toISOString().split('T')[0];
      const log = (stats.activityHistory || []).find(l => l.date === dateStr);
      days.push({
        date: dateStr,
        dayName: d.toLocaleDateString('zh-CN', { weekday: 'narrow' }),
        count: log ? log.count : 0
      });
    }
    return days;
  }, [stats.activityHistory]);

  const scheduledReview = useMemo(() => 
    vocab
      .filter(w => w.nextReview <= Date.now())
      .sort((a, b) => a.nextReview - b.nextReview),
    [vocab]
  );

  const currentWord = useMemo(() => {
    if (selectedWordId) {
      return vocab.find(v => v.id === selectedWordId);
    }
    if (scheduledReview.length > 0) {
      return scheduledReview[0];
    }
    if (isExtraReview && extraReviewPool.length > 0) {
      return extraReviewPool[0];
    }
    return null;
  }, [scheduledReview, extraReviewPool, selectedWordId, vocab, isExtraReview]);

  const filteredVocab = useMemo(() => {
    return vocab
      .filter(w => w.word.toLowerCase().includes(searchQuery.toLowerCase()) || w.translation.includes(searchQuery))
      .sort((a, b) => b.createdAt - a.createdAt);
  }, [vocab, searchQuery]);

  // --- Effects ---
  useEffect(() => {
    const savedVocab = localStorage.getItem(STORAGE_KEY);
    const savedStats = localStorage.getItem(STATS_KEY);
    if (savedVocab) setVocab(JSON.parse(savedVocab));
    if (savedStats) setStats(JSON.parse(savedStats));
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(vocab));
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
  }, [vocab, stats]);

  // --- Actions ---
  const handleExport = () => {
    const data = {
      vocab,
      stats,
      exportedAt: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `smartvocab_backup_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string);
        if (data.vocab && Array.isArray(data.vocab)) {
          const importedVocab: WordCard[] = data.vocab;
          const currentVocabMap = new Map(vocab.map(w => [w.word.toLowerCase(), w]));
          
          let addedCount = 0;
          let mergedCount = 0;

          importedVocab.forEach(newWord => {
            const key = newWord.word.toLowerCase();
            const existing = currentVocabMap.get(key);
            if (existing) {
              // Merge strategy: Keep the one with higher level (more mastered)
              if (newWord.level >= existing.level) {
                currentVocabMap.set(key, { ...newWord, id: existing.id }); // Keep existing ID for continuity
              }
              mergedCount++;
            } else {
              currentVocabMap.set(key, newWord);
              addedCount++;
            }
          });

          const newVocabList = Array.from(currentVocabMap.values());
          setVocab(newVocabList);

          // Merge Stats
          if (data.stats) {
            setStats(prev => {
              const mergedHistory = [...(prev.activityHistory || [])];
              (data.stats.activityHistory || []).forEach((item: ActivityLog) => {
                const idx = mergedHistory.findIndex(h => h.date === item.date);
                if (idx > -1) {
                  mergedHistory[idx].count = Math.max(mergedHistory[idx].count, item.count);
                } else {
                  mergedHistory.push(item);
                }
              });
              
              return {
                ...prev,
                totalReviewed: Math.max(prev.totalReviewed || 0, data.stats.totalReviewed || 0),
                streak: Math.max(prev.streak, data.stats.streak || 0),
                activityHistory: mergedHistory.sort((a, b) => a.date.localeCompare(b.date))
              };
            });
          }

          alert(`增量导入完成！\n新增单词：${addedCount} 个\n合并重复词：${mergedCount} 个`);
        } else {
          alert('无效的备份文件格式');
        }
      } catch (err) {
        alert('文件解析失败');
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const updateStudyActivity = () => {
    const today = new Date().toISOString().split('T')[0];
    setStats(prev => {
      const history = [...(prev.activityHistory || [])];
      const todayIndex = history.findIndex(l => l.date === today);
      if (todayIndex > -1) {
        history[todayIndex].count += 1;
      } else {
        history.push({ date: today, count: 1 });
      }
      if (history.length > 60) history.shift();

      let newStreak = prev.streak;
      if (prev.lastStudyDate !== today) {
        const yesterday = new Date(); yesterday.setDate(yesterday.getDate() - 1);
        const yesterdayStr = yesterday.toISOString().split('T')[0];
        newStreak = prev.lastStudyDate === yesterdayStr ? prev.streak + 1 : 1;
      }

      return {
        ...prev,
        lastStudyDate: today,
        streak: newStreak,
        totalReviewed: (prev.totalReviewed || 0) + 1,
        activityHistory: history
      };
    });
  };

  const startExtraReview = () => {
    setIsExtraReview(true);
    setSelectedWordId(null);
    setExtraReviewPool([...vocab].sort(() => 0.5 - Math.random()).slice(0, 10));
  };

  const playTTS = async (text: string, id: string) => {
    if (playingId) return;
    setPlayingId(id);
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') await ctx.resume();
      const response = await aiRef.current.models.generateContent({
        model: MODEL_TTS,
        contents: [{ parts: [{ text }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
        },
      });
      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        const audioBuffer = await decodeAudioData(decodeBase64(base64Audio), ctx);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        source.onended = () => setPlayingId(null);
        source.start();
      } else setPlayingId(null);
    } catch { setPlayingId(null); }
  };

  const addNewWord = async () => {
    if (!input.trim() || loadingStep !== 0) return;
    const wordToSearch = input.trim();
    setLoadingStep(1); 
    try {
      const resStage1 = await aiRef.current.models.generateContent({
        model: MODEL_TEXT,
        contents: `Translate "${wordToSearch}" to Chinese and provide US IPA. JSON format: {phonetic, translation}`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              phonetic: { type: Type.STRING },
              translation: { type: Type.STRING }
            },
            required: ["translation"]
          }
        }
      });
      const s1Data = JSON.parse(resStage1.text);
      const tempId = Date.now().toString();
      const newEntry: WordCard = {
        id: tempId,
        word: wordToSearch,
        phonetic: s1Data.phonetic || '',
        translation: s1Data.translation || '',
        examples: [],
        level: 0,
        nextReview: Date.now() + 30000, 
        createdAt: Date.now(),
        isEnriching: true,
        enrichStep: 1
      };
      setVocab(prev => [newEntry, ...prev]);
      setInput('');
      setSelectedWordId(tempId);
      setActiveTab('study');
      setLoadingStep(0); 
      enrichWordData(tempId, wordToSearch);
    } catch (err) {
      alert("词库连接失败");
      setLoadingStep(0);
    }
  };

  const enrichWordData = async (id: string, word: string) => {
    setVocab(prev => prev.map(w => w.id === id ? { ...w, enrichStep: 2 } : w));
    try {
      const response = await aiRef.current.models.generateContent({
        model: MODEL_TEXT,
        contents: `Enrich English word "${word}". Return JSON: {
          examples: [2 strings],
          etymology: "short Chinese text",
          derivatives: [{word, pos, trans}],
          collocations: [{phrase, trans}]
        }`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              examples: { type: Type.ARRAY, items: { type: Type.STRING } },
              etymology: { type: Type.STRING },
              derivatives: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { word: { type: Type.STRING }, pos: { type: Type.STRING }, trans: { type: Type.STRING } } } },
              collocations: { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { phrase: { type: Type.STRING }, trans: { type: Type.STRING } } } }
            }
          }
        }
      });
      const data = JSON.parse(response.text);
      setVocab(prev => prev.map(w => w.id === id ? { ...w, ...data, isEnriching: false, enrichStep: 3 } : w));
    } catch (e) {
      setVocab(prev => prev.map(w => w.id === id ? { ...w, isEnriching: false } : w));
    }
  };

  const updateMemory = (id: string, remembered: boolean) => {
    updateStudyActivity();
    const intervals = [0, 1, 2, 4, 7, 15, 30]; 
    setVocab(prev => prev.map(w => {
      if (w.id === id) {
        const nextLevel = remembered ? Math.min(w.level + 1, 6) : 0;
        const nextTime = remembered 
          ? Date.now() + (intervals[nextLevel] * 24 * 3600 * 1000)
          : Date.now() + 300000; 
        return { ...w, level: nextLevel, nextReview: nextTime };
      }
      return w;
    }));
    
    if (selectedWordId === id) {
      setSelectedWordId(null);
      setActiveTab('study');
    } else if (isExtraReview) {
      setExtraReviewPool(prev => prev.filter(p => p.id !== id));
      if (extraReviewPool.length <= 1) setIsExtraReview(false);
    }
  };

  return (
    <div className="max-w-md mx-auto h-screen flex flex-col bg-[#f2f3f5] overflow-hidden font-sans text-gray-800">
      <header className="px-6 pt-10 pb-4 flex justify-between items-center z-10 sticky top-0 bg-[#f2f3f5]/80 backdrop-blur-md">
        <div className="flex items-center gap-2">
           <span className="text-xs font-bold text-gray-400">
             {activeTab === 'study' ? (currentWord ? `正在复习 (待完成 ${scheduledReview.length})` : '任务已完成') : 
              activeTab === 'list' ? `词库总览 (${vocab.length})` : 
              activeTab === 'stats' ? '学习洞察' : ''}
           </span>
        </div>
        {activeTab === 'study' && (
          <button onClick={() => { setActiveTab('study'); setSelectedWordId(null); setIsExtraReview(false); }} className="text-xs font-bold text-emerald-500 hover:text-emerald-600 transition-colors">
            {scheduledReview.length > 0 ? `待复习 ${scheduledReview.length}` : '回到首页'}
          </button>
        )}
      </header>

      <main className="flex-grow flex flex-col px-4 overflow-y-auto custom-scrollbar pb-32">
        {activeTab === 'study' && currentWord ? (
          <div className="flex flex-col h-full animate-in fade-in slide-in-from-bottom-2 duration-500" key={currentWord.id}>
            <div className="py-6 px-2">
              <h1 className="text-5xl font-bold mb-3 tracking-tight flex items-center gap-3">
                {currentWord.word}
                {currentWord.isEnriching && (
                  <div className="flex gap-1 items-center">
                    <div className="w-2 h-2 rounded-full bg-orange-400 animate-bounce delay-75"></div>
                    <div className="w-2 h-2 rounded-full bg-orange-300 animate-bounce delay-150"></div>
                  </div>
                )}
                {!currentWord.isEnriching && <div className="w-2 h-2 rounded-full bg-emerald-300"></div>}
              </h1>
              <div className="flex items-center gap-2 mb-4">
                <button onClick={() => playTTS(currentWord.word, currentWord.id)} className="bg-gray-200/50 px-2 py-0.5 rounded-md flex items-center gap-1.5 text-xs font-medium text-gray-500 active:scale-90 transition-transform">
                  <span className="text-[10px]">美</span>
                  {playingId === currentWord.id ? <i className="fas fa-circle-notch fa-spin"></i> : <i className="fas fa-volume-up"></i>}
                </button>
                <span className="text-gray-400 text-sm">{currentWord.phonetic}</span>
              </div>
              <p className="text-xl font-bold text-gray-700 leading-tight">{currentWord.translation}</p>
            </div>

            {currentWord.isEnriching && (
              <div className="px-2 mb-4">
                <div className="bg-orange-50/50 border border-orange-100 rounded-xl p-3 flex items-center justify-between">
                  <span className="text-[11px] font-bold text-orange-600 flex items-center gap-2">
                    <i className="fas fa-microchip animate-pulse"></i>
                    {currentWord.enrichStep === 2 ? "正在深度解构词源与搭配..." : "正在获取核心解释..."}
                  </span>
                  <div className="flex gap-0.5">
                    {[1, 2, 3].map(step => (
                      <div key={step} className={`w-3 h-1 rounded-full transition-all duration-500 ${currentWord.enrichStep && currentWord.enrichStep >= step ? 'bg-orange-400' : 'bg-gray-200'}`}></div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="bg-white/70 backdrop-blur-md rounded-2xl p-5 mb-5 shadow-sm border border-white/50 flex items-start gap-4">
               {currentWord.isEnriching && (!currentWord.examples || currentWord.examples.length === 0) ? (
                 <div className="flex-grow space-y-2 animate-pulse">
                   <div className="h-4 bg-gray-200 rounded w-full"></div>
                   <div className="h-4 bg-gray-100 rounded w-3/4"></div>
                 </div>
               ) : (
                 <>
                   <button 
                     onClick={() => playTTS(currentWord.examples[0] || '', 'ex-1')} 
                     className="flex-shrink-0 w-9 h-9 mt-0.5 rounded-full bg-emerald-50 text-emerald-500 flex items-center justify-center active:bg-emerald-100 active:scale-90 transition-all shadow-sm"
                   >
                     {playingId === 'ex-1' ? <i className="fas fa-circle-notch fa-spin text-xs"></i> : <i className="fas fa-volume-up text-xs"></i>}
                   </button>
                   <div className="flex-grow min-w-0">
                     <p className="text-[17px] leading-[1.6] font-medium text-gray-800 break-words tracking-tight italic">
                       "{currentWord.examples[0]}"
                     </p>
                   </div>
                 </>
               )}
            </div>

            <div className="bg-white/60 backdrop-blur-md rounded-2xl flex-grow overflow-hidden flex flex-col shadow-sm border border-white/50 mb-10 min-h-[300px]">
               <div className="flex border-b border-gray-100 px-4 pt-4 pb-2 gap-6">
                 {['collocations', 'derivatives', 'etymology'].map(tid => (
                   <button key={tid} onClick={() => setDetailTab(tid as any)} className={`text-sm font-bold transition-all relative pb-2 ${detailTab === tid ? 'text-gray-800' : 'text-gray-400'}`}>
                     {tid === 'collocations' ? '搭配' : tid === 'derivatives' ? '派生' : '词根'}
                     {detailTab === tid && <div className="absolute bottom-0 left-0 w-full h-0.5 bg-emerald-400 rounded-full"></div>}
                   </button>
                 ))}
               </div>
               
               <div className="p-5 overflow-y-auto flex-grow custom-scrollbar">
                 {currentWord.isEnriching && (!currentWord.collocations) ? (
                    <div className="space-y-4 animate-pulse">
                      <div className="h-6 bg-gray-50 rounded w-3/4"></div>
                      <div className="h-6 bg-gray-50 rounded w-1/2"></div>
                      <div className="h-6 bg-gray-50 rounded w-2/3"></div>
                      <p className="text-[10px] text-gray-300 text-center uppercase tracking-[0.2em] mt-8">Intelligence Enriching</p>
                    </div>
                 ) : (
                   <>
                    {detailTab === 'collocations' && (
                      <div className="space-y-4">
                        {currentWord.collocations?.map((c, i) => (
                          <div key={i} className="border-b border-dashed border-gray-200 pb-2 last:border-0">
                            <p className="text-lg hover:text-emerald-600 transition-colors inline-block">{c.phrase}</p>
                            <p className="text-gray-500 text-sm mt-1">{c.trans}</p>
                          </div>
                        ))}
                      </div>
                    )}
                    {detailTab === 'derivatives' && (
                      <div className="space-y-3">
                        {currentWord.derivatives?.map((d, i) => (
                          <div key={i} className="flex justify-between items-center p-3 rounded-xl hover:bg-white/50 transition-colors">
                            <div><span className="font-bold text-gray-700 mr-2">{d.word}</span><span className="text-[10px] text-gray-400 italic">{d.pos}.</span></div>
                            <span className="text-sm text-gray-500">{d.trans}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {detailTab === 'etymology' && (
                      <div className="bg-gray-50/50 p-4 rounded-xl border border-gray-100">
                        <p className="text-sm leading-relaxed text-gray-600 italic">{currentWord.etymology || "暂无深度词源信息"}</p>
                      </div>
                    )}
                   </>
                 )}
               </div>
            </div>

            <div className="sticky bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-[#f2f3f5] via-[#f2f3f5] to-transparent z-10 flex gap-4">
              <button 
                onClick={() => updateMemory(currentWord.id, false)} 
                className="flex-1 bg-white border border-rose-100 text-rose-500 py-4 rounded-3xl text-lg font-bold shadow-sm active:scale-[0.97] transition-all"
              >
                没记住
              </button>
              <button 
                onClick={() => updateMemory(currentWord.id, true)} 
                className="flex-[2] bg-emerald-500 hover:bg-emerald-600 py-4 rounded-3xl text-white text-lg font-bold shadow-xl shadow-emerald-200/50 active:scale-[0.97] transition-all flex items-center justify-center gap-3 group"
              >
                {selectedWordId ? (
                  <>
                    <i className="fas fa-arrow-left text-sm group-hover:-translate-x-1 transition-transform"></i>
                    <span>返回词库</span>
                  </>
                ) : (
                  <>
                    <span>记住了，下一词</span>
                    <i className="fas fa-chevron-right text-sm group-hover:translate-x-1 transition-transform"></i>
                  </>
                )}
              </button>
            </div>
          </div>
        ) : activeTab === 'list' ? (
          <div className="flex flex-col h-full animate-in fade-in duration-500">
             <div className="flex justify-between items-center mb-6">
                <div className="relative flex-grow">
                  <i className="fas fa-search absolute left-4 top-1/2 -translate-y-1/2 text-gray-300"></i>
                  <input value={searchQuery} onChange={e => setSearchQuery(e.target.value)} placeholder="搜索词库..." className="w-full bg-white border border-gray-100 rounded-2xl py-3 pl-12 pr-4 outline-none focus:ring-2 focus:ring-emerald-400 transition-all text-sm shadow-sm" />
                </div>
                <div className="flex gap-2 ml-3">
                  <button onClick={handleExport} className="w-10 h-10 bg-white rounded-xl border border-gray-100 flex items-center justify-center text-gray-400 hover:text-emerald-500 hover:border-emerald-100 shadow-sm transition-all" title="导出备份">
                    <i className="fas fa-download text-sm"></i>
                  </button>
                  <button onClick={() => fileInputRef.current?.click()} className="w-10 h-10 bg-white rounded-xl border border-gray-100 flex items-center justify-center text-gray-400 hover:text-indigo-500 hover:border-indigo-100 shadow-sm transition-all" title="增量导入备份">
                    <i className="fas fa-plus text-sm"></i>
                  </button>
                  <input type="file" ref={fileInputRef} onChange={handleImport} accept=".json" className="hidden" />
                </div>
             </div>
             <div className="flex-grow space-y-3 pb-10">
               {filteredVocab.map(item => (
                 <button 
                   key={item.id} 
                   className="w-full text-left bg-white/60 backdrop-blur-sm p-4 rounded-2xl border border-white/50 flex justify-between items-center group active:bg-white/80 active:scale-[0.98] transition-all shadow-sm" 
                   onClick={() => { 
                     setSelectedWordId(item.id); 
                     setActiveTab('study'); 
                   }}
                 >
                    <div className="flex-grow">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-bold text-gray-800">{item.word}</h4>
                        {item.isEnriching && <i className="fas fa-magic fa-spin text-[8px] text-orange-400 ml-1"></i>}
                        <div className="flex gap-0.5 ml-2">
                          {[...Array(6)].map((_, i) => <div key={i} className={`w-1 h-1 rounded-full ${i < item.level ? 'bg-emerald-400' : 'bg-gray-100'}`}></div>)}
                        </div>
                      </div>
                      <p className="text-xs text-gray-400 truncate max-w-[200px]">{item.translation}</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <i className="fas fa-chevron-right text-[10px] text-gray-300 group-hover:text-emerald-400 transition-colors"></i>
                      <button 
                        onClick={(e) => { 
                          e.stopPropagation(); 
                          if(confirm('删除此词？')) setVocab(v => v.filter(w => w.id !== item.id)); 
                        }} 
                        className="text-gray-200 hover:text-red-400 p-2"
                      >
                        <i className="fas fa-trash-alt text-xs"></i>
                      </button>
                    </div>
                 </button>
               ))}
               {filteredVocab.length === 0 && <div className="text-center py-20 text-gray-400 text-sm">词库空空如也</div>}
             </div>
          </div>
        ) : activeTab === 'stats' ? (
          <div className="space-y-6 animate-in fade-in duration-500 pb-10">
             <div className="grid grid-cols-2 gap-4">
               <div className="bg-gradient-to-br from-emerald-400 to-emerald-600 p-6 rounded-[2.5rem] shadow-lg shadow-emerald-200/50 flex flex-col text-white">
                 <div className="flex justify-between items-start mb-4">
                   <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                     <i className="fas fa-award"></i>
                   </div>
                 </div>
                 <span className="text-[10px] font-black uppercase tracking-widest opacity-70 mb-1">已掌握词汇</span>
                 <span className="text-4xl font-black">{masterCount}</span>
               </div>
               <div className="bg-gradient-to-br from-orange-400 to-rose-500 p-6 rounded-[2.5rem] shadow-lg shadow-orange-200/50 flex flex-col text-white">
                 <div className="flex justify-between items-start mb-4">
                   <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                     <i className="fas fa-fire"></i>
                   </div>
                 </div>
                 <span className="text-[10px] font-black uppercase tracking-widest opacity-70 mb-1">当前连胜</span>
                 <span className="text-4xl font-black">{stats.streak}</span>
               </div>
             </div>

             <div className="bg-white p-6 rounded-[2rem] shadow-sm border border-white/50">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest">最近 7 日活跃度</h3>
                  <span className="text-[10px] font-bold text-emerald-500">累计复习 {stats.totalReviewed || 0} 次</span>
                </div>
                <div className="flex justify-between items-center gap-1">
                  {recentActivity.map((day, i) => (
                    <div key={i} className="flex flex-col items-center gap-2 flex-1">
                      <div className="relative w-full aspect-square rounded-xl bg-gray-50 overflow-hidden group">
                        <div 
                          className="absolute bottom-0 left-0 w-full bg-emerald-400 transition-all duration-1000 group-hover:bg-emerald-500"
                          style={{ height: `${Math.min(day.count * 10, 100)}%`, opacity: day.count > 0 ? 0.3 + (day.count * 0.1) : 0 }}
                        ></div>
                        <div className="absolute inset-0 flex items-center justify-center">
                           <span className={`text-[10px] font-black ${day.count > 5 ? 'text-emerald-700' : 'text-gray-300'}`}>
                             {day.count > 0 ? day.count : ''}
                           </span>
                        </div>
                      </div>
                      <span className={`text-[10px] font-black uppercase ${day.date === new Date().toISOString().split('T')[0] ? 'text-emerald-500' : 'text-gray-300'}`}>
                        {day.dayName}
                      </span>
                    </div>
                  ))}
                </div>
             </div>

             <div className="bg-white p-6 rounded-[2rem] shadow-sm border border-white/50">
                <h3 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-8">遗忘曲线分布</h3>
                <div className="flex items-end justify-between h-40 gap-3">
                  {distribution.map((d, i) => (
                    <div key={i} className="flex flex-col items-center flex-1 h-full">
                      <div className="w-full bg-gray-50/50 rounded-2xl relative flex items-end h-32 group">
                        <div 
                          className={`w-full transition-all duration-700 rounded-2xl ${i < 2 ? 'bg-rose-300' : i < 5 ? 'bg-amber-300' : 'bg-emerald-400'}`} 
                          style={{ height: `${d.ratio}%` }}
                        >
                        </div>
                      </div>
                      <span className="text-[10px] font-bold text-gray-400 mt-3">{d.label}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-8 pt-6 border-t border-gray-50 flex justify-between items-center">
                   <div className="flex flex-col">
                      <span className="text-[10px] font-black text-gray-300 uppercase">词库总量</span>
                      <span className="text-xl font-black text-gray-700">{vocab.length}</span>
                   </div>
                   <div className="flex flex-col items-end">
                      <span className="text-[10px] font-black text-gray-300 uppercase">掌握率</span>
                      <span className="text-xl font-black text-emerald-500">{vocab.length ? Math.round((masterCount / vocab.length) * 100) : 0}%</span>
                   </div>
                </div>
             </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center min-h-[70vh] text-center px-6">
             <div className="w-32 h-32 bg-white rounded-full flex items-center justify-center mb-8 shadow-inner relative">
               <i className="fas fa-seedling text-4xl text-emerald-300"></i>
               {scheduledReview.length > 0 && <div className="absolute top-0 right-0 bg-orange-500 text-white text-[10px] font-black px-2 py-1 rounded-full shadow-lg">{scheduledReview.length}</div>}
             </div>
             <h2 className="text-2xl font-black mb-2">准备好学习了吗？</h2>
             <p className="text-gray-400 text-sm mb-12">今日共有 {scheduledReview.length} 个单词待复习</p>
             <div className="w-full space-y-4">
               <button onClick={() => setActiveTab('add')} className="w-full bg-gray-900 text-white py-5 rounded-3xl font-bold shadow-xl active:scale-95 transition-all flex items-center justify-center gap-3">录入新单词</button>
               {vocab.length > 0 && <button onClick={() => { if(scheduledReview.length === 0) startExtraReview(); setActiveTab('study'); }} className="w-full bg-white text-gray-700 py-5 rounded-3xl font-bold border border-gray-100 shadow-sm active:scale-95 transition-all">
                 {scheduledReview.length > 0 ? `开始复习 (${scheduledReview.length})` : '自由温故'}
               </button>}
             </div>
          </div>
        )}
      </main>

      <nav className="fixed bottom-0 left-0 right-0 max-w-md mx-auto flex justify-around p-4 pb-8 bg-white/60 backdrop-blur-2xl border-t border-white/40 rounded-t-[2.5rem] z-50">
         <button onClick={() => { setActiveTab('study'); setSelectedWordId(null); setIsExtraReview(false); }} className={`p-4 rounded-2xl transition-all ${activeTab === 'study' ? 'bg-white shadow-md text-emerald-500 scale-110' : 'text-gray-400'}`}><i className="fas fa-brain text-xl"></i></button>
         <button onClick={() => { setActiveTab('stats'); setSelectedWordId(null); }} className={`p-4 rounded-2xl transition-all ${activeTab === 'stats' ? 'bg-white shadow-md text-emerald-500 scale-110' : 'text-gray-400'}`}><i className="fas fa-chart-line text-xl"></i></button>
         <button onClick={() => { setActiveTab('list'); setSelectedWordId(null); }} className={`p-4 rounded-2xl transition-all ${activeTab === 'list' ? 'bg-white shadow-md text-emerald-500 scale-110' : 'text-gray-400'}`}><i className="fas fa-list text-xl"></i></button>
         <button onClick={() => setActiveTab('add')} className={`p-4 rounded-2xl text-gray-400 hover:text-indigo-600 transition-colors`}><i className="fas fa-plus-circle text-xl"></i></button>
      </nav>

      <div className={`fixed inset-0 z-[60] bg-black/20 backdrop-blur-sm transition-all duration-300 ${activeTab === 'add' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        <div className={`absolute bottom-0 left-0 right-0 max-w-md mx-auto bg-white rounded-t-[3rem] p-8 pb-12 transition-transform duration-500 shadow-2xl ${activeTab === 'add' ? 'translate-y-0' : 'translate-y-full'}`}>
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-black">录入单词</h2>
            <button onClick={() => setActiveTab('study')} className="text-gray-300 hover:text-gray-600 p-2"><i className="fas fa-times text-lg"></i></button>
          </div>
          <div className="relative mb-6 group">
            <input 
              value={input} 
              onChange={e => setInput(e.target.value)}
              className="w-full p-5 bg-gray-50 border-none rounded-[1.5rem] outline-none focus:ring-2 focus:ring-emerald-400 transition-all text-xl" 
              placeholder="输入单词..." 
              autoFocus
              onKeyDown={e => e.key === 'Enter' && addNewWord()}
            />
            {loadingStep !== 0 && (
              <div className="absolute right-4 top-1/2 -translate-y-1/2 flex gap-1">
                <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-bounce"></div>
                <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-bounce delay-75"></div>
                <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-bounce delay-150"></div>
              </div>
            )}
          </div>
          <div className="flex flex-col gap-4">
            <div className={`flex items-center gap-3 px-2 transition-opacity duration-300 ${loadingStep !== 0 ? 'opacity-100' : 'opacity-0'}`}>
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${loadingStep >= 1 ? 'bg-emerald-500 text-white' : 'bg-gray-100 text-gray-400'}`}>
                {loadingStep > 1 ? <i className="fas fa-check"></i> : "1"}
              </div>
              <span className={`text-xs font-bold ${loadingStep === 1 ? 'text-gray-800' : 'text-gray-400'}`}>
                {loadingStep === 1 ? "正在检索基础含义..." : "已完成基础检索"}
              </span>
            </div>
            <button 
              onClick={addNewWord} 
              disabled={loadingStep !== 0 || !input}
              className="w-full bg-indigo-600 text-white py-5 rounded-[1.5rem] font-bold disabled:bg-gray-100 disabled:text-gray-400 active:scale-95 transition-all shadow-lg shadow-indigo-100 mt-2"
            >
              {loadingStep === 0 ? '立即分析并添加' : 'AI 处理中...'}
            </button>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
        .animate-in { animation: fadeIn 0.4s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        body { position: fixed; width: 100%; height: 100%; overflow: hidden; }
        #root { height: 100%; overflow: hidden; }
      `}</style>
    </div>
  );
};

createRoot(document.getElementById('root')!).render(<App />);
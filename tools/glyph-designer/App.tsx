
import React, { useState, useEffect, useCallback } from 'react';
import { 
  Type, 
  Circle, 
  Minus, 
  CornerUpLeft, 
  Trash2, 
  Code as CodeIcon, 
  Download,
  Info,
  LayoutGrid,
  Settings2,
  Undo,
  Redo
} from 'lucide-react';
import GridCanvas from './components/GridCanvas';
import { GridState, ToolType } from './types';
import { formatPythonCode } from './utils';
import { CHARACTER_PRESETS, applyPreset } from './presets';

const App: React.FC = () => {
  const [gridSize, setGridSize] = useState({ rows: 4, cols: 4 });
  const [state, setState] = useState<GridState>({
    circles: new Set<string>(),
    vSegments: new Set<string>(),
    hSegments: new Set<string>(),
    arcs: new Set<string>(),
    dots: new Set<string>()
  });
  
  // Undo/Redo History
  const [past, setPast] = useState<GridState[]>([]);
  const [future, setFuture] = useState<GridState[]>([]);

  const [activeTool, setActiveTool] = useState<ToolType>(ToolType.Circle);
  const [pythonCode, setPythonCode] = useState<string>("");

  useEffect(() => {
    setPythonCode(formatPythonCode(state));
  }, [state]);

  // Handle state updates with history tracking
  const updateState = useCallback((newState: GridState) => {
    setPast(prev => [...prev, state]);
    setFuture([]);
    setState(newState);
  }, [state]);

  const undo = () => {
    if (past.length === 0) return;
    const previous = past[past.length - 1];
    const newPast = past.slice(0, past.length - 1);
    
    setFuture(prev => [state, ...prev]);
    setPast(newPast);
    setState(previous);
  };

  const redo = () => {
    if (future.length === 0) return;
    const next = future[0];
    const newFuture = future.slice(1);

    setPast(prev => [...prev, state]);
    setFuture(newFuture);
    setState(next);
  };

  const clearCanvas = () => {
    if (confirm("Clear all elements?")) {
      updateState({
        circles: new Set(),
        vSegments: new Set(),
        hSegments: new Set(),
        arcs: new Set(),
        dots: new Set()
      });
    }
  };

  const loadCharacter = (presetName: string) => {
    const preset = CHARACTER_PRESETS.find(p => p.name === presetName);
    if (preset) {
      updateState(applyPreset(preset));
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(pythonCode);
    alert("Python code copied to clipboard!");
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 font-sans">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-black p-2 rounded-lg">
            <Type className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">GlyphForge <span className="text-slate-400 font-light">Designer</span></h1>
            <p className="text-xs text-slate-500 font-medium">Interactive Typeface Prototyper</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center border-r border-slate-200 pr-4 gap-1">
            <button 
              onClick={undo}
              disabled={past.length === 0}
              title="Undo"
              className="p-2 text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent rounded-lg transition-all"
            >
              <Undo className="w-5 h-5" />
            </button>
            <button 
              onClick={redo}
              disabled={future.length === 0}
              title="Redo"
              className="p-2 text-slate-600 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent rounded-lg transition-all"
            >
              <Redo className="w-5 h-5" />
            </button>
          </div>

          <button 
            onClick={clearCanvas}
            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-red-500 hover:bg-red-50 rounded-lg transition-all"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
          <button 
            onClick={copyToClipboard}
            className="flex items-center gap-2 px-5 py-2 text-sm font-bold bg-slate-900 text-white hover:bg-slate-800 active:scale-95 rounded-lg transition-all shadow-md"
          >
            <Download className="w-4 h-4" />
            Copy Code
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Sidebar / Tools */}
        <aside className="w-full md:w-80 bg-white border-r border-slate-200 p-6 flex flex-col gap-8 order-2 md:order-1 overflow-y-auto">
          
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em]">Grid Config</h2>
              <Settings2 className="w-3.5 h-3.5 text-slate-300" />
            </div>
            <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-xl border border-slate-100">
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1">Rows</label>
                <input 
                  type="number" 
                  min="2" 
                  max="10" 
                  value={gridSize.rows} 
                  onChange={(e) => setGridSize(prev => ({ ...prev, rows: parseInt(e.target.value) || 2 }))}
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 text-sm font-bold focus:ring-2 focus:ring-blue-500 outline-none"
                />
              </div>
              <div>
                <label className="block text-[10px] font-bold text-slate-500 uppercase mb-1">Cols</label>
                <input 
                  type="number" 
                  min="2" 
                  max="10" 
                  value={gridSize.cols} 
                  onChange={(e) => setGridSize(prev => ({ ...prev, cols: parseInt(e.target.value) || 2 }))}
                  className="w-full bg-white border border-slate-200 rounded-lg px-3 py-1.5 text-sm font-bold focus:ring-2 focus:ring-blue-500 outline-none"
                />
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-4">Toolbar</h2>
            <div className="grid grid-cols-1 gap-2">
              {[
                { id: ToolType.Circle, label: 'Circle', icon: Circle, desc: 'Full circle elements' },
                { id: ToolType.Arc, label: 'Quarter Arc', icon: CornerUpLeft, desc: '90° curve quadrants' },
                { id: ToolType.Segment, label: 'Tangent Line', icon: Minus, desc: 'Straight connections' },
                { id: ToolType.Dot, label: 'Anchor Dot', icon: Info, desc: 'Small terminal points' },
              ].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTool(t.id)}
                  className={`flex items-start gap-4 p-4 rounded-xl transition-all border-2 ${
                    activeTool === t.id 
                      ? 'bg-blue-50 border-blue-500 text-blue-700 shadow-sm' 
                      : 'bg-white border-transparent text-slate-600 hover:bg-slate-50 hover:border-slate-200'
                  }`}
                >
                  <t.icon className={`w-5 h-5 mt-0.5 ${activeTool === t.id ? 'text-blue-600' : 'text-slate-400'}`} />
                  <div className="text-left">
                    <p className="font-bold text-sm leading-none mb-1">{t.label}</p>
                    <p className={`text-[10px] ${activeTool === t.id ? 'text-blue-500' : 'text-slate-400'}`}>{t.desc}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em]">Quick Presets</h2>
              <LayoutGrid className="w-3 h-3 text-slate-300" />
            </div>
            <div className="grid grid-cols-4 gap-2">
              {CHARACTER_PRESETS.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => loadCharacter(preset.name)}
                  className="aspect-square flex items-center justify-center text-sm font-black border-2 border-slate-100 rounded-xl hover:bg-slate-900 hover:text-white hover:border-slate-900 transition-all text-slate-700 bg-white shadow-sm active:scale-90"
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-auto p-4 bg-slate-50 rounded-2xl border border-slate-200">
            <h3 className="text-xs font-bold mb-2 flex items-center gap-2 text-slate-800">
              <Info className="w-3.5 h-3.5 text-blue-500" />
              Pro Tip
            </h3>
            <p className="text-[11px] text-slate-500 leading-normal">
              Place dots "between" circles by clicking mid-way points. Use Tangents for smooth vertical or horizontal bridges.
            </p>
          </div>
        </aside>

        {/* Workspace */}
        <section className="flex-1 overflow-auto p-8 flex items-center justify-center bg-slate-100 order-1 md:order-2 relative">
          <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '40px 40px' }}></div>
          <GridCanvas 
            state={state} 
            tool={activeTool} 
            onChange={updateState} 
            rows={gridSize.rows}
            cols={gridSize.cols}
          />
        </section>

        {/* Code Panel */}
        <section className="w-full md:w-[480px] bg-slate-900 border-l border-slate-800 flex flex-col order-3">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/80 backdrop-blur sticky top-0">
            <div className="flex items-center gap-2 text-slate-300">
              <CodeIcon className="w-4 h-4 text-blue-500" />
              <span className="text-[11px] font-bold uppercase tracking-widest">Generator Script Snippet</span>
            </div>
            <div className="flex gap-1">
              <div className="w-2 h-2 rounded-full bg-red-500/50"></div>
              <div className="w-2 h-2 rounded-full bg-yellow-500/50"></div>
              <div className="w-2 h-2 rounded-full bg-green-500/50"></div>
            </div>
          </div>
          <div className="flex-1 overflow-auto p-6 font-mono selection:bg-blue-500/30">
            <pre className="text-[11px] text-blue-300 leading-relaxed">
              <code>{pythonCode}</code>
            </pre>
          </div>
          <div className="p-4 bg-black/40 border-t border-slate-800 text-slate-500 text-[10px] font-mono">
            # Grid: {gridSize.cols}x{gridSize.rows} | R=40 | DX=80 | DY=80
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-3 px-6 text-center text-[10px] font-bold text-slate-400 uppercase tracking-widest shrink-0">
        Engineered for GlyphForge v2.5
      </footer>
    </div>
  );
};

export default App;

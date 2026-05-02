import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import { get, set } from 'idb-keyval';
import { 
  Camera, 
  Play, 
  CheckCircle2, 
  AlertCircle, 
  Database, 
  Zap, 
  RefreshCw, 
  Trash2, 
  Target,
  ShieldCheck,
  ShieldAlert,
  ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// Configuration
const CLASSES = ['GOOD', 'DEFECT'];
const IMAGE_SIZE = 224;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [model, setModel] = useState<mobilenet.MobileNet | null>(null);
  const [classifier, setClassifier] = useState<tf.LayersModel | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [mode, setMode] = useState<'onboarding' | 'collection' | 'training' | 'detection'>('onboarding');
  const [samples, setSamples] = useState<{ [key: string]: tf.Tensor[] }>({ GOOD: [], DEFECT: [] });
  const [isTraining, setIsTraining] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [lastPredTime, setLastPredTime] = useState(0);

  // Initialize MobileNet
  useEffect(() => {
    const init = async () => {
      await tf.ready();
      const loadedModel = await mobilenet.load({ version: 2, alpha: 1.0 });
      setModel(loadedModel);
      setIsReady(true);

      // Try to load saved classifier
      try {
        const savedModelData = await get('vision-classifier');
        if (savedModelData) {
          // Note: Full TFJS model persistence in IndexedDB is complex.
          // For this simplicity, we re-train or rely on memory.
          // Real persistence would use customModel.save('indexeddb://vision-classifier')
        }
      } catch (e) {
        console.error("Persistence error:", e);
      }
    };
    init();
  }, []);

  // WebCam Setup
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: IMAGE_SIZE, height: IMAGE_SIZE },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraActive(true);
        if (mode === 'onboarding') setMode('collection');
      }
    } catch (err) {
      console.error("Camera access denied:", err);
    }
  };

  // Data Collection
  const captureSample = useCallback(async (label: string) => {
    if (!model || !videoRef.current) return;
    const activation = model.infer(videoRef.current, true);
    setSamples(prev => ({
      ...prev,
      [label]: [...prev[label], activation]
    }));
  }, [model]);

  // Model Training
  const trainClassifier = async () => {
    if (samples.GOOD.length < 5 || samples.DEFECT.length < 5) return;
    setIsTraining(true);
    
    const labelData: number[] = [];
    const trainingData: tf.Tensor[] = [];
    samples.GOOD.forEach(t => { trainingData.push(t); labelData.push(0); });
    samples.DEFECT.forEach(t => { trainingData.push(t); labelData.push(1); });

    const xTrain = tf.concat(trainingData);
    const yTrain = tf.oneHot(tf.tensor1d(labelData, 'int32'), 2);

    const customModel = tf.sequential();
    customModel.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [trainingData[0].shape[1]] }));
    customModel.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
    customModel.compile({ optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    await customModel.fit(xTrain, yTrain, { epochs: 20 });
    
    setClassifier(customModel);
    
    // Cleanup
    xTrain.dispose();
    yTrain.dispose();
    
    setIsTraining(false);
    setMode('detection');
  };

  // Detection Loop
  useEffect(() => {
    let requestRef: number;
    const detect = async () => {
      if (mode === 'detection' && classifier && model && videoRef.current && isCameraActive) {
        const start = performance.now();
        const activation = model.infer(videoRef.current, true);
        const predictionTensor = classifier.predict(activation) as tf.Tensor;
        const probabilities = await predictionTensor.data();
        setPrediction(probabilities[0] > probabilities[1] ? 'GOOD' : 'DEFECT');
        setLastPredTime(Math.round(performance.now() - start));
      }
      requestRef = requestAnimationFrame(detect);
    };
    if (mode === 'detection') requestRef = requestAnimationFrame(detect);
    return () => cancelAnimationFrame(requestRef);
  }, [mode, classifier, model, isCameraActive]);

  const resetProject = () => {
    setSamples({ GOOD: [], DEFECT: [] });
    setClassifier(null);
    setPrediction(null);
    setMode('collection');
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 font-sans overflow-x-hidden selection:bg-indigo-500/30">
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/10 blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8 md:py-12 flex flex-col min-h-screen">
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-12">
          <div className="animate-in fade-in slide-in-from-left duration-700">
            <div className="flex items-center gap-3 mb-3">
              <div className="bg-indigo-600 p-2 rounded-xl shadow-lg shadow-indigo-500/20">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <span className="text-[10px] font-black tracking-widest text-indigo-400 uppercase">Edge AI Platform</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-black tracking-tight bg-gradient-to-br from-white to-slate-400 bg-clip-text text-transparent">
              VisionInspect AI
            </h1>
          </div>

          <div className="flex items-center gap-6 bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-5 rounded-3xl">
            <div className="flex flex-col items-end">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1">Architecture</span>
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-black text-slate-200 uppercase">On-Device JS</span>
              </div>
            </div>
            <div className="h-10 w-px bg-slate-800" />
            <div className="flex flex-col items-end text-sm">
                <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-1 font-sans">Latency</span>
                <span className="font-mono text-indigo-400">Local Only</span>
            </div>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-10 flex-grow">
          <section className="lg:col-span-7 space-y-6">
            <div className="relative aspect-video lg:aspect-square bg-slate-900 rounded-[2.5rem] overflow-hidden border-2 border-slate-800 shadow-2xl group">
              {!isCameraActive && (
                <div className="absolute inset-0 z-20 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center p-8 text-center">
                  <div className="w-24 h-24 bg-indigo-600/10 border border-indigo-500/20 rounded-full flex items-center justify-center mb-6">
                    <Camera className="w-10 h-10 text-indigo-400 animate-pulse" />
                  </div>
                  <h2 className="text-2xl font-black mb-2 uppercase tracking-tight">Camera Required</h2>
                  <p className="text-slate-400 mb-8 max-w-xs text-sm">Grant hardware access to begin local quality inspection.</p>
                  <button onClick={startCamera} disabled={!isReady} className="group flex items-center gap-3 px-10 py-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-full font-black text-sm uppercase tracking-widest transition-all shadow-xl shadow-indigo-600/20 active:scale-95">
                    Initialize System <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                  </button>
                </div>
              )}

              <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover grayscale-[10%]" />

              <div className="absolute inset-x-0 top-0 p-6 flex justify-between items-start pointer-events-none">
                <div className={`px-4 py-1.5 rounded-lg border backdrop-blur-md text-[10px] font-black tracking-widest uppercase ${isCameraActive ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-700 text-slate-500'}`}>
                  {isCameraActive ? '• LIVE STREAM' : 'OFFLINE'}
                </div>
              </div>

              <AnimatePresence>
                {mode === 'detection' && prediction && (
                  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className={`absolute inset-0 pointer-events-none flex flex-col justify-end p-8 ${prediction === 'GOOD' ? 'bg-emerald-500/5' : 'bg-rose-500/5'}`}>
                    <div className={`flex items-center gap-4 p-6 rounded-3xl border-2 backdrop-blur-2xl ${prediction === 'GOOD' ? 'bg-emerald-500/10 border-emerald-500/40' : 'bg-rose-500/10 border-rose-500/40'}`}>
                       {prediction === 'GOOD' ? <ShieldCheck className="w-12 h-12 text-emerald-400" /> : <ShieldAlert className="w-12 h-12 text-rose-400" />}
                       <div className="flex-1">
                          <span className="text-[10px] block font-extrabold uppercase tracking-widest opacity-60">Inspection Result</span>
                          <span className={`text-5xl font-black tracking-tighter ${prediction === 'GOOD' ? 'text-emerald-400' : 'text-rose-400'}`}>{prediction}</span>
                       </div>
                       <div className="text-right">
                          <div className="text-2xl font-black font-mono">{lastPredTime}ms</div>
                          <div className="text-[8px] uppercase font-bold text-slate-500 tracking-widest">Processing</div>
                       </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="grid grid-cols-3 gap-4">
               <button onClick={() => setMode('collection')} className="flex flex-col items-center justify-center gap-2 p-6 bg-slate-900 border border-slate-800 rounded-[2rem] hover:bg-slate-800 transition-all text-slate-500 hover:text-indigo-400">
                  <RefreshCw className="w-6 h-6" />
                  <span className="text-[9px] font-black uppercase tracking-widest">Recalibrate</span>
               </button>
               <button className="flex flex-col items-center justify-center gap-2 p-6 bg-slate-900 border border-slate-800 rounded-[2rem] hover:bg-slate-800 transition-all text-slate-500 hover:text-white">
                  <Target className="w-6 h-6" />
                  <span className="text-[9px] font-black uppercase tracking-widest">Analyze</span>
               </button>
               <button onClick={resetProject} className="flex flex-col items-center justify-center gap-2 p-6 bg-rose-500/5 border border-rose-500/10 rounded-[2rem] hover:bg-rose-500/10 transition-all text-rose-500/40 hover:text-rose-500">
                  <Trash2 className="w-6 h-6" />
                  <span className="text-[9px] font-black uppercase tracking-widest">Reset</span>
               </button>
            </div>
          </section>

          <aside className="lg:col-span-5 space-y-8">
            <div className="flex gap-2 p-1.5 bg-slate-900 border border-slate-800 rounded-2xl">
              {(['onboarding', 'collection', 'training', 'detection'] as const).map(m => (
                <button key={m} disabled={mode === 'onboarding' && m !== 'onboarding'} className={`flex-1 py-3 text-[10px] font-black uppercase tracking-widest rounded-xl transition-all ${mode === m ? 'bg-indigo-600 text-white' : 'text-slate-600'}`}>
                  {m === 'onboarding' ? 'START' : m}
                </button>
              ))}
            </div>

            <AnimatePresence mode="wait">
              {mode === 'onboarding' && (
                <motion.div key="onboarding" className="bg-indigo-600/5 border border-indigo-500/20 p-10 rounded-[2.5rem] text-center">
                  <div className="inline-flex p-6 bg-indigo-600 rounded-full mb-8 shadow-2xl">
                    <Target className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-black mb-4 uppercase tracking-tighter">Machine Learning Wizard</h3>
                  <p className="text-slate-400 text-sm leading-relaxed mb-10">We need to show the machine what "Correct" looks like versus "Defective". We'll capture 20 samples per category.</p>
                  <button onClick={startCamera} className="w-full py-5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-full font-black uppercase text-xs tracking-widest">Start Onboarding</button>
                </motion.div>
              )}

              {mode === 'collection' && (
                <motion.div key="collection" className="bg-slate-900 border border-slate-800 p-8 rounded-[2.5rem] space-y-8">
                  <div className="flex items-center gap-4">
                    <Database className="w-6 h-6 text-indigo-400" />
                    <h3 className="text-xl font-black uppercase tracking-tight">Dataset Acquisition</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="p-6 bg-slate-950 rounded-3xl border border-emerald-500/20">
                       <div className="flex justify-between items-center mb-6">
                         <span className="text-[10px] font-black text-emerald-400 tracking-widest uppercase">Target: GOOD</span>
                         <span className="font-mono text-emerald-400 font-bold">{samples.GOOD.length} Samples</span>
                       </div>
                       <button onClick={() => captureSample('GOOD')} className="w-full py-4 bg-emerald-500 text-slate-950 rounded-2xl font-black uppercase text-xs tracking-widest hover:scale-105 active:scale-95 transition-all">Capture Good</button>
                    </div>
                    <div className="p-6 bg-slate-950 rounded-3xl border border-rose-500/20">
                       <div className="flex justify-between items-center mb-6">
                         <span className="text-[10px] font-black text-rose-400 tracking-widest uppercase">Target: DEFECT</span>
                         <span className="font-mono text-rose-400 font-bold">{samples.DEFECT.length} Samples</span>
                       </div>
                       <button onClick={() => captureSample('DEFECT')} className="w-full py-4 bg-rose-500 text-white rounded-2xl font-black uppercase text-xs tracking-widest hover:scale-105 active:scale-95 transition-all">Capture Defect</button>
                    </div>
                  </div>
                  <button onClick={() => setMode('training')} disabled={samples.GOOD.length < 5 || samples.DEFECT.length < 5} className="w-full py-5 bg-indigo-600 disabled:opacity-30 text-white rounded-2xl font-black uppercase text-xs tracking-widest shadow-xl">Start Neural Training</button>
                </motion.div>
              )}

              {mode === 'training' && (
                <motion.div key="training" className="bg-slate-900 border border-slate-800 p-10 rounded-[2.5rem] text-center">
                  <RefreshCw className={`w-12 h-12 mx-auto mb-8 text-indigo-400 ${isTraining ? 'animate-spin' : ''}`} />
                  <h3 className="text-2xl font-black mb-4 uppercase tracking-tighter">Neural Optimization</h3>
                  <p className="text-slate-400 text-sm mb-10">We are fine-tuning the model layers to your specific item environment. This process is fully private and local.</p>
                  <button onClick={trainClassifier} disabled={isTraining} className="w-full py-6 bg-indigo-600 disabled:bg-slate-700 text-white rounded-2xl font-black uppercase text-xs tracking-[0.2em]">{isTraining ? 'Optimizing Tensors...' : 'Start Edge Training'}</button>
                </motion.div>
              )}

              {mode === 'detection' && (
                <motion.div key="detection" className="bg-slate-900 border border-slate-800 p-8 rounded-[2.5rem] space-y-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                       <Zap className="w-5 h-5 text-indigo-400" />
                       <h3 className="font-black uppercase tracking-tight">Active Engine</h3>
                    </div>
                    <div className="px-3 py-1 bg-indigo-500/10 text-[9px] text-indigo-400 uppercase font-black rounded-lg">Operational</div>
                  </div>
                  <div className="p-8 bg-slate-950 rounded-3xl border border-slate-800 text-center">
                    <span className="text-[10px] block font-black uppercase tracking-widest text-slate-500 mb-4">Inspection Monitor</span>
                    <span className={`text-5xl font-black tracking-tighter ${prediction === 'GOOD' ? 'text-emerald-400' : 'text-rose-400'}`}>{prediction || 'IDLE'}</span>
                  </div>
                  <button onClick={() => setMode('collection')} className="w-full py-4 border-2 border-slate-700 hover:border-slate-500 text-slate-500 hover:text-white rounded-2xl font-black uppercase text-[10px] tracking-widest transition-all">Retrain Model</button>
                </motion.div>
              )}
            </AnimatePresence>
          </aside>
        </main>
      </div>
    </div>
  );
}

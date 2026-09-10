import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. AI & MACHINE LEARNING ENGINE
# ==========================================
class PeakLoadEngine:
    def __init__(self):
        self.model = HistGradientBoostingRegressor(random_state=42, max_iter=100)
        self.is_trained = False
        self.metrics = {}
        self.historical_data = None
        self.threshold = 0.0
        self.peak_tariff = 12.0
        self.offpeak_tariff = 3.0

    def generate_demo_data(self, days=60):
        np.random.seed(42)
        end_date = pd.Timestamp.now().floor('h')  # Fixed: changed 'H' to 'h'
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='h')  # Fixed: changed 'H' to 'h'
        
        df = pd.DataFrame({'timestamp': dates})
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        base_load = 4.0
        daily_pattern = np.sin((df['hour'] - 6) * (np.pi / 12)) * 3
        daily_pattern = np.where(daily_pattern < 0, 0, daily_pattern)
        weekend_factor = np.where(df['is_weekend'] == 1, 0.7, 1.0)
        
        df['temperature'] = 20 + np.sin((df['hour'] - 8) * (np.pi / 12)) * 10 + np.random.normal(0, 2, len(df))
        temp_impact = np.where(df['temperature'] > 28, (df['temperature'] - 28) * 0.5, 0)
        
        df['energy_consumption'] = (base_load + daily_pattern + temp_impact) * weekend_factor
        df['energy_consumption'] += np.random.normal(0, 0.5, len(df))
        
        spike_indices = np.random.choice(df.index, size=int(len(df)*0.02), replace=False)
        df.loc[spike_indices, 'energy_consumption'] += np.random.uniform(2.0, 5.0, len(spike_indices))
        df['energy_consumption'] = df['energy_consumption'].clip(lower=1.0)
        
        self.historical_data = df
        return df

    def preprocess_features(self, df):
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        if 'temperature' not in df.columns:
            df['temperature'] = 25.0 
        return df[['hour', 'day_of_week', 'is_weekend', 'temperature']]

    def train(self):
        self.generate_demo_data()
        data = self.historical_data.copy()
        X = self.preprocess_features(data)
        y = data['energy_consumption']
        
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        
        self.metrics = {
            'mae': round(mean_absolute_error(y_test, preds), 2),
            'rmse': round(root_mean_squared_error(y_test, preds), 2),
            'r2': round(r2_score(y_test, preds) * 100, 2)
        }
        self.threshold = np.percentile(y, 90)
        self.is_trained = True

    def forecast_next_24h(self):
        if not self.is_trained:
            self.train()
            
        last_timestamp = self.historical_data['timestamp'].max()
        future_dates = [last_timestamp + timedelta(hours=i) for i in range(1, 25)]
        future_df = pd.DataFrame({'timestamp': future_dates})
        future_df['temperature'] = 25 + np.sin((future_df['timestamp'].dt.hour - 8) * (np.pi / 12)) * 8
        
        X_future = self.preprocess_features(future_df)
        predictions = self.model.predict(X_future)
        
        results, peaks = [], []
        for i, dt in enumerate(future_dates):
            pred_val = float(predictions[i])
            is_peak = pred_val > self.threshold
            severity = "NORMAL"
            if is_peak:
                severity = "CRITICAL" if pred_val > self.threshold * 1.15 else "HIGH"
                peaks.append({"time": dt.strftime('%H:%M'), "demand": round(pred_val, 2), "severity": severity})
                
            results.append({
                "time": dt.strftime('%H:%M'),
                "predicted_demand": round(pred_val, 2),
                "threshold": round(self.threshold, 2),
                "is_peak": is_peak
            })
        return results, peaks

    def generate_recommendations(self, peaks):
        if not peaks:
            return []
        rec_peaks = [p for p in peaks if p['severity'] == 'CRITICAL'] or peaks
        recs = []
        for p in rec_peaks[:2]:
            recs.extend([
                {"load_type": "EV Charging", "current": f"{p['time']}", "recommended": "01:00 - 04:00", "reduction": 3.6},
                {"load_type": "Water Heating", "current": f"{p['time']}", "recommended": "Shift 2 hrs early", "reduction": 1.5}
            ])
        return recs

    def simulate_shift(self, shift_amount_kw):
        forecast, _ = self.forecast_next_24h()
        df = pd.DataFrame(forecast)
        
        original_peak = df['predicted_demand'].max()
        df['optimized_demand'] = df['predicted_demand']
        
        peak_mask = df['predicted_demand'] > self.threshold
        total_shifted = 0
        
        for idx in df[peak_mask].index:
            avail = min(shift_amount_kw, df.loc[idx, 'optimized_demand'] - (self.threshold * 0.8))
            if avail > 0:
                df.loc[idx, 'optimized_demand'] -= avail
                total_shifted += avail
                
        for idx in df.index:
            hour = int(df.loc[idx, 'time'].split(':')[0])
            if 1 <= hour <= 5 and total_shifted > 0:
                df.loc[idx, 'optimized_demand'] += (total_shifted / 4)
                
        new_peak = df['optimized_demand'].max()
        return {
            "chart_data": df[['time', 'predicted_demand', 'optimized_demand', 'threshold']].to_dict(orient='records'),
            "impact": {
                "original_peak_kw": round(original_peak, 2),
                "new_peak_kw": round(new_peak, 2),
                "reduction_pct": round(((original_peak - new_peak) / original_peak) * 100, 1) if original_peak > 0 else 0,
                "cost_savings_inr": round(total_shifted * (self.peak_tariff - self.offpeak_tariff), 2)
            }
        }

# ==========================================
# 2. FASTAPI BACKEND
# ==========================================
app = FastAPI(title="PeakGuard AI")
engine = PeakLoadEngine()

class SimRequest(BaseModel):
    shift_amount_kw: float

@app.on_event("startup")
def startup():
    engine.train()

@app.get("/api/dashboard")
def get_dashboard_data():
    forecast, peaks = engine.forecast_next_24h()
    return {
        "current_load": round(float(engine.historical_data['energy_consumption'].iloc[-1]), 2),
        "metrics": engine.metrics,
        "forecast": forecast,
        "peaks": peaks,
        "recommendations": engine.generate_recommendations(peaks)
    }

@app.post("/api/simulate")
def simulate(req: SimRequest):
    return engine.simulate_shift(req.shift_amount_kw)

# ==========================================
# 3. REACT FRONTEND (Embedded HTML/JS)
# ==========================================
@app.get("/")
def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PeakGuard AI</title>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/prop-types/prop-types.min.js"></script>
        <script src="https://unpkg.com/recharts/umd/Recharts.js"></script>
        <script>
            tailwind.config = { theme: { extend: { colors: { brand: '#10b981', dark: '#0f172a', card: '#1e293b' } } } }
        </script>
        <style>body { background-color: #0f172a; color: #f1f5f9; font-family: system-ui, sans-serif; }</style>
    </head>
    <body>
        <div id="root"></div>
        <script type="text/babel">
            const { useState, useEffect } = React;
            const { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } = window.Recharts;

            function App() {
                const [data, setData] = useState(null);
                const [tab, setTab] = useState('dashboard');
                const [simShift, setSimShift] = useState(2.0);
                const [simResult, setSimResult] = useState(null);

                useEffect(() => { fetch('/api/dashboard').then(res => res.json()).then(setData); }, []);
                useEffect(() => {
                    if (tab === 'simulate' && data) {
                        fetch('/api/simulate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({shift_amount_kw: parseFloat(simShift)})
                        }).then(res => res.json()).then(setSimResult);
                    }
                }, [tab, simShift, data]);

                if (!data) return <div className="flex h-screen items-center justify-center text-brand text-xl">⚡ Loading AI Engine...</div>;

                return (
                    <div className="min-h-screen flex flex-col">
                        <header className="bg-card border-b border-slate-700 p-4 flex justify-between items-center">
                            <div className="text-brand font-bold text-xl flex items-center gap-2">⚡ PeakGuard AI</div>
                            <nav className="flex gap-4">
                                <button onClick={() => setTab('dashboard')} className={`px-4 py-2 rounded ${tab==='dashboard' ? 'bg-brand text-white' : 'text-slate-400'}`}>Dashboard</button>
                                <button onClick={() => setTab('simulate')} className={`px-4 py-2 rounded ${tab==='simulate' ? 'bg-brand text-white' : 'text-slate-400'}`}>Simulator</button>
                            </nav>
                        </header>
                        <main className="p-6 max-w-7xl mx-auto w-full space-y-6">
                            {tab === 'dashboard' && (
                                <>
                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                        <div className="bg-card p-5 rounded-xl border border-slate-700">
                                            <div className="text-slate-400 text-sm">Current Load</div>
                                            <div className="text-2xl font-bold text-blue-400">{data.current_load} kW</div>
                                        </div>
                                        <div className="bg-card p-5 rounded-xl border border-slate-700">
                                            <div className="text-slate-400 text-sm">Next Peak</div>
                                            <div className="text-2xl font-bold text-red-400">{data.peaks.length ? data.peaks[0].time : 'None'}</div>
                                        </div>
                                        <div className="bg-card p-5 rounded-xl border border-slate-700">
                                            <div className="text-slate-400 text-sm">Model Accuracy (R²)</div>
                                            <div className="text-2xl font-bold text-brand">{data.metrics.r2}%</div>
                                        </div>
                                        <div className="bg-card p-5 rounded-xl border border-slate-700">
                                            <div className="text-slate-400 text-sm">System Status</div>
                                            <div className="text-2xl font-bold text-green-400">✅ Online</div>
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                        <div className="lg:col-span-2 bg-card p-5 rounded-xl border border-slate-700">
                                            <h2 className="text-lg font-semibold mb-4">📊 24-Hour AI Demand Forecast</h2>
                                            <div className="h-[300px] w-full">
                                                <ResponsiveContainer>
                                                    <LineChart data={data.forecast}>
                                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                                        <XAxis dataKey="time" stroke="#94a3b8" />
                                                        <YAxis stroke="#94a3b8" />
                                                        <Tooltip contentStyle={{backgroundColor: '#1e293b', borderColor: '#334155'}} />
                                                        <ReferenceLine y={data.forecast[0].threshold} stroke="#ef4444" strokeDasharray="5 5" label={{position: 'top', value: 'Peak Threshold', fill: '#ef4444'}} />
                                                        <Line type="monotone" dataKey="predicted_demand" name="Demand (kW)" stroke="#10b981" strokeWidth={3} dot={false} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>
                                        <div className="space-y-6">
                                            <div className="bg-card p-5 rounded-xl border border-slate-700">
                                                <h2 className="text-lg font-semibold mb-4">⚠️ Peak Alerts</h2>
                                                {data.peaks.length === 0 ? <p className="text-slate-400">No peaks predicted.</p> : 
                                                    data.peaks.map((p, i) => (
                                                        <div key={i} className="p-3 mb-2 rounded bg-red-900/20 border border-red-800 flex justify-between">
                                                            <div><div className="font-bold text-red-400">{p.severity}</div><div className="text-sm text-red-200">{p.time}</div></div>
                                                            <div className="text-xl font-bold text-red-400">{p.demand} kW</div>
                                                        </div>
                                                    ))
                                                }
                                            </div>
                                            <div className="bg-card p-5 rounded-xl border border-slate-700">
                                                <h2 className="text-lg font-semibold mb-4">⚙️ AI Recommendations</h2>
                                                {data.recommendations.map((r, i) => (
                                                    <div key={i} className="border-l-2 border-brand pl-3 mb-4">
                                                        <div className="font-medium">{r.load_type}</div>
                                                        <div className="text-sm text-slate-400">Shift from <span className="text-red-400">{r.current}</span> to <span className="text-brand">{r.recommended}</span></div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                </>
                            )}
                            {tab === 'simulate' && simResult && (
                                <div className="bg-card p-5 rounded-xl border border-slate-700">
                                    <h2 className="text-xl font-bold mb-6">🎛️ What-If Load Shifting Simulator</h2>
                                    <div className="flex gap-8 items-center bg-dark p-4 rounded-lg border border-slate-700 mb-6">
                                        <div className="w-1/2">
                                            <label className="block mb-2 text-slate-400">Shiftable Load (kW): <span className="text-white font-bold">{simShift}</span></label>
                                            <input type="range" min="0" max="6" step="0.5" value={simShift} onChange={e => setSimShift(e.target.value)} className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer" />
                                        </div>
                                        <div className="w-1/2 flex gap-4">
                                            <div className="bg-card p-3 flex-1 rounded border border-slate-700 text-center">
                                                <div className="text-xs text-slate-400">Peak Reduction</div>
                                                <div className="text-xl font-bold text-brand">{simResult.impact.reduction_pct}%</div>
                                            </div>
                                            <div className="bg-card p-3 flex-1 rounded border border-slate-700 text-center">
                                                <div className="text-xs text-slate-400">Est. Savings</div>
                                                <div className="text-xl font-bold text-green-400">₹{simResult.impact.cost_savings_inr}</div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="h-[400px] w-full">
                                        <ResponsiveContainer>
                                            <AreaChart data={simResult.chart_data}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                                <XAxis dataKey="time" stroke="#94a3b8" />
                                                <YAxis stroke="#94a3b8" />
                                                <Tooltip contentStyle={{backgroundColor: '#1e293b', borderColor: '#334155'}} />
                                                <ReferenceLine y={simResult.chart_data[0].threshold} stroke="#94a3b8" strokeDasharray="5 5" />
                                                <Area type="monotone" dataKey="predicted_demand" name="Original" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} />
                                                <Area type="monotone" dataKey="optimized_demand" name="Optimized" stroke="#10b981" fill="#10b981" fillOpacity={0.4} />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            )}
                        </main>
                    </div>
                );
            }
            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(<App />);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
      uvicorn.run(app, host="0.0.0.0", port=8000)
   
import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

const AREAS = ["manitoba", "new-york", "ontario", "quebec_p33c", "manitoba_sk"];
const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api";
const API_URL = `${API_BASE}/simulation_data.json`;

function usePolling(url, intervalMs = 5000) {
  const [data, setData] = React.useState(null);
  const [err, setErr] = React.useState(null);
  React.useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        const j = await r.json();
        if (alive) { setData(j); setErr(null); }
      } catch (e) {
        if (alive) setErr(String(e));
      }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => { alive = false; clearInterval(id); };
  }, [url, intervalMs]);
  return { data, err };
}

function StatCard({ title, value, subtitle }) {
  return <div className="card"><div className="title">{title}</div><div className="value">{value}</div>{subtitle && <div className="subtle">{subtitle}</div>}</div>;
}

function AreaCard({name, loadMW, priceF, priceA, socPct, src, windowLabel}) {
  return (
    <div className="card">
      <h4 style={{margin:0}}>{name}</h4>
      <div className="grid" style={{gridTemplateColumns:"1fr 1fr", marginTop:8}}>
        <div className="subtle">Load</div><div className="subtle" style={{textAlign:"right"}}>{(loadMW ?? 0).toFixed(2)} MW</div>
        <div className="subtle">Price (F)</div><div className="subtle" style={{textAlign:"right"}}>${(priceF ?? 0).toFixed(2)}/MWh</div>
        <div className="subtle">Price (A)</div><div className="subtle" style={{textAlign:"right"}}>${(priceA ?? 0).toFixed(2)}/MWh</div>
        <div className="subtle">Prediction</div><div className="subtle" style={{textAlign:"right"}}>{src ?? "?"} · {windowLabel ?? "unknown"}</div>
        <div className="subtle">SOC</div><div className="subtle" style={{textAlign:"right"}}>{(socPct ?? 0).toFixed(1)}%</div>
      </div>
    </div>
  );
}

function LineBlock({ title, series }) {
  const labels = series[0]?.data?.map((p) => p.x) ?? [];
  const merged = labels.map((x, i) => {
    const o = { x };
    series.forEach((s) => { o[s.name] = s.data[i]?.y ?? null; });
    return o;
  });
  return (
    <div className="card">
      <div className="title">{title}</div>
      <div style={{ height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={merged}><XAxis dataKey="x" hide /><YAxis /><Tooltip /><Legend />{series.map((s) => <Line key={s.name} type="monotone" dataKey={s.name} dot={false} strokeWidth={2} />)}</LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function App() {
  const [page, setPage] = React.useState("operations");
  const { data, err } = usePolling(API_URL, 5000);
  const [modeBusy, setModeBusy] = React.useState(false);

  const setMode = async (mode) => {
    setModeBusy(true);
    try {
      await fetch(`${API_BASE}/mode/${mode}`, { method: "POST" });
    } finally {
      setModeBusy(false);
    }
  };

  if (err) return <div className="container"><div className="card"><div className="title">Error</div><pre className="subtle">{err}</pre></div></div>;
  if (!data || data.status === "starting") return <div className="container"><span className="badge">Starting… waiting for first step</span></div>;

  const step = data.step ?? 0;
  const running = data.status === "running";
  const current = data.current || {};
  const previous = data.previous || null;
  const areaLoads = current.perAreaLoadMW || {};
  const soc = data.soc || {};
  const priceF = current.areaPriceForecast || {};
  const priceA = current.areaPriceActual || {};
  const forecastSource = current.forecastSource || {};
  const predictionWindow = current.predictionWindow || {};
  const dollars = n => (n ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 });

  const steps = data.hist?.steps || [];
  const labels = steps.map((_, i) => i.toString());
  const totalLoad = data.hist?.total_load || [];
  const totalSavings = data.hist?.total_savings_forecast || [];
  const priceHist = data.hist?.area_price || {};
  const areaLoadHist = data.hist?.area_load || {};

  const loadSeries = [{ name: "Total Load (MW)", data: labels.map((x, i) => ({ x, y: totalLoad[i] ?? null })) }];
  const savingsSeries = [{ name: "Savings (forecast $)", data: labels.map((x, i) => ({ x, y: totalSavings[i] ?? null })) }];
  const priceSeries = AREAS.map((a) => ({ name: `Price ${a} (F)`, data: (priceHist[a] ?? []).map((y, i) => ({ x: labels[i], y })) }));
  const perAreaSeries = AREAS.map((a) => ({ name: `Load ${a} (MW)`, data: (areaLoadHist[a] ?? []).map((y, i) => ({ x: labels[i], y })) }));
  const modeSeries = ["short", "medium", "long"].map((m, idx) => ({ name: m, data: modeHist.map((v, i) => ({ x: labels[i], y: v === m ? idx + 1 : null })) }));

  return (
    <div className="container">
      <div className="header" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <h1 style={{ margin: 0 }}>DC Optimization Dashboard</h1>
        <span className="badge">{running ? "Running" : data.status} — step {step + 1}</span>
        <span className="badge">{current?.time?.date} {current?.time?.hhmm}</span>
        <span className="badge">Mode: {data.activeOptimizationMode} ({data.modeSelection})</span>
        <button className="badge" onClick={() => setPage("operations")}>Operations</button>
        <button className="badge" onClick={() => setPage("comparison")}>Forecast vs No-Forecast</button>
        <button className="badge" disabled={modeBusy} onClick={() => setMode("auto")}>auto</button>
        <button className="badge" disabled={modeBusy} onClick={() => setMode("short")}>short</button>
        <button className="badge" disabled={modeBusy} onClick={() => setMode("medium")}>medium</button>
        <button className="badge" disabled={modeBusy} onClick={() => setMode("long")}>long</button>
      </div>

      <div className="grid grid-4" style={{marginTop:16}}>
        <StatCard title="Total Load (MW)" value={(data.totalLoad ?? 0).toFixed(2)} />
        <StatCard title="Cost (forecast cum.)" value={`$${dollars(data.totalCost)}`} />
        <StatCard title="Savings (forecast cum.)" value={`$${dollars(data.totalSavings)}`} />
        <StatCard title="Realized Savings (actual cum.)" value={`$${dollars(data.realizedSavings)}`} />
      </div>

      <h3 style={{marginTop:24}}>Per-area (current step)</h3>
      <div className="grid" style={{gridTemplateColumns:"repeat(auto-fit, minmax(230px, 1fr))"}}>
        {AREAS.map(a => <AreaCard key={a} name={a} loadMW={areaLoads[a]} priceF={priceF[a]} priceA={priceA[a]} socPct={soc[a]} src={forecastSource[a]} windowLabel={predictionWindow[a]} />)}
      </div>

      <h3 style={{marginTop:24}}>Current vs Previous Step</h3>
      <div className="grid grid-2">
        <div className="card">
          <div className="title">Current step prices</div>
          <div style={{marginTop:10, overflowX:"auto"}}>
            <table className="table">
              <thead><tr><th>Area</th><th>Forecast $/MWh</th><th>Actual $/MWh</th><th>Source</th><th>Window</th></tr></thead>
              <tbody>
                {AREAS.map(a => (
                  <tr key={a}>
                    <td style={{fontWeight:600}}>{a}</td>
                    <td>${(priceF?.[a] ?? 0).toFixed(2)}</td>
                    <td>${(priceA?.[a] ?? 0).toFixed(2)}</td>
                    <td>{current?.forecastSource?.[a] ?? "?"}</td>
                    <td>{current?.predictionWindow?.[a] ?? "unknown"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="grid grid-2" style={{ marginTop: 20 }}>
            <LineBlock title="Savings (forecast $)" series={savingsSeries} />
            <LineBlock title="Optimization mode over time" series={modeSeries} />
          </div>
        </>
      ) : (
        <>
          <div className="grid grid-4" style={{ marginTop: 16 }}>
            <StatCard title="Total Load (MW)" value={(data.totalLoad ?? 0).toFixed(2)} />
            <StatCard title="Cost (forecast cum.)" value={`$${dollars(data.totalCost)}`} />
            <StatCard title="Savings (forecast cum.)" value={`$${dollars(data.totalSavings)}`} />
            <StatCard title="Realized Savings (actual cum.)" value={`$${dollars(data.realizedSavings)}`} />
          </div>
        </div>

        {previous && (
          <>
            <div className="card">
              <div className="title">Previous step prices</div>
              <div style={{marginTop:10, overflowX:"auto"}}>
                <table className="table">
                  <thead><tr><th>Area</th><th>Forecast $/MWh</th><th>Actual $/MWh</th><th>Source</th><th>Window</th></tr></thead>
                  <tbody>
                    {AREAS.map(a => (
                      <tr key={a}>
                        <td style={{fontWeight:600}}>{a}</td>
                        <td>${(previous.areaPriceForecast?.[a] ?? 0).toFixed(2)}</td>
                        <td>${(previous.areaPriceActual?.[a] ?? 0).toFixed(2)}</td>
                        <td>{previous.forecastSource?.[a] ?? "?"}</td>
                        <td>{previous.predictionWindow?.[a] ?? "unknown"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card">
              <div className="title">Previous step costs</div>
              <div className="grid" style={{gridTemplateColumns:"1fr 1fr", marginTop:10}}>
                <div className="subtle">Forecast (opt)</div><div className="subtle" style={{textAlign:"right"}}>${dollars(previous?.stepCosts?.forecast_opt)}</div>
                <div className="subtle">Forecast (equal)</div><div className="subtle" style={{textAlign:"right"}}>${dollars(previous?.stepCosts?.forecast_equal)}</div>
                <div className="subtle">Forecast savings</div><div className="subtle" style={{textAlign:"right"}}>${dollars(previous?.stepCosts?.forecast_savings)}</div>
                <div className="subtle">Realized (opt)</div><div className="subtle" style={{textAlign:"right"}}>${dollars(previous?.stepCosts?.realized_opt)}</div>
                <div className="subtle">Realized (equal)</div><div className="subtle" style={{textAlign:"right"}}>${dollars(previous?.stepCosts?.realized_equal)}</div>
                <div className="subtle">Realized savings</div><div className="subtle" style={{textAlign:"right"}}>${dollars(previous?.stepCosts?.realized_savings)}</div>
              </div>
            </div>
          </>
        )}
      </div>

          <h3 style={{ marginTop: 24 }}>Per-area (current step)</h3>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))" }}>
            {AREAS.map((a) => <AreaCard key={a} name={a} loadMW={areaLoads[a]} priceF={priceF[a]} priceA={priceA[a]} socPct={soc[a]} src={forecastSource[a]} windowLabel={predictionWindow[a]} />)}
          </div>

          <h3 style={{ marginTop: 24 }}>Current vs Previous Step</h3>
          <div className="grid grid-2">
            <div className="card"><div className="title">Current step prices</div><div style={{ marginTop: 10, overflowX: "auto" }}><table className="table"><thead><tr><th>Area</th><th>Forecast</th><th>Actual</th><th>Source</th><th>Window</th></tr></thead><tbody>{AREAS.map((a) => <tr key={a}><td style={{ fontWeight: 600 }}>{a}</td><td>${(priceF?.[a] ?? 0).toFixed(2)}</td><td>${(priceA?.[a] ?? 0).toFixed(2)}</td><td>{current?.forecastSource?.[a] ?? "?"}</td><td>{current?.predictionWindow?.[a] ?? "unknown"}</td></tr>)}</tbody></table></div></div>
            <div className="card"><div className="title">Current step costs</div><div className="grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 10 }}><div className="subtle">Forecast (opt)</div><div className="subtle" style={{ textAlign: "right" }}>${dollars(current?.stepCosts?.forecast_opt)}</div><div className="subtle">Forecast (equal)</div><div className="subtle" style={{ textAlign: "right" }}>${dollars(current?.stepCosts?.forecast_equal)}</div><div className="subtle">Forecast savings</div><div className="subtle" style={{ textAlign: "right" }}>${dollars(current?.stepCosts?.forecast_savings)}</div><div className="subtle">Realized savings</div><div className="subtle" style={{ textAlign: "right" }}>${dollars(current?.stepCosts?.realized_savings)}</div><div className="subtle">Degradation</div><div className="subtle" style={{ textAlign: "right" }}>${dollars(current?.stepCosts?.degradation)}</div></div></div>
            {previous && <div className="card"><div className="title">Previous step source/window</div><div style={{ marginTop: 10, overflowX: "auto" }}><table className="table"><thead><tr><th>Area</th><th>Source</th><th>Window</th></tr></thead><tbody>{AREAS.map((a) => <tr key={a}><td>{a}</td><td>{previous.forecastSource?.[a] ?? "?"}</td><td>{previous.predictionWindow?.[a] ?? "unknown"}</td></tr>)}</tbody></table></div></div>}
          </div>

          <h3 style={{ marginTop: 24 }}>Live charts (~5h)</h3>
          <div className="grid grid-2"><LineBlock title="Total Load (MW)" series={loadSeries} /><LineBlock title="Savings (forecast $)" series={savingsSeries} /><LineBlock title="Price per area (forecast)" series={priceSeries} /><LineBlock title="Area loads (MW)" series={perAreaSeries} /></div>
        </>
      )}

      <hr style={{ border: 0, borderTop: "1px solid #1c2545", margin: "20px 0" }} />
      <small style={{ fontFamily: "monospace" }}>API: {API_URL}</small>
    </div>
  );
}

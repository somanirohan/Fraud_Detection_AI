import { LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import { useEffect, useState } from "react";
import axios from "axios";

export default function RiskTrend() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/risk-trend").then(res => setData(res.data));
  }, []);

  return (
    <div className="card">
      <h3>Fraud Risk Trend</h3>
      <LineChart width={350} height={250} data={data}>
        <XAxis dataKey="id" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="risk" stroke="#ec4899" />
      </LineChart>
    </div>
  );
}



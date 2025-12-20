import { ScatterChart, Scatter, XAxis, YAxis, Tooltip } from "recharts";
import { useEffect, useState } from "react";
import axios from "axios";

export default function AmountRiskScatter() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/amount-risk").then(res => setData(res.data));
  }, []);

  return (
    <div className="card">
      <h3>Amount vs Risk</h3>
      <ScatterChart width={350} height={250}>
        <XAxis dataKey="amount" name="Amount" />
        <YAxis dataKey="risk" name="Risk" />
        <Tooltip />
        <Scatter data={data} fill="#22c55e" />
      </ScatterChart>
    </div>
  );
}

import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";
import { useEffect, useState } from "react";
import axios from "axios";

export default function RiskBar() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/risk-bar").then(res => {
      setData([
        { name: "High", value: res.data.High },
        { name: "Medium", value: res.data.Medium },
        { name: "Low", value: res.data.Low }
      ]);
    });
  }, []);

  return (
    <div className="card">
      <h3>Transactions by Risk</h3>
      <BarChart width={300} height={250} data={data}>
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="value" fill="#7c3aed" />
      </BarChart>
    </div>
  );
}

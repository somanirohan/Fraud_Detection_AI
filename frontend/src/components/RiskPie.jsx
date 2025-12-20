import { PieChart, Pie, Cell, Tooltip } from "recharts";
import { useEffect, useState } from "react";
import axios from "axios";

export default function RiskPie() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/stats").then(res => {
      setData([
        { name: "High", value: res.data.high },
        { name: "Medium", value: res.data.medium },
        { name: "Low", value: res.data.low }
      ]);
    });
  }, []);

  return (
    <div className="card">
      <h3>Risk Distribution</h3>
      <PieChart width={300} height={300}>
        <Pie data={data} dataKey="value" outerRadius={100}>
          <Cell fill="#ff4d4f"/>
          <Cell fill="#faad14"/>
          <Cell fill="#52c41a"/>
        </Pie>
        <Tooltip/>
      </PieChart>
    </div>
  );
}

import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";
import { useEffect, useState } from "react";
import axios from "axios";

export default function HourlyFraud() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/hourly-fraud").then(res => {
      const arr = Object.entries(res.data).map(([h, v]) => ({
        hour: h,
        count: v
      }));
      setData(arr);
    });
  }, []);

  return (
    <div className="card full">
      <h3>Transactions by Hour</h3>
      <BarChart width={700} height={250} data={data}>
        <XAxis dataKey="hour" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" fill="#facc15" />
      </BarChart>
    </div>
  );
}

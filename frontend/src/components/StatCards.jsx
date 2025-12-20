import { useEffect, useState } from "react";
import axios from "axios";

export default function StatCards() {
  const [stats, setStats] = useState({
    high: 0,
    medium: 0,
    low: 0
  });

  useEffect(() => {
    axios
      .get("http://127.0.0.1:8000/stats")
      .then((res) => setStats(res.data))
      .catch((err) => console.error("Stats fetch error", err));
  }, []);

  return (
    <div className="card full">
      <h3>Risk Summary</h3>

      <div className="stats">
        <div className="stat red">
          High Risk
          <b>{stats.high}</b>
        </div>

        <div className="stat yellow">
          Medium Risk
          <b>{stats.medium}</b>
        </div>

        <div className="stat green">
          Low Risk
          <b>{stats.low}</b>
        </div>
      </div>
    </div>
  );
}

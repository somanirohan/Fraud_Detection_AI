import { useEffect, useState } from "react";
import axios from "axios";

export default function TransactionsTable() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/transactions")
      .then(res => setData(res.data));
  }, []);

  return (
    <div className="card">
      <h3>Transactions</h3>
      <table>
        <thead>
          <tr>
            <th>Amount</th>
            <th>Risk</th>
            <th>Probability</th>
          </tr>
        </thead>
        <tbody>
          {data.map(t => (
            <tr key={t.id}>
              <td>{t.amount}</td>
              <td>{t.risk_level}</td>
              <td>{t.fraud_probability}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

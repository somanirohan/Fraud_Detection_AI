import { useState } from "react";
import axios from "axios";

export default function FraudForm() {
  const [form, setForm] = useState({
    amount: "",
    transaction_hour: "",
    is_new_device: 0,
    location_change: 0,
    daily_txn_count: ""
  });

  const submit = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:8000/analyze", {
        ...form,
        amount: Number(form.amount),
        transaction_hour: Number(form.transaction_hour),
        daily_txn_count: Number(form.daily_txn_count)
      });

      console.log("API Response:", res.data);

      alert(
        `Transaction Stored\n\nRisk: ${res.data.risk_level}\nProbability: ${res.data.fraud_probability}`
      );

      // OPTIONAL: refresh stats / charts after insert
      window.dispatchEvent(new Event("update-data"));

    } catch (err) {
      console.error("API Error:", err);
      alert("Failed to analyze transaction, check console");
    }
  };

  return (
    <div className="card">
      <h3>Add Transaction</h3>

      <input
        placeholder="Amount"
        onChange={(e) => setForm({ ...form, amount: e.target.value })}
      />

      <input
        placeholder="Hour (0-23)"
        onChange={(e) => setForm({ ...form, transaction_hour: e.target.value })}
      />

      <input
        placeholder="Daily Txn Count"
        onChange={(e) => setForm({ ...form, daily_txn_count: e.target.value })}
      />

      <select
        onChange={(e) => setForm({ ...form, is_new_device: Number(e.target.value) })}
      >
        <option value={0}>Same Device</option>
        <option value={1}>New Device</option>
      </select>

      <select
        onChange={(e) => setForm({ ...form, location_change: Number(e.target.value) })}
      >
        <option value={0}>Same Location</option>
        <option value={1}>Location Changed</option>
      </select>

      <button onClick={submit}>Analyze & Store</button>
    </div>
  );
}

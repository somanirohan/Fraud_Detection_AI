import axios from "axios";
import { useState } from "react";

export default function CsvUpload() {
  const [file, setFile] = useState(null);

  const upload = async () => {
    const form = new FormData();
    form.append("file", file);
    await axios.post("http://127.0.0.1:8000/upload-csv", form);
    alert("CSV stored");
  };

  return (
    <div className="card">
      <h3>Upload CSV</h3>
      <input type="file" accept=".csv" onChange={e => setFile(e.target.files[0])}/>
      <button onClick={upload}>Upload</button>
    </div>
  );
}

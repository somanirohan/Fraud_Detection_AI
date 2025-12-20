import Navbar from "./Navbar";
import StatCards from "./StatCards";
import FraudForm from "./FraudForm";
import CsvUpload from "./CsvUpload";
import RiskPie from "./RiskPie";
import TransactionsTable from "./TransactionsTable";
import RiskBar from "./RiskBar";
import RiskTrend from "./RiskTrend";
import AmountRiskScatter from "./AmountRiskScatter";
import HourlyFraud from "./HourlyFraud";

export default function Dashboard() {
  return (
    <>
      <Navbar />
      <div className="layout">
        <StatCards />
        <FraudForm />
        <CsvUpload />
        <RiskPie />
        <RiskBar />
        <RiskTrend />
        <AmountRiskScatter />
        <HourlyFraud />
        <TransactionsTable />
      </div>
    </>
  );
}

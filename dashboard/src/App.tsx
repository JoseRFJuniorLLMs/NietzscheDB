import { Routes, Route, Navigate } from "react-router-dom"
import { DashboardLayout } from "./layouts/DashboardLayout"
import { OverviewPage } from "./pages/OverviewPage"
import { CollectionsPage } from "./pages/CollectionsPage"
import { DataExplorerPage } from "./pages/DataExplorerPage"
import { NodesPage } from "./pages/NodesPage"
import { SettingsPage } from "./pages/SettingsPage"
import { GraphExplorerPage } from "./pages/GraphExplorerPage"
import QueryPage from "./pages/QueryPage"
import AlgorithmsPage from "./pages/AlgorithmsPage"
import AgencyPage from "./pages/AgencyPage"
import SleepPage from "./pages/SleepPage"
import BackupPage from "./pages/BackupPage"
import MonitoringPage from "./pages/MonitoringPage"

function App() {
  return (
    <Routes>
      <Route element={<DashboardLayout />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/collections" element={<CollectionsPage />} />
        <Route path="/nodes" element={<NodesPage />} />
        <Route path="/explorer" element={<DataExplorerPage />} />
        <Route path="/graph" element={<GraphExplorerPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/query" element={<QueryPage />} />
        <Route path="/algorithms" element={<AlgorithmsPage />} />
        <Route path="/agency" element={<AgencyPage />} />
        <Route path="/sleep" element={<SleepPage />} />
        <Route path="/backup" element={<BackupPage />} />
        <Route path="/monitoring" element={<MonitoringPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App

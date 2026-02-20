import { Routes, Route, Navigate } from "react-router-dom"
import { DashboardLayout } from "./layouts/DashboardLayout"
import { OverviewPage } from "./pages/OverviewPage"
import { CollectionsPage } from "./pages/CollectionsPage"
import { DataExplorerPage } from "./pages/DataExplorerPage"
import { NodesPage } from "./pages/NodesPage"
import { SettingsPage } from "./pages/SettingsPage"
import { GraphExplorerPage } from "./pages/GraphExplorerPage"

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
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App

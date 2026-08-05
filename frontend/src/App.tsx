import SimulatorPage from "./features/simulator/SimulatorPage";
import { TrainingSocketProvider } from "./hooks/useTrainingSocket";

export default function App() {
  return (
    <TrainingSocketProvider>
      <SimulatorPage />
    </TrainingSocketProvider>
  );
}
